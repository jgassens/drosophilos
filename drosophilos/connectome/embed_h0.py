"""Stage H0: embed a minimal circuit in real MCNS wiring under Profile 2 rules.

Circuit (dual-rail, one bit each for a and b):
    input buffer / register : four 2-neuron excitatory loops (rails a1, a0, b1, b0);
                              a circulating spike holds the value
    arithmetic              : dual-rail AND (the carry of a half adder):
                              y1 = a1 AND b1 (threshold gate, two sub-threshold inputs)
                              y0 = a0 OR  b0 (threshold gate, each input supra-threshold)
    completion detector     : E fires on y1 or y0 (word complete, exactly one rail)
    return path             : E -> inhibitory neuron(s) F -> loop members: RESET, then READY

Profile 2: no added or rerouted edges. Allowed: scale the weight of an existing edge
within 0 <= q <= k_max * count * 16 (never flip sign), zero documented parasitic edges.
Every neuron here is a real MCNS bodyId; every edge is an anatomical edge.
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from ..sim.model import QUANTA_PER_SYNAPSE, Params
from .mcns import MCNS


@dataclass(frozen=True)
class Policy:
    profile: int = 2
    k_max: float = 4.0  # 0 <= q <= k_max * count * QUANTA_PER_SYNAPSE
    loop_margin: float = 1.4  # loop regeneration drive = margin * single-pulse need
    and_fraction: float = 0.65  # each AND input = fraction * sustained-rate need (two > need, one < need)
    or_margin: float = 2.0  # OR / completion / reset-drive inputs = margin * sustained-rate need
    reset_factor: float = 1.5  # |reset| per hit member = factor * loop drive; measured: a single
    #   inhibitory spike of 1.5x on BOTH members stops the loop at every phase (one member needs
    #   3x and still misses phases; 4-pulse trains need only 1x on one member)
    reset_both_members: bool = True
    zero_parasitic: bool = True  # zero anatomical edges among circuit neurons that are not designed
    excitatory_nt: tuple[str, ...] = ("acetylcholine",)
    inhibitory_nt: tuple[str, ...] = ("gaba", "glutamate")

    def req_count(self, q: float) -> int:
        """Smallest anatomical synapse count whose bounded weight can reach |q| quanta."""
        return int(math.ceil(abs(q) / (self.k_max * QUANTA_PER_SYNAPSE)))


def peak_response(params: Params, horizon_ms: float = 60.0) -> tuple[float, float]:
    """(peak V excursion per mV of g jump, time-to-peak ms) under the discrete update."""
    a, c, k = params.constants()
    V, g = 0.0, 1.0
    best, t_best = 0.0, 0.0
    for s in range(int(horizon_ms / params.dt)):
        V = V * a + g * k
        g = g * c
        if V > best:
            best, t_best = V, (s + 1) * params.dt
    return best, t_best


def needed_quanta(params: Params) -> float:
    """Quanta of a single synchronous input that just reaches threshold at its peak."""
    peak, _ = peak_response(params)
    gap = params.V_th - params.E_L
    return gap / peak / params.w_unit


def loop_period_steps(params: Params, loop_quanta: int) -> tuple[int, float]:
    """Simulate one isolated 2-neuron loop; return (steady period in steps, first-spike latency ms)."""
    from .._loop_probe import probe_loop  # local import to avoid a sim<->connectome import cycle at load

    return probe_loop(params, loop_quanta)


@dataclass
class Targets:
    needed_single: float  # quanta of one synchronous pulse that just reaches threshold
    needed_rate: float  # quanta per loop pulse whose sustained train just reaches threshold
    period_steps: int
    loop: int
    and_in: int
    or_in: int
    completion: int  # single-pulse: C->E, D->E and E->F each fire their target alone
    reset: int  # negative, per hit member

    @classmethod
    def from_policy(cls, params: Params, policy: Policy) -> "Targets":
        nq = needed_quanta(params)
        loop = int(math.ceil(policy.loop_margin * nq))
        period, _ = loop_period_steps(params, loop)
        # mean drive of a periodic train: g_mean = q * w_unit * tau_s / P ; V settles at E_L + g_mean
        gap = params.V_th - params.E_L
        needed_rate = gap * (period * params.dt) / (params.w_unit * params.tau_s)
        return cls(
            needed_single=nq,
            needed_rate=needed_rate,
            period_steps=period,
            loop=loop,
            and_in=int(math.ceil(policy.and_fraction * needed_rate)),
            or_in=int(math.ceil(policy.or_margin * needed_rate)),
            # gate outputs fire slowly (an AND barely above threshold fires ~30 Hz), so the
            # completion and reset-drive edges must work in single-pulse mode
            completion=loop,
            reset=-int(math.ceil(policy.reset_factor * loop)),
        )


@dataclass
class DesignedEdge:
    pre: int  # neuron index
    post: int
    count: int  # anatomical synapses
    quanta: int  # bounded target
    role: str


@dataclass
class Embedding:
    loops: dict[str, tuple[int, int]]  # rail -> (driver member, partner member)
    gate_and: int
    gate_or: int
    completion: int
    resets: dict[str, tuple[list[int], list[int]]]  # rail -> (inhibitory neurons, members hit)
    edges: list[DesignedEdge]
    parasitic: list[DesignedEdge] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def neurons(self) -> list[int]:
        s = set()
        for u, v in self.loops.values():
            s.update((u, v))
        s.update((self.gate_and, self.gate_or, self.completion))
        for fs, _ in self.resets.values():
            s.update(fs)
        return sorted(s)

    def roles(self) -> dict[int, str]:
        r = {}
        for rail, (u, v) in self.loops.items():
            r[u] = f"loop_{rail}_driver"
            r[v] = f"loop_{rail}_partner"
        r[self.gate_and] = "and_y1"
        r[self.gate_or] = "or_y0"
        r[self.completion] = "completion"
        for rail, (fs, _) in self.resets.items():
            for k, f in enumerate(fs):
                r[f] = f"reset_{rail}_{k}"
        return r

    def min_margin(self) -> float:
        """min over designed edges of (k_max*count*16)/|q| relative headroom, as count/req."""
        return min(e.count * QUANTA_PER_SYNAPSE / abs(e.quanta) for e in self.edges)


def _count_matrix(m: MCNS) -> sp.csr_matrix:
    return sp.csr_matrix((m.count.astype(np.int32), (m.pre, m.post)), shape=(m.n, m.n))


def find_embeddings(
    m: MCNS,
    params: Params = Params(),
    policy: Policy = Policy(),
    max_solutions: int = 200,
    time_limit_s: float = 300.0,
    verbose: bool = True,
) -> tuple[list[Embedding], dict]:
    """Enumerate complete Profile 2 embeddings of the H0 circuit.

    Reset rule: for each loop, every member must receive at least |reset| quanta of
    bounded inhibition from inhibitory neurons that E drives with >= completion quanta;
    several inhibitory neurons may share one member (their count is the relay overhead).
    """
    t0 = time.time()
    tg = Targets.from_policy(params, policy)
    c_loop, c_and, c_or = policy.req_count(tg.loop), policy.req_count(tg.and_in), policy.req_count(tg.or_in)
    c_e = policy.req_count(tg.completion)
    cap = policy.k_max * QUANTA_PER_SYNAPSE  # quanta available per anatomical synapse
    nt = m.neurons["nt"].to_numpy()
    exc = np.isin(nt, policy.excitatory_nt)
    inh = np.isin(nt, policy.inhibitory_nt)
    n = m.n
    C = _count_matrix(m)
    CT = C.tocsc()

    # 1. loops: mutual excitatory pairs with both directions >= c_loop
    keep = (m.count >= c_loop) & exc[m.pre] & exc[m.post]
    A = sp.csr_matrix((np.ones(keep.sum(), np.int8), (m.pre[keep], m.post[keep])), shape=(n, n))
    M = sp.triu(A.multiply(A.T), k=1).tocoo()
    loops = list(zip(M.row.tolist(), M.col.tolist()))
    n_loops = len(loops)
    loop_of: dict[int, list[int]] = {}
    for li, (u, v) in enumerate(loops):
        loop_of.setdefault(u, []).append(li)
        loop_of.setdefault(v, []).append(li)

    # 2. reset feasibility, vectorised:
    #    EF[E, f] = 1 if E drives inhibitory f with >= c_e synapses
    #    Hm[f, x] = available inhibitory quanta from f onto neuron x (capped bound)
    keep = (m.count >= c_e) & exc[m.pre] & inh[m.post]
    EF = sp.csr_matrix((np.ones(keep.sum(), np.int8), (m.pre[keep], m.post[keep])), shape=(n, n))
    keep = inh[m.pre]
    Hq = sp.csr_matrix((np.minimum(m.count[keep] * cap, abs(tg.reset) * 4).astype(np.float64),
                        (m.pre[keep], m.post[keep])), shape=(n, n))
    # per (E, x): total inhibitory quanta E can bring onto x through its inhibitory partners
    EX = (EF.astype(np.float64) @ Hq).tocsr()  # n x n, sparse
    Hq_csc = Hq.tocsc()  # column access: inhibitory inputs of a member, with row indices
    # loop li is resettable from E iff EX[E,u] >= |reset| and EX[E,v] >= |reset|
    members = np.array(loops, dtype=np.int64)  # (n_loops, 2)

    stats = {
        "targets": tg.__dict__,
        "required_counts": {"loop": c_loop, "and_in": c_and, "or_in": c_or, "completion": c_e,
                            "reset_quanta_per_member": abs(tg.reset)},
        "n_loops": n_loops,
    }
    if verbose:
        print(f"[h0] targets {tg}; required counts loop>={c_loop} and>={c_and} or>={c_or} E>={c_e}; "
              f"reset >= {abs(tg.reset)} quanta per member via E-driven inhibitory neurons")
        print(f"[h0] {n_loops} candidate loops (mutual excitatory pairs >= {c_loop})")

    def strong_inputs(x: int, thr: int):
        col = CT.getcol(x)
        keep = col.data >= thr
        return col.indices[keep], col.data[keep]

    def loop_inputs(x: int, thr: int) -> dict[int, tuple[int, int]]:
        js, cs = strong_inputs(x, thr)
        out: dict[int, tuple[int, int]] = {}
        for j, cnt in zip(js.tolist(), cs.tolist()):
            if not exc[j]:
                continue
            for li in loop_of.get(j, ()):
                if li not in out or cnt > out[li][1]:
                    out[li] = (j, cnt)
        return out

    indeg_and = np.bincount(m.post[(m.count >= c_and) & exc[m.pre]], minlength=n)
    indeg_e = np.bincount(m.post[(m.count >= c_e) & exc[m.pre]], minlength=n)
    E_candidates = np.nonzero(exc & (indeg_e >= 2))[0]
    stats["n_E_candidates"] = int(len(E_candidates))
    if verbose:
        print(f"[h0] {len(E_candidates)} completion candidates (excitatory, >=2 excitatory inputs >= {c_e})")

    cache_and: dict[int, dict] = {}
    cache_or: dict[int, dict] = {}
    solutions: list[Embedding] = []
    n_E_with_gates = 0
    n_E_with_resettable = 0
    n_assign = 0

    def resettable_loops(E: int) -> np.ndarray:
        row = EX.getrow(E)
        avail = np.zeros(n)
        avail[row.indices] = row.data
        a0 = avail[members[:, 0]] >= abs(tg.reset)
        a1 = avail[members[:, 1]] >= abs(tg.reset)
        ok = (a0 & a1) if policy.reset_both_members else (a0 | a1)
        return np.nonzero(ok)[0]

    def reset_set(E: int, li: int, forbidden: set) -> tuple[list[int], list[DesignedEdge]] | None:
        """inhibitory neurons driven by E that together cover both members with >= |reset|."""
        fs = EF.getrow(E).indices
        u, v = loops[li]
        chosen, edges = [], []
        targets_x = (u, v) if policy.reset_both_members else sorted((u, v), key=lambda x: -EX[E, x])[:1]
        for x in targets_x:
            need = abs(tg.reset)
            col = Hq_csc.getcol(x)
            fset = set(fs.tolist())
            cand = [(f, q) for f, q in zip(col.indices.tolist(), col.data.tolist()) if f in fset and f not in forbidden]
            cand.sort(key=lambda t: -t[1])
            for f, q in cand:
                if need <= 0:
                    break
                take = min(q, need)
                edges.append(DesignedEdge(f, x, int(C[f, x]), -int(take), "reset"))
                if f not in chosen:
                    chosen.append(f)
                need -= take
            if need > 0:
                return None
        return chosen, edges

    for E in E_candidates.tolist():
        if time.time() - t0 > time_limit_s or len(solutions) >= max_solutions:
            break
        RL = resettable_loops(E)
        if len(RL) < 4:
            continue
        n_E_with_resettable += 1
        RLs = set(RL.tolist())
        U, _ = strong_inputs(E, c_e)
        U = [j for j in U.tolist() if exc[j] and j != E]
        if len(U) < 2:
            continue
        found_for_E = False
        for Cn, Dn in itertools.permutations(U, 2):
            if indeg_and[Cn] < 2:
                continue
            if Cn not in cache_and:
                cache_and[Cn] = loop_inputs(Cn, c_and)
            LC = {li: t for li, t in cache_and[Cn].items() if li in RLs}
            if len(LC) < 2:
                continue
            if Dn not in cache_or:
                cache_or[Dn] = loop_inputs(Dn, c_or)
            LD = {li: t for li, t in cache_or[Dn].items() if li in RLs}
            if len(LD) < 2:
                continue
            n_E_with_gates += 1
            forbidden = {Cn, Dn, E}
            tries_for_pair = 0
            for a1, b1 in itertools.combinations(sorted(LC), 2):
                if tries_for_pair > 400:
                    break
                ma, mb = set(loops[a1]), set(loops[b1])
                if ma & mb or (ma | mb) & forbidden:
                    continue
                for a0, b0 in itertools.combinations(sorted(LD), 2):
                    m0, m1 = set(loops[a0]), set(loops[b0])
                    if m0 & m1 or (m0 | m1) & (ma | mb | forbidden):
                        continue
                    n_assign += 1
                    tries_for_pair += 1
                    if tries_for_pair > 400:
                        break
                    members_all = ma | mb | m0 | m1
                    used = set(forbidden) | members_all
                    resets, reset_edges = {}, []
                    ok = True
                    for rail, li in (("a1", a1), ("b1", b1), ("a0", a0), ("b0", b0)):
                        r = reset_set(E, li, used)
                        if r is None:
                            ok = False
                            break
                        fs, es = r
                        resets[rail] = (fs, es)  # inhibitory neurons may serve several latches
                    if not ok:
                        continue

                    def driver(li, gate, thr):
                        u, v = loops[li]
                        cu = C[u, gate] if C[u, gate] >= thr else -1
                        cv = C[v, gate] if C[v, gate] >= thr else -1
                        return (u, v) if cu >= cv else (v, u)

                    loops_d = {"a1": driver(a1, Cn, c_and), "b1": driver(b1, Cn, c_and),
                               "a0": driver(a0, Dn, c_or), "b0": driver(b0, Dn, c_or)}
                    edges = []
                    for rail, (u, v) in loops_d.items():
                        edges.append(DesignedEdge(u, v, int(C[u, v]), tg.loop, f"loop_{rail}"))
                        edges.append(DesignedEdge(v, u, int(C[v, u]), tg.loop, f"loop_{rail}"))
                    for rail in ("a1", "b1"):
                        u = loops_d[rail][0]
                        edges.append(DesignedEdge(u, Cn, int(C[u, Cn]), tg.and_in, f"and_{rail}"))
                    for rail in ("a0", "b0"):
                        u = loops_d[rail][0]
                        edges.append(DesignedEdge(u, Dn, int(C[u, Dn]), tg.or_in, f"or_{rail}"))
                    edges.append(DesignedEdge(Cn, E, int(C[Cn, E]), tg.completion, "completion_y1"))
                    edges.append(DesignedEdge(Dn, E, int(C[Dn, E]), tg.completion, "completion_y0"))
                    resets_out = {}
                    for rail, (fs, es) in resets.items():
                        for f in fs:
                            edges.append(DesignedEdge(E, f, int(C[E, f]), tg.completion, f"reset_drive_{rail}"))
                        for e in es:
                            edges.append(DesignedEdge(e.pre, e.post, e.count, e.quanta, f"reset_{rail}"))
                        resets_out[rail] = (fs, sorted({e.post for e in es}))
                    emb = Embedding(loops_d, Cn, Dn, E, resets_out, edges)
                    emb.parasitic = _parasitic_edges(C, emb, {(e.pre, e.post) for e in edges})
                    emb.stats = {"E": E, "min_margin": emb.min_margin(), "n_parasitic": len(emb.parasitic),
                                 "parasitic_quanta_abs": int(sum(abs(p.quanta) for p in emb.parasitic)),
                                 "n_reset_neurons": sum(len(fs) for fs, _ in resets_out.values())}
                    solutions.append(emb)
                    found_for_E = True
                    break
                if found_for_E:
                    break
            if found_for_E or len(solutions) >= max_solutions:
                break
    stats.update({
        "n_E_with_four_resettable_loops": n_E_with_resettable,
        "n_E_with_gate_pairs": n_E_with_gates,
        "n_loop_assignments_tried": n_assign,
        "n_solutions": len(solutions),
        "search_seconds": round(time.time() - t0, 1),
        "truncated": len(solutions) >= max_solutions or (time.time() - t0) > time_limit_s,
    })
    if verbose:
        print(f"[h0] {len(solutions)} full embeddings in {stats['search_seconds']} s "
              f"(E with 4+ resettable loops: {n_E_with_resettable}; with gate pairs: {n_E_with_gates}; "
              f"assignments tried: {n_assign})")
    return solutions, stats


def _parasitic_edges(C: sp.csr_matrix, emb: Embedding, designed: set) -> list[DesignedEdge]:
    ns = emb.neurons
    sub = C[ns][:, ns].tocoo()
    out = []
    for i, j, cnt in zip(sub.row.tolist(), sub.col.tolist(), sub.data.tolist()):
        pre, post = ns[i], ns[j]
        if (pre, post) in designed or pre == post:
            continue
        out.append(DesignedEdge(pre, post, int(cnt), int(cnt) * QUANTA_PER_SYNAPSE, "parasitic"))
    return out


def isolation_cost(m: MCNS, neurons: list[int]) -> dict:
    """What Profile 2 silencing would have to zero for the circuit to stop broadcasting:
    anatomical edges and synapses from circuit neurons to non-circuit neurons, and the
    reverse (inputs the surround can deliver)."""
    circ = np.zeros(m.n, dtype=bool)
    circ[np.asarray(neurons)] = True
    out = circ[m.pre] & ~circ[m.post]
    inp = ~circ[m.pre] & circ[m.post]
    return {
        "out_edges": int(out.sum()),
        "out_synapses": int(m.count[out].sum()),
        "out_edges_ge_24": int((out & (m.count >= 24)).sum()),
        "in_edges": int(inp.sum()),
        "in_synapses": int(m.count[inp].sum()),
        "in_edges_ge_24": int((inp & (m.count >= 24)).sum()),
    }


def best_embedding(solutions: list[Embedding], m: MCNS | None = None) -> Embedding:
    """Prefer robust weights (largest min margin), then the smallest isolation cost when a
    connectome is given, then fewest / weakest parasitic edges."""
    if m is not None:
        for e in solutions:
            if "out_synapses" not in e.stats:
                e.stats.update(isolation_cost(m, e.neurons))
        return max(solutions, key=lambda e: (round(e.min_margin(), 1), -e.stats["out_synapses"], -e.stats["parasitic_quanta_abs"]))
    return max(solutions, key=lambda e: (e.min_margin(), -e.stats["parasitic_quanta_abs"]))
