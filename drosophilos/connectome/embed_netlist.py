"""Stage H, first tool: place a compiled netlist onto real MCNS neurons under Profile 2 rules
(anatomical edges only, each designed synapse carried by an anatomical edge of the right
sign whose count, scaled at most k_max times, reaches the designed quanta), and report how
much of the circuit the fly's wiring can carry.

Greedy with forward checking, no backtracking: neurons are placed most-constrained first
(latch pairs, then whoever has the most placed neighbours); a neuron's candidates are the
real neurons adjacent, with enough synapses and the right sign, to every placed neighbour.
A neuron with no candidate stays unplaced and its edges count as missing — the measure of
what would need Profile 3 additions. H0 placed a hand-designed 15-neuron circuit this way
by hand; this is the same rule set applied to a netlist of hundreds or thousands.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from ..lib.netlist import Netlist
from .embed_h0 import Policy
from .mcns import MCNS


@dataclass
class Placement:
    mapping: dict  # designed neuron -> real neuron
    unplaced: list
    carried: int  # designed edges carried by an anatomical edge
    missing: list  # (src, dst, quanta, reason)
    parasitic: int  # anatomical edges among the placed neurons that are not designed (to zero)
    seconds: float
    order_note: str = ""
    by_role: dict = field(default_factory=dict)

    def summary(self, net: Netlist) -> dict:
        return {"neurons": net.n, "placed": len(self.mapping), "unplaced": len(self.unplaced), "edges": net.nnz,
                "carried": self.carried, "missing": len(self.missing), "carried_fraction": round(self.carried / max(1, net.nnz), 4),
                "parasitic_to_zero": self.parasitic, "seconds": round(self.seconds, 1)}


def _count_matrix(m: MCNS):
    C = sp.csr_matrix((m.count.astype(np.int32), (m.pre, m.post)), shape=(m.n, m.n))
    return C, C.tocsc()


def place_netlist(net: Netlist, m: MCNS, policy: Policy = Policy(), verbose: bool = False, time_limit_s: float = 600.0,
                  lookahead: int = 1) -> Placement:  # lookahead did not help on the adder (7 % at 64 vs 10 % greedy): the misses are structural
    t0 = time.time()
    n = net.n
    src, dst, q = np.asarray(net.src), np.asarray(net.dst), np.asarray(net.quanta)
    req = np.array([policy.req_count(x) for x in q], dtype=np.int32)
    nt = m.neurons["nt"].to_numpy()
    exc_mask = np.isin(nt, policy.excitatory_nt)
    inh_mask = np.isin(nt, policy.inhibitory_nt)
    C, CT = _count_matrix(m)
    # designed neuron signs (Dale's law)
    sign = np.zeros(n, np.int8)
    for s_, x in zip(src, q):
        sg = 1 if x > 0 else -1
        if sign[s_] == 0:
            sign[s_] = sg
        elif sign[s_] != sg:
            sign[s_] = 2  # mixed: unplaceable as one neuron
    out_edges = [[] for _ in range(n)]  # (dst, req)
    in_edges = [[] for _ in range(n)]  # (src, req)
    for e, (s_, d_) in enumerate(zip(src, dst)):
        out_edges[s_].append((d_, req[e]))
        in_edges[d_].append((s_, req[e]))
    # latch pairs: mutual edges
    pair_of = {}
    for s_, d_ in zip(src, dst):
        if s_ < d_ and any(x == s_ for x, _ in out_edges[d_]):
            pair_of[s_] = d_; pair_of[d_] = s_
    used = np.zeros(m.n, bool)
    mapping: dict[int, int] = {}
    unplaced: list[int] = []
    degree = np.array([len(out_edges[i]) + len(in_edges[i]) for i in range(n)])

    def sign_ok(cands, d):
        if sign[d] == 1:
            return cands[exc_mask[cands]]
        if sign[d] == -1:
            return cands[inh_mask[cands]]
        if sign[d] == 2:
            return cands[:0]
        return cands  # no outputs: any neuron

    def candidates(d):
        """Real neurons adjacent, strongly enough and with the right signs, to every placed neighbour of d."""
        cand = None
        for p, r in in_edges[d]:  # p -> d
            if p in mapping:
                row = C.getrow(mapping[p])
                ok = row.indices[row.data >= r]
                cand = ok if cand is None else np.intersect1d(cand, ok, assume_unique=False)
                if cand.size == 0:
                    return cand
        for p, r in out_edges[d]:  # d -> p
            if p in mapping:
                col = CT.getcol(mapping[p])
                ok = col.indices[col.data >= r]
                cand = ok if cand is None else np.intersect1d(cand, ok, assume_unique=False)
                if cand.size == 0:
                    return cand
        if cand is None:
            return None
        cand = cand[~used[cand]]
        return sign_ok(cand, d)

    def free_start(d):
        """A first neuron with no placed neighbour: a latch pair's members need a strong mutual partner."""
        if d in pair_of:
            r = max(rr for x, rr in out_edges[d] if x == pair_of[d])
            r2 = max(rr for x, rr in out_edges[pair_of[d]] if x == d)
            keep = (m.count >= min(r, r2)) & exc_mask[m.pre] & exc_mask[m.post] & ~used[m.pre] & ~used[m.post]
            A = sp.csr_matrix((np.ones(keep.sum(), np.int8), (m.pre[keep], m.post[keep])), shape=(m.n, m.n))
            M = sp.triu(A.multiply(A.T), k=1).tocoo()
            if M.nnz == 0:
                return None
            k = np.random.default_rng(len(mapping)).integers(0, M.nnz)
            return int(M.row[k])
        pool = np.flatnonzero((exc_mask if sign[d] == 1 else inh_mask if sign[d] == -1 else np.ones(m.n, bool)) & ~used)
        return int(pool[np.random.default_rng(len(mapping)).integers(0, pool.size)]) if pool.size else None

    # order: a heap of (-placed neighbours, -degree, id), updated as neighbours get placed
    placed_nb = np.zeros(n, np.int32)
    heap = [(0, -int(degree[i]), i) for i in range(n)]
    heapq.heapify(heap)
    done = np.zeros(n, bool)
    while heap:
        negp, negd, d = heapq.heappop(heap)
        if done[d]:
            continue
        if -negp != placed_nb[d]:  # stale entry
            heapq.heappush(heap, (-int(placed_nb[d]), negd, d))
            continue
        done[d] = True
        if time.time() - t0 > time_limit_s:
            unplaced.append(d)
            continue
        cand = candidates(d)
        if cand is None:
            r = free_start(d)
        elif cand.size == 0:
            r = None
        else:  # one-step lookahead: the candidate that leaves the most unplaced neighbours a compatible partner
            r = int(cand[0])
            if cand.size > 1:
                cand = cand[:lookahead] if cand.size > lookahead else cand
                best, r = -1, int(cand[0])
                nbrs_out = [(x, rr) for x, rr in out_edges[d] if x not in mapping]  # d -> x needs C[r, f(x)] >= rr
                nbrs_in = [(x, rr) for x, rr in in_edges[d] if x not in mapping]  # x -> d needs C[f(x), r] >= rr
                for c_ in cand.tolist():
                    row = C.getrow(c_); col = CT.getcol(c_)
                    score = 0
                    for x, rr in nbrs_out:
                        ok = row.indices[(row.data >= rr) & ~used[row.indices]]
                        score += int(sign_ok(ok, x).size > 0)
                    for x, rr in nbrs_in:
                        ok = col.indices[(col.data >= rr) & ~used[col.indices]]
                        score += int(sign_ok(ok, x).size > 0)
                    if score > best:
                        best, r = score, int(c_)
        if r is None:
            unplaced.append(d)
        else:
            mapping[d] = r
            used[r] = True
            for p, _ in in_edges[d] + out_edges[d]:
                if not done[p]:
                    placed_nb[p] += 1
                    heapq.heappush(heap, (-int(placed_nb[p]), -int(degree[p]), p))
        if verbose and len(mapping) % 200 == 0:
            print(f"  placed {len(mapping)} unplaced {len(unplaced)} in {time.time() - t0:.0f} s", flush=True)
    # audit
    carried, missing = 0, []
    for e, (s_, d_) in enumerate(zip(src, dst)):
        if s_ not in mapping or d_ not in mapping:
            missing.append((int(s_), int(d_), int(q[e]), "endpoint unplaced"))
            continue
        c = C[mapping[s_], mapping[d_]]
        if c >= req[e] and ((q[e] > 0 and exc_mask[mapping[s_]]) or (q[e] < 0 and inh_mask[mapping[s_]])):
            carried += 1
        else:
            missing.append((int(s_), int(d_), int(q[e]), "weak or wrong-sign edge"))
    placed = np.array(sorted(mapping.values()))
    sub = C[placed][:, placed]
    parasitic = int(sub.nnz - carried)
    pl = Placement(mapping, unplaced, carried, missing, parasitic, time.time() - t0)
    roles = net.roles
    by_role: dict = {}
    for d in range(n):
        key = roles[d].split(".")[-1] if "." in roles[d] else roles[d]
        key = "".join(ch for ch in key if not ch.isdigit())
        by_role.setdefault(key, [0, 0])
        by_role[key][0 if d in mapping else 1] += 1
    pl.by_role = by_role
    return pl
