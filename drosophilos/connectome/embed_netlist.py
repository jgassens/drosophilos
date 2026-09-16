"""Stage H: place a compiled netlist onto real MCNS neurons under Profile 2 rules (anatomical
edges only; each designed synapse carried by an anatomical edge of the right sign whose
count, scaled at most k_max times, reaches the designed quanta) and report how much of the
circuit the fly's wiring can carry. Results on the 4-bit adder: docs/h1_placement.md.

Motif search (`place_netlist`), a maximum-coverage search: every designed edge carried is a
point, and a motif that cannot be placed whole is placed as well as it can be rather than
dropped.

1. Motifs are read off the netlist's structure: latches (mutual excitatory pairs), flip-flop
   pairs (mutual inhibitory pairs whose members carry a tonic bias, protocol.flipflop), flip-flop
   triples (a flip-flop pair plus its excitatory proxy p: a biased excitatory neuron whose only
   designed input is v's inhibition, add_flipflop(proxy=True); p's outputs are ordinary
   excitatory edges the search carries), relays
   (an excitatory relay plus the inhibitory interneuron(s) that share its source and hold it
   down), delay chains (maximal paths of in-degree-one excitatory neurons), hubs (neurons
   whose out-degree or in-degree exceeds `hub_deg` in that direction; their many edges are
   soft: scored, not enforced, since no single real neuron carries them) and singles.
2. Every designed neuron gets two candidate sets of real neurons. The loose layer: the right
   transmitter, enough strong partners, and membership in a real instance of its own motif
   (a mutual pair -- for a flip-flop a mutual inhibitory pair whose members each have an
   excitatory driver, for a flip-flop triple such a pair plus an excitatory neuron the member
   hosting v inhibits at the proxy's threshold -- a relay with an inhibitor, a walk long enough
   for its chain). The strict
   layer adds the context (a source of k relays needs k real relays; a chain's predecessor a
   walk one edge longer) and arc consistency over the hard edges.
3. Depth-first search over motifs, most constrained first (fewest strict candidates), each
   motif assigned as a unit: a latch or flip-flop as a real mutual pair, a flip-flop triple as a
   real (pair, proxy), a relay as a real (E, I) pair, a
   chain as a real path found by depth-first search over strong cholinergic edges, a hub by
   coverage of its partners' candidates. Candidates are ranked by a lookahead (does every
   neighbouring motif keep a complete instance? plus hub edges satisfied, minus an isolation
   penalty: `isolation_weight` times the log of each host's anatomical input synapses, plus a
   quarter of the log of its outputs -- the exposure the whole-brain run showed floods a host
   chosen for coverage alone, docs/h1_placement.md). Assigning a motif forward-checks every
   unplaced hard neighbour; a candidate that empties a domain is rejected while another exists. A motif with no complete instance
   triggers bounded chronological backtracking; if that fails it is placed from the loose
   layer (its own edges kept, its neighbours' needs dropped) or, node by node, where it
   carries the most edges, as a soft anchor that prunes nobody's domain.
4. Repair: each motif is taken out and put back where it carries the most edges given
   everything else placed, until nothing improves.
5. Restarts (hubs last / hubs first, varied tie-breaking) inside the time limit; the placement
   carrying the most designed edges is returned, with an audit of every designed edge,
   the parasitic anatomical edges among the hosts (to zero), motif outcomes and the missing
   edges by class.

`hub_split_analysis` reports what splitting a hub into several real neurons would take;
`place_netlist_greedy` is the first tool (neuron by neuron, no backtracking; 10 % of the
adder), kept as the baseline.
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
    carried: int  # distinct designed (src, dst) pairs carried by an anatomical edge
    missing: list  # (src, dst, quanta, reason) per distinct missing pair
    parasitic: int  # anatomical edges among the placed neurons that are not designed (to zero)
    seconds: float
    edges: int = 0  # distinct designed (src, dst) pairs (duplicate netlist synapses on the same pair merged)
    order_note: str = ""
    by_role: dict = field(default_factory=dict)
    motifs: dict = field(default_factory=dict)  # kind -> {"complete": n, "partial": n, "unplaced": n} by what is carried
    motifs_search: dict = field(default_factory=dict)  # kind -> how the search placed them (complete / partial fallback / skipped)
    missing_by_class: dict = field(default_factory=dict)  # (src role, dst role) -> [missing, total]
    strategy: str = ""
    isolation_weight: float = 0.0
    exposure: dict = field(default_factory=dict)  # the hosts' anatomical synapses from / to neurons outside the circuit (exact)
    objective: float = 0.0  # carried - isolation_weight * sum over hosts of (log1p(in_ext) + 0.25 * log1p(out_ext)) - cap * unplaced
    ffpair_drivers: dict = field(default_factory=dict)  # flip-flop hosts: {"hosts": placed members, "with_driver": those with an excitatory input >= the loop threshold}
    proxy_readout: dict = field(default_factory=dict)  # flip-flop proxies: {"proxies": designed, "placed": on a host, "edges": designed edges out of a proxy, "carried": of those carried}

    def summary(self, net: Netlist) -> dict:
        ex = self.exposure
        return {"neurons": net.n, "placed": len(self.mapping), "unplaced": len(self.unplaced),
                "edges": self.edges, "synapses": net.nnz,
                "carried": self.carried, "missing": len(self.missing), "carried_fraction": round(self.carried / max(1, self.edges), 4),
                "parasitic_to_zero": self.parasitic, "seconds": round(self.seconds, 1), "strategy": self.strategy,
                "isolation_weight": self.isolation_weight, "objective": round(self.objective, 3),
                "external_in_edges": ex.get("in_edges", 0), "external_out_edges": ex.get("out_edges", 0),
                "external_in_synapses": ex.get("in_total", 0), "external_out_synapses": ex.get("out_total", 0),
                "external_in_mean": ex.get("in_mean", 0.0), "external_out_mean": ex.get("out_mean", 0.0),
                "external_in_max": ex.get("in_max", 0), "external_out_max": ex.get("out_max", 0),
                "external_in_p90": ex.get("in_p90", 0.0), "external_out_p90": ex.get("out_p90", 0.0),
                "ffpair_hosts": self.ffpair_drivers.get("hosts", 0), "ffpair_hosts_with_driver": self.ffpair_drivers.get("with_driver", 0),
                "proxies": self.proxy_readout.get("proxies", 0), "proxies_placed": self.proxy_readout.get("placed", 0),
                "proxy_readout_edges": self.proxy_readout.get("edges", 0), "proxy_readout_carried": self.proxy_readout.get("carried", 0)}


OUT_EXPOSURE_WEIGHT = 0.25  # outputs into the surround count a quarter of inputs from it (inputs are what floods a host)


def isolation_penalty(A: "_Anatomy", isolation_weight: float) -> np.ndarray:
    """Per real neuron: isolation_weight * (log1p(input synapses) + 0.25 * log1p(output synapses)),
    from the static anatomical totals (the circuit set is not known while the placement grows;
    the correction for edges among hosts is small and the audit reports the exact figure). The
    natural log makes isolation_weight = 1 mean that one carried designed edge (+1 in every
    ranking score) is worth a factor e in a host's input synapse count."""
    if isolation_weight == 0.0:
        return np.zeros(A.n, np.float64)
    return isolation_weight * (np.log1p(A.in_syn) + OUT_EXPOSURE_WEIGHT * np.log1p(A.out_syn))


def penalty_cap(A: "_Anatomy", isolation_weight: float) -> float:
    """The penalty charged for a designed neuron left unplaced: the noisiest real neuron's. The
    weight ranks hosts; it must never make "no host" the cheapest choice (an unplaced neuron
    becomes a synthetic one in the image, which is worse than any host), so every comparison
    between assignments that place different nodes charges the missing ones at this cap."""
    if isolation_weight == 0.0:
        return 0.0
    return isolation_weight * (np.log1p(A.in_syn.max()) + OUT_EXPOSURE_WEIGHT * np.log1p(A.out_syn.max()))


def exposure_of(A: "_Anatomy", hosts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact per-host (external input synapses, external output synapses, external input edges,
    external output edges): the anatomical totals minus what lies among the hosts themselves
    (the same count as embed_h0.isolation_cost)."""
    hosts = np.asarray(hosts, dtype=np.int64)
    if hosts.size == 0:
        z = np.zeros(0, np.int64)
        return z, z, z, z
    sub = A.C[hosts][:, hosts]
    in_int = np.asarray(sub.sum(axis=0)).ravel().astype(np.int64)
    out_int = np.asarray(sub.sum(axis=1)).ravel().astype(np.int64)
    ones = sub.copy(); ones.data[:] = 1
    in_edges = np.diff(A.CT.indptr)[hosts] - np.asarray(ones.sum(axis=0)).ravel().astype(np.int64)
    out_edges = np.diff(A.C.indptr)[hosts] - np.asarray(ones.sum(axis=1)).ravel().astype(np.int64)
    return A.in_syn[hosts].astype(np.int64) - in_int, A.out_syn[hosts].astype(np.int64) - out_int, in_edges, out_edges


def exposure_summary(in_ext: np.ndarray, out_ext: np.ndarray, in_edges: np.ndarray | None = None,
                     out_edges: np.ndarray | None = None) -> dict:
    if in_ext.size == 0:
        return {"hosts": 0, "in_total": 0, "out_total": 0, "in_mean": 0.0, "out_mean": 0.0,
                "in_max": 0, "out_max": 0, "in_p90": 0.0, "out_p90": 0.0, "in_edges": 0, "out_edges": 0}
    return {"hosts": int(in_ext.size),
            "in_edges": int(in_edges.sum()) if in_edges is not None else 0,  # what the whole-brain control zeroes
            "out_edges": int(out_edges.sum()) if out_edges is not None else 0,
            "in_total": int(in_ext.sum()), "out_total": int(out_ext.sum()),
            "in_mean": round(float(in_ext.mean()), 1), "out_mean": round(float(out_ext.mean()), 1),
            "in_max": int(in_ext.max()), "out_max": int(out_ext.max()),
            "in_p90": round(float(np.percentile(in_ext, 90)), 1), "out_p90": round(float(np.percentile(out_ext, 90)), 1)}


def role_key(role: str) -> str:
    key = role.split(".")[-1] if "." in role else role
    return "".join(ch for ch in key if not ch.isdigit())


def _count_matrix(m: MCNS):
    C = sp.csr_matrix((m.count.astype(np.int32), (m.pre, m.post)), shape=(m.n, m.n))
    C.sum_duplicates()
    C.sort_indices()
    CT = C.tocsc()
    CT.sort_indices()
    return C, CT


# ----------------------------------------------------------------------------------------
# anatomy: thresholded views of the connectome and the masks of its motif instances
# ----------------------------------------------------------------------------------------
class _Anatomy:
    def __init__(self, m: MCNS, policy: Policy):
        self.m, self.n = m, m.n
        self.C, self.CT = _count_matrix(m)
        nt = m.neurons["nt"].to_numpy()
        self.exc = np.isin(nt, policy.excitatory_nt)
        self.inh = np.isin(nt, policy.inhibitory_nt)
        self._A: dict = {}
        self._deg: dict = {}
        self._walk: dict = {}
        self._memo: dict = {}
        self.outdeg24 = np.bincount(m.pre[m.count >= 24], minlength=self.n)  # isolation cost proxy (H0: >= 24 fires alone)
        # exposure: every neuron's anatomical in-degree and out-degree in synapses (all partners)
        self.in_syn = np.bincount(m.post, weights=m.count, minlength=self.n).astype(np.int64)
        self.out_syn = np.bincount(m.pre, weights=m.count, minlength=self.n).astype(np.int64)

    def row(self, r: int, thr: int) -> np.ndarray:
        a, b = self.C.indptr[r], self.C.indptr[r + 1]
        idx, dat = self.C.indices[a:b], self.C.data[a:b]
        return idx[dat >= thr]

    def col(self, r: int, thr: int) -> np.ndarray:
        a, b = self.CT.indptr[r], self.CT.indptr[r + 1]
        idx, dat = self.CT.indices[a:b], self.CT.data[a:b]
        return idx[dat >= thr]

    def count(self, r: int, s: int) -> int:
        a, b = self.C.indptr[r], self.C.indptr[r + 1]
        k = np.searchsorted(self.C.indices[a:b], s)
        return int(self.C.data[a + k]) if k < b - a and self.C.indices[a + k] == s else 0

    def A(self, thr: int, pre: str | None = None, post: str | None = None):
        """Binary csr of edges with count >= thr, optionally restricted to excitatory ("exc") or
        inhibitory ("inh") pre / post neurons, and its transpose."""
        key = (thr, pre, post)
        if key not in self._A:
            m = self.m
            keep = m.count >= thr
            if pre is not None:
                keep &= (self.exc if pre == "exc" else self.inh)[m.pre]
            if post is not None:
                keep &= (self.exc if post == "exc" else self.inh)[m.post]
            A = sp.csr_matrix((np.ones(int(keep.sum()), np.int8), (m.pre[keep], m.post[keep])), shape=(self.n, self.n))
            self._A[key] = (A, A.T.tocsr())
        return self._A[key]

    def outdeg(self, thr: int) -> np.ndarray:
        if ("out", thr) not in self._deg:
            self._deg[("out", thr)] = np.bincount(self.m.pre[self.m.count >= thr], minlength=self.n)
        return self._deg[("out", thr)]

    def indeg(self, thr: int, sign: int) -> np.ndarray:
        if ("in", thr, sign) not in self._deg:
            keep = (self.m.count >= thr) & ((self.exc if sign > 0 else self.inh)[self.m.pre])
            self._deg[("in", thr, sign)] = np.bincount(self.m.post[keep], minlength=self.n)
        return self._deg[("in", thr, sign)]

    def mutual(self, thr: int):
        """(mask of excitatory neurons in a mutual excitatory pair >= thr both ways, the pairs)."""
        if ("mutual", thr) not in self._memo:
            A, _ = self.A(thr, "exc", "exc")
            M = sp.triu(A.multiply(A.T), k=1).tocoo()
            mask = np.zeros(self.n, bool)
            mask[M.row] = True
            mask[M.col] = True
            self._memo[("mutual", thr)] = (mask, np.stack([M.row, M.col], 1).astype(np.int32))
        return self._memo[("mutual", thr)]

    def mutual_inh(self, thr: int, driven: bool = True):
        """(mask of inhibitory neurons in a mutual inhibitory pair >= thr both ways, the pairs):
        the flip-flop's hosts. With `driven`, only pairs whose members each have at least one
        excitatory input >= thr (from outside the pair: the partner is inhibitory, so any such
        input is), the driver a tonic bias stands in for -- bench/h1_inhpairs.py's count."""
        key = ("mutual_inh", thr, driven)
        if key not in self._memo:
            A, _ = self.A(thr, "inh", "inh")
            M = sp.triu(A.multiply(A.T), k=1).tocoo()
            pairs = np.stack([M.row, M.col], 1).astype(np.int32)
            if driven and len(pairs):
                drv = self.indeg(thr, 1) > 0
                pairs = pairs[drv[pairs[:, 0]] & drv[pairs[:, 1]]]
            mask = np.zeros(self.n, bool)
            mask[pairs.ravel()] = True
            self._memo[key] = (mask, pairs)
        return self._memo[key]

    def proxy_hosts(self, thr: int, r_vp: int, driven: bool = True):
        """(mask of pair members that inhibit some excitatory neuron >= r_vp, mask of excitatory
        neurons so inhibited by a member): the flip-flop triple's hosts for v and p, the pair
        drawn from mutual_inh(thr, driven) -- bench/h1_inhpairs.py's loose proxy_candidates. The
        strict variant there (a proxy with an excitatory output >= thr) is what p's own designed
        outputs impose through the degree masks, so it is not repeated here."""
        key = ("proxy_hosts", thr, r_vp, driven)
        if key not in self._memo:
            mm, _ = self.mutual_inh(thr, driven)
            VP, _ = self.A(r_vp, "inh", "exc")  # member -> excitatory target
            members = np.flatnonzero(mm)
            sub = VP[members]
            has_proxy = np.zeros(self.n, bool)
            has_proxy[members[np.diff(sub.indptr) > 0]] = True
            proxy = np.zeros(self.n, bool)
            proxy[np.unique(sub.indices)] = True
            self._memo[key] = (has_proxy, proxy)
        return self._memo[key]

    def relay(self, r_se: int, r_si: int, r_ie: int):
        """Masks (S, E, I) of the real source/relay/inhibitor triples: S -> E >= r_se (exc -> exc),
        S -> I >= r_si (exc -> inh), I -> E >= r_ie (inh -> exc)."""
        key = ("relay", r_se, r_si, r_ie)
        if key not in self._memo:
            SE, _ = self.A(r_se, "exc", "exc")
            SI, _ = self.A(r_si, "exc", "inh")
            IE, _ = self.A(r_ie, "inh", "exc")
            SIE = (SI.astype(np.int32) @ IE.astype(np.int32)).tocsr()
            T = SE.multiply(SIE > 0).tocsr()  # (S, E) pairs with some I
            T.eliminate_zeros()
            s_count = np.diff(T.indptr).astype(np.int32)  # distinct relays each source can drive
            T = T.tocoo()
            s_mask = s_count > 0
            e_mask = np.zeros(self.n, bool); e_mask[T.col] = True
            # I: some S with S->I and some E with I->E and S->E
            SIT = SI.T.tocsr().astype(np.int32)  # I x S
            IS_E = (SIT @ SE.astype(np.int32)).tocsr()  # I x E via S
            TI = IE.multiply(IS_E > 0).tocoo()
            i_mask = np.zeros(self.n, bool); i_mask[TI.row] = True
            self._memo[key] = (s_mask, e_mask, i_mask)
            self._memo[key + ("count",)] = s_count
        return self._memo[key]

    def relay_count(self, r_se: int, r_si: int, r_ie: int) -> np.ndarray:
        self.relay(r_se, r_si, r_ie)
        return self._memo[("relay", r_se, r_si, r_ie, "count")]

    def walk(self, thr: int, k: int, forward: bool) -> np.ndarray:
        """Excitatory neurons from which (forward) or to which (backward) a walk of k
        excitatory edges >= thr exists; an upper bound on a simple path, used to prune."""
        key = (thr, forward)
        if key not in self._walk:
            A, AT = self.A(thr, "exc", "exc")
            self._walk[key] = [(A if forward else AT), [self.exc.copy()]]
        M, reach = self._walk[key]
        while len(reach) <= k:
            reach.append((M @ reach[-1].astype(np.int32)) > 0)
        return reach[k]


# ----------------------------------------------------------------------------------------
# design: the netlist's structure (signs, hard/soft edges, motifs)
# ----------------------------------------------------------------------------------------
@dataclass
class Motif:
    kind: str  # latch | ffpair | fftriple | relay | chain | hub | single
    nodes: tuple
    edges: list = field(default_factory=list)  # internal hard edge indices
    reqs: dict = field(default_factory=dict)  # kind-specific


class _Design:
    def __init__(self, net: Netlist, policy: Policy, hub_deg: int, hard_hubs=(), ff_driver: bool = True):
        self.net = net
        n = self.n = net.n
        self.src = np.asarray(net.src, dtype=np.int64)
        self.dst = np.asarray(net.dst, dtype=np.int64)
        self.q = np.asarray(net.quanta, dtype=np.int64)
        self.req = np.array([policy.req_count(x) for x in self.q], dtype=np.int32)
        self.esign = np.where(self.q > 0, 1, -1).astype(np.int8)
        self.bias = np.zeros(n, np.float64)  # the netlist's per-neuron tonic bias (mV); parallel to roles
        self.bias[: len(net.bias)] = np.asarray(net.bias, dtype=np.float64)[:n]
        self.ff_driver = ff_driver
        # neuron signs: majority of outgoing edges (Dale's law); a mixed neuron keeps its majority
        # sign and its minority-sign edges can never be carried. A flip-flop member (a biased
        # neuron in a mutual inhibitory pair with another biased neuron) is inhibitory whatever
        # else it drives: its loop is what it is, and an excitatory readout edge from it (an edge
        # relay reading `.u`) is the minority a GABA host cannot carry.
        pos = np.bincount(self.src, weights=(self.q > 0), minlength=n)
        neg = np.bincount(self.src, weights=(self.q < 0), minlength=n)
        self.sign = np.where(pos + neg == 0, 0, np.where(pos >= neg, 1, -1)).astype(np.int8)
        self.ff_members = self._ff_members()
        self.sign[list(self.ff_members)] = -1
        # a flip-flop proxy (biased, excitatory, its only input a member's inhibition) is
        # excitatory even when it drives nothing yet: its host must be a cholinergic neuron
        self.ff_proxy_of = self._ff_proxies()
        self.sign[list(self.ff_proxy_of)] = 1
        self.mixed = [int(d) for d in np.flatnonzero((pos > 0) & (neg > 0))]
        outdeg = np.bincount(self.src, minlength=n)
        indeg = np.bincount(self.dst, minlength=n)
        self.out_hub = outdeg > hub_deg
        self.in_hub = indeg > hub_deg
        self.hub = self.out_hub | self.in_hub
        sign_ok = self.esign == self.sign[self.src]
        self.hard = sign_ok & ~self.out_hub[self.src] & ~self.in_hub[self.dst]
        anchored = np.zeros(n, bool)  # hubs whose many edges are enforced (a hub-anchored search)
        for h in hard_hubs:
            anchored[net.roles.index(h) if isinstance(h, str) else int(h)] = True
        self.hard |= sign_ok & anchored[self.src] & self.out_hub[self.src]
        self.hard |= sign_ok & anchored[self.dst] & self.in_hub[self.dst]
        self.soft = sign_ok & ~self.hard
        self.impossible = ~sign_ok
        self.hard_out = [[] for _ in range(n)]
        self.hard_in = [[] for _ in range(n)]
        self.soft_out = [[] for _ in range(n)]
        self.soft_in = [[] for _ in range(n)]
        for e in range(len(self.src)):
            s, d = int(self.src[e]), int(self.dst[e])
            if self.hard[e]:
                self.hard_out[s].append(e); self.hard_in[d].append(e)
            elif self.soft[e]:
                self.soft_out[s].append(e); self.soft_in[d].append(e)
        self.motifs: list[Motif] = []
        self.motif_of = np.full(n, -1, np.int64)
        self._find_motifs()

    def _edge(self, s: int, d: int):
        for e in self.hard_out[s]:
            if self.dst[e] == d:
                return e
        return None

    def _ff_members(self) -> set:
        """Designed neurons that are flip-flop members: biased, with an inhibitory edge to a
        biased neuron that inhibits them back (raw edges, before the sign vote)."""
        inh_pairs = set()
        for e in np.flatnonzero((self.q < 0) & (self.bias[self.src] != 0) & (self.bias[self.dst] != 0)).tolist():
            inh_pairs.add((int(self.src[e]), int(self.dst[e])))
        members = set()
        for s, d in inh_pairs:
            if (d, s) in inh_pairs:
                members.add(s); members.add(d)
        return members

    def _ff_proxies(self) -> dict:
        """Designed proxy -> the member that inhibits it: biased neurons outside the pairs whose
        only input (raw edges) is an inhibitory synapse from a flip-flop member and whose outputs
        (if any) are all excitatory -- add_flipflop's p (inhibited by v) or q (inhibited by u)."""
        n = self.n
        indeg = np.bincount(self.dst, minlength=n)
        neg_out = np.bincount(self.src, weights=(self.q < 0), minlength=n)
        out: dict = {}
        for e in np.flatnonzero((self.q < 0) & (self.bias[self.dst] != 0)).tolist():
            s, d = int(self.src[e]), int(self.dst[e])
            if s in self.ff_members and d not in self.ff_members and indeg[d] == 1 and neg_out[d] == 0:
                out[d] = s
        return out

    def _find_motifs(self):
        n = self.n
        taken = np.zeros(n, bool)

        def add(kind, nodes, edges, reqs):
            mi = len(self.motifs)
            self.motifs.append(Motif(kind, tuple(int(x) for x in nodes), list(edges), reqs))
            for x in nodes:
                taken[x] = True
                self.motif_of[x] = mi

        for d in np.flatnonzero(self.hub):
            add("hub", (d,), [], {})
        # latches: mutual excitatory hard pairs
        for u in range(n):
            if taken[u] or self.sign[u] != 1:
                continue
            for e in self.hard_out[u]:
                v = int(self.dst[e])
                if v > u and not taken[v] and self.sign[v] == 1:
                    e2 = self._edge(v, u)
                    if e2 is not None:
                        add("latch", (u, v), [e, e2], {"r_uv": int(self.req[e]), "r_vu": int(self.req[e2])})
                        break
        # flip-flop pairs: mutual inhibitory hard pairs between biased neurons (protocol.flipflop);
        # the loop's quanta are whatever the netlist gives (the flip-flop's default is 1.0x loop),
        # the bias is what marks the pair. Placed on real mutual inhibitory pairs whose members
        # each have an excitatory driver (ff_driver), as a unit like a latch. A pair with a proxy
        # (add_flipflop(proxy=True): p, biased and excitatory, its only input v's inhibition) is
        # the triple (u, v, p), nodes ordered so that nodes[1] is the member inhibiting p; it is
        # placed as a unit on a real (driven pair, excitatory neuron the v host inhibits). The
        # SET proxy p (on v) is preferred when a pair has both p and q; the other stays a single
        # joined to its member by an ordinary hard edge.
        for u in sorted(self.ff_members):
            if taken[u]:
                continue
            for e in self.hard_out[u]:
                v = int(self.dst[e])
                if v > u and not taken[v] and v in self.ff_members and self.q[e] < 0:
                    e2 = self._edge(v, u)
                    if e2 is not None and self.q[e2] < 0:
                        proxy = None
                        for inhibitor, other, e_ov, e_vo in ((v, u, e, e2), (u, v, e2, e)):
                            for e3 in self.hard_out[inhibitor]:
                                p = int(self.dst[e3])
                                if not taken[p] and self.ff_proxy_of.get(p) == inhibitor and self.q[e3] < 0:
                                    proxy = (inhibitor, other, e_ov, e_vo, e3, p)
                                    break
                            if proxy is not None:
                                break
                        if proxy is None:
                            add("ffpair", (u, v), [e, e2], {"r_uv": int(self.req[e]), "r_vu": int(self.req[e2]),
                                                             "driver": bool(self.ff_driver),
                                                             "bias": (float(self.bias[u]), float(self.bias[v]))})
                        else:
                            inhibitor, other, e_ov, e_vo, e3, p = proxy
                            add("fftriple", (other, inhibitor, p), [e_ov, e_vo, e3],
                                {"r_uv": int(self.req[e_ov]), "r_vu": int(self.req[e_vo]), "r_vp": int(self.req[e3]),
                                 "driver": bool(self.ff_driver),
                                 "bias": (float(self.bias[other]), float(self.bias[inhibitor]), float(self.bias[p]))})
                        break
        # relays: excitatory E with inhibitory I whose only hard output is E and that shares a source with E
        for E in range(n):
            if taken[E] or self.sign[E] != 1:
                continue
            srcs = {int(self.src[e]): e for e in self.hard_in[E]}
            inhibitors = []
            for e in self.hard_in[E]:
                I = int(self.src[e])
                if self.sign[I] != -1 or taken[I] or len(self.hard_out[I]) != 1:
                    continue
                shared = [(S, es, self._edge(S, I)) for S, es in srcs.items() if self.sign[S] == 1 and self._edge(S, I) is not None]
                if shared:
                    S, e_se, e_si = shared[0]
                    inhibitors.append((I, e, int(self.req[e]), S, int(self.req[e_se]), int(self.req[e_si])))
            if inhibitors:
                add("relay", (E,) + tuple(I for I, *_ in inhibitors), [e for _, e, *_ in inhibitors],
                    {"inhibitors": inhibitors})
        # chains: maximal paths of excitatory neurons with exactly one hard (excitatory) input
        eligible = np.zeros(n, bool)
        for c in range(n):
            if taken[c] or self.sign[c] != 1 or len(self.hard_in[c]) != 1:
                continue
            p = int(self.src[self.hard_in[c][0]])
            eligible[c] = self.sign[p] == 1
        for c in np.flatnonzero(eligible):
            if taken[c]:
                continue
            p = int(self.src[self.hard_in[c][0]])
            if eligible[p] and not taken[p] and len(self.hard_out[p]) == 1:
                continue  # not a chain start
            nodes, edges = [int(c)], []
            cur = int(c)
            while len(self.hard_out[cur]) == 1:
                nxt = int(self.dst[self.hard_out[cur][0]])
                if not eligible[nxt] or taken[nxt] or nxt in nodes:
                    break
                edges.append(self.hard_out[cur][0])
                nodes.append(nxt)
                cur = nxt
            if len(nodes) >= 2:
                add("chain", nodes, edges, {"thr": int(min(self.req[e] for e in edges))})
            else:
                add("single", nodes, [], {})
        for d in range(n):
            if not taken[d]:
                add("single", (d,), [], {})


# ----------------------------------------------------------------------------------------
# the search
# ----------------------------------------------------------------------------------------
class _Search:
    KILL = 2.0  # penalty for a candidate that empties an unplaced neighbour's domain

    def __init__(self, D: _Design, A: _Anatomy, rng: np.random.Generator, hub_first: bool, cand_cap: int,
                 chain_budget: int, max_backtracks: int, backtrack_depth: int, deadline: float,
                 mac_max: int = 0, order: str = "mrv", reject_kills: bool = True, soft_w: float = 1.0,
                 isolation_weight: float = 0.0):
        self.D, self.A, self.rng = D, A, rng
        self.iso_w = isolation_weight
        self.iso_pen = isolation_penalty(A, isolation_weight)  # per real neuron; every ranking subtracts it
        self.pen_unplaced = penalty_cap(A, isolation_weight)  # a node left out is charged the noisiest neuron's penalty
        self.hub_first, self.cand_cap, self.chain_budget = hub_first, cand_cap, chain_budget
        self.max_bt, self.bt_depth, self.deadline = max_backtracks, backtrack_depth, deadline
        self.mac_max = mac_max
        self.reserve_max = 48  # chain paths keep out of neighbour domains this small
        self.order, self.reject_kills, self.soft_w = order, reject_kills, soft_w
        self.n_real = A.n
        self.real = np.full(D.n, -1, np.int64)  # designed -> real, -1 unplaced
        self.used = np.zeros(A.n, bool)
        self.onpath = np.zeros(A.n, bool)
        self.dom: list = [None] * D.n
        self.trail: list = []  # (designed node, previous domain)
        self.outcome: dict = {}  # motif index -> complete | partial | skipped
        self.backtracks = 0
        self.hub_row: dict = {}  # placed hub designed id -> dense count row/col
        self.soft_anchor: set = set()  # nodes placed by best fit: scored, never enforced
        self._static_domains()

    # ---- static domains ------------------------------------------------------------
    def _static_domains(self):
        D, A = self.D, self.A
        n = D.n
        if getattr(D, "static_dom", None) is not None:  # computed once per design, shared by restarts
            self.dom, self.dom1 = list(D.static_dom), list(D.static_dom1)
            return
        masks: list = [None] * n
        fan: dict = {}
        for d in range(n):
            mask = (A.exc if D.sign[d] == 1 else A.inh if D.sign[d] == -1 else np.ones(A.n, bool)).copy()
            reqs = sorted(int(D.req[e]) for e in D.hard_out[d])
            for r in set(reqs):
                mask &= A.outdeg(r) >= sum(1 for x in reqs if x >= r)
            for sgn in (1, -1):
                reqs = sorted(int(D.req[e]) for e in D.hard_in[d] if D.sign[D.src[e]] == sgn)
                for r in set(reqs):
                    mask &= A.indeg(r, sgn) >= sum(1 for x in reqs if x >= r)
            masks[d] = mask
        # layer 1: the motif's own instances (a real mutual pair, a real relay with an inhibitor,
        # a walk long enough for the chain); the fallback for a motif that cannot be completed in
        # context still draws from this layer, so its own edges survive
        for mo in D.motifs:
            if mo.kind == "latch":
                mm, _ = A.mutual(min(mo.reqs["r_uv"], mo.reqs["r_vu"]))
                for x in mo.nodes:
                    masks[x] &= mm
            elif mo.kind == "ffpair":
                mm, _ = A.mutual_inh(min(mo.reqs["r_uv"], mo.reqs["r_vu"]), mo.reqs["driver"])
                for x in mo.nodes:
                    masks[x] &= mm
            elif mo.kind == "fftriple":
                # loose: the pair on any real driven pair (a triple that cannot be completed
                # still lands its pair), p on an excitatory neuron some such member inhibits
                u, v, pp = mo.nodes
                thr = min(mo.reqs["r_uv"], mo.reqs["r_vu"])
                mm, _ = A.mutual_inh(thr, mo.reqs["driver"])
                _, proxy = A.proxy_hosts(thr, mo.reqs["r_vp"], mo.reqs["driver"])
                masks[u] &= mm
                masks[v] &= mm
                masks[pp] &= proxy
            elif mo.kind == "relay":
                E = mo.nodes[0]
                for I, e, r_ie, S, r_se, r_si in mo.reqs["inhibitors"]:
                    s_m, e_m, i_m = A.relay(r_se, r_si, r_ie)
                    masks[E] &= e_m
                    masks[I] &= i_m
                    fan[(S, r_se, r_si, r_ie)] = fan.get((S, r_se, r_si, r_ie), 0) + 1
            elif mo.kind == "chain":
                L, thr = len(mo.nodes), mo.reqs["thr"]
                for i, x in enumerate(mo.nodes):
                    masks[x] &= A.walk(thr, L - 1 - i, True)
                    masks[x] &= A.walk(thr, i, False)
        self.dom1 = [np.flatnonzero(mk).astype(np.int32) for mk in masks]
        # layer 2: the context (a source of k relays needs k real relays, a chain's predecessor a
        # walk one edge longer) and arc consistency over the hard edges; the search proper
        for mo in D.motifs:
            if mo.kind == "fftriple":  # strict: v on a member that inhibits some excitatory neuron
                thr = min(mo.reqs["r_uv"], mo.reqs["r_vu"])
                has_proxy, _ = A.proxy_hosts(thr, mo.reqs["r_vp"], mo.reqs["driver"])
                masks[mo.nodes[1]] &= has_proxy
            if mo.kind == "chain":
                L, thr = len(mo.nodes), mo.reqs["thr"]
                for e in D.hard_in[mo.nodes[0]]:
                    p = int(D.src[e])
                    if D.sign[p] == 1:
                        masks[p] &= A.walk(min(thr, int(D.req[e])), L, True)
        for (S, r_se, r_si, r_ie), k in fan.items():
            masks[S] &= A.relay_count(r_se, r_si, r_ie) >= k
        self.dom = [np.flatnonzero(mk).astype(np.int32) for mk in masks]
        self._arc_consistency(rounds=2)
        D.static_dom, D.static_dom1 = list(self.dom), list(self.dom1)

    def _arc_consistency(self, rounds: int):
        """Drop candidates with no compatible partner in a hard neighbour's domain."""
        D, A = self.D, self.A
        for _ in range(rounds):
            changed = False
            for e in np.flatnonzero(D.hard):
                s, d, r = int(D.src[e]), int(D.dst[e]), int(D.req[e])
                Ar, ArT = A.A(r)
                dense = np.zeros(A.n, np.int32)
                dense[self.dom[d]] = 1
                supp = (Ar @ dense) > 0
                new = self.dom[s][supp[self.dom[s]]]
                if len(new) < len(self.dom[s]):
                    self.dom[s] = new; changed = True
                dense[:] = 0
                dense[self.dom[s]] = 1
                supp = (ArT @ dense) > 0
                new = self.dom[d][supp[self.dom[d]]]
                if len(new) < len(self.dom[d]):
                    self.dom[d] = new; changed = True
            if not changed:
                break

    # ---- assignment and forward checking --------------------------------------------
    def _free(self, arr: np.ndarray) -> np.ndarray:
        return arr[~self.used[arr]]

    def assign(self, cand: dict, soft: bool = False) -> list:
        """Assign a motif's nodes; forward-check every unplaced hard neighbour, then propagate
        arc consistency from any domain that shrank to at most `mac_max` candidates (a small
        domain constrains its neighbours cheaply; a large one is left to forward checking).
        Returns the unplaced nodes whose domains emptied (the assignment's kills).
        `soft`: a best-fit placement that already misses edges; it is rewarded in scoring
        but prunes no domain, so it cannot drag its unplaced neighbours down with it."""
        D, A = self.D, self.A
        queue: list = []
        for d, r in cand.items():
            self.real[d] = r
            self.used[r] = True
            if D.hub[d]:
                self.hub_row[d] = (A.C[r].toarray().ravel(), A.CT[:, r].toarray().ravel())
            if soft:
                self.soft_anchor.add(d)
        if soft:
            return []
        for d, r in cand.items():
            for e in D.hard_out[d]:
                x = int(D.dst[e])
                if self.real[x] < 0:
                    self._shrink(x, A.row(r, int(D.req[e])), queue)
            for e in D.hard_in[d]:
                x = int(D.src[e])
                if self.real[x] < 0:
                    self._shrink(x, A.col(r, int(D.req[e])), queue)
        killed: list = []
        while queue:
            x = queue.pop()
            free = self._free(self.dom[x])
            if free.size == 0:
                if x not in killed:
                    killed.append(x)
                continue
            if free.size > self.mac_max:
                continue
            for e in D.hard_out[x]:
                y = int(D.dst[e])
                if self.real[y] < 0:
                    r = int(D.req[e])
                    self._shrink(y, np.unique(np.concatenate([A.row(a, r) for a in free.tolist()])), queue)
            for e in D.hard_in[x]:
                y = int(D.src[e])
                if self.real[y] < 0:
                    r = int(D.req[e])
                    self._shrink(y, np.unique(np.concatenate([A.col(a, r) for a in free.tolist()])), queue)
        return killed

    def _shrink(self, x: int, support: np.ndarray, queue: list):
        new = np.intersect1d(self.dom[x], support, assume_unique=True)
        new1 = np.intersect1d(self.dom1[x], support, assume_unique=True)
        if new.size < self.dom[x].size or new1.size < self.dom1[x].size:
            self.trail.append((x, self.dom[x], self.dom1[x]))
            self.dom[x], self.dom1[x] = new, new1
            queue.append(x)

    def undo_to(self, mark: int, nodes):
        while len(self.trail) > mark:
            x, old, old1 = self.trail.pop()
            self.dom[x], self.dom1[x] = old, old1
        for d in nodes:
            if self.real[d] >= 0:
                self.used[self.real[d]] = False
                self.real[d] = -1
                self.hub_row.pop(d, None)
                self.soft_anchor.discard(d)

    # ---- scoring ------------------------------------------------------------------------
    def feasible(self, mi: int, pd: dict, doms: list) -> bool:
        """Does motif `mi` still have a complete real instance when its nodes' domains are
        replaced by `pd` (the domains a tentative assignment would leave)?"""
        D, A = self.D, self.A
        mo = D.motifs[mi]
        dn = {x: pd[x] if x in pd else self._free(doms[x]) for x in mo.nodes}
        if any(v.size == 0 for v in dn.values()):
            return False
        if mo.kind in ("latch", "ffpair"):
            u, v = mo.nodes
            r_uv, r_vu = mo.reqs["r_uv"], mo.reqs["r_vu"]
            du, dv = dn[u], dn[v]
            if du.size > dv.size:
                du, dv, r_uv, r_vu = dv, du, r_vu, r_uv
            for a in du[: self.cand_cap].tolist():
                bs = np.intersect1d(A.row(a, r_uv), A.col(a, r_vu), assume_unique=True)
                bs = np.intersect1d(bs, dv, assume_unique=True)
                if bs.size > 1 or (bs.size == 1 and bs[0] != a):
                    return True
            return False
        if mo.kind == "fftriple":
            u, v, pp = mo.nodes
            dp = dn[pp]
            for a, b in self._pair_instances(dn[u], dn[v], mo.reqs["r_uv"], mo.reqs["r_vu"], self.cand_cap):
                xs = np.intersect1d(A.row(b, mo.reqs["r_vp"]), dp, assume_unique=True)
                if xs.size and ((xs != a) & (xs != b)).any():
                    return True
            return False
        if mo.kind == "relay":
            E = mo.nodes[0]
            for e_real in dn[E][: self.cand_cap].tolist():
                if all(np.intersect1d(dn[I], A.col(e_real, r_ie), assume_unique=True).size
                       for I, _, r_ie, *_ in mo.reqs["inhibitors"]):
                    return True
            return False
        return True  # chain: walk masks are already in the domains; single / hub: non-empty

    def score(self, assign: dict, doms: list | None = None) -> float:
        """Value ordering: +1 per unplaced neighbour motif that keeps a complete instance,
        -KILL per one that loses it, +1 per soft (hub) edge satisfied, minus the isolation
        penalty of each host (isolation_weight * log-exposure) and an isolation tie-break
        (strong outgoing partners the Profile 2 image would have to zero)."""
        D, A = self.D, self.A
        doms = self.dom if doms is None else doms
        s = 0.0
        pruned: dict = {}
        for d, r in assign.items():
            for e in D.hard_out[d]:
                x = int(D.dst[e])
                if x in assign:
                    continue
                if self.real[x] >= 0:
                    if x in self.soft_anchor and A.count(r, int(self.real[x])) >= D.req[e]:
                        s += 1.0
                    continue
                pruned[x] = np.intersect1d(pruned.get(x, doms[x]), A.row(r, int(D.req[e])), assume_unique=True)
            for e in D.hard_in[d]:
                x = int(D.src[e])
                if x in assign:
                    continue
                if self.real[x] >= 0:
                    if x in self.soft_anchor and A.count(int(self.real[x]), r) >= D.req[e]:
                        s += 1.0
                    continue
                pruned[x] = np.intersect1d(pruned.get(x, doms[x]), A.col(r, int(D.req[e])), assume_unique=True)
            for e in D.soft_out[d]:  # d is a hub: reward the placed targets it reaches
                x = int(D.dst[e])
                if self.real[x] >= 0 and A.count(r, int(self.real[x])) >= D.req[e]:
                    s += self.soft_w
            for e in D.soft_in[d]:
                h = int(D.src[e])
                if h in self.hub_row and self.hub_row[h][0][r] >= D.req[e]:
                    s += self.soft_w
                elif self.real[h] >= 0 and A.count(int(self.real[h]), r) >= D.req[e]:
                    s += self.soft_w
            s -= 1e-4 * A.outdeg24[r] + self.iso_pen[r]
        by_motif: dict = {}
        for x, px in pruned.items():
            by_motif.setdefault(int(D.motif_of[x]), {})[x] = self._free(px)
        for mi, pd in by_motif.items():
            s += len(pd) if self.feasible(mi, pd, doms) else -self.KILL * len(pd)
        return s

    def _ranked(self, assigns: list, doms: list | None = None) -> list:
        if not assigns:
            return assigns
        sc = np.array([self.score(a, doms) for a in assigns])
        sc = sc + 1e-6 * self.rng.random(len(sc))
        return [assigns[i] for i in np.argsort(-sc, kind="stable")]

    def _hub_order(self, x: int, dx: np.ndarray) -> np.ndarray:
        """Candidates of x reached by a placed hub's soft edge first, then the quietest
        (smallest isolation penalty), then a random order (so restarts explore different
        regions)."""
        if dx.size <= 1:
            return dx
        pri = self.rng.random(dx.size) - self.iso_pen[dx]
        for e in self.D.soft_in[x]:
            h = int(self.D.src[e])
            if h in self.hub_row:
                pri += 10.0 * (self.hub_row[h][0][dx] >= self.D.req[e])
        for e in self.D.soft_out[x]:
            h = int(self.D.dst[e])
            if h in self.hub_row:
                pri += 10.0 * (self.hub_row[h][1][dx] >= self.D.req[e])
        return dx[np.argsort(-pri, kind="stable")]

    # ---- candidate generators (tier 0: complete instances) -------------------------
    def _pair_instances(self, du: np.ndarray, dv: np.ndarray, r_uv: int, r_vu: int, cap: int | None = None):
        """Real (a, b) with a in du, b in dv, a -> b >= r_uv and b -> a >= r_vu, a != b, enumerated
        from the smaller side (in the order given); at most `cap` outer neurons are tried."""
        if du.size == 0 or dv.size == 0:
            return
        if du.size <= dv.size:
            for a in (du if cap is None else du[:cap]).tolist():
                bs = np.intersect1d(self.A.row(a, r_uv), self.A.col(a, r_vu), assume_unique=True)
                bs = np.intersect1d(bs, dv, assume_unique=True)
                for b in bs.tolist():
                    if b != a:
                        yield a, int(b)
        else:
            for b in (dv if cap is None else dv[:cap]).tolist():
                as_ = np.intersect1d(self.A.col(b, r_uv), self.A.row(b, r_vu), assume_unique=True)
                as_ = np.intersect1d(as_, du, assume_unique=True)
                for a in as_.tolist():
                    if a != b:
                        yield int(a), b

    def cands_latch(self, mo: Motif, doms: list):
        """Real mutual pairs for a latch or a flip-flop pair (the domains are already sign-masked
        and restricted to members of real pairs of the motif's kind)."""
        u, v = mo.nodes
        du, dv = self._free(doms[u]), self._free(doms[v])
        if du.size == 0 or dv.size == 0:
            return []
        du, dv = self._hub_order(u, du), self._hub_order(v, dv)
        out = []
        for a, b in self._pair_instances(du, dv, mo.reqs["r_uv"], mo.reqs["r_vu"]):
            out.append({u: a, v: b})
            if len(out) >= self.cand_cap:
                break
        return self._ranked(out, doms)

    def cands_triple(self, mo: Motif, doms: list):
        """Real (pair, proxy) instances for a flip-flop triple: a real mutual inhibitory pair
        (a, b) as cands_latch finds them, b hosting the member that inhibits the proxy, plus an
        excitatory neuron b inhibits at the proxy's threshold, from p's domain and distinct from
        both members. Pairs without a proxy in p's (forward-checked) domain are passed over
        rather than counted against the candidate cap; each pair contributes its few best
        proxies (as a relay contributes its inhibitors)."""
        u, v, pp = mo.nodes
        du, dv, dp = self._free(doms[u]), self._free(doms[v]), self._free(doms[pp])
        if du.size == 0 or dv.size == 0 or dp.size == 0:
            return []
        du, dv = self._hub_order(u, du), self._hub_order(v, dv)
        r_vp = mo.reqs["r_vp"]
        per_pair = max(2, self.cand_cap // 8)
        out = []
        for a, b in self._pair_instances(du, dv, mo.reqs["r_uv"], mo.reqs["r_vu"], cap=4 * self.cand_cap):
            xs = np.intersect1d(self.A.row(b, r_vp), dp, assume_unique=True)
            xs = xs[(xs != a) & (xs != b)]
            if xs.size == 0:
                continue
            xs = self._hub_order(pp, xs)[:per_pair]
            out.extend({u: a, v: b, pp: int(x)} for x in xs.tolist())
            if len(out) >= self.cand_cap:
                break
        return self._ranked(out, doms)

    def cands_relay(self, mo: Motif, doms: list):
        E = mo.nodes[0]
        inh = mo.reqs["inhibitors"]
        out = []
        dE = self._hub_order(E, self._free(doms[E]))
        for e_real in dE.tolist():
            assign = {E: e_real}
            ok = True
            for I, e, r_ie, *_ in inh:
                cs = np.intersect1d(self._free(doms[I]), self.A.col(e_real, r_ie), assume_unique=True)
                cs = cs[~np.isin(cs, list(assign.values()))]
                if cs.size == 0:
                    ok = False
                    break
                best = self._ranked([{I: int(c)} for c in cs[: max(4, self.cand_cap // 8)].tolist()], doms)[0]
                assign.update(best)
            if ok:
                out.append(assign)
                if len(out) >= self.cand_cap:
                    break
        return self._ranked(out, doms)

    def cands_single(self, mo: Motif, doms: list):
        x = mo.nodes[0]
        dx = self._hub_order(x, self._free(doms[x]))[: self.cand_cap]
        return self._ranked([{x: int(c)} for c in dx.tolist()], doms)

    def cands_hub(self, mo: Motif, doms: list):
        """A hub's candidates ranked by how many of its soft targets/sources still have a
        compatible partner among the hub's strong partners (a sum of sparse mat-vecs)."""
        D, A = self.D, self.A
        h = mo.nodes[0]
        dh = self._free(doms[h])
        if dh.size == 0:
            return []
        cover = np.zeros(A.n, np.float64)
        dense = np.zeros(A.n, np.int32)
        for e in D.soft_out[h] + D.hard_out[h]:
            x, r = int(D.dst[e]), int(D.req[e])
            Ar, _ = A.A(r)
            dense[:] = 0
            if self.real[x] >= 0:
                dense[self.real[x]] = 1
            else:
                dense[doms[x]] = 1
            cover += (Ar @ dense) > 0
        for e in D.soft_in[h] + D.hard_in[h]:
            x, r = int(D.src[e]), int(D.req[e])
            _, ArT = A.A(r)
            dense[:] = 0
            if self.real[x] >= 0:
                dense[self.real[x]] = 1
            else:
                dense[doms[x]] = 1
            cover += (ArT @ dense) > 0
        sc = cover[dh] - 1e-4 * A.outdeg24[dh] - self.iso_pen[dh]
        order = np.argsort(-sc, kind="stable")[: self.cand_cap]
        return [{h: int(dh[i])} for i in order if cover[dh[i]] > 0]

    def cands_chain(self, mo: Motif, doms: list, want_partial: bool = False):
        """Real paths for a designed chain by depth-first search over strong excitatory
        edges, each position restricted to its node's domain, walk-reach pruning the rest.
        Forward from the start when its domain is the tighter one, else backward from the
        end; the candidate budget is spread over the tight end's options so that the paths
        differ where it matters. Neurons that sit in a small domain of an unplaced neighbour
        of the chain are reserved for that neighbour. With `want_partial`, the longest prefix
        (or suffix) found instead."""
        D, A = self.D, self.A
        nodes = list(mo.nodes)
        L = len(nodes)
        reqs = [int(D.req[e]) for e in mo.edges]  # reqs[i]: nodes[i] -> nodes[i+1]
        forward = len(self._free(doms[nodes[0]])) <= len(self._free(doms[nodes[-1]]))
        if not forward:
            nodes = nodes[::-1]
            reqs = reqs[::-1]
        thr = mo.reqs["thr"]
        neigh = A.row if forward else A.col
        reserved = np.zeros(A.n, bool)
        for x in nodes:
            for e in D.hard_out[x] + D.hard_in[x]:
                y = int(D.dst[e]) if D.src[e] == x else int(D.src[e])
                if self.real[y] < 0 and y not in mo.nodes:
                    fy = self._free(doms[y])
                    if fy.size <= self.reserve_max:
                        reserved[fy] = True

        def options(i: int, prev: int | None) -> np.ndarray:
            base = self._free(doms[nodes[i]])
            if prev is not None:
                base = np.intersect1d(base, neigh(prev, reqs[i - 1]), assume_unique=True)
            base = base[~self.onpath[base] & ~reserved[base]]
            if i < L - 1:
                base = base[A.walk(thr, L - 1 - i, forward)[base]]
            if base.size > 1:
                deg = (A.outdeg(thr)[base] if forward else A.indeg(thr, 1)[base]).astype(np.float64)
                deg -= self.iso_pen[base]
                for e in D.soft_in[nodes[i]]:  # a placed hub's soft edge onto this hop is worth an edge
                    h = int(D.src[e])
                    if h in self.hub_row:
                        deg += 1000.0 * (self.hub_row[h][0][base] >= D.req[e])
                base = base[np.argsort(-(deg + self.rng.random(base.size)), kind="stable")]
            return base

        starts = options(0, None).tolist()
        per_start = max(1, self.cand_cap // max(1, len(starts)))
        best_partial: list[int] = []
        total = 0
        total_steps = 0
        total_step_budget = 8 * self.chain_budget  # a hard cap across all starts, on top of each start's own budget
        steps_since_check = 0
        expired = False
        for s0 in starts:
            if expired or total_steps >= total_step_budget:
                break
            budget = self.chain_budget
            path: list[int] = []
            stack = [iter([s0])]
            found = 0
            while stack and budget > 0 and total_steps < total_step_budget:
                nxt = next(stack[-1], None)
                if nxt is None:
                    stack.pop()
                    if path:
                        self.onpath[path.pop()] = False
                    continue
                budget -= 1
                total_steps += 1
                steps_since_check += 1
                if steps_since_check >= 500:
                    steps_since_check = 0
                    if time.time() > self.deadline:
                        expired = True
                        break
                path.append(nxt)
                self.onpath[nxt] = True
                if len(path) > len(best_partial):
                    best_partial = list(path)
                if len(path) == L:
                    found += 1
                    total += 1
                    if not want_partial:
                        yield {nodes[i]: path[i] for i in range(L)}
                    self.onpath[path.pop()] = False
                    if found >= per_start:
                        break
                    continue
                stack.append(iter(options(len(path), nxt).tolist()))
            for x in path:
                self.onpath[x] = False
            if total >= self.cand_cap or expired:
                break
        if want_partial and 2 <= len(best_partial) < L:
            yield {nodes[i]: best_partial[i] for i in range(len(best_partial))}

    def candidates(self, mo: Motif, doms: list | None = None) -> list:
        doms = self.dom if doms is None else doms
        if mo.kind in ("latch", "ffpair"):
            return self.cands_latch(mo, doms)
        if mo.kind == "fftriple":
            return self.cands_triple(mo, doms)
        if mo.kind == "relay":
            return self.cands_relay(mo, doms)
        if mo.kind == "chain":
            paths = list(self.cands_chain(mo, doms))
            if self.iso_w:  # every complete path carries the same edges: the quietest first (DFS order on ties)
                paths.sort(key=lambda c: float(self.iso_pen[list(c.values())].sum()))
            return paths
        if mo.kind == "hub":
            return self.cands_hub(mo, doms)
        return self.cands_single(mo, doms)

    # ---- variable ordering -----------------------------------------------------------
    KIND_RANK = {"hub": 0, "latch": 1, "ffpair": 1, "fftriple": 1, "relay": 2, "chain": 3, "single": 4}

    def pick_next(self, done: np.ndarray):
        """Most constrained motif first: the one whose tightest node has the fewest free
        candidates (hubs first under the hub-first strategy, never under hub-last)."""
        D = self.D
        best, best_key = None, None
        for mi, mo in enumerate(D.motifs):
            if done[mi]:
                continue
            if mo.kind == "hub":
                if not self.hub_first:
                    continue
                tight = -1
            elif self.order == "mrv":
                tight = min(int(self._free(self.dom[x]).size) for x in mo.nodes)
            else:  # "connected": most placed hard neighbours first
                tight = 0
                for x in mo.nodes:
                    for e in D.hard_out[x]:
                        tight -= self.real[D.dst[e]] >= 0
                    for e in D.hard_in[x]:
                        tight -= self.real[D.src[e]] >= 0
            key = (tight, self.KIND_RANK[mo.kind], -len(mo.nodes), mi)
            if best_key is None or key < best_key:
                best, best_key = mi, key
        return best

    # ---- the depth-first search over motifs ----------------------------------------
    def try_candidates(self, cands: list, start: int):
        """Assign the first candidate from `start` that empties no domain; if every one does,
        assign the one with the fewest kills. Returns (index, kills) or (None, None)."""
        fallback = None
        for k in range(start, len(cands)):
            mark = len(self.trail)
            killed = self.assign(cands[k])
            if not killed or not self.reject_kills:
                return k, []
            if fallback is None or len(killed) < fallback[1]:
                fallback = (k, len(killed))
            self.undo_to(mark, cands[k].keys())
        if fallback is None:
            return None, None
        killed = self.assign(cands[fallback[0]])
        return fallback[0], killed

    def run(self):
        D = self.D
        n_mot = len(D.motifs)
        done = np.zeros(n_mot, bool)
        frames: list = []  # [motif, candidates, next index, trail mark, assigned nodes, tier]
        bt_left = np.full(n_mot, self.max_bt, np.int32)
        while True:
            if time.time() > self.deadline:
                break
            mi = self.pick_next(done)
            if mi is None:
                break
            mo = D.motifs[mi]
            cands = self.candidates(mo)
            mark = len(self.trail)
            k, killed = self.try_candidates(cands, 0)
            if k is not None and not killed:
                done[mi] = True
                self.outcome[mi] = "complete"
                frames.append([mi, cands, k + 1, mark, list(cands[k].keys()), 0])
                continue
            if k is not None:  # every candidate kills something: undo, try to repair upstream first
                self.undo_to(mark, cands[k].keys())
            resolved = False
            depth = 0
            while bt_left[mi] > 0 and frames and depth < self.bt_depth and time.time() < self.deadline:
                f = frames[-1]
                if f[5] != 0:  # a frame placed by a fallback has no alternatives worth trying
                    break
                self.undo_to(f[3], f[4])
                done[f[0]] = False
                self.outcome.pop(f[0], None)
                bt_left[mi] -= 1
                self.backtracks += 1
                k2, killed2 = self.try_candidates(f[1], f[2])
                if k2 is not None and not killed2:
                    f[2], f[4] = k2 + 1, list(f[1][k2].keys())
                    done[f[0]] = True
                    self.outcome[f[0]] = "complete"
                    resolved = True
                    break
                if k2 is not None:
                    self.undo_to(f[3], f[1][k2].keys())
                frames.pop()
                depth += 1
            if resolved:
                continue
            # give up on a clean placement: the least damaging complete candidate if there is
            # one, else the loose-layer / best-fit placement of relax_motif
            mark = len(self.trail)
            cands = self.candidates(mo)
            k, killed = self.try_candidates(cands, 0)
            if k is not None:
                self.outcome[mi] = "complete"
                frames.append([mi, cands, k + 1, mark, list(cands[k].keys()), 1])
            else:
                placed = self.relax_motif(mo)
                self.outcome[mi] = "partial" if placed else "skipped"
                frames.append([mi, [], 0, mark, placed, 1])
            done[mi] = True
        for mi in range(n_mot):
            if not done[mi]:
                self.outcome[mi] = "skipped"

    # ---- relaxed placement: a node where it carries the most edges to placed neighbours ---
    def coverage_pick(self, d: int):
        """The free real neuron of d's sign carrying the most of d's edges to placed
        neighbours (hard and soft) net of its isolation penalty, fewest strong outgoing
        partners on ties; None if no neuron carries any."""
        D, A = self.D, self.A
        votes: list = []
        for e in D.hard_out[d] + D.soft_out[d]:
            x = int(D.dst[e])
            if self.real[x] >= 0:
                votes.append(A.col(int(self.real[x]), int(D.req[e])))
        for e in D.hard_in[d] + D.soft_in[d]:
            x = int(D.src[e])
            if self.real[x] >= 0:
                votes.append(A.row(int(self.real[x]), int(D.req[e])))
        if not votes:
            return None
        cnt = np.bincount(np.concatenate(votes), minlength=A.n)
        sign_mask = A.exc if D.sign[d] == 1 else A.inh if D.sign[d] == -1 else np.ones(A.n, bool)
        cnt = np.where(sign_mask & ~self.used, cnt, 0)
        if cnt.max() == 0:
            return None
        ok = np.flatnonzero(cnt > 0)
        val = cnt[ok] - self.iso_pen[ok]
        top = ok[val >= val.max() - 1e-9]
        return int(top[np.argmin(A.outdeg24[top])])

    def relax_motif(self, mo: Motif) -> list:
        """Place a motif that has no complete instance in context: first as a whole from the
        loose domains (the motif's own edges and its edges to placed neighbours honoured, its
        unplaced neighbours' needs ignored), else node by node from the loose domains, else
        where the node carries the most edges. Returns the nodes placed."""
        placed = []
        if mo.kind != "chain":
            cands = self.candidates(mo, self.dom1)
            if cands:
                self.assign(cands[0])
                return list(cands[0].keys())
        else:
            cands = list(self.cands_chain(mo, self.dom1))
            partial = cands[0] if cands else next(self.cands_chain(mo, self.dom1, want_partial=True), None)
            if partial:
                self.assign(partial)
                placed = list(partial.keys())
        rest = [x for x in mo.nodes if self.real[x] < 0]
        order = sorted(rest, key=lambda x: (self._free(self.dom1[x]).size == 0, -len(self.D.hard_out[x]) - len(self.D.hard_in[x])))
        for x in order:
            fx = self._hub_order(x, self._free(self.dom1[x]))[: self.cand_cap]
            if fx.size:
                cand = self._ranked([{x: int(c)} for c in fx.tolist()], self.dom1)[0]
                self.assign(cand)
            else:
                r = self.coverage_pick(x)
                if r is None:
                    continue
                self.assign({x: r}, soft=True)
            placed.append(x)
        return placed

    # ---- repair: re-place each motif knowing both sides of its neighbourhood ---------------
    def carried_by(self, assign: dict) -> int:
        """Designed edges (hard and soft) that `assign` would carry: among its own nodes and to
        placed neighbours outside it."""
        D, A = self.D, self.A
        n = 0
        for d, r in assign.items():
            for e in D.hard_out[d] + D.soft_out[d]:
                x = int(D.dst[e])
                rx = assign.get(x, self.real[x] if self.real[x] >= 0 else -1)
                if rx >= 0 and A.count(r, int(rx)) >= D.req[e]:
                    n += 1
            for e in D.hard_in[d] + D.soft_in[d]:
                x = int(D.src[e])
                if x in assign:
                    continue  # counted from the source side
                if self.real[x] >= 0 and A.count(int(self.real[x]), r) >= D.req[e]:
                    n += 1
        return n

    def fc_domain(self, x: int, base: np.ndarray, exclude) -> np.ndarray:
        """`base` restricted to neurons compatible with every placed hard neighbour of x."""
        D, A = self.D, self.A
        dom = base
        for e in D.hard_out[x]:
            y = int(D.dst[e])
            if self.real[y] >= 0 and y not in exclude:
                dom = np.intersect1d(dom, A.col(int(self.real[y]), int(D.req[e])), assume_unique=True)
        for e in D.hard_in[x]:
            y = int(D.src[e])
            if self.real[y] >= 0 and y not in exclude:
                dom = np.intersect1d(dom, A.row(int(self.real[y]), int(D.req[e])), assume_unique=True)
        return dom

    def value(self, assign: dict, nodes) -> float:
        """The repair's objective for one motif: edges `assign` carries minus its hosts' isolation
        penalty, with every node of the motif that `assign` leaves out charged at the cap
        (assignments placing different subsets of the motif are compared on the same footing)."""
        pen = sum(float(self.iso_pen[int(assign[x])]) if x in assign else self.pen_unplaced for x in nodes)
        return self.carried_by(assign) - pen

    def improve(self, rounds: int = 4) -> int:
        """Large-neighbourhood repair: take each motif out in turn and put it back where it
        carries the most edges net of its hosts' isolation penalty, given everything else
        placed (a complete instance compatible with all placed neighbours if one exists, else
        the best node-by-node fit); keep the change only if that objective is strictly larger.
        Monotone in the objective; returns the edges gained (net, so it can be negative when
        isolation_weight > 0 and a quieter host carries one edge fewer)."""
        D = self.D
        gained = 0
        for _ in range(rounds):
            round_gain = 0
            round_obj = 0.0
            for mi, mo in enumerate(D.motifs):
                if time.time() > self.deadline:
                    return gained
                if mo.kind == "hub":
                    continue
                cur = {x: int(self.real[x]) for x in mo.nodes if self.real[x] >= 0}
                cur_n = self.carried_by(cur)
                cur_v = self.value(cur, mo.nodes)
                for x, r in cur.items():
                    self.real[x] = -1
                    self.used[r] = False
                doms = list(self.dom1)
                for x in mo.nodes:
                    doms[x] = self.fc_domain(x, D.static_dom1[x], mo.nodes)
                cands = self.candidates(mo, doms)
                best, best_n, best_v = cur, cur_n, cur_v
                for c in cands[: self.cand_cap]:
                    v = self.value(c, mo.nodes)
                    if v > best_v + 1e-9:
                        best, best_n, best_v = c, self.carried_by(c), v
                if best is cur:  # node by node: the best free neuron for each node in turn
                    alt: dict = {}
                    for x in sorted(mo.nodes, key=lambda x: -len(D.hard_out[x]) - len(D.hard_in[x])):
                        r = self.coverage_pick(x)
                        if r is not None:
                            alt[x] = r
                            self.used[r] = True
                            self.real[x] = r
                    for x, r in alt.items():
                        self.real[x] = -1
                        self.used[r] = False
                    v = self.value(alt, mo.nodes)
                    if v > best_v + 1e-9:
                        best, best_n, best_v = alt, self.carried_by(alt), v
                for x, r in best.items():
                    self.real[x] = r
                    self.used[r] = True
                round_gain += best_n - cur_n
                round_obj += best_v - cur_v
            gained += round_gain
            if round_obj <= 0:
                break
        return gained

    # ---- relaxed pass: every remaining neuron where it carries the most edges -----------
    def relaxed(self):
        D = self.D
        while time.time() < self.deadline:
            best = None
            for d in np.flatnonzero(self.real == -1).tolist():
                gain = 0
                for e in D.hard_out[d] + D.soft_out[d]:
                    gain += self.real[D.dst[e]] >= 0
                for e in D.hard_in[d] + D.soft_in[d]:
                    gain += self.real[D.src[e]] >= 0
                if gain and (best is None or gain > best[0]):
                    best = (gain, d)
            if best is None:
                break
            d = best[1]
            r = self.coverage_pick(d)
            if r is None:
                self.real[d] = -2  # nothing carries an edge: leave unplaced, do not revisit
                continue
            self.assign({d: r}, soft=True)
        self.real[self.real == -2] = -1


# ----------------------------------------------------------------------------------------
# audit
# ----------------------------------------------------------------------------------------
def audit(net: Netlist, D: _Design, A: _Anatomy, real: np.ndarray, seconds: float, outcome: dict, strategy: str,
          policy: Policy, isolation_weight: float = 0.0) -> Placement:
    """A designed neuron can hold the same (src, dst) synapse twice (the netlist does not merge
    duplicates); a real anatomical edge carries both at once, so the audit works on distinct
    pairs with their quanta summed (the neuron's actual input to that anatomical edge), not on
    raw netlist rows -- else one anatomical edge is counted as two carried designed edges and
    the parasitic count is under-counted by the same amount."""
    mapping = {int(d): int(r) for d, r in enumerate(real) if r >= 0}
    unplaced = [int(d) for d in range(D.n) if real[d] < 0]
    roles = net.roles
    pairs: dict[tuple[int, int], int] = {}
    for e in range(len(D.src)):
        key = (int(D.src[e]), int(D.dst[e]))
        pairs[key] = pairs.get(key, 0) + int(D.q[e])
    carried, missing = 0, []
    missing_by_class: dict = {}
    for (s, d), q in pairs.items():
        cls = (role_key(roles[s]), role_key(roles[d]))
        missing_by_class.setdefault(cls, [0, 0])[1] += 1
        esign = 1 if q > 0 else -1
        if real[s] < 0 or real[d] < 0:
            reason = "endpoint unplaced"
        elif esign != D.sign[s]:
            reason = "wrong sign (mixed-sign designed neuron)"
        elif not (A.exc[real[s]] if D.sign[s] == 1 else A.inh[real[s]]):
            reason = "sign"  # the chosen host neuron's own transmitter does not match the designed sign
        elif A.count(int(real[s]), int(real[d])) >= policy.req_count(q):
            carried += 1
            continue
        else:
            reason = "no anatomical edge strong enough"
        missing.append((s, d, q, reason))
        missing_by_class[cls][0] += 1
    placed = np.array(sorted(mapping.values()), dtype=np.int64)
    parasitic = int(A.C[placed][:, placed].nnz - carried) if placed.size else 0
    pl = Placement(mapping, unplaced, carried, missing, parasitic, seconds, edges=len(pairs), strategy=strategy,
                   isolation_weight=isolation_weight)
    # exposure: what the rest of the brain can deliver to the hosts and receive from them (exact:
    # the anatomical totals less the synapses among the hosts), and the objective the search ranks by
    in_ext, out_ext, in_edges, out_edges = exposure_of(A, placed)
    pl.exposure = exposure_summary(in_ext, out_ext, in_edges, out_edges)
    pl.objective = float(carried - isolation_weight * (np.log1p(in_ext) + OUT_EXPOSURE_WEIGHT * np.log1p(out_ext)).sum()
                         - len(unplaced) * penalty_cap(A, isolation_weight))
    by_role: dict = {}
    for d in range(D.n):
        key = role_key(roles[d])
        by_role.setdefault(key, [0, 0])
        by_role[key][0 if real[d] >= 0 else 1] += 1
    pl.by_role = by_role
    motifs: dict = {}
    by_search: dict = {}
    carried_edge = np.zeros(len(D.src), bool)
    for e in range(len(D.src)):
        s_, d_ = int(D.src[e]), int(D.dst[e])
        carried_edge[e] = real[s_] >= 0 and real[d_] >= 0 and not D.impossible[e] and A.count(int(real[s_]), int(real[d_])) >= D.req[e]
    for mi, mo in enumerate(D.motifs):
        n_placed = sum(real[x] >= 0 for x in mo.nodes)
        if n_placed == len(mo.nodes) and all(carried_edge[e] for e in mo.edges):
            o = "complete"  # every node placed and every one of the motif's own edges carried
        elif n_placed:
            o = "partial"
        else:
            o = "unplaced"
        motifs.setdefault(mo.kind, {})
        motifs[mo.kind][o] = motifs[mo.kind].get(o, 0) + 1
        so = outcome.get(mi, "skipped")
        by_search.setdefault(mo.kind, {})
        by_search[mo.kind][so] = by_search[mo.kind].get(so, 0) + 1
    pl.motifs = motifs
    pl.motifs_search = by_search
    # flip-flop hosts with an anatomical driver (an excitatory input >= the loop threshold): the
    # bias is a parameter edit and needs none, so this is reported, not required
    ff_hosts, ff_drv = 0, 0
    for mo in D.motifs:
        if mo.kind not in ("ffpair", "fftriple"):
            continue
        thr = min(mo.reqs["r_uv"], mo.reqs["r_vu"])
        for x in mo.nodes[:2]:
            if real[x] >= 0:
                ff_hosts += 1
                ff_drv += int(A.indeg(thr, 1)[int(real[x])] > 0)
    if ff_hosts:
        pl.ffpair_drivers = {"hosts": ff_hosts, "with_driver": ff_drv}
    # the proxies' readout: every designed edge out of a flip-flop proxy (to its relays, vetoes)
    # is excitatory from an excitatory neuron, so a host can carry it; how many are carried
    if D.ff_proxy_of:
        prox = set(D.ff_proxy_of)
        n_edges = sum(1 for (s_, _) in pairs if s_ in prox)
        n_missing = sum(1 for s_, _, _, _ in missing if s_ in prox)
        pl.proxy_readout = {"proxies": len(prox), "placed": sum(1 for x in prox if real[x] >= 0),
                            "edges": n_edges, "carried": n_edges - n_missing}
    pl.missing_by_class = {f"{a} -> {b}": v for (a, b), v in sorted(missing_by_class.items(), key=lambda kv: -kv[1][0])}
    return pl


# ----------------------------------------------------------------------------------------
# entry points
# ----------------------------------------------------------------------------------------
def place_netlist(net: Netlist, m: MCNS, policy: Policy = Policy(), verbose: bool = False, time_limit_s: float = 600.0,
                  hub_deg: int = 20, restarts: int = 8, cand_cap: int = 48, chain_budget: int = 20_000,
                  max_backtracks: int = 8, backtrack_depth: int = 3, repair_rounds: int = 6, seed: int = 0,
                  strategies=("hub_last", "hub_first", "hub_last", "hub_last"), hard_hubs=(),
                  order: str = "mrv", reject_kills: bool = True, soft_w: float = 1.0, mac_max: int = 0,
                  isolation_weight: float = 0.0, ff_driver: bool = True) -> Placement:
    """Motif-level placement with backtracking and forward checking (module docstring), then
    a large-neighbourhood repair. Runs up to `restarts` descents, cycling `strategies`, within
    `time_limit_s`; returns the placement that carries the most designed edges.
    `hard_hubs`: roles (or ids) of hubs whose many edges are enforced rather than scored (a
    hub-anchored search: its targets are pulled into the hub's neighbourhood).
    `order`: variable ordering ("mrv" most-constrained-first, or "connected": most placed hard
    neighbours first). `reject_kills`: reject a candidate that empties an unplaced neighbour's
    domain when another exists. `soft_w`: weight of a satisfied soft (hub) edge in scoring.
    `mac_max`: propagate arc consistency from any domain that shrinks to at most this many
    candidates (0: forward checking only, no propagation).
    `isolation_weight`: every candidate ranking (domain order, lookahead score, hub coverage,
    coverage_pick and the repair) subtracts isolation_weight * (log1p(input synapses) +
    0.25 * log1p(output synapses)) per host, so that a quieter host -- one the rest of the brain
    reaches less -- wins over a noisier one carrying up to that many fewer edges; 1 means one
    carried edge is worth a factor e in a host's input count. The weight ranks hosts and never
    refuses one: a designed neuron left unplaced is charged the noisiest neuron's penalty. The
    placement returned is the restart with the largest objective (carried edges minus the exact
    penalty of its hosts, minus that cap per unplaced neuron); at 0 (the default) this is the
    most edges carried, as before. Placement.exposure and summary report the hosts' external
    input / output synapses and edges (total, mean, max, p90).
    `ff_driver`: a flip-flop pair's hosts are drawn from real mutual inhibitory pairs whose
    members each have an excitatory input >= the loop threshold (the driver the designed bias
    stands in for; anatomically unnecessary since the bias is a parameter edit, but the pool
    bench/h1_inhpairs.py counted); False draws from every mutual inhibitory pair. Either way
    Placement.ffpair_drivers reports how many chosen hosts have such a driver."""
    t0 = time.time()
    A = _Anatomy(m, policy)
    D = _Design(net, policy, hub_deg, hard_hubs, ff_driver=ff_driver)
    if verbose:
        kinds = {}
        for mo in D.motifs:
            kinds[mo.kind] = kinds.get(mo.kind, 0) + 1
        print(f"[place] {net.n} neurons, {net.nnz} edges; hard {int(D.hard.sum())} soft {int(D.soft.sum())} "
              f"impossible {int(D.impossible.sum())}; motifs {kinds}; mixed-sign {D.mixed}", flush=True)
    best: Placement | None = None
    deadline = t0 + time_limit_s
    per_run = time_limit_s / max(1, restarts)
    for k in range(restarts):
        if time.time() > deadline - 1:
            break
        strategy = strategies[k % len(strategies)]
        rng = np.random.default_rng(seed + k)
        run_deadline = min(deadline, time.time() + per_run) if k < restarts - 1 else deadline
        S = _Search(D, A, rng, hub_first=(strategy == "hub_first"), cand_cap=cand_cap, chain_budget=chain_budget,
                    max_backtracks=max_backtracks, backtrack_depth=backtrack_depth, deadline=run_deadline,
                    mac_max=mac_max, order=order, reject_kills=reject_kills, soft_w=soft_w, isolation_weight=isolation_weight)
        t1 = time.time()
        S.run()
        S.relaxed()
        pre = audit(net, D, A, S.real, 0.0, S.outcome, "", policy, isolation_weight)
        before = pre.carried
        gained = S.improve(rounds=repair_rounds)
        pl = audit(net, D, A, S.real, time.time() - t0, S.outcome, f"{strategy} seed {seed + k}", policy, isolation_weight)
        pl.order_note = f"backtracks {S.backtracks}; {before} carried before repair, +{gained} repaired; descent {time.time() - t1:.1f} s"
        if verbose:
            print(f"[place] restart {k} ({strategy}): carried {before} -> {pl.carried}/{net.nnz} placed {len(pl.mapping)}/{net.n} "
                  f"backtracks {S.backtracks} in {time.time() - t1:.1f} s; external inputs mean {pre.exposure['in_mean']} -> {pl.exposure['in_mean']} "
                  f"objective {pl.objective:.1f}; motifs {pl.motifs}", flush=True)
        if best is None or pl.objective > best.objective or (pl.objective == best.objective and pl.carried > best.carried):
            best = pl
    best.seconds = time.time() - t0
    return best


def host_types(pl: Placement, net: Netlist, m: MCNS, top: int = 6) -> dict:
    """Which real cell types host each designed role (the top few by count)."""
    types = m.neurons["type"].fillna("?").to_numpy()
    out: dict = {}
    for d, r in pl.mapping.items():
        out.setdefault(role_key(net.roles[d]), {})
        t = str(types[r])
        out[role_key(net.roles[d])][t] = out[role_key(net.roles[d])].get(t, 0) + 1
    return {k: dict(sorted(v.items(), key=lambda kv: -kv[1])[:top]) for k, v in out.items()}


def hub_split_analysis(net: Netlist, m: MCNS, pl: Placement, policy: Policy = Policy(), hub_deg: int = 20) -> dict:
    """What splitting each hub into several real neurons would take, given the placement:
    a greedy set cover of the hub's placed partners by real neurons of the hub's sign (each
    copy carries the hub's edges to the partners it reaches), and for every copy how many of
    the hub's own placed inputs (outputs, for an input hub) reach it. Reported, not applied:
    the mapping stays one real neuron per designed neuron."""
    A = _Anatomy(m, policy)
    D = _Design(net, policy, hub_deg)
    types = m.neurons["type"].fillna("?").to_numpy()
    taken = np.zeros(A.n, bool)
    taken[list(pl.mapping.values())] = True
    out: dict = {}
    for h in np.flatnonzero(D.hub).tolist():
        many_out = D.out_hub[h]
        partners = [(int(D.dst[e]) if many_out else int(D.src[e]), int(D.req[e])) for e in (D.soft_out[h] if many_out else D.soft_in[h])]
        placed = [(x, r) for x, r in partners if x in pl.mapping]
        few = [(int(D.src[e]) if many_out else int(D.dst[e]), int(D.req[e])) for e in (D.hard_in[h] + D.soft_in[h] if many_out else D.hard_out[h] + D.soft_out[h])]
        few = [(x, r) for x, r in few if x in pl.mapping]
        sign_mask = A.exc if D.sign[h] == 1 else A.inh if D.sign[h] == -1 else np.ones(A.n, bool)
        uncovered = list(placed)
        copies = []
        while uncovered:
            votes = [A.col(pl.mapping[x], r) if many_out else A.row(pl.mapping[x], r) for x, r in uncovered]
            cnt = np.bincount(np.concatenate(votes), minlength=A.n)
            cnt = np.where(sign_mask & ~taken, cnt, 0)
            if cnt.max() == 0:
                break
            c = int(np.argmax(cnt))
            taken[c] = True
            covered = [(x, r) for x, r in uncovered if (A.count(c, pl.mapping[x]) if many_out else A.count(pl.mapping[x], c)) >= r]
            uncovered = [p for p in uncovered if p not in covered]
            inputs_ok = sum(1 for x, r in few if (A.count(pl.mapping[x], c) if many_out else A.count(c, pl.mapping[x])) >= r)
            copies.append({"real": c, "type": str(types[c]), "covers": len(covered), "own_inputs_reached": f"{inputs_ok}/{len(few)}"})
        out[net.roles[h]] = {"partners": len(partners), "partners_placed": len(placed), "copies": len(copies),
                             "covered": len(placed) - len(uncovered), "uncoverable": len(uncovered), "detail": copies}
    return out


def place_netlist_greedy(net: Netlist, m: MCNS, policy: Policy = Policy(), verbose: bool = False,
                         time_limit_s: float = 600.0) -> Placement:
    """The first tool: neuron by neuron, most constrained first, no backtracking (baseline)."""
    t0 = time.time()
    n = net.n
    src, dst, q = np.asarray(net.src), np.asarray(net.dst), np.asarray(net.quanta)
    req = np.array([policy.req_count(x) for x in q], dtype=np.int32)
    nt = m.neurons["nt"].to_numpy()
    exc_mask = np.isin(nt, policy.excitatory_nt)
    inh_mask = np.isin(nt, policy.inhibitory_nt)
    C, CT = _count_matrix(m)
    sign = np.zeros(n, np.int8)
    for s_, x in zip(src, q):
        sg = 1 if x > 0 else -1
        if sign[s_] == 0:
            sign[s_] = sg
        elif sign[s_] != sg:
            sign[s_] = 2
    out_edges = [[] for _ in range(n)]
    in_edges = [[] for _ in range(n)]
    for e, (s_, d_) in enumerate(zip(src, dst)):
        out_edges[s_].append((d_, req[e]))
        in_edges[d_].append((s_, req[e]))
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
        return cands

    def candidates(d):
        cand = None
        for p, r in in_edges[d]:
            if p in mapping:
                row = C.getrow(mapping[p])
                ok = row.indices[row.data >= r]
                cand = ok if cand is None else np.intersect1d(cand, ok)
                if cand.size == 0:
                    return cand
        for p, r in out_edges[d]:
            if p in mapping:
                col = CT.getcol(mapping[p])
                ok = col.indices[col.data >= r]
                cand = ok if cand is None else np.intersect1d(cand, ok)
                if cand.size == 0:
                    return cand
        if cand is None:
            return None
        return sign_ok(cand[~used[cand]], d)

    def free_start(d):
        if d in pair_of:
            r = max(rr for x, rr in out_edges[d] if x == pair_of[d])
            r2 = max(rr for x, rr in out_edges[pair_of[d]] if x == d)
            keep = (m.count >= min(r, r2)) & exc_mask[m.pre] & exc_mask[m.post] & ~used[m.pre] & ~used[m.post]
            Ab = sp.csr_matrix((np.ones(keep.sum(), np.int8), (m.pre[keep], m.post[keep])), shape=(m.n, m.n))
            M = sp.triu(Ab.multiply(Ab.T), k=1).tocoo()
            if M.nnz == 0:
                return None
            k = np.random.default_rng(len(mapping)).integers(0, M.nnz)
            return int(M.row[k])
        pool = np.flatnonzero((exc_mask if sign[d] == 1 else inh_mask if sign[d] == -1 else np.ones(m.n, bool)) & ~used)
        return int(pool[np.random.default_rng(len(mapping)).integers(0, pool.size)]) if pool.size else None

    placed_nb = np.zeros(n, np.int32)
    heap = [(0, -int(degree[i]), i) for i in range(n)]
    heapq.heapify(heap)
    done = np.zeros(n, bool)
    while heap:
        negp, negd, d = heapq.heappop(heap)
        if done[d]:
            continue
        if -negp != placed_nb[d]:
            heapq.heappush(heap, (-int(placed_nb[d]), negd, d))
            continue
        done[d] = True
        if time.time() - t0 > time_limit_s:
            unplaced.append(d)
            continue
        cand = candidates(d)
        r = free_start(d) if cand is None else (None if cand.size == 0 else int(cand[0]))
        if r is None:
            unplaced.append(d)
        else:
            mapping[d] = r
            used[r] = True
            for p, _ in in_edges[d] + out_edges[d]:
                if not done[p]:
                    placed_nb[p] += 1
                    heapq.heappush(heap, (-int(placed_nb[p]), -int(degree[p]), p))
    real = np.full(n, -1, np.int64)
    for d, r in mapping.items():
        real[d] = r
    return audit(net, _Design(net, policy, hub_deg=10**9), _Anatomy(m, policy), real, time.time() - t0, {}, "greedy", policy)
