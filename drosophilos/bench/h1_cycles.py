"""Stage H1: if a latch were a directed CYCLE of 3 or 4 excitatory neurons instead of a
reciprocal pair (`bench/h1_cell.py`'s `latch_capacity`), how many could MCNS hold?

A reciprocal pair is a 2-hop loop: A fires B, B fires A, period two hops. A 3- or 4-cycle
(A -> B -> C -> A, or A -> B -> C -> D -> A) is the same kind of self-sustaining loop with a
longer period -- each member fires once per lap instead of once every other lap. This bench
counts the connectome's supply of such cycles at the same k_max weight bounds as the pair
ceiling (docs/capacity_doom.md section 5): how many exist, and -- the number that actually
bounds a one-neuron-per-role circuit -- the largest set of *vertex-disjoint* cycles a greedy
packing finds.

TIMING A REVIEWER MUST CHECK BEFORE BUILDING ON THIS: this bench only counts supply, not
whether the timing works. A 2-hop pair loop's period is what H0 measured at ~4.7 ms (213 Hz).
A 3- or 4-hop cycle at the same ~5.3 ms/hop is a 16-21 ms period -- three to four times
slower. Every timing contract written against the pair's period (relay recovery windows,
veto windows, kill-train timing; `lib/contracts.py`) is derived from that ~4.7 ms figure and
would need re-deriving from scratch for a 3- or 4-hop loop, not simply rescaled, since
recovery and settling times are properties of the membrane and synapse dynamics, not of the
loop that drives them. Nothing here re-measures that; a cycle-based latch is not a drop-in
replacement for a pair-based one until it is.

Counts, for k_max in (4, 8, 16) on the thresholded cholinergic -> cholinergic adjacency A
(same threshold as `latch_capacity`'s pair loop: `Policy(k_max).req_count(3621)` synapses
each edge):

  3-cycles: trace(A^3) / 3 is exact (no closed length-3 walk can revisit a vertex without a
    self-loop, and MCNS has none after `count >= thr` is enforced with sign held fixed).
  4-cycles: trace(A^4) over-counts degenerate closed walks that reuse a reciprocal (mutual)
    edge -- "there and back" twice on one pair, or twice through a shared hub vertex on two
    different mutual pairs. Both are counted here exactly and subtracted (`_four_cycle_count`),
    so the reported figure is an exact simple-cycle count, not an estimate.

Packing: a large set of vertex-disjoint cycles is not found by counting -- it is a maximum
independent set problem restricted to short cycles, NP-hard in general. `_greedy_pack` does
several random-order greedy passes (find a cycle at each unused start vertex by bounded
random DFS, take it, move on) and keeps the largest; reported as a lower bound, the same way
`latch_capacity`'s matching is exact but this packing is not.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

PYTHON = "/Users/jeremiahgassensmith/programming/drosophilos/.venv/bin/python"
LOOP_QUANTA = 3621  # same loop threshold as bench.h1_cell.latch_capacity
K_MAX_VALUES = (4, 8, 16)
RESTARTS = 8
BRANCH_CAP = 30
TIME_BUDGET_S = 60.0  # per (k_max, cycle-length-set) packing; several restarts inside it


def _drop_self_loops(A: sp.csr_matrix) -> sp.csr_matrix:
    """A handful of MCNS neurons synapse onto themselves (101 raw, one survives >= 29 synapses
    at k_max 8); a self-loop is not a cycle and breaks the no-self-loop trace identities below."""
    A = A.tocsr(copy=True)
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def _adjacency(A: sp.csr_matrix) -> dict:
    """CSR -> {node: frozenset(successors)} for every node with an out-edge."""
    A = A.tocsr()
    A.sort_indices()
    adj = {}
    for i in range(A.shape[0]):
        a, b = A.indptr[i], A.indptr[i + 1]
        if b > a:
            adj[i] = frozenset(A.indices[a:b].tolist())
    return adj


def _three_cycle_count(A: sp.csr_matrix) -> int:
    """trace(A^3) / 3, exact for a simple digraph without self-loops (no closed 3-walk can
    revisit a vertex without one)."""
    Ai = A.astype(np.int64)
    A2 = Ai @ Ai
    trace3 = int(A2.multiply(Ai.T).sum())
    assert trace3 % 3 == 0, trace3
    return trace3 // 3


def _four_cycle_count(A: sp.csr_matrix) -> tuple[int, int]:
    """(walk count trace(A^4), exact simple 4-cycle count). Every closed 4-walk v0 v1 v2 v3 v0
    with no self-loop either has all four vertices distinct (a genuine simple 4-cycle, counted
    4x in the trace -- once per starting vertex) or reuses one mutual (reciprocal) edge:
      v0 == v2 (or symmetrically v1 == v3): "there and back" on mutual pair (v0,v1) then again
        on mutual pair (v0,v3) [or the mirror, centred at v1] -- 2 * sum_v d(d-1) walks, d the
        mutual-degree of the repeated vertex;
      v0==v2 AND v1==v3: pure back-and-forth on a single mutual edge -- 2 * (#mutual pairs)
        walks (each ordered direction of each pair gives one such walk).
    So trace(A^4) = 4*C4 + 2*sum_v d(v)*(d(v)-1) + 2*P, solved for C4."""
    Ai = A.astype(np.int64)
    A2 = Ai @ Ai
    walks4 = int(A2.multiply(A2.T).sum())
    Mu = A.multiply(A.T)  # symmetric mutual-pair adjacency
    deg_mu = np.asarray(Mu.sum(axis=1)).ravel()
    p_pairs = int(Mu.nnz) // 2
    degenerate = int(2 * np.sum(deg_mu * (deg_mu - 1)) + 2 * p_pairs)
    c4 = (walks4 - degenerate) // 4
    assert (walks4 - degenerate) % 4 == 0, (walks4, degenerate)
    return walks4, int(c4)


def _find_cycle(v0: int, length: int, adj: dict, used: set, rng: random.Random, branch_cap: int) -> list | None:
    """Bounded random DFS for one simple directed cycle of exactly `length` vertices starting
    and ending at v0, avoiding vertices in `used`. Returns the vertex list, or None."""
    path = [v0]

    def rec() -> bool:
        if len(path) == length:
            return v0 in adj.get(path[-1], ())
        cand = [c for c in adj.get(path[-1], ()) if c not in used and c not in path]
        if len(cand) > branch_cap:
            cand = rng.sample(cand, branch_cap)
        else:
            rng.shuffle(cand)
        for c in cand:
            path.append(c)
            if rec():
                return True
            path.pop()
        return False

    return list(path) if rec() else None


def _greedy_pack_once(adj: dict, lengths: tuple[int, ...], rng: random.Random, branch_cap: int) -> list[list[int]]:
    """One random-order greedy pass: visit nodes in random order, take the first cycle found
    (trying `lengths` in the given order at each start vertex), mark its members used."""
    used: set = set()
    cycles: list = []
    order = list(adj.keys())
    rng.shuffle(order)
    for v0 in order:
        if v0 in used:
            continue
        for length in lengths:
            c = _find_cycle(v0, length, adj, used, rng, branch_cap)
            if c is not None:
                cycles.append(c)
                used.update(c)
                break
    return cycles


def greedy_pack(adj: dict, lengths: tuple[int, ...], restarts: int = RESTARTS, branch_cap: int = BRANCH_CAP,
                time_budget_s: float = TIME_BUDGET_S, seed: int = 0) -> list[list[int]]:
    """Best of several random-order greedy passes (a lower bound on the maximum vertex-disjoint
    packing of cycles with length in `lengths`); stops early if `time_budget_s` is exceeded."""
    rng = random.Random(seed)
    best: list = []
    t0 = time.time()
    for r in range(restarts):
        cycles = _greedy_pack_once(adj, lengths, random.Random(rng.random()), branch_cap)
        if len(cycles) > len(best):
            best = cycles
        if time.time() - t0 > time_budget_s:
            break
    return best


def _member_stats(cycles: list[list[int]], m, A_) -> dict:
    """Superclass counts and mean external-input synapses of a disjoint cycle set's members."""
    if not cycles:
        return {"members": 0, "superclass_counts": {}, "mean_external_in_synapses": 0.0}
    members = sorted({v for c in cycles for v in c})
    sc = m.neurons["superclass"].fillna("?").to_numpy()
    counts: dict = {}
    for v in members:
        s = str(sc[v])
        counts[s] = counts.get(s, 0) + 1
    mean_in = float(np.mean(A_.in_syn[members]))
    return {"members": len(members),
            "superclass_counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
            "mean_external_in_synapses": round(mean_in, 1)}


def cycle_capacity(m, k_max_values: tuple = K_MAX_VALUES, loop_quanta: int = LOOP_QUANTA,
                    restarts: int = RESTARTS, branch_cap: int = BRANCH_CAP, time_budget_s: float = TIME_BUDGET_S,
                    verbose: bool = True) -> dict:
    """Per k_max: pair reference (from `h1_cell.latch_capacity`), 3-cycle and 4-cycle exact
    counts and disjoint-packing lower bounds, a mixed (length 2-4) disjoint packing, and the
    mixed set's member superclasses and mean external input synapses."""
    from ..connectome.embed_h0 import Policy
    from ..connectome.embed_netlist import _Anatomy
    from .h1_cell import latch_capacity
    pairs = latch_capacity(m, k_max_values, loop_quanta)
    out: dict = {}
    for k in k_max_values:
        t0 = time.time()
        pol = Policy(k_max=float(k))
        A_ = _Anatomy(m, pol)
        thr = pol.req_count(loop_quanta)
        A, _ = A_.A(thr, "exc", "exc")
        A = _drop_self_loops(A)
        n3 = _three_cycle_count(A)
        walks4, n4 = _four_cycle_count(A)
        adj = _adjacency(A)
        pack3 = greedy_pack(adj, (3,), restarts, branch_cap, time_budget_s, seed=k)
        pack4 = greedy_pack(adj, (4,), restarts, branch_cap, time_budget_s, seed=k + 1000)
        pack_mixed = greedy_pack(adj, (2, 3, 4), restarts, branch_cap, time_budget_s, seed=k + 2000)
        length_counts: dict = {}
        for c in pack_mixed:
            length_counts[len(c)] = length_counts.get(len(c), 0) + 1
        stats = _member_stats(pack_mixed, m, A_)
        rec = {
            "threshold": thr,
            "graph_edges": int(A.nnz), "graph_nodes_with_out_edge": len(adj),
            "pairs": pairs[str(k)],
            "three_cycle": {"count": n3, "disjoint": len(pack3)},
            "four_cycle": {"walk_count": walks4, "count": n4, "disjoint": len(pack4)},
            "mixed": {"disjoint": len(pack_mixed), "by_length": dict(sorted(length_counts.items())),
                      "superclass_counts": stats["superclass_counts"],
                      "mean_external_in_synapses": stats["mean_external_in_synapses"]},
            "seconds": round(time.time() - t0, 1),
        }
        out[str(k)] = rec
        if verbose:
            print(f"[h1_cycles] k_max={k} thr={thr}: pairs disjoint {pairs[str(k)]['max_disjoint_pairs']}, "
                  f"3-cycles {n3} (disjoint {len(pack3)}), 4-cycles {n4} (disjoint {len(pack4)}), "
                  f"mixed disjoint {len(pack_mixed)} in {rec['seconds']} s", flush=True)
    return out


def print_table(results: dict) -> None:
    header = f"{'k_max':>6} {'thr':>4} {'pairs':>8} {'3-cyc':>10} {'3-disj':>7} {'4-cyc':>10} {'4-disj':>7} {'mixed-disj':>10}"
    print(header)
    print("-" * len(header))
    for k, rec in results.items():
        print(f"{k:>6} {rec['threshold']:>4} {rec['pairs']['max_disjoint_pairs']:>8} "
              f"{rec['three_cycle']['count']:>10} {rec['three_cycle']['disjoint']:>7} "
              f"{rec['four_cycle']['count']:>10} {rec['four_cycle']['disjoint']:>7} "
              f"{rec['mixed']['disjoint']:>10}")
    print()
    for k, rec in results.items():
        print(f"k_max {k}: mixed disjoint set superclasses {rec['mixed']['superclass_counts']}, "
              f"mean external in-synapses {rec['mixed']['mean_external_in_synapses']}")


def run_all(path: Path | None = None, restarts: int = RESTARTS, branch_cap: int = BRANCH_CAP,
            time_budget_s: float = TIME_BUDGET_S, verbose: bool = True) -> dict:
    from ..connectome.mcns import load_mcns
    m = load_mcns()
    results = cycle_capacity(m, restarts=restarts, branch_cap=branch_cap, time_budget_s=time_budget_s, verbose=verbose)
    if path is not None:
        path.write_text(json.dumps(results, indent=1))
    print_table(results)
    return results


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--restarts", type=int, default=RESTARTS)
    ap.add_argument("--branch-cap", type=int, default=BRANCH_CAP)
    ap.add_argument("--time-budget", type=float, default=TIME_BUDGET_S)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    path = args.out or Path(__file__).resolve().parents[2] / "docs" / "h1_cycles.json"
    run_all(path=path, restarts=args.restarts, branch_cap=args.branch_cap, time_budget_s=args.time_budget)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
