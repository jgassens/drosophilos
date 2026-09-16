"""Supply of mutual-inhibition pairs as flip-flop candidates -- counting only.

The shuffled controls (docs/h1_placement.md, "What the fly's wiring contributes") found that
many of the fly's strongest reciprocal pairs are between *inhibitory* neurons, useless to the
current two-neuron excitatory latch (`_Anatomy.mutual`, `bench/h1_cell.py:latch_capacity`).
This bench asks how many of those inh-inh pairs could instead anchor a different circuit: a
flip-flop built from two mutually inhibiting neurons, each with a tonic excitatory driver, so
that exactly one of the pair fires at a time (winner-take-all bistability).

This counts SUPPLY ONLY. An inhibition-based latch is a different circuit from the excitatory
loop the placer builds today -- it needs a steady external drive on both members (not a single
kick) and its state is read out as *silence versus firing*, not as which of two excitatory
loops is circulating. Nothing in this file designs, places or simulates that circuit; it only
asks how many candidate (pair + driver + readout) triples the connectome's wiring offers at a
given synapse-count threshold, the same way `latch_capacity` counts excitatory pairs without
placing them.

For k_max in (4, 8, 16) (thresholds 57, 29, 15 synapses; `Policy.req_count` at the loop
quantum of `latch_capacity`), on the thresholded adjacency:
  1. reciprocal pairs by sign class: exc-exc (the current latch, for reference), inh-inh (both
     GABA or glutamate), and mixed (anything else, including a neuron whose transmitter is
     neither in Policy.excitatory_nt nor Policy.inhibitory_nt);
  2. for inh-inh pairs, how many have EACH member driven -- at least one excitatory input of
     >= threshold synapses from a neuron outside the pair (a tonic driver candidate) -- and,
     of those, how many also have EACH member read out -- at least one inhibitory output of
     >= threshold to a neuron outside the pair;
  3. the maximum matching (disjoint pairs) of the inh-inh graph, and of the driven-and-readout
     subgraph (`networkx.max_weight_matching`, `maxcardinality=True`, as `latch_capacity` uses);
  4. for the disjoint driven-and-readout set: superclass counts, the top 8 cell types, mean
     external input synapses per member ("exposure", `embed_netlist.exposure_of` -- the
     anatomical in-degree in synapses from outside the pair, unthresholded), and how many
     members lie in the antennal lobe. Antennal-lobe membership is read off the `class` column
     of the neurons table (ALPN / ALLN / ALIN / ALON -- the fly's antennal-lobe cell-type
     classes recorded there; `superclass` does not distinguish the antennal lobe from the rest
     of the central brain).

  5. the PROXY readout (docs/a1_flipflop.md, "The excitatory proxy"): a flip-flop's members
     are inhibitory, so no reader can take `u` as the excitatory rail it expects; the design
     adds p, an EXCITATORY neuron inhibited by v, that fires exactly while SET. On the connectome
     that is an inhibitory edge >= thr from a pair member to an excitatory neuron outside the pair
     (the proxy host); the strict variant also asks that host for an excitatory output >= thr to a
     neuron outside the pair, so the proxy can drive a relay. Counted for the driven inh-inh
     pairs: how many have >= 1 loose / strict proxy candidate, the number of candidate (pair,
     proxy) triples, and the largest DISJOINT set of triples (no neuron in two pairs, no proxy
     shared -- a 3-set packing, solved exactly with scipy's MILP where it finishes within
     MILP_TIME_LIMIT_S and otherwise reported as greedy lower / matching upper bounds), with the
     superclass counts of the chosen pairs and the mean external input synapses of the chosen
     proxies (`exposure_of` over the proxies, so the pair's own edge into the proxy is counted:
     it is an external input to the proxy that the design keeps).

Writes docs/h1_inhpairs.json, one table per k_max; `python -m drosophilos.bench.h1_inhpairs`
reproduces it and prints the table.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

K_MAX_VALUES = (4, 8, 16)
LOOP_QUANTA = 3621  # matches bench/h1_cell.py:latch_capacity
TOP_TYPES = 8
AL_COLUMN = "class"
AL_PREFIX = "AL"  # ALPN, ALLN, ALIN, ALON
MILP_TIME_LIMIT_S = 480.0  # exact (pair, proxy) packing (k_max 16: ~200-300 s); past this the greedy / bound pair is reported


def _mutual_pairs(Adir):
    """Reciprocal (a, b) pairs, a < b, in a directed binary adjacency (csr, csr.T)."""
    import scipy.sparse as sp
    A, AT = Adir
    M = sp.triu(A.multiply(AT), k=1).tocoo()
    return np.stack([M.row, M.col], axis=1).astype(np.int64) if M.nnz else np.zeros((0, 2), np.int64)


def sign_classes(A, pairs: np.ndarray) -> dict:
    """Partition reciprocal pairs by the transmitter sign of their two members."""
    if len(pairs) == 0:
        return {"exc_exc": 0, "inh_inh": 0, "mixed": 0}
    exc = A.exc[pairs]
    inh = A.inh[pairs]
    ee = exc[:, 0] & exc[:, 1]
    ii = inh[:, 0] & inh[:, 1]
    return {"exc_exc": int(ee.sum()), "inh_inh": int(ii.sum()), "mixed": int((~(ee | ii)).sum())}


def driven_mask(A, pairs: np.ndarray, thr: int) -> tuple[np.ndarray, np.ndarray]:
    """(driver, driver_and_readout) boolean masks over `pairs` (both members inh-inh).

    driver: each member has >= 1 excitatory input >= thr from a neuron outside the pair. The
    other pair member cannot supply this (it is inhibitory), so no explicit exclusion is needed.
    readout: each member also has >= 1 inhibitory output >= thr to a neuron outside the pair.
    Every inh-inh pair already has an edge to its partner at >= thr (that is why it is a mutual
    pair), so "outside the pair" is the member's total >= thr out-degree minus that one edge.
    """
    if len(pairs) == 0:
        z = np.zeros(0, bool)
        return z, z
    indeg_exc = A.indeg(thr, +1)
    outdeg = A.outdeg(thr)
    driver = (indeg_exc[pairs[:, 0]] > 0) & (indeg_exc[pairs[:, 1]] > 0)
    readout = ((outdeg[pairs[:, 0]] - 1) > 0) & ((outdeg[pairs[:, 1]] - 1) > 0)
    return driver, driver & readout


def max_matching(pairs: np.ndarray) -> int:
    import networkx as nx
    if len(pairs) == 0:
        return 0
    G = nx.Graph()
    G.add_edges_from(map(tuple, pairs.tolist()))
    return len(nx.max_weight_matching(G, maxcardinality=True))


def matched_members(pairs: np.ndarray) -> list[int]:
    """The neurons actually used by a maximum matching of `pairs` (disjoint by construction)."""
    import networkx as nx
    if len(pairs) == 0:
        return []
    G = nx.Graph()
    G.add_edges_from(map(tuple, pairs.tolist()))
    matching = nx.max_weight_matching(G, maxcardinality=True)
    members: list[int] = []
    for a, b in matching:
        members.extend((int(a), int(b)))
    return members


def driven_disjoint_report(m, A, members: list[int]) -> dict:
    """Superclass / cell-type / exposure / antennal-lobe breakdown of a disjoint member set."""
    from ..connectome.embed_netlist import exposure_of
    if not members:
        return {"pairs": 0, "members": 0, "superclass_counts": {}, "top_cell_types": {},
                "mean_external_input_synapses": 0.0, "antennal_lobe_members": 0,
                "antennal_lobe_column": AL_COLUMN}
    idx = np.array(members, dtype=np.int64)
    rows = m.neurons.iloc[idx]
    superclass_counts = {str(k): int(v) for k, v in rows["superclass"].value_counts().items()}
    top_cell_types = {str(k): int(v) for k, v in rows["type"].value_counts().head(TOP_TYPES).items()}
    in_ext, _, _, _ = exposure_of(A, idx)
    al_members = int(rows[AL_COLUMN].astype(str).str.startswith(AL_PREFIX).sum())
    return {
        "pairs": len(idx) // 2, "members": len(idx),
        "superclass_counts": superclass_counts, "top_cell_types": top_cell_types,
        "mean_external_input_synapses": round(float(in_ext.mean()), 1),
        "antennal_lobe_members": al_members, "antennal_lobe_column": AL_COLUMN,
        "antennal_lobe_prefix": AL_PREFIX,
    }


# ---- the excitatory proxy: (pair, proxy) triples ---------------------------------------

def proxy_candidates(A, pairs: np.ndarray, thr: int, strict: bool) -> tuple[np.ndarray, np.ndarray]:
    """(pair_index, proxy) rows: every excitatory neuron outside the pair that a member inhibits
    with >= thr synapses. `strict` keeps only proxies with an excitatory output >= thr to a
    neuron outside the pair (the relay it will drive)."""
    if len(pairs) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    Ainh_exc, _ = A.A(thr, "inh", "exc")  # member -> excitatory target
    rows_i, rows_x = [], []
    for side in (0, 1):
        sub = Ainh_exc[pairs[:, side]].tocoo()  # row = pair index, col = proxy
        rows_i.append(sub.row.astype(np.int64))
        rows_x.append(sub.col.astype(np.int64))
    pi, px = np.concatenate(rows_i), np.concatenate(rows_x)
    keep = (px != pairs[pi, 0]) & (px != pairs[pi, 1])  # a member is inhibitory, so this never triggers; kept for the record
    pi, px = pi[keep], px[keep]
    if strict:
        Aexc, _ = A.A(thr, "exc", None)  # proxy -> anything, excitatory pre
        outdeg_exc = np.asarray(Aexc.sum(axis=1)).ravel().astype(np.int64)
        # outputs into the pair itself do not count: subtract the proxy's edges to a and b
        into_pair = np.asarray(Aexc[px, pairs[pi, 0]]).ravel() + np.asarray(Aexc[px, pairs[pi, 1]]).ravel()
        ok = (outdeg_exc[px] - into_pair) > 0
        pi, px = pi[ok], px[ok]
    key = np.unique(pi * (A.n + 1) + px)  # dedupe (both members may inhibit the same proxy)
    return key // (A.n + 1), key % (A.n + 1)


def pack_triples(pairs: np.ndarray, pi: np.ndarray, px: np.ndarray, n: int,
                 time_limit_s: float = MILP_TIME_LIMIT_S, proxy_cost: np.ndarray | None = None) -> dict:
    """Largest set of (pair, proxy) triples with disjoint pairs and unshared proxies: a 3-set
    packing. Exact by MILP (scipy/HiGHS) when it proves optimality in time; the greedy lower
    bound and the pair-matching upper bound (disjoint pairs among those with any proxy) are
    always reported. `proxy_cost` (per neuron, >= 0): once the pairs are chosen, their proxies
    are re-assigned by a min-weight bipartite matching so the packing uses the cheapest proxies
    it can without losing a triple (a fractional tie-break inside the MILP stalls HiGHS at
    k_max 16). Returns {"chosen": [(pair index, proxy)], "method", bounds}."""
    if len(pi) == 0:
        return {"chosen": [], "method": "empty", "greedy_lower_bound": 0, "pair_matching_upper_bound": 0}
    cand_pairs = np.unique(pi)
    upper = max_matching(pairs[cand_pairs])
    # greedy: pairs with the fewest proxies first, each takes its least-contended free proxy
    n_prox = np.bincount(pi, minlength=len(pairs))
    prox_load = np.bincount(px, minlength=n)
    order = np.argsort(pi, kind="stable")
    pi_s, px_s = pi[order], px[order]
    starts, ends = np.searchsorted(pi_s, cand_pairs), np.searchsorted(pi_s, cand_pairs, side="right")
    used = np.zeros(n, bool)
    greedy: list[tuple[int, int]] = []
    for k in np.argsort(n_prox[cand_pairs], kind="stable"):
        j = int(cand_pairs[k])
        a, b = pairs[j]
        if used[a] or used[b]:
            continue
        opts = px_s[starts[k]:ends[k]]
        opts = opts[~used[opts]]
        if len(opts) == 0:
            continue
        x = int(opts[np.argmin(prox_load[opts])])
        used[a] = used[b] = used[x] = True
        greedy.append((j, x))
    out = {"chosen": greedy, "method": "greedy", "greedy_lower_bound": len(greedy),
           "pair_matching_upper_bound": int(upper)}
    if len(greedy) == upper:
        out["method"] = "greedy (meets the pair-matching bound)"
    else:
        try:
            import scipy.sparse as sp
            from scipy.optimize import Bounds, LinearConstraint, milp
            T = len(pi)
            rows = np.concatenate([pairs[pi, 0], pairs[pi, 1], px])  # every neuron a triple touches: <= 1 triple each
            cols = np.concatenate([np.arange(T)] * 3)
            touched, rows = np.unique(rows, return_inverse=True)
            M = sp.csr_matrix((np.ones(3 * T), (rows, cols)), shape=(len(touched), T))
            res = milp(c=-np.ones(T), constraints=LinearConstraint(M, 0, 1), integrality=np.ones(T),
                       bounds=Bounds(0, 1), options={"time_limit": time_limit_s})
            if res.x is not None:
                sol = [(int(pi[t]), int(px[t])) for t in np.flatnonzero(res.x > 0.5)]
                if len(sol) >= len(greedy):
                    out["chosen"] = sol
                    out["method"] = "milp optimal" if res.status == 0 else f"milp time-limited (gap <= {upper - len(sol)})"
        except Exception as e:  # noqa: BLE001 - the bounds still stand
            out["method"] = f"greedy (milp unavailable: {type(e).__name__})"
    if proxy_cost is not None and out["chosen"]:
        out["chosen"] = reassign_proxies(out["chosen"], pi, px, proxy_cost, n)
    return out


def reassign_proxies(chosen: list[tuple[int, int]], pi: np.ndarray, px: np.ndarray, cost: np.ndarray, n: int) -> list[tuple[int, int]]:
    """Keep the chosen pairs; give each the cheapest proxy a full bipartite matching allows
    (scipy's min_weight_full_bipartite_matching; the existing assignment proves one exists)."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import min_weight_full_bipartite_matching
    cp = np.array([j for j, _ in chosen], dtype=np.int64)
    row_of = {int(j): r for r, j in enumerate(cp)}
    keep = np.isin(pi, cp)
    rows = np.array([row_of[int(j)] for j in pi[keep]], dtype=np.int64)
    cols = px[keep]
    used_cols, cols = np.unique(cols, return_inverse=True)
    B = sp.csr_matrix((cost[used_cols[cols]] + 1.0, (rows, cols)), shape=(len(cp), len(used_cols)))
    try:
        r, c = min_weight_full_bipartite_matching(B)
    except ValueError:  # no full matching found by the solver: keep what we had
        return chosen
    return [(int(cp[i]), int(used_cols[k])) for i, k in zip(r, c)]


def proxy_report(m, A, pairs: np.ndarray, thr: int, strict: bool) -> dict:
    """Supply of (driven inh-inh pair, excitatory proxy) triples at one threshold."""
    from ..connectome.embed_netlist import exposure_of
    pi, px = proxy_candidates(A, pairs, thr, strict)
    pack = pack_triples(pairs, pi, px, A.n, proxy_cost=A.in_syn.astype(float))  # quietest proxies among the maximum packings
    out = {
        "pairs_with_a_proxy": int(len(np.unique(pi))),
        "candidate_triples": int(len(pi)),
        "distinct_proxies": int(len(np.unique(px))),
        "mean_proxies_per_pair": round(float(len(pi) / max(1, len(np.unique(pi)))), 1),
        "disjoint_triples": len(pack["chosen"]),
        "packing_method": pack["method"],
        "greedy_lower_bound": pack["greedy_lower_bound"],
        "pair_matching_upper_bound": pack["pair_matching_upper_bound"],
        "pairs_superclass_counts": {}, "pairs_top_cell_types": {}, "proxies_superclass_counts": {},
        "proxies_top_cell_types": {}, "proxies_mean_external_input_synapses": 0.0,
        "pairs_mean_external_input_synapses": 0.0,
    }
    if pack["chosen"]:
        cp = np.array([j for j, _ in pack["chosen"]], dtype=np.int64)
        cx = np.array([x for _, x in pack["chosen"]], dtype=np.int64)
        members = pairs[cp].ravel()
        for key, idx in (("pairs", members), ("proxies", cx)):
            rows = m.neurons.iloc[idx]
            out[f"{key}_superclass_counts"] = {str(k): int(v) for k, v in rows["superclass"].value_counts().items()}
            out[f"{key}_top_cell_types"] = {str(k): int(v) for k, v in rows["type"].value_counts().head(TOP_TYPES).items()}
        # exposure of each chosen host set on its own: a proxy's input from its pair member is
        # external to the proxy set and is counted (it is the edge the design uses)
        out["pairs_mean_external_input_synapses"] = round(float(exposure_of(A, members)[0].mean()), 1)
        out["proxies_mean_external_input_synapses"] = round(float(exposure_of(A, cx)[0].mean()), 1)
    return out


def inhpairs_table(m, k_max: float, loop_quanta: int = LOOP_QUANTA) -> dict:
    from ..connectome.embed_h0 import Policy
    from ..connectome.embed_netlist import _Anatomy
    policy = Policy(k_max=float(k_max))
    A = _Anatomy(m, policy)
    thr = policy.req_count(loop_quanta)

    pairs_all = _mutual_pairs(A.A(thr))
    sign_counts = sign_classes(A, pairs_all)

    exc = A.exc[pairs_all] if len(pairs_all) else np.zeros((0, 2), bool)
    inh = A.inh[pairs_all] if len(pairs_all) else np.zeros((0, 2), bool)
    ii_mask = (inh[:, 0] & inh[:, 1]) if len(pairs_all) else np.zeros(0, bool)
    ii_pairs = pairs_all[ii_mask]

    driver, driven_readout = driven_mask(A, ii_pairs, thr)
    n_driver = int(driver.sum())
    n_driven_readout = int(driven_readout.sum())

    disjoint_ii = max_matching(ii_pairs)
    driven_pairs = ii_pairs[driven_readout]
    disjoint_driven = max_matching(driven_pairs)
    members = matched_members(driven_pairs)
    disjoint_report = driven_disjoint_report(m, A, members)
    driver_pairs = ii_pairs[driver]
    proxy = {"loose": proxy_report(m, A, driver_pairs, thr, strict=False),
             "strict": proxy_report(m, A, driver_pairs, thr, strict=True)}

    return {
        "threshold": thr,
        "reciprocal_pairs": sign_counts,
        "inh_inh": {
            "pairs": int(len(ii_pairs)),
            "driven_each_member": n_driver,
            "driven_and_readout_each_member": n_driven_readout,
        },
        "max_disjoint_pairs": {
            "inh_inh": disjoint_ii,
            "driven_and_readout": disjoint_driven,
        },
        "driven_disjoint_set": disjoint_report,
        "proxy_readout": proxy,
    }


def print_table(out: dict) -> None:
    """One row per k_max: the pair supply, then the (pair, proxy) supply loose / strict."""
    cols = ("k_max", "thr", "inh_inh", "driven", "disjoint_driven", "pairs_w_proxy L/S", "triples L/S",
            "disjoint_triples L/S", "method L/S", "proxy_ext_in L/S")
    rows = []
    for k, t in out.items():
        L, S = t["proxy_readout"]["loose"], t["proxy_readout"]["strict"]
        rows.append((k, t["threshold"], t["inh_inh"]["pairs"], t["inh_inh"]["driven_each_member"],
                     t["max_disjoint_pairs"]["driven_and_readout"],
                     f'{L["pairs_with_a_proxy"]}/{S["pairs_with_a_proxy"]}',
                     f'{L["candidate_triples"]}/{S["candidate_triples"]}',
                     f'{L["disjoint_triples"]}/{S["disjoint_triples"]}',
                     f'{L["packing_method"].split(" (")[0]}/{S["packing_method"].split(" (")[0]}',
                     f'{L["proxies_mean_external_input_synapses"]}/{S["proxies_mean_external_input_synapses"]}'))
    w = [max(len(str(c)), *(len(str(r[i])) for r in rows)) for i, c in enumerate(cols)]
    print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(cols)))
    for r in rows:
        print("  ".join(str(v).ljust(w[i]) for i, v in enumerate(r)))


def run_all(k_max_values=K_MAX_VALUES, loop_quanta: int = LOOP_QUANTA, m=None) -> dict:
    from ..connectome.mcns import load_mcns
    if m is None:
        m = load_mcns()
    return {str(k): inhpairs_table(m, k, loop_quanta) for k in k_max_values}


def main() -> None:
    path = Path(__file__).resolve().parents[2] / "docs" / "h1_inhpairs.json"
    out = run_all()
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print_table(out)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
