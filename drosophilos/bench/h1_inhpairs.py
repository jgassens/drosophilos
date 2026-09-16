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

Writes docs/h1_inhpairs.json, one table per k_max; `python -m drosophilos.bench.h1_inhpairs`
reproduces it.
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
    }


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
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
