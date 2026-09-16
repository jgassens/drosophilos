"""Stage H1: is the 61 % carried fraction (docs/h1_placement.md) a property of the fly's actual
wiring, or of any graph with the same degrees?

Three control connectomes, each keeping something about MCNS and randomising the rest:

  'configuration' -- rewires who connects to whom, holding every neuron's out-synapse count and
                     in-synapse count fixed (a directed, weighted configuration model: a
                     double-edge-swap Markov chain run within buckets of equal synapse count, so
                     a swap can never change either endpoint's total; self-loops the swap would
                     create are rejected in place, so the only source of drift is duplicate
                     (pre, post) pairs produced by two different swaps landing on the same target,
                     merged by summing their counts -- this never moves a degree since the summed
                     weight still belongs to the same two nodes, but is reported for the record).
  'weights'       -- keeps the graph (which pre connects to which post) and permutes the synapse
                     counts among the existing edges: same adjacency, different strengths.
  'transmitter'   -- keeps the graph and the weights, and permutes each neuron's transmitter
                     label among neurons of its own superclass: same wiring, scrambled sign.

`shuffle_mcns` returns a real `MCNS` (same dataclass, so `place_netlist` and `latch_capacity`
need no changes) with the same neuron table shape and the same edge/synapse totals. `run_all`
measures latch supply (k_max 4) and places the 4-bit adder on each control and on the real MCNS,
writing docs/h1_shuffle.json. `python -m drosophilos.bench.h1_shuffle` reproduces it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from ..connectome.mcns import INHIBITORY, MCNS

MODES = ("configuration", "weights", "transmitter")
SEEDS = (0, 1)
K_MAX = 4.0
RESTARTS = 2
TIME_LIMIT_S = 300.0
MIN_SWAP_MULTIPLE = 10  # swap attempts per bucket, as a multiple of the bucket's own edge count
ROUNDS_PER_BUCKET = 24  # each round proposes len(bucket)//2 swaps -> ~12x the bucket's edges


# ---------------------------------------------------------------------------------------
# mode 'configuration': weighted directed double-edge-swap, bucketed by synapse count so a
# swap can never change either endpoint's out- or in-synapse total (see module docstring)
# ---------------------------------------------------------------------------------------
def _swap_bucket(pre: np.ndarray, post: np.ndarray, rng: np.random.Generator, rounds: int) -> np.ndarray:
    """In-place double-edge-swap of `post` among edges sharing one synapse-count bucket.
    Each round pairs up a random permutation of the bucket and proposes swapping the two
    edges' targets; a proposal that would create a self-loop is rejected and the pair is
    left untouched. Returns the (possibly) mutated `post`."""
    k = len(pre)
    if k < 2:
        return post
    post = post.copy()
    for _ in range(rounds):
        perm = rng.permutation(k)
        if k % 2:
            perm = perm[:-1]
        i, j = perm[0::2], perm[1::2]
        new_i, new_j = post[j], post[i]
        ok = (pre[i] != new_i) & (pre[j] != new_j)
        post[i[ok]] = new_i[ok]
        post[j[ok]] = new_j[ok]
    return post


def _configuration_shuffle(pre: np.ndarray, post: np.ndarray, count: np.ndarray,
                           rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Bucket edges by synapse count, double-edge-swap within each bucket, then merge any
    (pre, post) pairs two different swaps happened to land on by summing their counts. Returns
    the shuffled (pre, post, count) and a report of how far the degree sequences moved (0 unless
    merging or a pre-existing self-loop removed one)."""
    order = np.argsort(count, kind="stable")
    pre_s, post_s, count_s = pre[order], post[order], count[order]
    bucket_starts = np.flatnonzero(np.diff(count_s, prepend=count_s[0] - 1))
    bucket_ends = np.r_[bucket_starts[1:], len(count_s)]
    new_post = post_s.copy()
    for a, b in zip(bucket_starts, bucket_ends):
        new_post[a:b] = _swap_bucket(pre_s[a:b], post_s[a:b], rng, ROUNDS_PER_BUCKET)

    out_before = np.bincount(pre_s, weights=count_s, minlength=n)
    in_before = np.bincount(post_s, weights=count_s, minlength=n)

    self_loop = pre_s == new_post
    n_self_loops = int(self_loop.sum())
    keep = ~self_loop
    C = sp.csr_matrix((count_s[keep], (pre_s[keep], new_post[keep])), shape=(n, n))
    n_before_merge = int(keep.sum())
    C.sum_duplicates()
    coo = C.tocoo()
    n_after_merge = coo.nnz
    out_after = np.bincount(coo.row, weights=coo.data, minlength=n)
    in_after = np.bincount(coo.col, weights=coo.data, minlength=n)
    report = {
        "self_loops_dropped": n_self_loops,
        "multi_edges_merged": n_before_merge - n_after_merge,
        "out_synapse_l1_drift": float(np.abs(out_after - out_before).sum()),
        "in_synapse_l1_drift": float(np.abs(in_after - in_before).sum()),
        "neurons_with_out_drift": int((out_after != out_before).sum()),
        "neurons_with_in_drift": int((in_after != in_before).sum()),
    }
    return coo.row.astype(np.int32), coo.col.astype(np.int32), coo.data.astype(np.int32), report


# ---------------------------------------------------------------------------------------
# mode 'weights': same adjacency, permuted synapse counts
# ---------------------------------------------------------------------------------------
def _weight_shuffle(count: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return rng.permutation(count)


# ---------------------------------------------------------------------------------------
# mode 'transmitter': permute nt labels within each superclass; sign follows nt
# ---------------------------------------------------------------------------------------
def _transmitter_shuffle(neurons, rng: np.random.Generator):
    neurons = neurons.copy(deep=True)
    nt = neurons["nt"].to_numpy(copy=True)
    new_nt = nt.copy()
    for idx in neurons.groupby("superclass", sort=False).indices.values():
        idx = np.asarray(idx)
        if len(idx) > 1:
            new_nt[idx] = nt[idx][rng.permutation(len(idx))]
    neurons["nt"] = new_nt
    if "sign" in neurons.columns:
        neurons["sign"] = np.where(np.isin(new_nt, list(INHIBITORY)), -1, 1).astype(np.int8)
    return neurons


def shuffle_mcns(m: MCNS, seed: int, mode: str) -> MCNS:
    """A control connectome with the same neuron table shape and the same edge/synapse totals
    as `m`, randomised per `mode` (module docstring). Returns a real `MCNS`."""
    rng = np.random.default_rng(seed)
    if mode == "configuration":
        pre, post, count, _ = _configuration_shuffle(m.pre, m.post, m.count, rng, m.n)
        neurons = m.neurons.copy(deep=True)
    elif mode == "weights":
        pre, post = m.pre.copy(), m.post.copy()
        count = _weight_shuffle(m.count, rng)
        neurons = m.neurons.copy(deep=True)
    elif mode == "transmitter":
        pre, post, count = m.pre.copy(), m.post.copy(), m.count.copy()
        neurons = _transmitter_shuffle(m.neurons, rng)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return MCNS(neurons, pre, post, count, min_syn=m.min_syn, version=f"{m.version} [{mode} seed={seed}]")


def configuration_drift(m: MCNS, seed: int) -> dict:
    """The degree-sequence drift report for a 'configuration' shuffle at `seed` (module
    docstring): computed separately from `shuffle_mcns` so that function's return stays a plain
    `MCNS`."""
    rng = np.random.default_rng(seed)
    _, _, _, report = _configuration_shuffle(m.pre, m.post, m.count, rng, m.n)
    return report


# ---------------------------------------------------------------------------------------
# measurement: latch supply and adder placement, real MCNS beside each control
# ---------------------------------------------------------------------------------------
def _adder_netlist():
    from ..lib.adder import build_adder_channel
    from ..sim.model import Params
    return build_adder_channel(Params(), 4, ordered=True, watchdog_hops=100).net


def measure(m: MCNS, label: str, verbose: bool = True) -> dict:
    """latch_capacity at k_max 4 and one adder placement, in the shape docs/h1_shuffle.json
    stores for one (mode, seed) row (or the real connectome)."""
    from ..bench.h1_cell import latch_capacity
    from ..connectome.embed_h0 import Policy
    from ..connectome.embed_netlist import _Design, place_netlist

    policy = Policy(k_max=K_MAX)
    lc = latch_capacity(m, k_max_values=(int(K_MAX),))["4"]

    net = _adder_netlist()
    D = _Design(net, policy, hub_deg=20)
    hard_total, hub_total = int(D.hard.sum()), int(D.soft.sum())
    pairs = {}
    for e in range(len(D.src)):
        key = (int(D.src[e]), int(D.dst[e]))
        if key not in pairs:
            pairs[key] = "hub" if D.soft[e] else ("impossible" if D.impossible[e] else "hard")
    hard_pairs = sum(1 for c in pairs.values() if c == "hard")
    hub_pairs = sum(1 for c in pairs.values() if c == "hub")

    t0 = time.time()
    pl = place_netlist(net, m, policy=policy, restarts=RESTARTS, time_limit_s=TIME_LIMIT_S, seed=0, verbose=verbose)
    seconds = time.time() - t0
    missing = {(int(s), int(d)) for s, d, _, _ in pl.missing}
    hard_carried = sum(1 for k, c in pairs.items() if c == "hard" and k not in missing)
    hub_carried = sum(1 for k, c in pairs.items() if c == "hub" and k not in missing)

    if verbose:
        print(f"[h1_shuffle] {label}: mutual {lc['mutual_pairs']} disjoint {lc['max_disjoint_pairs']} "
              f"strong_core {lc['strong_core']} | carried {pl.carried}/{pl.edges} "
              f"placed {len(pl.mapping)}/{net.n} in {seconds:.1f}s", flush=True)
    return {
        "label": label,
        "latch_capacity": {"threshold": lc["threshold"], "mutual_pairs": lc["mutual_pairs"],
                           "max_disjoint_pairs": lc["max_disjoint_pairs"], "strong_core": lc["strong_core"]},
        "placement": {
            "carried": pl.carried, "edges": pl.edges, "carried_fraction": round(pl.carried / max(1, pl.edges), 4),
            "hard": {"carried": hard_carried, "total": hard_pairs},
            "hub": {"carried": hub_carried, "total": hub_pairs},
            "neurons_placed": len(pl.mapping), "neurons_total": net.n, "seconds": round(seconds, 1),
        },
    }


def run_all(data_dir=None, path: Path | None = None, verbose: bool = True) -> dict:
    from ..connectome.mcns import load_mcns
    m = load_mcns(**({"data_dir": data_dir} if data_dir is not None else {}))
    results: dict = {"k_max": K_MAX, "restarts": RESTARTS, "time_limit_s": TIME_LIMIT_S, "seeds": list(SEEDS),
                     "real": measure(m, "real MCNS", verbose=verbose), "controls": []}
    for mode in MODES:
        for seed in SEEDS:
            label = f"{mode} seed {seed}"
            sm = shuffle_mcns(m, seed, mode)
            row = measure(sm, label, verbose=verbose)
            row["mode"], row["seed"] = mode, seed
            if mode == "configuration":
                row["degree_drift"] = configuration_drift(m, seed)
            results["controls"].append(row)
    if path is not None:
        path.write_text(json.dumps(results, indent=1))
    return results


def print_table(results: dict) -> None:
    rows = [{"mode": "real", "seed": "-", **results["real"]["latch_capacity"], **results["real"]["placement"]}]
    for c in results["controls"]:
        rows.append({"mode": c["mode"], "seed": c["seed"], **c["latch_capacity"], **c["placement"]})
    header = ["mode", "seed", "mutual_pairs", "max_disjoint_pairs", "strong_core", "carried", "edges", "hard", "hub", "neurons_placed"]
    widths = {h: max(len(h), *(len(str(r.get(h, ""))) for r in rows)) for h in header}
    print(" | ".join(h.ljust(widths[h]) for h in header))
    for r in rows:
        hard = f"{r['hard']['carried']}/{r['hard']['total']}" if isinstance(r.get("hard"), dict) else ""
        hub = f"{r['hub']['carried']}/{r['hub']['total']}" if isinstance(r.get("hub"), dict) else ""
        vals = {**r, "hard": hard, "hub": hub}
        print(" | ".join(str(vals.get(h, "")).ljust(widths[h]) for h in header))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    path = args.out or Path(__file__).resolve().parents[2] / "docs" / "h1_shuffle.json"
    results = run_all(path=path, verbose=not args.quiet)
    print_table(results)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
