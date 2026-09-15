"""Stage H1 weight-bound sweep: how the 4-bit adder's placement and simulation change with
`k_max` (the H0 Policy bound: an anatomical edge carries a designed edge of |q| quanta when
count * 16 * k >= |q|, k <= k_max). Read first: docs/h1_placement.md (measured at k_max = 4).

For k_max in (4, 8, 16, 32): places the adder netlist (`place_netlist`, restarts=4,
time_limit_s=300, seed=0, best of the restarts), records the placement's coverage and the
image's simulation under three conditions:
  A: every missing edge added, parasitic zeroed (the mapping computes)
  B: carried edges only, parasitic zeroed (does the bare placement compute?)
  G: carried + only the three broadcast hubs' missing edges added, parasitic zeroed
     (condition C's selector, at this k_max)

Writes docs/h1_sweep.json; `python -m drosophilos.bench.h1_sweep` reproduces it (four
placements at up to 300 s each, four images x three conditions x 20 additions each).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..connectome.embed_h0 import Policy
from ..connectome.embed_image import BROADCAST, build_image, from_sources, none_missing, run_h1_conditions
from ..connectome.embed_netlist import host_types, place_netlist
from ..connectome.mcns import MCNS, load_mcns
from ..lib.adder import build_adder_channel
from ..sim.model import Params

K_MAX_VALUES = (4, 8, 16, 32)
HUBS = BROADCAST  # ("Q.reset_inh", "P.reset_inh", "P.wd.cancel_inh")
LOOP_QUANTA = 3621  # the latch loop's designed quanta (docs/h1_placement.md): the parasitic
#   threshold a report reads as "could this edge alone carry a latch loop at this k_max?"


def k_histogram(scales: dict, k_max: float, n_bins: int = 8) -> dict:
    """Histogram of the carried edges' scale k = |q| / (count * 16), as fractions of this
    sweep point's k_max (embed_image.SCALE_BINS is fixed at <=4.0 and would silently drop
    everything above it once k_max > 4)."""
    edges = [k_max * (i + 1) / n_bins for i in range(n_bins)]
    hist = {}
    lo = 0.0
    for hi in edges:
        hist[f"({lo:.2f}, {hi:.2f}]"] = sum(1 for k in scales.values() if lo < k <= hi)
        lo = hi
    return hist


def run_one(k_max: float, m: MCNS, verbose: bool = True) -> dict:
    policy = Policy(k_max=float(k_max))
    ch = build_adder_channel(Params(), 4, ordered=True, watchdog_hops=100)
    net = ch.net

    t0 = time.time()
    pl = place_netlist(net, m, policy=policy, restarts=4, time_limit_s=300.0, seed=0, verbose=False)
    search_seconds = time.time() - t0

    hub_ids = {net.roles.index(h) for h in HUBS}
    missing_pairs = {(s, d) for s, d, _, _ in pl.missing}
    hub_total = hub_carried = hard_total = hard_carried = 0
    for s, d in zip(net.src, net.dst):
        s, d = int(s), int(d)
        is_hub = s in hub_ids or d in hub_ids
        carried = (s, d) not in missing_pairs and s in pl.mapping and d in pl.mapping
        if is_hub:
            hub_total += 1
            hub_carried += carried
        else:
            hard_total += 1
            hard_carried += carried

    conditions = {
        "A": ("all designed edges (missing ones as Profile 3), parasitic zeroed", lambda e: True, policy.k_max),
        "B": ("carried edges only", none_missing, policy.k_max),
        "G": ("carried + the three broadcast hubs' missing edges", from_sources(HUBS), policy.k_max),
    }
    conditions = {k: (desc, sel, Policy(k_max=float(k_max), zero_parasitic=True)) for k, (desc, sel, _) in conditions.items()}
    res = run_h1_conditions(ch, m, pl, width=4, n_cases=20, seed=0, conditions=conditions, verbose=verbose)

    img_A = build_image(net, m, pl, Policy(k_max=float(k_max), zero_parasitic=True), profile3=lambda e: True)
    loop_req = policy.req_count(LOOP_QUANTA)

    out = {
        "k_max": k_max,
        "placement": {
            "carried": pl.carried, "carried_fraction": round(pl.carried / net.nnz, 4),
            "hard_carried": hard_carried, "hard_total": hard_total,
            "hub_carried": hub_carried, "hub_total": hub_total,
            "neurons_placed": len(pl.mapping), "neurons_total": net.n,
            "parasitic_to_zero": pl.parasitic,
            "parasitic_ge_loop_threshold": sum(1 for e in img_A.parasitic if e.count >= loop_req),
            "search_seconds": round(search_seconds, 1),
        },
        "host_types": host_types(pl, net, m),
        "k_histogram": k_histogram(img_A.scales, float(k_max)),
        "k_observed_max": img_A.counts["scale_max"],
        "conditions": {
            name: {"correct": r["correct"], "n": r["n"], "profile3_edges": r["profile3_edges"],
                   "parasitic_kept": r["parasitic_kept"], "no_accept": r["no_accept"],
                   "chained_correct": r.get("chained_correct")}
            for name, r in res.items()
        },
    }
    if res["B"]["correct"] < res["B"]["n"]:
        out["missing_by_class_condition_B"] = dict(sorted(pl.missing_by_class.items(), key=lambda kv: -kv[1][0])[:15])
    return out


def run_sweep(k_max_values=K_MAX_VALUES, verbose: bool = True) -> dict:
    m = load_mcns()
    results = {}
    for k in k_max_values:
        if verbose:
            print(f"[h1_sweep] k_max = {k}", flush=True)
        results[str(k)] = run_one(k, m, verbose=verbose)
    return results


def main() -> None:
    out = run_sweep()
    path = Path(__file__).resolve().parents[2] / "docs" / "h1_sweep.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
