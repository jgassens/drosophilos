"""Stage H1 isolation sweep: what the 4-bit adder's placement gives up in coverage when the
search is told to prefer hosts the rest of the brain reaches less. Read first: the last section
of docs/h1_placement.md -- inside the whole brain the coverage-only placement computes nothing
while its hosts' 327,497 anatomical inputs are live; those hosts average ~530 external inputs
each, because the search never looked at exposure.

For isolation_weight in (0, 0.5, 1, 2, 4): places the adder netlist (`place_netlist`,
restarts=4, time_limit_s=300, seed=0, k_max 4, best restart by objective = carried edges minus
the weighted log-exposure of the hosts), records the coverage (total / hard / hub carried),
neurons placed, parasitic edges, the hosts' external input and output edges (what the
whole-brain control zeroes) and synapses (total, mean, max, p90), the hosts' superclasses and
the search time; then builds the condition-A image and runs the isolated conditions A (every
missing edge added, parasitic zeroed) and B (carried edges only) over 20 additions, as
bench/h1_sweep.py does.

Writes docs/h1_isolation.json and saves the mapping of the weight with the smallest external
input total whose condition A still computes 20/20 as docs/h1_placement_mapping_isolated.json
(the format `embed_image.load_mapping` reads; `bench/h1_fullgraph.py --mapping` takes it into
the whole brain -- that run is not part of this script). `python -m drosophilos.bench.h1_isolation`
reproduces it: five placements at up to 300 s each and five images x two conditions x 20
additions, all on the isolated ~614-neuron image.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..connectome.embed_h0 import Policy
from ..connectome.embed_image import BROADCAST, build_image, none_missing, run_h1_conditions, save_mapping
from ..connectome.embed_netlist import Placement, host_types, place_netlist
from ..connectome.mcns import MCNS, load_mcns
from ..lib.adder import build_adder_channel
from ..lib.netlist import Netlist
from ..sim.model import Params

WEIGHTS = (0.0, 0.5, 1.0, 2.0, 4.0)
K_MAX = 4.0
HUBS = BROADCAST  # ("Q.reset_inh", "P.reset_inh", "P.wd.cancel_inh")
RESTARTS, TIME_LIMIT_S, SEED, N_CASES = 4, 300.0, 0, 20
DOCS = Path(__file__).resolve().parents[2] / "docs"


def adder():
    return build_adder_channel(Params(), 4, ordered=True, watchdog_hops=100)


def coverage_split(net: Netlist, pl: Placement) -> dict:
    """Carried designed edges split as bench/h1_sweep.py does: "hub" = an edge touching one of
    the three broadcast neurons, "hard" = every other designed edge (netlist rows, not merged)."""
    hub_ids = {net.roles.index(h) for h in HUBS}
    missing_pairs = {(s, d) for s, d, _, _ in pl.missing}
    out = {"hub_carried": 0, "hub_total": 0, "hard_carried": 0, "hard_total": 0}
    for s, d in zip(net.src, net.dst):
        s, d = int(s), int(d)
        carried = (s, d) not in missing_pairs and s in pl.mapping and d in pl.mapping
        key = "hub" if (s in hub_ids or d in hub_ids) else "hard"
        out[f"{key}_total"] += 1
        out[f"{key}_carried"] += int(carried)
    return out


def host_superclasses(pl: Placement, m: MCNS) -> dict:
    sc = m.neurons["superclass"].fillna("?").to_numpy()
    counts: dict = {}
    for r in pl.mapping.values():
        counts[str(sc[r])] = counts.get(str(sc[r]), 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def run_one(weight: float, m: MCNS, verbose: bool = True) -> tuple[dict, Placement]:
    policy = Policy(k_max=K_MAX)
    ch = adder()
    net = ch.net
    t0 = time.time()
    pl = place_netlist(net, m, policy=policy, restarts=RESTARTS, time_limit_s=TIME_LIMIT_S, seed=SEED, verbose=False,
                       isolation_weight=weight)
    search_seconds = time.time() - t0

    zero = Policy(k_max=K_MAX, zero_parasitic=True)
    conditions = {
        "A": ("all designed edges (missing ones as Profile 3), parasitic zeroed", lambda e: True, zero),
        "B": ("carried edges only", none_missing, zero),
    }
    res = run_h1_conditions(ch, m, pl, width=4, n_cases=N_CASES, seed=SEED, conditions=conditions, verbose=verbose)
    img_A = build_image(net, m, pl, zero, profile3=lambda e: True)

    ex = pl.exposure
    out = {
        "isolation_weight": weight,
        "placement": {
            "carried": pl.carried, "edges": pl.edges, "carried_fraction": round(pl.carried / max(1, pl.edges), 4),
            **coverage_split(net, pl),
            "neurons_placed": len(pl.mapping), "neurons_total": net.n,
            "parasitic_to_zero": pl.parasitic,
            "objective": round(pl.objective, 2), "strategy": pl.strategy, "order_note": pl.order_note,
            "search_seconds": round(search_seconds, 1),
        },
        "exposure": {
            "external_in_edges": ex["in_edges"], "external_out_edges": ex["out_edges"],  # h1_fullgraph zeroes these
            "external_in_synapses": ex["in_total"], "external_out_synapses": ex["out_total"],
            "external_in_mean": ex["in_mean"], "external_out_mean": ex["out_mean"],
            "external_in_max": ex["in_max"], "external_out_max": ex["out_max"],
            "external_in_p90": ex["in_p90"], "external_out_p90": ex["out_p90"],
        },
        "host_superclasses": host_superclasses(pl, m),
        "host_types": host_types(pl, net, m),
        "image_A": {"profile3_edges": img_A.counts["profile3_edges"], "parasitic_zeroed": img_A.counts["parasitic_zeroed"],
                    "synthetic_neurons": img_A.counts["synthetic_neurons"], "scale_max": img_A.counts["scale_max"]},
        "conditions": {
            name: {"correct": r["correct"], "n": r["n"], "wrong": r["wrong"], "faults": r["faults"], "timeouts": r["timeouts"],
                   "no_accept": r["no_accept"], "profile3_edges": r["profile3_edges"], "parasitic_kept": r["parasitic_kept"],
                   "chained_correct": r.get("chained_correct")}
            for name, r in res.items()
        },
        "missing_by_class_top": dict(sorted(pl.missing_by_class.items(), key=lambda kv: -kv[1][0])[:10]),
    }
    return out, pl


def table(results: dict) -> str:
    head = ("| weight | carried | hard | hub | placed | parasitic | ext. input edges | ext. input synapses (mean / p90 / max) "
            "| ext. output edges | ext. output synapses (mean) | A | B | seconds |")
    rows = [head, "|" + "---|" * 13]
    for w, r in results.items():
        p, e, c = r["placement"], r["exposure"], r["conditions"]
        rows.append(f"| {w} | {p['carried']} / {p['edges']} ({100 * p['carried_fraction']:.1f} %) "
                    f"| {p['hard_carried']} / {p['hard_total']} | {p['hub_carried']} / {p['hub_total']} "
                    f"| {p['neurons_placed']} / {p['neurons_total']} | {p['parasitic_to_zero']:,} "
                    f"| {e['external_in_edges']:,} "
                    f"| {e['external_in_synapses']:,} ({e['external_in_mean']} / {e['external_in_p90']} / {e['external_in_max']:,}) "
                    f"| {e['external_out_edges']:,} | {e['external_out_synapses']:,} ({e['external_out_mean']}) "
                    f"| {c['A']['correct']}/{c['A']['n']} | {c['B']['correct']}/{c['B']['n']} | {p['search_seconds']} |")
    return "\n".join(rows)


def pick_isolated(results: dict) -> str | None:
    """The weight with the smallest external input total whose condition A computes every sum."""
    ok = [(r["exposure"]["external_in_synapses"], w) for w, r in results.items()
          if r["conditions"]["A"]["correct"] == r["conditions"]["A"]["n"]]
    return min(ok)[1] if ok else None


def run_sweep(weights=WEIGHTS, verbose: bool = True, out_dir: Path = DOCS) -> dict:
    m = load_mcns()
    results: dict = {}
    placements: dict = {}
    for w in weights:
        if verbose:
            print(f"[h1_isolation] isolation_weight = {w}", flush=True)
        results[str(w)], placements[str(w)] = run_one(w, m, verbose=verbose)
        if verbose:
            p, e = results[str(w)]["placement"], results[str(w)]["exposure"]
            print(f"[h1_isolation] w={w}: carried {p['carried']}/{p['edges']} placed {p['neurons_placed']} "
                  f"external inputs {e['external_in_synapses']:,} (mean {e['external_in_mean']}) "
                  f"outputs {e['external_out_synapses']:,} in {p['search_seconds']} s", flush=True)
    chosen = pick_isolated(results)
    out = {
        "netlist": "build_adder_channel(Params(), 4, ordered=True, watchdog_hops=100)",
        "search": f"place_netlist(net, m, Policy(k_max={K_MAX:g}), restarts={RESTARTS}, time_limit_s={TIME_LIMIT_S:g}, seed={SEED}, "
                  f"isolation_weight=w); best restart by objective (carried - w * sum(log1p(in_ext) + 0.25 log1p(out_ext)))",
        "conditions": f"isolated image, {N_CASES} additions, seed {SEED}: A all missing edges added and parasitic zeroed; B carried only",
        "isolated_mapping": {"weight": chosen, "rule": "smallest external input total whose condition A computes every sum",
                             "path": "docs/h1_placement_mapping_isolated.json"},
        "results": results,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "h1_isolation.json").write_text(json.dumps(out, indent=1))
    if chosen is not None:
        ch = adder()
        pl = placements[chosen]
        save_mapping(out_dir / "h1_placement_mapping_isolated.json", ch.net, m, pl, meta={
            "netlist": out["netlist"], "connectome": m.version,
            "search": f"place_netlist(net, m, Policy(k_max={K_MAX:g}), restarts={RESTARTS}, time_limit_s={TIME_LIMIT_S:g}, "
                      f"seed={SEED}, isolation_weight={chosen})",
            "isolation_weight": float(chosen), "exposure": pl.exposure,
            "chosen_because": out["isolated_mapping"]["rule"],
        })
    print(table(results))
    print(f"isolated mapping: isolation_weight = {chosen}")
    return out


def main() -> None:
    run_sweep()
    print(f"wrote {DOCS / 'h1_isolation.json'}")


if __name__ == "__main__":
    main()
