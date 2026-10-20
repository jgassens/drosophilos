"""Stage H: place and simulate one regional configuration of the ordered flip-flop adder.

Reproduce with `python -m drosophilos.bench.h_stage_place --region optic_lobe
--isolation-weight 1 --cases 50 --out data/stage_h/place/optic_lobe_w1.json
--mapping-out data/stage_h/place/optic_lobe_w1_mapping.json [--data-dir DIR]`.
Configuration F is the default; --no-config-f builds the earlier flip-flop adder.
Writes placement, Profile 3, whole-brain/regional exposure and isolated A/B simulation
results to JSON, plus a mapping with the builder metadata needed for whole-brain runs.
`--summarize data/stage_h/place/*.json --table-out ranking.md` ranks saved configurations.
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np

from ..connectome.embed_image import (adder_cases, build_image, h1_conditions, run_h1_conditions,
                                     save_mapping, simulate_channel)
from ..connectome.embed_netlist import Policy, _Anatomy, exposure_of, place_netlist
from ..connectome.mcns import MCNS, load_mcns
from ..lib.adder import build_adder_channel
from ..protocol.latch import DALE_BIASED_VERSION, DALE_FEEDBACK_VERSION, DALE_REFRACTORY_VERSION
from ..sim.model import Params

REGIONS = {
    "optic_lobe": ("ol_intrinsic", "ol_sensory", "visual_projection", "visual_centrifugal"),
    "central_brain": ("cb_intrinsic", "cb_sensory", "cb_sensory_tbc", "cb_motor", "cb_endocrine",
                      "descending_neuron"),
    "vnc": ("vnc_intrinsic", "vnc_sensory", "vnc_sensory_tbc", "vnc_motor", "vnc_efferent",
            "vnc_endocrine", "vnc_tbc", "ascending_neuron", "sensory_ascending"),
}
K_MAX = 4.0


def region_mask(neurons, region: str) -> np.ndarray:
    """Select positional MCNS neuron indices without slicing the anatomical graph."""
    if region not in ("all", *REGIONS):
        raise ValueError(f"unknown region {region!r}")
    assigned = {s for values in REGIONS.values() for s in values}
    unknown = sorted(set(neurons["superclass"].fillna("<null>").astype(str)) - assigned)
    if unknown:
        warnings.warn(f"superclass values not assigned to a region: {', '.join(unknown)}", UserWarning, stacklevel=2)
    return np.ones(len(neurons), bool) if region == "all" else neurons["superclass"].isin(REGIONS[region]).to_numpy()


def mapping_meta(region: str, isolation_weight: float, restarts: int, seed: int,
                 config_f: bool = True, width: int = 4, latch_feedback: str = "autapse") -> dict:
    """Top-level save_mapping contract shared with the whole-brain bench."""
    if latch_feedback not in ("autapse", "dale_private", "dale_biased", "dale_refractory"):
        raise ValueError(f"unsupported latch_feedback: {latch_feedback!r}")
    if latch_feedback != "autapse" and not config_f:
        raise ValueError(f"{latch_feedback} latch_feedback requires configuration F")
    meta = {"builder": "build_adder_channel", "width": int(width), "ordered": True,
            "watchdog_hops": 100, "storage": "flipflop", "rate_robust": bool(config_f),
            "experimental_autapse": bool(config_f), "ordered_xor_oneshot": bool(config_f),
            "latch_feedback": str(latch_feedback),
            "k_max": K_MAX, "isolation_weight": float(isolation_weight), "region": str(region),
            "restarts": int(restarts), "seed": int(seed)}
    if latch_feedback == "dale_private":
        meta["dale_feedback_version"] = DALE_FEEDBACK_VERSION
    elif latch_feedback == "dale_biased":
        meta["dale_biased_version"] = DALE_BIASED_VERSION
    elif latch_feedback == "dale_refractory":
        meta["dale_refractory_version"] = DALE_REFRACTORY_VERSION
    return meta


def stats(values) -> dict:
    values = np.asarray(values, np.int64)
    return {"mean": float(values.mean()) if values.size else 0.0,
            "median": float(np.median(values)) if values.size else 0.0,
            "p90": float(np.percentile(values, 90)) if values.size else 0.0,
            "max": int(values.max()) if values.size else 0, "total": int(values.sum())}


def host_exposure(m: MCNS, hosts, mask: np.ndarray, policy: Policy) -> dict:
    """Per-host exposure from/to all non-hosts, and from/to neurons outside the region.

    Every anatomical edge is retained; only the partner selection changes for the second
    set of statistics. Edge counts are distinct anatomical pairs, synapses are their counts.
    """
    A = _Anatomy(m, policy)
    hosts = np.asarray(sorted(hosts), np.int64)
    whole = exposure_of(A, hosts)
    # Slice the small host boundary first, rather than copying the entire outside graph.
    incoming = A.C[:, hosts][~mask]
    outgoing = A.C[hosts][:, ~mask]
    outside = (np.asarray(incoming.sum(axis=0)).ravel(), np.asarray(outgoing.sum(axis=1)).ravel(),
               np.asarray(incoming.getnnz(axis=0)).ravel(), np.asarray(outgoing.getnnz(axis=1)).ravel())
    names = ("input_synapses", "output_synapses", "input_edges", "output_edges")
    return {"hosts": int(hosts.size),
            "outside_hosts": {name: stats(v) for name, v in zip(names, whole)},
            "outside_region": {name: stats(v) for name, v in zip(names, outside)}}


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _counts(series, top: int | None = None) -> dict:
    items = sorted(((str(k), int(v)) for k, v in series.fillna("?").value_counts().items()),
                   key=lambda kv: (-kv[1], kv[0]))
    return dict(items if top is None else items[:top])


def run_one(m: MCNS, region: str, isolation_weight: float, n_cases: int = 50, restarts: int = 4,
            time_limit_s: float = 600.0, seed: int = 0, conditions: str = "AB", config_f: bool = True,
            mapping_path: Path | None = None, width: int = 4, latch_feedback: str = "autapse") -> dict:
    """One regional placement and isolated A/B runs. Width is configurable for synthetic tests."""
    if not conditions or set(conditions) - set("AB"):
        raise ValueError("conditions must select A and/or B")
    if n_cases < 1 or restarts < 1 or time_limit_s <= 1:
        raise ValueError("cases and restarts must be positive; time limit must exceed one second")
    mask = region_mask(m.neurons, region)
    if not mask.any():
        raise ValueError(f"region {region!r} contains no neurons")
    meta = mapping_meta(region, isolation_weight, restarts, seed, config_f, width, latch_feedback)
    feedback_version = (
        {"dale_feedback_version": meta["dale_feedback_version"]}
        if latch_feedback == "dale_private" else
        {"dale_biased_version": meta["dale_biased_version"]}
        if latch_feedback == "dale_biased" else
        {"dale_refractory_version": meta["dale_refractory_version"]}
        if latch_feedback == "dale_refractory" else {})
    ch = build_adder_channel(Params(), width, ordered=True, watchdog_hops=100, storage="flipflop",
                             rate_robust=config_f, experimental_autapse=config_f, ordered_xor_oneshot=config_f,
                             latch_feedback=latch_feedback, **feedback_version)
    net = ch.net
    policy = Policy(k_max=K_MAX)
    t0 = time.monotonic()
    pl = place_netlist(net, m, policy=policy, restarts=restarts, time_limit_s=time_limit_s, seed=seed,
                       isolation_weight=isolation_weight, allowed_hosts=mask, region=region, verbose=False)
    search_seconds = time.monotonic() - t0
    if mapping_path is not None:
        mapping_path = Path(mapping_path)
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        save_mapping(mapping_path, net, m, pl, meta)
    conds = h1_conditions(policy)
    # A supplies the complete Profile 3 accounting even in a B-only run.
    image_a = build_image(net, m, pl, conds["A"][2], profile3=conds["A"][1])
    classes = image_a.counts["profile3_by_class"]
    autapses = sum(v for k, v in classes.items() if k.startswith("autapse "))
    top = sorted(((k, v) for k, v in classes.items() if not k.startswith("autapse ")),
                 key=lambda kv: (-kv[1], kv[0]))[:10]
    hosts = sorted(pl.mapping.values())
    out = {**meta, "config_f": config_f, "n_cases": n_cases, "time_limit_s": time_limit_s,
           "region_superclasses": list(REGIONS[region]) if region != "all" else sorted(m.neurons["superclass"].dropna().astype(str).unique()),
           "region_neurons": int(mask.sum()), "placement": pl.summary(net),
           "carried_edge_pairs": pl.carried, "designed_edge_pairs": pl.edges,
           "neurons_placed": len(pl.mapping), "neurons_unplaced": len(pl.unplaced),
           "profile3": {"total": image_a.counts["profile3_edges"], "autapses": autapses,
                        "non_autapses": image_a.counts["profile3_edges"] - autapses,
                        "top_non_autapse_classes": dict(top), "by_class": classes},
           "parasitic_edges_zeroed": image_a.counts["parasitic_zeroed"],
           "exposure": host_exposure(m, hosts, mask, policy),
           "hosts": {"superclasses": _counts(m.neurons.iloc[hosts]["superclass"]),
                     "top_cell_types": _counts(m.neurons.iloc[hosts]["type"], 10)},
           "search_seconds": search_seconds, "git_commit": git_commit()}
    print(f"[stage-h] {region}, w={isolation_weight}: carried {pl.carried}/{pl.edges}, "
          f"placed {len(hosts)}/{net.n}, Profile 3 {out['profile3']['total']}, {search_seconds:.1f} s", flush=True)
    selected = {k: v for k, v in conds.items() if k in conditions}
    fresh = run_h1_conditions(ch, m, pl, width=width, n_cases=n_cases, seed=seed, policy=policy, conditions=selected)
    _, words, expected = adder_cases(width, n_cases, seed)
    out["conditions"] = {}
    for name, (_, select, pol) in selected.items():
        image = image_a if name == "A" else build_image(net, m, pl, pol, profile3=select)
        chained = simulate_channel(ch, image, words, expected, fresh_each=False)
        chained.pop("records")
        # Fresh results keep the H1 tally; the complete chained tally includes failures too.
        fresh[name].pop("chained_correct", None)
        fresh[name].pop("chained_cycle_ms", None)
        out["conditions"][name] = {"fresh": fresh[name], "chained": chained}
    return out


def _all_correct(row: dict, condition: str) -> bool:
    result = row.get("conditions", {}).get(condition, {})
    n = row["n_cases"]
    return all(result.get(mode, {}).get("correct", 0) == n for mode in ("fresh", "chained"))


def _result_cell(row: dict, condition: str) -> str:
    result = row.get("conditions", {}).get(condition)
    if result is None:
        return "not run"
    parts = []
    for mode in ("fresh", "chained"):
        r = result[mode]
        latency = r.get("accept_ms", [])
        ms = f"{np.mean(latency):.1f}" if latency else "—"
        parts.append(f"{r['correct']}/{r['n']} (w/f/t/noA {r['wrong']}/{r['faults']}/{r['timeouts']}/{r['no_accept']}; {ms} ms)")
    return " / ".join(parts)


def summarize(paths, table_path: Path) -> list[dict]:
    """Rank results by A all-correct, coverage, Profile 3 count, then whole-brain input mean.

    Mapping JSONs are deliberately skipped so the documented *.json glob is safe.
    """
    rows = []
    for path in sorted(set(Path(p) for p in paths)):
        row = json.loads(path.read_text())
        if "mapping" in row:
            continue
        if "conditions" not in row or "profile3" not in row:
            raise ValueError(f"{path} is not a Stage H placement result")
        rows.append({**row, "source": str(path)})
    rows.sort(key=lambda r: (not _all_correct(r, "A"),
                             -r["carried_edge_pairs"] / max(1, r["designed_edge_pairs"]),
                             r["profile3"]["total"], r["exposure"]["outside_hosts"]["input_synapses"]["mean"],
                             r["source"]))
    lines = ["A all-correct means every fresh and chained case is correct. Results show fresh / chained: "
             "correct/requested (wrong/fault/timeout/no ACCEPT; mean ACCEPT latency). Exposure is per host; "
             "outside input/output refers to all non-host neurons.", "",
             "| Rank | Region | F | w | Pool | Placed / unplaced | Carried / designed (%) | P3 total / autapse / other | "
             "Parasitic zeroed | Outside input syn mean / p90 / max | Outside output syn mean | "
             "Outside in/out edges total | Beyond-region in/out syn mean | A | B | Search s | Restarts / seed | Git | Source |",
             "|" + " --- |" * 19]
    for rank, r in enumerate(rows, 1):
        ex = r["exposure"]["outside_hosts"]
        regional = r["exposure"]["outside_region"]
        p3 = r["profile3"]
        pct = 100 * r["carried_edge_pairs"] / max(1, r["designed_edge_pairs"])
        cells = [rank, r["region"], "yes" if r["config_f"] else "no", r["isolation_weight"], r["region_neurons"],
                 f"{r['neurons_placed']} / {r['neurons_unplaced']}",
                 f"{r['carried_edge_pairs']} / {r['designed_edge_pairs']} ({pct:.1f}%)",
                 f"{p3['total']} / {p3['autapses']} / {p3['non_autapses']}", r["parasitic_edges_zeroed"],
                 f"{ex['input_synapses']['mean']:.1f} / {ex['input_synapses']['p90']:.1f} / {ex['input_synapses']['max']}",
                 f"{ex['output_synapses']['mean']:.1f}", f"{ex['input_edges']['total']} / {ex['output_edges']['total']}",
                 f"{regional['input_synapses']['mean']:.1f} / {regional['output_synapses']['mean']:.1f}",
                 _result_cell(r, "A"), _result_cell(r, "B"), f"{r['search_seconds']:.1f}",
                 f"{r['restarts']} / {r['seed']}", (r.get("git_commit") or "unknown")[:12], r["source"]]
        lines.append("| " + " | ".join(str(c).replace("|", "\\|") for c in cells) + " |")
    table_path = Path(table_path)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text("\n".join(lines) + "\n")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--region", choices=("all", *REGIONS), default="all")
    ap.add_argument("--isolation-weight", type=float, default=0.0)
    ap.add_argument("--cases", type=int, default=50)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--mapping-out", type=Path)
    ap.add_argument("--data-dir", type=Path)
    ap.add_argument("--restarts", type=int, default=4)
    ap.add_argument("--time-limit", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conditions", default="AB")
    ap.add_argument("--no-config-f", action="store_true")
    ap.add_argument("--latch-feedback", choices=("autapse", "dale_private", "dale_biased", "dale_refractory"), default="autapse",
                    help=("dale_private and dale_biased are EXPERIMENTAL and NOT autapse replacements: "
                          "dale_private fails the three-sigma corners and reload; dale_biased passes the "
                          "512-corner and reload bar (0/0/0, 118/128) but FAILS the separate 44-step "
                          "rate floor (corner minimum interval 33 steps); dale_refractory is REJECTED: "
                          "corner failures 4/11/15, +700 reload 114/128, minimum interval 33 < 44"))
    ap.add_argument("--summarize", nargs="+", metavar="JSON")
    ap.add_argument("--table-out", type=Path)
    a = ap.parse_args()
    if a.summarize:
        if a.table_out is None:
            ap.error("--summarize requires --table-out")
        paths = [Path(p) for pattern in a.summarize for p in glob.glob(pattern)]
        if not paths:
            ap.error("--summarize matched no files")
        summarize(paths, a.table_out)
        print(f"wrote {a.table_out}")
        return
    if a.out is None or a.mapping_out is None:
        ap.error("placement requires --out and --mapping-out")
    if a.latch_feedback != "autapse" and a.no_config_f:
        ap.error(f"--latch-feedback {a.latch_feedback} requires configuration F")
    m = load_mcns(a.data_dir) if a.data_dir is not None else load_mcns()
    out = run_one(m, a.region, a.isolation_weight, a.cases, a.restarts, a.time_limit, a.seed,
                  a.conditions, not a.no_config_f, a.mapping_out, latch_feedback=a.latch_feedback)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(f"wrote {a.out} and {a.mapping_out}")


if __name__ == "__main__":
    main()
