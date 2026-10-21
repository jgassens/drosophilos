"""Summarize the LOCAL ECHO study without loading a connectome or simulating.

python -m drosophilos.bench.h_echo_report --place-dir data/stage_h/place \
    --fullgraph-dir data/stage_h/fullgraph --out docs/stage_h/echo_placement.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SETTINGS = {"w0": (0, 0, 0), **{f"out{w}": (0, w, 0) for w in (1, 2, 4, 8)},
            **{f"echo{w}": (0, 0, w) for w in (1, 2, 4)}}
LIVE_JOBS = {"w0": {"pub": 4048871, "cal": 4048872},
             "out1": {"pub": 4048877, "cal": 4048878},
             "out8": {"pub": 4048873, "cal": 4048874},
             "echo1": {"pub": 4048875, "cal": 4048876}}
CONTROLS = (
    ("w0", "sealed", "all_w0_calib_e025_i2_sealed.json", None),
    ("w0", "outputs-zeroed", "all_w0_calib_e025_i2_outzero_silent.json", None),
    ("echo1", "sealed", "all_echo1_cal_silent_sealed.json", 4049043),
    ("echo1", "outputs-zeroed", "all_echo1_cal_silent_outzero.json", 4049037),
    ("out8", "sealed", "all_out8_cal_silent_sealed.json", 4049044),
    ("out8", "outputs-zeroed", "all_out8_cal_silent_outzero.json", 4049038),
)
UNITS = ("abs_quanta", "count")
LAGS = (0, 1, 2, 3, 5, 10, 20, 30, 50)


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def distribution(values):
    return {"n": len(values), "median": float(np.median(values)) if values else None,
            "min": float(min(values)) if values else None,
            "max": float(max(values)) if values else None}


def lag_analysis(additions: list[dict]) -> dict:
    """Pearson correlations within each addition; positive lag means IN after OUT.

    Recenter both overlapping slices separately at every lag. Average correlations
    equally across eligible additions, never concatenate windows or pad their tails.
    Onset and return summaries also include constant series when defined.
    """
    result = {}
    for unit in UNITS:
        correlations = [[] for _ in range(51)]
        latencies, early, whole = [], [], []
        used = 0
        out_total = in_total = zero_bins = bins = near_in = 0
        for addition in additions:
            trace = addition["echo_trace"]
            if trace["bin_ms"] != 1:
                raise ValueError("lag analysis requires 1 ms bins")
            arrays = trace["per_ms"]
            outgoing = np.asarray(arrays["out"][unit], dtype=float)
            incoming = (np.asarray(arrays["in_exc"][unit], dtype=float)
                        + np.asarray(arrays["in_inh"][unit], dtype=float))
            if outgoing.ndim != 1 or outgoing.shape != incoming.shape:
                raise ValueError("OUT and IN must be equal-length one-dimensional arrays")
            if not np.all(np.isfinite(outgoing)) or not np.all(np.isfinite(incoming)):
                raise ValueError("non-finite trace")
            out_total += int(outgoing.sum())
            in_total += int(incoming.sum())
            bins += outgoing.size
            zero_bins += int(np.count_nonzero(outgoing == 0))
            prefix = np.concatenate(([0], np.cumsum(outgoing > 0)))
            recent_out = prefix[1:] - prefix[np.maximum(0, np.arange(outgoing.size) - 19)] > 0
            near_in += int(incoming[recent_out].sum())
            out_on = np.flatnonzero(outgoing > 0)
            in_on = np.flatnonzero(incoming > 0)
            if out_on.size and in_on.size:
                latencies.append(int(in_on[0] - out_on[0]))
            if out_on.size:
                window = slice(int(out_on[0]), int(out_on[0]) + 20)
                early.append(ratio(incoming[window].sum(), outgoing[window].sum()))
            if outgoing.sum():
                whole.append(ratio(incoming.sum(), outgoing.sum()))
            eligible = False
            for lag in range(51):
                if outgoing.size - lag < 3:
                    continue
                x = outgoing[:outgoing.size - lag]
                y = incoming[lag:]
                x, y = x - x.mean(), y - y.mean()
                norm = np.linalg.norm(x) * np.linalg.norm(y)
                if not norm:
                    continue
                correlations[lag].append(float(np.clip(np.dot(x, y) / norm, -1, 1)))
                eligible = True
            used += int(eligible)
        means = [float(np.mean(v)) if v else None for v in correlations]
        valid = [lag for lag, value in enumerate(means) if value is not None]
        peak = max(valid, key=lambda lag: means[lag]) if valid else None
        result[unit] = {
            "correlation": {
                "n_additions_used": used,
                "r_at_lag_ms": {str(lag): means[lag] for lag in LAGS},
                "n_at_lag_ms": {str(lag): len(correlations[lag]) for lag in range(51)},
                "mean_r_0_to_50_ms": means,
                "peak_lag_ms": peak, "peak_r": means[peak] if peak is not None else None,
            },
            "onset_latency_ms": distribution(latencies),
            "early_20_ms_return_ratio": distribution(early),
            "per_addition_whole_window_return_ratio": distribution(whole),
            "whole_window_return_ratio": ratio(in_total, out_total),
            "out_zero_bins": zero_bins, "total_bins": bins,
            "out_positive_bin_fraction": ratio(bins - zero_bins, bins),
            "recomputed_any_out_20_ms_fraction": ratio(near_in, in_total),
        }
    return result


def outcome_summary(data):
    summary = data["summary"]
    return {"correct": summary["correct"], "n": summary["n"],
            "outcomes": summary["outcomes"]}


def summarize_live(data):
    echo = data["echo_trace"]
    totals = {name: dict(echo[name]) for name in ("out", "in_exc", "in_inh")}
    totals["in"] = {unit: totals["in_exc"][unit] + totals["in_inh"][unit] for unit in UNITS}
    return {**outcome_summary(data), "args": data["args"], "params": data["params"],
            "weight_units": echo.get("weight_units"),
            "totals": totals,
            "return_ratio": {u: ratio(totals["in"][u], totals["out"][u]) for u in UNITS},
            "inhibitory_share_in_abs_quanta": ratio(totals["in_inh"]["abs_quanta"],
                                                    totals["in"]["abs_quanta"]),
            "stored_metrics_record_only": echo["metrics"],
            "derived_lag": lag_analysis(data["additions"])}


def read_optional(path):
    return json.loads(path.read_text()) if path.exists() else None


def build_report(place_dir: Path, fullgraph_dir: Path) -> dict:
    rows = []
    for tag, weights in SETTINGS.items():
        filename = f"all_{'w0_rescore' if tag == 'w0' else tag}.json"
        data = read_optional(place_dir / filename)
        row = {"placement": tag, "source": filename,
               "placement_job_id": 4048888 if tag == "w0" else None,
               "penalty": dict(zip(("isolation_in", "isolation_out", "echo_weight"), weights)),
               "carried": None, "designed_edges": None, "carried_fraction": None,
               "host_echo_paths": None, "host_echo_neurons": None,
               "live": {}, "live_sources": {}, "live_job_ids": {}, "live_status": {}}
        if data is not None:
            p = data["placement"]
            row.update(carried=p["carried"], designed_edges=p["edges"],
                       carried_fraction=ratio(p["carried"], p["edges"]),
                       host_echo_paths=p["host_echo_paths"], host_echo_neurons=p["host_echo_neurons"],
                       neurons=p["neurons"], region=data["region"], source_commit=data["git_commit"],
                       isolated_condition_A={k: data["conditions"]["A"]["fresh"][k]
                                             for k in ("correct", "n")})
        for surround in ("pub", "cal"):
            name = f"all_{tag}_echotrace_{surround}_silent.json"
            job = LIVE_JOBS.get(tag, {}).get(surround)
            live = read_optional(fullgraph_dir / name) if job is not None else None
            row["live"][surround] = summarize_live(live) if live is not None else None
            row["live_sources"][surround] = name if job is not None else None
            row["live_job_ids"][surround] = job
            row["live_status"][surround] = ("finished" if live is not None else
                                            "not finished" if job is not None else "not measured")
        rows.append(row)
    controls = []
    for tag, condition, filename, job in CONTROLS:
        data = read_optional(fullgraph_dir / filename)
        controls.append({"placement": tag, "condition": condition, "surround": "cal",
                         "source": filename, "job_id": job,
                         "status": "finished" if data is not None else "not finished",
                         "result": {**outcome_summary(data), "args": data["args"]}
                         if data is not None else None})
    return {"study": "LOCAL ECHO", "code_commit": "0b7b166",
            "source_directories": {"place": str(place_dir), "fullgraph": str(fullgraph_dir)},
            "surrounds": {"pub": {"exc": 1, "inh": 1, "cap": None, "drive": "silent"},
                          "cal": {"exc": 0.25, "inh": 2, "cap": None, "drive": "silent"}},
            "analysis": {"bin_ms": 1, "max_lag_ms": 50, "early_window_ms": 20,
                         "minimum_overlap_bins": 3,
                         "correlation": "Pearson per overlapping addition slice, equal addition mean; earliest peak tie",
                         "early_window": "OUT onset bin through onset+19 inclusive; truncate at addition end",
                         "undefined": "null; zero variance skipped separately at each lag",
                         "stored_near_window_true_cutoff_ms": [19.1, 20.9]},
            "placements": rows, "controls": controls}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--place-dir", required=True, type=Path)
    parser.add_argument("--fullgraph-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report = build_report(args.place_dir, args.fullgraph_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
