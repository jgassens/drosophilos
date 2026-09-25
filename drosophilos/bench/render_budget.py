"""Audit saved render_doom JSON budgets; never run a simulator or infer frame latency.

Example:
    python -m drosophilos.bench.render_budget docs/a2/doom2_40_h200.json \
        docs/a2/doom4_24na_h200.json --target-seconds 60

Only the Python standard library is required. Timings are amortized over the
requested frame count, not measured first-frame or input-to-display latency.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

STATUS_FIELDS = ("wrong", "missing", "bad_outputs", "faults", "timeouts")


def _number(data: Mapping[str, Any], key: str, *, integer: bool = False,
            allow_zero: bool = False) -> int | float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number")
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{key} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    if integer and int(value) != value:
        raise ValueError(f"{key} must be an integer")
    return int(value) if integer else float(value)


def summarize(data: Mapping[str, Any], *, target_seconds: float | None = None) -> dict:
    """Normalize an aggregate report without treating missing evidence as success."""
    if not isinstance(data, Mapping):
        raise ValueError("report must be a JSON object")
    frames = _number(data, "frames", integer=True)
    width = _number(data, "width", integer=True)
    height = _number(data, "height", integer=True)
    wall = _number(data, "wall_s")
    neural = _number(data, "neural_ms") / 1000.0
    status = {key: _number(data, key, integer=True, allow_zero=True)
              for key in STATUS_FIELDS if key in data}
    absent = [key for key in STATUS_FIELDS if key not in data]
    outcome = ("reported_failures" if any(status.values()) else
               "incomplete_status" if absent else "reported_clean")
    result = {
        "requested_frames": frames,
        "width": width,
        "height": height,
        "requested_pixels": frames * width * height,
        "outcome": outcome,
        "status": status,
        "missing_status_fields": absent,
        "run_wall_s": wall,
        "run_neural_s": neural,
        "amortized_wall_s_per_requested_frame": wall / frames,
        "amortized_neural_s_per_requested_frame": neural / frames,
        "wall_s_per_neural_s": wall / neural,
        "timing_scope": "aggregate run divided by requested frames; not measured frame latency",
        "notes": [
            "render_doom starts wall timing after compilation and circuit construction.",
            "Aggregate reports cannot establish first-frame, steady-state, or input-to-display latency.",
            "Different workloads, sizes, backends, or circuit configurations are not speedup comparisons.",
        ],
    }
    if "nodes" in data:
        result["nodes"] = _number(data, "nodes", integer=True)
    if "neurons_per_node" in data:
        result["neurons_per_node"] = _number(data, "neurons_per_node", integer=True)
        if "nodes" in result:
            result["total_neurons"] = result["nodes"] * result["neurons_per_node"]
    if target_seconds is not None:
        target = _number({"target_seconds": target_seconds}, "target_seconds")
        result["illustrative_target_s_per_frame"] = target
        result["wall_to_target_ratio"] = wall / frames / target
        result["notes"].append("The target ratio is arithmetic, not a predicted or achieved speedup.")
    if absent:
        result["notes"].append("Missing counters are unknown, not zero.")
    if outcome != "reported_clean":
        result["notes"].append("This report does not establish a clean completed render.")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--target-seconds", type=float, default=None,
                        help="illustrative wall seconds/frame; not a speedup prediction")
    args = parser.parse_args(argv)
    results = []
    try:
        for path in args.reports:
            report = json.loads(path.read_text(encoding="utf-8"))
            results.append({"report": str(path), **summarize(report, target_seconds=args.target_seconds)})
    except (OSError, ValueError, OverflowError) as exc:
        print(f"render_budget: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(results, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
