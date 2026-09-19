"""Small, repeatable comparisons for the performance campaign (``perf_campaign.md`` §3).

This is deliberately a driver rather than another benchmark implementation.  In particular,
``render_doom.render`` owns frame timing and ``run_pipeline_batched`` owns pipeline timing.
The JSON is the complete machine-readable record; the Markdown is a compact comparison view.
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from ..compiler.frontend_c import compile_c
from ..compiler.kernel import KernelSpec, compile_kernel, kernel_outputs, loop_body
from ..display.frame import PALETTE
from ..lib.kernel import build_pipeline, run_pipeline_batched
from ..sim.model import Params
from .kernel_campaign import block as campaign_block
from .render_doom import RenderConfig, render


ONE_KEY_VARIANT_KEYS = frozenset({"backend", "device", "dtype", "mul", "copies", "pacing"})
REPORT_COLUMNS = ("first-frame latency (neural s)", "first-frame latency (wall s)",
                  "subsequent frame intervals (neural s / wall s)", "total neural s", "wall s",
                  "wall/neural", "neurons", "edges", "spikes", "wrong", "missing", "duplicates",
                  "invalid", "faults", "timeouts", "truncated")


@dataclass(frozen=True)
class Variant:
    """One deliberately isolated comparison choice."""

    name: str
    overrides: dict[str, Any]

    def __post_init__(self) -> None:
        validate_variant(self.overrides)


@dataclass
class CampaignConfig:
    baseline: dict[str, Any]
    variants: list[Variant] = field(default_factory=list)

    def configurations(self) -> list[tuple[str, dict[str, Any], tuple[str, ...]]]:
        """Baseline plus each one-key variant; no hidden config merging is permitted."""
        validate_variant(self.baseline, baseline=True)
        out = [("baseline", dict(self.baseline), ())]
        for variant in self.variants:
            cfg = dict(self.baseline)
            cfg.update(variant.overrides)
            out.append((variant.name, cfg, (variant.name,)))
        return out


def validate_variant(overrides: dict[str, Any], *, baseline: bool = False) -> None:
    """Reject ambiguous comparisons early, including a two-key CLI/programmatic variant."""
    bad = set(overrides) - ONE_KEY_VARIANT_KEYS
    if bad:
        raise ValueError(f"unsupported variant key(s): {', '.join(sorted(bad))}")
    if not baseline and len(overrides) > 1:
        raise ValueError("a variant may override at most one key; split this comparison")


def default_variants(copies: Iterable[int]) -> list[Variant]:
    """The standard one- and eight-copy question, on the baseline's backend/device."""
    values = list(dict.fromkeys(int(x) for x in copies))
    return [Variant(f"copies-{n}", {"copies": n}) for n in values]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _frames(cfg: RenderConfig, directory: Path) -> list[Image.Image]:
    """Ask the existing oracle to write frames, then read the files it intentionally owns."""
    out = directory / "reference"
    render(replace(cfg, out=str(out), reference_only=True))
    return [Image.open(f"{out}_{f}_reference.png").convert("RGB").copy()
            for f in range(cfg.frames)]


def coverage_check(cfg: RenderConfig, *, directory: Path | None = None) -> dict[str, Any]:
    """Ensure selected level-2 inputs demonstrate changing state before timing anything.

    The error text names the sequence to alter so a future workload cannot accidentally turn a
    static screenshot into a performance result.
    """
    own_tmp = directory is None
    tmp = tempfile.TemporaryDirectory(prefix="drosophilos-perf-") if own_tmp else None
    root = Path(tmp.name) if tmp is not None else Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    try:
        images = _frames(cfg, root)
        pixels = [tuple(image.getdata()) for image in images]
        sequence = cfg.inputs
        if len(set(pixels)) == 1:
            raise RuntimeError(f"coverage failed for {cfg.source}: input sequence {sequence!r} does not change state; adjust it")
        if not any(pixels[i] != pixels[i + 1] for i in range(len(pixels) - 1)):
            raise RuntimeError(f"coverage failed for {cfg.source}: input sequence {sequence!r} has no later frame reflecting a change; adjust it")
        sprite_counts: list[int] | None = None
        if Path(cfg.source).name == "doom4.c":
            imp = {PALETTE[index] for index in (15, 10, 9)}
            sprite_counts = [sum(pixel in imp for pixel in frame) for frame in pixels]
            if not all(sprite_counts):
                raise RuntimeError(f"coverage failed for doom4: input sequence {sequence!r} has an invisible sprite; adjust it")
        return {"source": cfg.source, "inputs": sequence, "frame_different": [
            pixels[i] != pixels[i + 1] for i in range(len(pixels) - 1)], "sprite_pixels": sprite_counts}
    finally:
        if tmp is not None:
            tmp.cleanup()


def _primitive_spec(op: str, width: int = 8) -> tuple[KernelSpec, list[int]]:
    constants = {"ADD": 1, "AND": 0x55, "XOR": 0xA5, "MOV": 0}
    name = op.lower()
    # In the resident cell format MOV selects its B operand, so its stream token belongs in
    # B (the other primitive operations consume A and B in the conventional order).
    cell = ({"name": name, "op": "MOV", "a": ("const", "k"), "b": "input"}
            if op == "MOV" else {"name": name, "op": op, "a": "input", "b": ("const", "k")})
    spec = KernelSpec([cell],
                      {"k": constants[op]}, {}, "input", width)
    spec.outputs = [name]
    return spec, list(range(16))


def primitive_block(name: str, params: Params, *, mul: str, tokens: int) -> tuple[KernelSpec, Any, list[int], list[tuple[int, ...]]]:
    """Build the four single fixed-operation cells and the established composition blocks."""
    if name.startswith("cells-"):
        spec, values = _primitive_spec(name.removeprefix("cells-").upper())
    elif name == "perspective" and mul == "array":
        prog = compile_c(Path("examples/render2.c").read_text())
        spec = compile_kernel(prog, loop_body(prog), "col", params={"heading": 3}, mul="array")
        values = list(range(16))
    else:
        spec, _pl, values, _ref = campaign_block(name, params)
    # Kernel-campaign samples are eight tokens; make the fixed suite visibly repetitive.
    values = (list(values) * ((tokens + len(values) - 1) // len(values)))[:max(16, tokens)]
    pl = build_pipeline(params, spec.width, spec.cells, consts=spec.consts, mems=spec.mems, outputs=spec.outputs)
    return spec, pl, values, kernel_outputs(spec, values)


def _primitive_run(name: str, cfg: dict[str, Any], *, tokens: int, profile_steps: int = 0,
                   max_neural_s: float = 150) -> dict[str, Any]:
    import torch

    params = Params()
    spec, pl, values, reference = primitive_block(name, params, mul=cfg["mul"], tokens=tokens)
    copies = int(cfg["copies"])
    first_wall: float | None = None
    started = time.perf_counter()

    def observed(_node: int, _cell: str, _step: int, _value: int | None, wall: float) -> None:
        nonlocal first_wall
        if first_wall is None:
            first_wall = wall - started

    dtype = torch.float32 if cfg["dtype"] == "float32" else torch.float64
    # Capturing every neuron on node 0 is intentionally limited to short primitive runs.  It
    # adds trace-copy/memory cost, which is recorded rather than pretending spikes are free.
    outs, _sim, stats = run_pipeline_batched(
        pl, params, [list(values) for _ in range(copies)], max_ms=max_neural_s * 1000,
        device=cfg["device"], dtype=dtype, expect_outputs=[len(values) * len(spec.outputs)] * copies,
        progress=0, on_output=observed, profile_steps=profile_steps,
        capture_spikes=(0, range(pl.net.n)),
    )
    wrong = missing = duplicates = 0
    for node in range(copies):
        for output_index, cell in enumerate(spec.outputs):
            got = [value for _step, value in outs[node][cell]]
            want = [row[output_index] for row in reference]
            duplicates += max(0, len(got) - len(want))
            for index, expected in enumerate(want):
                if index >= len(got):
                    missing += 1
                elif got[index] != expected:
                    wrong += 1
    first = outs[0][spec.outputs[0]]
    load_step = stats["load_steps"][0][0] if stats["load_steps"][0] else None
    first_neural = None if not first or load_step is None else (first[0][0] - load_step) * params.dt / 1000
    intervals = [] if len(first) < 2 else [
        (first[i][0] - first[i - 1][0]) * params.dt / 1000 for i in range(1, len(first))]
    captured_steps, _captured_neurons = stats["captured_spikes"]
    neural = stats["neural_ms"] / 1000
    wall = stats["wall_s"]
    return {
        "level": "primitive", "workload": name, "config": dict(cfg), "tokens": len(values),
        "reference": reference, "outputs": [{cell: values_ for cell, values_ in node.items()} for node in outs],
        "first_neural_s": first_neural, "first_wall_s": first_wall,
        "interval_neural_s": intervals, "interval_wall_s": [], "neural_s": neural, "wall_s": wall,
        "wall_neural": None if not neural else wall / neural, "neurons": pl.net.n, "edges": pl.net.nnz,
        "spikes": int(len(captured_steps)), "spike_count_method": "capture_spikes(all neurons, node 0)",
        "spike_count_cost": "primitive-only trace capture; extra host memory/copy work",
        "wrong": wrong, "missing": missing, "duplicates": duplicates,
        "invalid": stats["bad_outputs"], "faults": stats["faults"], "timeouts": stats["timeouts"],
        "host_stalls": stats["host_stalls"], "truncated": stats["truncated"], "max_neural_s": max_neural_s,
        "t_load": stats["t_load"],
        "load_events": stats["load_events"], "profile": stats["profile"], "profiled": bool(profile_steps),
    }


def _render_run(workload: str, cfg: dict[str, Any], *, out: Path, profile_steps: int = 0,
                workloads: dict[str, dict[str, Any]] = None) -> dict[str, Any]:
    spec = (SMALL_WORKLOADS if workloads is None else workloads)[workload]
    rec = render(RenderConfig(source=spec["source"], width=spec["width"], height=spec["height"], frames=spec["frames"],
                              inputs=spec["inputs"], nodes=cfg["copies"], backend=cfg["backend"], device=cfg["device"],
                              dtype=cfg["dtype"], pacing=cfg["pacing"], mul=cfg["mul"], progress=0,
                              profile_steps=profile_steps, out=str(out)))
    timing = rec["timing"]
    steps = timing["frame_complete_step"]
    first_neural = None if not steps or steps[0] is None else steps[0] * Params().dt / 1000
    neural_intervals = [None if steps[i] is None or steps[i - 1] is None else
                        (steps[i] - steps[i - 1]) * Params().dt / 1000 for i in range(1, len(steps))]
    neural = rec["neural_ms"] / 1000
    wall = timing["t_total"]
    return {
        "level": "small", "workload": workload, "config": dict(cfg), "first_neural_s": first_neural,
        "first_wall_s": timing["first_frame_s"], "interval_neural_s": neural_intervals,
        "interval_wall_s": timing["frame_interval_s"][1:], "neural_s": neural, "wall_s": wall,
        "wall_neural": None if not neural else wall / neural, "neurons": rec["neurons_per_node"],
        "edges": rec["edges_per_node"], "spikes": None, "spike_count_method": "not captured for render",
        "wrong": rec["wrong"], "missing": rec["missing"], "duplicates": rec["duplicates"],
        "invalid": rec["invalid_outputs"], "faults": rec["faults"], "timeouts": rec["timeouts"],
        "host_stalls": rec["host_stalls"], "truncated": rec["truncated"], "timing": timing,
        "profile": rec["profile"], "profiled": bool(profile_steps), "reproducibility": rec,
    }


SMALL_WORKLOADS = {
    "doom2-small": {"source": "examples/doom2.c", "width": 8, "height": 5, "frames": 3, "inputs": "2,258,0"},
    "doom4-small": {"source": "examples/doom4.c", "width": 8, "height": 5, "frames": 3, "inputs": "6,256,0"},
}
HISTORICAL_WORKLOADS = {
    "doom2-historical": {"source": "examples/doom2.c", "width": 40, "height": 25, "frames": 2, "inputs": "2,258"},
    "doom4-historical": {"source": "examples/doom4.c", "width": 24, "height": 15, "frames": 2, "inputs": "6,256"},
}


def _median_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Headline values are derived only from the unprofiled repeat records."""
    def spread(key: str) -> dict[str, float | None]:
        values = [r[key] for r in records if r.get(key) is not None]
        return {"median": statistics.median(values), "min": min(values), "max": max(values)} if values else {"median": None, "min": None, "max": None}
    base = dict(records[0])
    base["repeat_summary"] = {"wall_s": spread("wall_s"), "first_wall_s": spread("first_wall_s"),
                              "first_neural_s": spread("first_neural_s")}
    base["headline"] = {"wall_s": base["repeat_summary"]["wall_s"]["median"],
                        "first_wall_s": base["repeat_summary"]["first_wall_s"]["median"],
                        "first_neural_s": base["repeat_summary"]["first_neural_s"]["median"]}
    return base


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, list):
        return ", ".join(_fmt(v) for v in value)
    return str(value)


def _section(records: list[dict[str, Any]], title: str) -> list[str]:
    lines = [f"### {title}", "", "| configuration | " + " | ".join(REPORT_COLUMNS) + " |",
             "|" + "---|" * (len(REPORT_COLUMNS) + 1)]
    if not records:
        lines.append("| no selected configurations |" + "—|" * len(REPORT_COLUMNS))
    for record in records:
        headline = record.get("headline", record)
        truncated = record.get("truncated")
        # A capped primitive block hits max_neural_s and reads as a host stall unless the
        # table names the cap explicitly; a plain truthy "truncated" looks like a measurement.
        if truncated and record.get("host_stalls") and record.get("max_neural_s") is not None:
            truncated = "stalled@cap"
        row = [f"{record['workload']} / {record['config_name']}", headline.get("first_neural_s"),
               headline.get("first_wall_s"), [record.get("interval_neural_s"), record.get("interval_wall_s")],
               record.get("neural_s"), headline.get("wall_s"), record.get("wall_neural"), record.get("neurons"),
               record.get("edges"), record.get("spikes"), record.get("wrong"), record.get("missing"),
               record.get("duplicates"), record.get("invalid"), record.get("faults"), record.get("timeouts"),
               truncated]
        lines.append("| " + " | ".join(_fmt(x) for x in row) + " |")
    return lines + [""]


def write_report(out: Path, report: dict[str, Any]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2, default=_jsonable) + "\n")
    status = ("partial — {done}/{planned} configurations, still running"
             .format(done=report.get("configurations_done"), planned=report.get("configurations_planned"))
             if not report.get("complete", True) else "complete")
    lines = ["# DrosophilOS performance comparison", "", f"Status: {status}.", "",
             "Headline medians/min/max come only from unprofiled repeats. Primitive spikes are captured "
             "for all neurons on node 0 only, adding trace-copy cost; renders do not capture spikes.", ""]
    for level in ("primitive", "small", "historical"):
        level_records = [r for r in report["summaries"] if r["level"] == level]
        if not level_records:
            continue
        lines += [f"## {level}", ""]
        sim = [r for r in level_records if r["comparison"] == "simulator-only"]
        circuit = [r for r in level_records if r["comparison"] == "circuit-only"]
        combined = list(level_records)
        lines += _section(sim, "Unchanged-circuit simulator-speed comparisons")
        lines += _section(circuit, "Changed-circuit comparisons")
        lines += _section(combined, "Combined")
    out.with_suffix(".md").write_text("\n".join(lines))


def _expand_config_sets(level: str, workload: str,
                        configs: list[tuple[str, dict[str, Any], tuple[str, ...]]],
                        ) -> list[tuple[str, dict[str, Any], tuple[str, ...], str]]:
    """The multiplier is a circuit choice only for the perspective kernel.  Each named
    variant itself changes exactly one key; pairing it with a copies variant is the
    explicit combined comparison, not an accidental two-key variant."""
    config_sets: list[tuple[str, dict[str, Any], tuple[str, ...], str]] = []
    for config_name, cfg, variant_names in configs:
        config_sets.append((config_name, cfg, variant_names, "simulator-only"))
        if level == "primitive" and workload == "perspective":
            alternative = "pipelined" if cfg["mul"] == "array" else "array"
            changed = dict(cfg); changed["mul"] = alternative
            tags = variant_names + (f"mul-{alternative}",)
            comparison = "circuit-only" if not variant_names else "combined"
            config_sets.append((f"{config_name}+mul-{alternative}", changed, tags, comparison))
    return config_sets


CPU_RENDER_COST_NOTE = ("small/historical renders are cluster workloads (slurm/submit.sh); an "
                        "8x5x3-frame doom2 render took ~2h43m on four CPU copies per RESULTS.md")

# Sizing constants only, not measurements: per-token neural seconds for primitive blocks (the
# 16-bit perspective multiplier is far more expensive than the single-cell/fanout/tick blocks),
# and per-pixel-per-frame neural seconds for small/historical renders at one copy.
_PRIMITIVE_NEURAL_S_PER_TOKEN = {"perspective": 6.6}
_PRIMITIVE_NEURAL_S_PER_TOKEN_DEFAULT = 1.2
_RENDER_NEURAL_S_PER_PIXEL_FRAME = {"doom4": 30.0, "doom2": 15.0}


def _estimate_neural_s(level: str, workload: str, cfg: dict[str, Any], args: argparse.Namespace) -> float:
    """A sizing estimate for --dry-run, not a measurement: see docs/perf_campaign_suite.md."""
    if level == "primitive":
        per_token = _PRIMITIVE_NEURAL_S_PER_TOKEN.get(workload, _PRIMITIVE_NEURAL_S_PER_TOKEN_DEFAULT)
        return min(args.tokens * per_token, args.primitive_max_s)
    workloads = SMALL_WORKLOADS if level == "small" else HISTORICAL_WORKLOADS
    spec = workloads[workload]
    per_pixel_frame = next(s for name, s in _RENDER_NEURAL_S_PER_PIXEL_FRAME.items() if name in spec["source"])
    copies = int(cfg["copies"])
    return spec["width"] * spec["height"] * spec["frames"] * per_pixel_frame / copies


def run_campaign(args: argparse.Namespace) -> dict[str, Any]:
    levels = [x.strip() for x in args.levels.split(",") if x.strip()]
    invalid = set(levels) - {"primitive", "small", "historical"}
    if invalid:
        raise ValueError(f"unknown level(s): {', '.join(sorted(invalid))}")
    if "historical" in levels and not args.historical:
        raise ValueError("historical workloads are H200-only and require --historical")
    gated_levels = {"small", "historical"} & set(levels)
    if gated_levels and args.device == "cpu" and not args.allow_cpu_renders:
        raise ValueError(
            f"{', '.join(sorted(gated_levels))} on --device cpu requires --allow-cpu-renders: "
            f"{CPU_RENDER_COST_NOTE}")
    copies = [int(x) for x in args.copies.split(",") if x]
    if not copies or any(x < 1 for x in copies):
        raise ValueError("--copies must be positive integers")
    baseline = {"backend": args.backend, "device": args.device, "dtype": args.dtype, "mul": args.mul,
                "copies": copies[0], "pacing": args.pacing}
    campaign = CampaignConfig(baseline, [Variant(f"copies-{n}", {"copies": n}) for n in copies[1:]])
    configs = campaign.configurations()
    planned = [{"name": name, **cfg} for name, cfg, _variants in configs]

    work: list[tuple[str, str]] = []
    if "primitive" in levels:
        requested_blocks = [x.strip() for x in args.primitive_blocks.split(",") if x.strip()]
        expanded = []
        for name in requested_blocks:
            expanded += ["cells-add", "cells-and", "cells-xor", "cells-mov"] if name == "cells" else [name]
        allowed = {"cells-add", "cells-and", "cells-xor", "cells-mov", "fanout", "perspective", "tick"}
        unknown_blocks = set(expanded) - allowed
        if unknown_blocks:
            raise ValueError(f"unknown primitive block(s): {', '.join(sorted(unknown_blocks))}")
        work += [("primitive", n) for n in expanded]
    if "small" in levels:
        work += [("small", n) for n in SMALL_WORKLOADS]
    # Historical is intentionally represented and gated.  Do not accidentally make it part
    # of an ordinary laptop run.
    if "historical" in levels:
        work += [("historical", n) for n in HISTORICAL_WORKLOADS]

    if args.dry_run:
        estimates = []
        for level, workload in work:
            for config_name, cfg, _variant_names, comparison in _expand_config_sets(level, workload, configs):
                estimates.append({"level": level, "workload": workload, "config_name": config_name,
                                  "comparison": comparison,
                                  "estimated_neural_s": _estimate_neural_s(level, workload, cfg, args)})
        return {"dry_run": True, "levels": levels, "configurations": planned,
                "historical_note": "H200-only; omitted unless --historical",
                "cost_estimate": estimates,
                "cost_estimate_note": ("sizing estimate only, not a measurement; "
                                       "wall ≈ 2–15× neural depending on device")}

    report: dict[str, Any] = {"campaign": "docs/perf_campaign.md §3", "generated_at": time.time(),
                              "arguments": vars(args), "historical_note": "H200-only and off by default",
                              "coverage": [], "records": [], "summaries": [],
                              "complete": False, "configurations_done": 0,
                              "configurations_planned": sum(
                                  len(_expand_config_sets(level, workload, configs)) for level, workload in work)}
    out = Path(args.out)
    if "small" in levels:
        for workload, item in SMALL_WORKLOADS.items():
            report["coverage"].append(coverage_check(RenderConfig(**item)))

    for level, workload in work:
        for config_name, cfg, variant_names, comparison in _expand_config_sets(level, workload, configs):
            repeats = 1 if level == "historical" else args.repeats
            batch: list[dict[str, Any]] = []
            for repeat in range(repeats):
                if level == "primitive":
                    record = _primitive_run(workload, cfg, tokens=args.tokens, max_neural_s=args.primitive_max_s)
                elif level == "small":
                    record = _render_run(workload, cfg, out=out.parent / f"{out.name}_{workload}_{config_name}_{repeat}")
                else:
                    item = HISTORICAL_WORKLOADS[workload]
                    record = _render_run(workload, cfg, out=out.parent / f"{out.name}_{workload}_{config_name}_{repeat}",
                                         workloads=HISTORICAL_WORKLOADS)
                    record["level"] = "historical"
                    record["historical_workload"] = item
                record.update({"config_name": config_name, "repeat": repeat, "profiled": False,
                               "comparison": comparison})
                report["records"].append(record); batch.append(record)
            summary = _median_summary(batch)
            summary.update({"config_name": config_name, "comparison": comparison})
            report["summaries"].append(summary)
            if args.profile_steps:
                if level == "primitive":
                    profile_record = _primitive_run(workload, cfg, tokens=args.tokens,
                                                     profile_steps=args.profile_steps,
                                                     max_neural_s=args.primitive_max_s)
                elif level == "small":
                    profile_record = _render_run(workload, cfg, out=out.parent / f"{out.name}_{workload}_{config_name}_profile",
                                                 profile_steps=args.profile_steps)
                else:
                    continue
                profile_record.update({"config_name": config_name, "repeat": None, "profiled": True,
                                       "headline_excluded": True, "comparison": comparison})
                report["records"].append(profile_record)
            # Write after every completed configuration so a job killed by its time limit
            # (a laptop run or a Slurm job hitting --time) still leaves the results so far.
            report["configurations_done"] += 1
            write_report(out, report)
    report["complete"] = True
    write_report(out, report)
    return report


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--levels", default="primitive", help="comma-separated: primitive, small, historical "
                    "(small/historical are cluster workloads; see --allow-cpu-renders)")
    ap.add_argument("--backend", choices=("torch",), default="torch", help="batched runner backend")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    ap.add_argument("--mul", choices=("array", "pipelined"), default="array")
    ap.add_argument("--copies", default="1,8")
    ap.add_argument("--pacing", choices=("host", "neural"), default="host")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--tokens", type=int, default=16, help="primitive tokens per block (minimum 16)")
    ap.add_argument("--primitive-blocks", default="cells,fanout,tick",
                    help="comma-separated primitive blocks; 'cells' expands to ADD, AND, XOR, MOV; "
                    "'perspective' is selectable but not in the default smoke set (~6.6 neural s/token)")
    ap.add_argument("--primitive-max-s", type=float, default=150,
                    help="neural-time cap per primitive block, sized to its token count (default 150s; "
                    "the 16-bit perspective multiplier needs ~6.6 neural s/token)")
    ap.add_argument("--profile-steps", type=int, default=0)
    ap.add_argument("--historical", action="store_true", help="enable H200-only historical workloads")
    ap.add_argument("--allow-cpu-renders", action="store_true",
                    help="permit small/historical levels on --device cpu; these are cluster workloads "
                    "(slurm/submit.sh) — an 8x5x3-frame doom2 render took ~2h43m on four CPU copies")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="data/perf/campaign")
    return ap


def main() -> None:
    args = _parser().parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be at least one")
    if args.tokens < 1:
        raise SystemExit("--tokens must be positive")
    try:
        result = run_campaign(args)
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from None
    if args.dry_run:
        print(json.dumps(result, indent=2))
    else:
        print(f"wrote {Path(args.out).with_suffix('.json')} and {Path(args.out).with_suffix('.md')}")


if __name__ == "__main__":
    main()
