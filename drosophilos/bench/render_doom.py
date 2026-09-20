"""Render the Doom-shaped examples on resident neural dataflow kernels."""

from __future__ import annotations

import argparse
import json
import signal
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

from ..compiler.frontend_c import compile_c
from ..compiler.kernel import compile_program, kernel_outputs
from ..display.frame import write_png
from ..lib.kernel import build_pipeline, run_pipeline, run_pipeline_batched
from ..sim.model import D_MAX, Params
from ..sim.profile import Profiler
from .repro import record as repro_record


@dataclass
class RenderConfig:
    source: str = "examples/doom1.c"
    width: int = 40
    height: int = 25
    frames: int = 2
    inputs: str = "0,2,258,256"
    nodes: int = 8
    device: str = "cpu"
    max_ms: float = 3600000
    out: str = "data/a2/doom1"
    json: str | None = None
    pacing: str = "host"
    fp32: bool = False
    progress: float = 300
    mul: str = "array"
    pixel_stream: str = "p"
    tick_stream: str = "f"
    params: str | None = None
    reference_only: bool = False
    backend: str | None = None
    dtype: str = "float64"
    profile_steps: int = 0
    wall_limit: float = 0.0
    graph_steps: int = 0
    observe_every: int | None = None
    delivery: str = "auto"
    datapath: str = "generic"


def _effective_backend(cfg: RenderConfig):
    """Validate and return backend, torch device/dtype names, simulator, and GPU."""
    import torch

    backend = cfg.backend or ("torch" if cfg.nodes > 1 or cfg.device != "cpu" else "ref")
    dtype_name = "float32" if cfg.fp32 else cfg.dtype
    if cfg.fp32:
        warnings.warn("--fp32 is deprecated; use --dtype float32", DeprecationWarning, stacklevel=3)
    if backend not in ("ref", "torch", "torch-fast"):
        raise SystemExit(f"unsupported backend {backend!r}; choose ref, torch or torch-fast")
    if dtype_name not in ("float64", "float32"):
        raise SystemExit(f"unsupported dtype {dtype_name!r}; choose float64 or float32")
    try:
        device = torch.device(cfg.device)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"invalid torch device {cfg.device!r}: {exc}") from None
    if backend == "ref" and device.type != "cpu":
        raise SystemExit("the ref backend supports only --device cpu")
    if backend == "ref" and dtype_name != "float64":
        raise SystemExit("the ref backend supports only --dtype float64")
    if backend in ("torch", "torch-fast") and device.type == "mps" and dtype_name == "float64":
        raise SystemExit("Torch on MPS does not support --dtype float64; use float32")
    if backend != "torch-fast" and (cfg.graph_steps or cfg.observe_every or cfg.delivery != "auto"):
        raise SystemExit("--graph-steps, --observe-every and --delivery apply to --backend torch-fast only")
    if cfg.delivery not in ("auto", "sparse", "scatter"):
        raise SystemExit("--delivery must be auto, sparse or scatter")
    if cfg.graph_steps < 0 or cfg.graph_steps > D_MAX + 1:
        raise SystemExit(f"--graph-steps must lie in [0, {D_MAX + 1}]")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested, but CUDA is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("--device mps requested, but MPS is not available")
    if backend in ("torch", "torch-fast") and device.type not in ("cpu", "cuda", "mps"):
        raise SystemExit(f"the {backend} backend does not support --device {device.type}")
    gpu_name = None
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
    elif device.type == "mps":
        gpu_name = "Apple Metal Performance Shaders"
    simulator = {"ref": "RefSim", "torch": "TorchSim", "torch-fast": "FastSim"}[backend]
    torch_dtype = torch.float32 if dtype_name == "float32" else torch.float64
    return backend, str(device), dtype_name, torch_dtype, simulator, gpu_name


def count_duplicates(ks, scheds: list, outs: list) -> int:
    """Outputs delivered beyond what each node's schedule owes: an output cell owes one word
    per token of its stream. Cells carry the stream *key* (`ks.stream_keys[name]`, e.g.
    `input:f`) while schedules carry the stream *name* (`f`); the lookup goes through the map."""
    name_of_key = {key: name for name, key in ks.stream_keys.items()}
    cell_stream = {c["name"]: name_of_key[c["stream"]] for c in ks.cells if c.get("op") != "RING" and "stream" in c}
    dup = 0
    for sched, node_outs in zip(scheds, outs):
        counts = {name: 0 for name in ks.streams}
        for stream, _value, _barrier in sched:
            counts[stream] += 1
        for cell_name, values in node_outs.items():
            dup += max(0, len(values) - counts[cell_stream[cell_name]])
    return dup


def render(cfg: RenderConfig) -> dict:
    """Compile, render, simulate, and return the complete benchmark record."""
    total_started = time.perf_counter()
    compile_s = 0.0
    source_text = Path(cfg.source).read_text()

    # Reject backend requests before doing expensive compiler work.
    effective = None if cfg.reference_only else _effective_backend(cfg)
    t = time.perf_counter()
    prog = compile_c(source_text)
    compile_s += time.perf_counter() - t
    W, H, B, F = cfg.width, cfg.height, cfg.nodes, cfg.frames
    per_node_cols = -(-W // B)
    if cfg.pacing == "neural" and W % B:
        raise SystemExit(f"neural pacing: the width {W} must be a multiple of the copies {B}")
    initial_state = json.loads(cfg.params) if cfg.params else None
    pix_s, tick_s = cfg.pixel_stream, cfg.tick_stream

    t = time.perf_counter()
    ks = compile_program(prog, params=initial_state, pacing="host", mul=cfg.mul)
    compile_s += time.perf_counter() - t
    col_streams = [s for s in ks.streams if s not in (pix_s, tick_s)]
    if cfg.pacing == "neural":
        t = time.perf_counter()
        ks = compile_program(prog, params=initial_state, pacing=cfg.pacing, mul=cfg.mul,
                             counts={**{s: per_node_cols for s in col_streams},
                                     pix_s: per_node_cols * H})
        compile_s += time.perf_counter() - t
    pix_key, tick_key = ks.stream_keys[pix_s], ks.stream_keys[tick_s]
    sx, sy = 160 // W, 100 // H
    cols = [x * sx for x in range(W)]
    rows = [y * sy for y in range(H)]
    ticks = [int(v) for v in cfg.inputs.split(",")][:F]
    if len(ticks) < F:
        raise SystemExit(f"--inputs provides {len(ticks)} tick values for {F} requested frames")
    at = lambda token: ((token & 255) // sx, (token >> 8) // sy)
    cell_by_name = {c["name"]: c for c in ks.cells if c["op"] != "RING"}
    pixel_cell = next(o for o in ks.outputs if cell_by_name[o]["stream"] == pix_key)
    store_cells = {s: [o for o in ks.outputs if cell_by_name[o]["stream"] == ks.stream_keys[s]]
                   for s in col_streams}
    tick_cells = [o for o in ks.outputs if cell_by_name[o]["stream"] == tick_key]

    # The oracle and reference images are unchanged from the original driver.
    ref_sched = []
    for f in range(F):
        for stream in col_streams:
            ref_sched += [(stream, c) for c in cols]
        ref_sched += [(pix_s, (r << 8) | c) for r in rows for c in cols]
        ref_sched.append((tick_s, ticks[f]))
    ref_out = kernel_outputs(ks, ref_sched)
    ref_frames, cursor = [], 0
    for f in range(F):
        cursor += len(col_streams) * len(cols)
        frame = {at((r << 8) | c): ref_out[cursor + j][0]
                 for j, (r, c) in enumerate((r, c) for r in rows for c in cols)}
        ref_frames.append(frame)
        cursor += len(rows) * len(cols) + 1
        write_png(frame, W, H, f"{cfg.out}_{f}_reference.png")

    if cfg.reference_only:
        return {
            "reference_only": True,
            "source_path": cfg.source,
            "width": W,
            "height": H,
            "frames_requested": F,
            "frames_completed": F,
            "out": cfg.out,
            "timing": {"t_compile": compile_s, "t_total": time.perf_counter() - total_started},
        }

    backend, device, dtype_name, torch_dtype, simulator_name, gpu_name = effective
    deal = [cols[b::B] for b in range(B)]
    scheds, expect, sched_meta = [], [], []
    for b in range(B):
        schedule, metadata, owed = [], [], 0
        for f in range(F):
            barrier = (lambda x: x) if cfg.pacing == "host" else (lambda x: 0)
            for stream in col_streams:
                for c in deal[b]:
                    schedule.append((stream, c, barrier(owed)))
                    metadata.append((f, "column"))
                owed += len(deal[b]) * len(store_cells[stream])
            for c in deal[b]:
                for r in rows:
                    schedule.append((pix_s, (r << 8) | c, barrier(owed)))
                    metadata.append((f, "pixel"))
            owed += len(deal[b]) * len(rows)
            schedule.append((tick_s, ticks[f], barrier(owed)))
            metadata.append((f, "tick"))
            owed += len(tick_cells)
        scheds.append(schedule)
        sched_meta.append(metadata)
        expect.append(owed)

    P = Params()
    t = time.perf_counter()
    pl = build_pipeline(P, prog.width, ks.cells, consts=ks.consts, mems=ks.mems,
                        outputs=ks.outputs, streams=ks.streams, phases=ks.phases,
                        datapath=cfg.datapath)
    compile_s += time.perf_counter() - t

    pixel_values = [[] for _ in range(B)]
    pixel_valid = [[True] * F for _ in range(B)]
    pixel_seen = [[0] * F for _ in range(B)]
    tick_seen = {(b, cell): [False] * F for b in range(B) for cell in tick_cells}
    tick_ord = {(b, cell): 0 for b in range(B) for cell in tick_cells}
    frame_wall_abs: list[float | None] = [None] * F
    frame_steps: list[int | None] = [None] * F
    commit_wall_abs: list[float | None] = [None] * F
    commit_steps: list[int | None] = [None] * F
    written: set[int] = set()
    image_profiler = Profiler(bool(cfg.profile_steps))

    def pixels_per_frame(node: int) -> int:
        return len(deal[node]) * len(rows)

    def assembled_frame(f: int) -> dict:
        got = {}
        for b in range(B):
            count = pixels_per_frame(b)
            values = pixel_values[b][f * count:(f + 1) * count]
            for (c, r), value in zip(((c, r) for c in deal[b] for r in rows), values):
                if value is not None:
                    got[at((r << 8) | c)] = value
        return got

    def write_neural_frame(f: int) -> None:
        if f in written:
            return
        if image_profiler.enabled:
            with image_profiler.region("image_write"):
                write_png(assembled_frame(f), W, H, f"{cfg.out}_{f}_neural.png")
        else:
            write_png(assembled_frame(f), W, H, f"{cfg.out}_{f}_neural.png")
        written.add(f)

    def on_output(node: int, cell_name: str, step: int, value: int | None, wall_s: float) -> None:
        if cell_name == pixel_cell:
            ordinal = len(pixel_values[node])
            pixel_values[node].append(value)
            count = pixels_per_frame(node)
            if count:
                f = ordinal // count
                if f < F:
                    pixel_seen[node][f] += 1
                    pixel_valid[node][f] = pixel_valid[node][f] and value is not None
                    if (frame_wall_abs[f] is None and
                            all(pixel_seen[b][f] >= pixels_per_frame(b) and pixel_valid[b][f]
                                for b in range(B))):
                        frame_wall_abs[f] = wall_s
                        frame_steps[f] = step
                        write_neural_frame(f)
        if cell_name in tick_cells:
            f = tick_ord[(node, cell_name)]
            tick_ord[(node, cell_name)] += 1
            if f < F:
                tick_seen[(node, cell_name)][f] = True
                if (commit_wall_abs[f] is None and
                        all(tick_seen[(b, cell)][f] for b in range(B) for cell in tick_cells)):
                    commit_wall_abs[f] = wall_s
                    commit_steps[f] = step

    def on_progress(report: dict) -> None:
        print(f"progress: {report['elapsed_s']:.0f}s elapsed, {report['neural_ms']:.0f} ms neural, "
              f"{report['steps_per_s']:.0f} steps/s, outputs {report['outputs']}/{report['expected_outputs']}, "
              f"nodes done {report['nodes_done']}/{report['total_nodes']}, "
              f"faults {report['faults']}, timeouts {report['timeouts']}", flush=True)

    interrupted = {"value": False}

    def stop_requested() -> bool:
        return interrupted["value"] or bool(cfg.wall_limit and
                                             time.perf_counter() - total_started >= cfg.wall_limit)

    old_handlers = {}
    if threading.current_thread() is threading.main_thread():
        def handle_signal(_signum, _frame):
            interrupted["value"] = True

        for sig in (signal.SIGTERM, signal.SIGINT):
            old_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, handle_signal)

    run_started = time.perf_counter()
    try:
        if backend == "ref":
            _, sim, stats = run_pipeline(
                pl, P, scheds[0], max_ms=cfg.max_ms, expect_outputs=expect[0],
                on_output=on_output, should_stop=stop_requested,
            )
            outs = [stats["outputs_by_cell"]]
            load_events = [stats["load_events"]]
            profile = {"profiled_steps": 0, "regions": {}}
        else:
            outs, sim, stats = run_pipeline_batched(
                pl, P, scheds, max_ms=cfg.max_ms, device=device,
                expect_outputs=expect, dtype=torch_dtype,
                progress=(cfg.progress, on_progress) if cfg.progress else None,
                on_output=on_output, profile_steps=cfg.profile_steps,
                should_stop=stop_requested, backend=backend,
                graph_steps=cfg.graph_steps, observe_every=cfg.observe_every, delivery=cfg.delivery,
            )
            load_events = stats["load_events"]
            profile = dict(stats["profile"])
    except BaseException:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        raise

    runner_finished = time.perf_counter()
    image_regions = image_profiler.regions()
    if "image_write" in image_regions:
        profile["regions"]["image_write"] = image_regions["image_write"]

    # Ensure partial and invalid frames are still materialized, with missing pixels black.
    for f in range(F):
        write_neural_frame(f)

    wrong = missing = duplicates = 0
    for b in range(B):
        count = pixels_per_frame(b)
        for f in range(F):
            start = f * count
            coords = [(c, r) for c in deal[b] for r in rows]
            for j, (c, r) in enumerate(coords):
                pos = start + j
                if pos >= len(pixel_values[b]):
                    missing += 1
                else:
                    value = pixel_values[b][pos]
                    if value is not None and value != ref_frames[f][at((r << 8) | c)]:
                        wrong += 1
    duplicates = count_duplicates(ks, scheds, outs)

    runtime_inputs = [{
        "index": f,
        "value": ticks[f],
        "injected_step": None,
        "injected_wall_s": None,
        "committed_step": commit_steps[f],
        "committed_wall_s": None if commit_wall_abs[f] is None else commit_wall_abs[f] - run_started,
        "first_frame_using_state": None,
        "first_frame_using_state_completion_s": None,
    } for f in range(F)]
    column_first_abs: list[float | None] = [None] * F
    tick_injections = [[] for _ in range(F)]
    for b, events in enumerate(load_events):
        for event in events:
            f, kind = sched_meta[b][event["schedule_index"]]
            if kind == "tick":
                tick_injections[f].append(event)
            elif kind == "column":
                wall = event["injected_wall_s"]
                if column_first_abs[f] is None or wall < column_first_abs[f]:
                    column_first_abs[f] = wall
    for f, events in enumerate(tick_injections):
        if events:
            runtime_inputs[f]["injected_step"] = max(e["injected_step"] for e in events)
            runtime_inputs[f]["injected_wall_s"] = max(e["injected_wall_s"] for e in events) - run_started
        committed = commit_wall_abs[f]
        if committed is not None:
            for next_frame, first_column in enumerate(column_first_abs):
                if first_column is not None and first_column > committed:
                    runtime_inputs[f]["first_frame_using_state"] = next_frame
                    if frame_wall_abs[next_frame] is not None:
                        runtime_inputs[f]["first_frame_using_state_completion_s"] = (
                            frame_wall_abs[next_frame] - run_started)
                    break

    frame_complete_s = [None if x is None else x - run_started for x in frame_wall_abs]
    frame_intervals = [None] * F
    for f in range(1, F):
        if frame_complete_s[f] is not None and frame_complete_s[f - 1] is not None:
            frame_intervals[f] = frame_complete_s[f] - frame_complete_s[f - 1]
    completed_indices = [f for f, wall in enumerate(frame_wall_abs) if wall is not None]
    tail_s = tail_steps = None
    if completed_indices:
        last = completed_indices[-1]
        tail_s = max(0.0, runner_finished - frame_wall_abs[last])
        tail_steps = max(0, sim.step_index - int(frame_steps[last]))

    timing = {
        "t_compile": compile_s,
        "t_load": stats["t_load"],
        "first_frame_s": frame_complete_s[0] if frame_complete_s else None,
        "frame_complete": frame_complete_s,
        "frame_complete_step": frame_steps,
        "frame_interval_s": frame_intervals,
        "runtime_inputs": runtime_inputs,
        "t_total": time.perf_counter() - total_started,
        "tail_s": tail_s,
        "tail_steps": tail_steps,
    }
    rec = repro_record(
        source_path=cfg.source, pl=pl, inputs=ticks, initial_state=initial_state,
        pacing=cfg.pacing, mul=cfg.mul, copies=B, width=W, height=H,
        frames_requested=F, params=P, backend=backend, device=device,
        dtype=dtype_name, simulator=simulator_name, gpu_name=gpu_name,
        perturbation=None, seed=None,
    )
    rec.update({
        "datapath": pl.datapath,
        "backend_selected_by_default": cfg.backend is None,
        # the effective simulator class and its execution mode, from the runner itself
        "simulator_effective": stats.get("simulator", simulator_name),
        "simulator_mode": {key: stats[key] for key in ("observe_every", "graph_steps", "graph_active",
                                                        "graph_fallback_reason", "delivery", "delays")
                           if key in stats},
        "kernel_cells": len(ks.cells),
        "frames_completed": len(completed_indices),
        "wrong": wrong,
        "missing": missing,
        "duplicates": duplicates,
        "invalid_outputs": stats["bad_outputs"],
        "faults": stats["faults"],
        "timeouts": stats["timeouts"],
        "host_stalls": stats["host_stalls"],
        "truncated": bool(stats["truncated"] or stop_requested()),
        "neural_ms": stats.get("neural_ms", sim.step_index * P.dt),
        "profiled_steps": profile.get("profiled_steps", 0),
        "profile": profile,
        "timing": timing,
        "out": cfg.out,
    })
    rec["truncated"] = bool(rec["truncated"] or stop_requested())
    for sig, handler in old_handlers.items():
        signal.signal(sig, handler)
    return rec


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=RenderConfig.source)
    ap.add_argument("--width", type=int, default=RenderConfig.width)
    ap.add_argument("--height", type=int, default=RenderConfig.height)
    ap.add_argument("--frames", type=int, default=RenderConfig.frames)
    ap.add_argument("--inputs", default=RenderConfig.inputs,
                    help="tick inputs per frame: turn in bits 0-5, forward 256 / back 512")
    ap.add_argument("--nodes", type=int, default=RenderConfig.nodes)
    ap.add_argument("--device", default=RenderConfig.device)
    ap.add_argument("--backend", choices=["ref", "torch", "torch-fast"], default=None)
    ap.add_argument("--dtype", choices=["float64", "float32"], default=RenderConfig.dtype)
    ap.add_argument("--fp32", action="store_true", help="deprecated alias for --dtype float32")
    ap.add_argument("--max-ms", type=float, default=RenderConfig.max_ms)
    ap.add_argument("--wall-limit", type=float, default=RenderConfig.wall_limit,
                    help="stop cleanly after this many wall seconds (0 disables)")
    ap.add_argument("--graph-steps", type=int, default=RenderConfig.graph_steps,
                    help="torch-fast only: run K-step blocks (a CUDA graph on CUDA); 0 = eager steps")
    ap.add_argument("--observe-every", type=int, default=RenderConfig.observe_every,
                    help="torch-fast only: device-to-host spike transfer interval in steps (default: the decode window)")
    ap.add_argument("--delivery", default=RenderConfig.delivery, choices=["auto", "sparse", "scatter"],
                    help="torch-fast only: synaptic delivery kernel (auto: CSR product on CUDA, edge-wise scatter elsewhere)")
    ap.add_argument("--datapath", default=RenderConfig.datapath, choices=["generic", "specialized"],
                    help="fixed-operation cell datapath implementation")
    ap.add_argument("--profile-steps", type=int, default=RenderConfig.profile_steps,
                    help="profile this many post-settle neural steps (0 disables)")
    ap.add_argument("--out", default=RenderConfig.out)
    ap.add_argument("--json", default=None)
    ap.add_argument("--pacing", default=RenderConfig.pacing, choices=["host", "neural"])
    ap.add_argument("--progress", type=float, default=RenderConfig.progress,
                    help="seconds between progress lines (0 disables)")
    ap.add_argument("--mul", default=RenderConfig.mul, choices=["array", "pipelined"])
    ap.add_argument("--pixel-stream", default=RenderConfig.pixel_stream)
    ap.add_argument("--tick-stream", default=RenderConfig.tick_stream)
    ap.add_argument("--params", default=None)
    ap.add_argument("--reference-only", action="store_true")
    return ap


def main() -> None:
    args = _parser().parse_args()
    cfg = RenderConfig(**vars(args))
    result = render(cfg)
    if result.get("reference_only"):
        print(f"reference: {cfg.width}x{cfg.height} x {cfg.frames} frames written to "
              f"{cfg.out}_<k>_reference.png", flush=True)
        return
    default_note = " (selected by default)" if result["backend_selected_by_default"] else ""
    gpu = f", {result['hardware']['gpu_name']}" if result["hardware"]["gpu_name"] else ""
    mode = result.get("simulator_mode") or {}
    mode_note = (f"; graph_steps {mode['graph_steps']} ({'CUDA graph' if mode.get('graph_active') else 'eager blocks'}), "
                 f"observe_every {mode['observe_every']}, delivery {mode['delivery']}" if "graph_steps" in mode else "")
    print(f"backend: {result['simulator_effective']} via {result['backend']}{default_note}; "
          f"device {result['device']}{gpu}; dtype {result['dtype']}{mode_note}", flush=True)
    print(f"kernels: {result['kernel_cells']} cells, {result['neurons_per_node']} neurons per node, "
          f"{result['copies']} nodes, {result['width']}x{result['height']} pixels x "
          f"{result['frames_requested']} frames, pacing {result['pacing']}", flush=True)
    timing = result["timing"]
    print(f"neural doom: {result['frames_completed']}/{result['frames_requested']} frames in "
          f"{result['neural_ms'] / 1000:.1f} s of neural time ({timing['t_total']:.1f} s total wall); "
          f"wrong {result['wrong']}, missing {result['missing']}, duplicates {result['duplicates']}, "
          f"invalid {result['invalid_outputs']}; tail {timing['tail_s']} s / {timing['tail_steps']} steps; "
          f"{cfg.out}_<k>_neural.png", flush=True)
    if result["host_stalls"]:
        print("host stall: max_ms was reached with work outstanding", flush=True)
    if result["truncated"]:
        print("truncated: signal or wall limit stopped the run", flush=True)
    if cfg.json:
        with open(cfg.json, "w") as fp:
            json.dump(result, fp, indent=1)


if __name__ == "__main__":
    main()
