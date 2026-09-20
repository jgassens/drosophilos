"""Event-level diagnostic of the multiplier regression (`docs/perf_campaign.md` §5).

Measured: the pipelined multiplier is 2.1× faster per token than the array multiplier on the
perspective kernel alone (Juno 412136: 3.2 s vs 6.9 s), yet inside doom4's column pass a pixel
cost 63 s with it against 30 s with the array (a3_kernels §11.2). Aggregate counters cannot say
why. This tool captures the handshake events of every multiplier cell, its producers and its
consumers on node 0 of a run — request arrival, START, ACT^d (operand sampling), stage
completion, commit request, commit, DONE, the consumers' requests and STARTs — and splits each
token's life into

    wait_input   = last needed request arrives  − previous DONE      (producer starvation)
    go           = START − max(last request, idle restored)          (the go chain)
    compute      = stage completion − START                          (the multiplier itself)
    commit_wait  = commit − completion                               (downstream backpressure:
                                                                      a reader still holds a
                                                                      pending request)
    drain        = DONE − commit                                     (the commit and reset)

so time to the first result, the intervals between results, time waiting for consumers and
time draining are separate numbers, as the campaign orders require. Pipelined rows are cells
of their own (`op == "MULP_ROW"`), so the table also says whether consecutive tokens overlap
across rows.

    python -m drosophilos.bench.mul_diag --kernel perspective --mul array --device cpu
    python -m drosophilos.bench.mul_diag --kernel doom --source examples/doom4.c --width 8 --height 5 \
        --frames 1 --inputs 6 --mul pipelined --device cuda --out data/perf/muldiag_doom4_pipelined

The doom schedule mirrors `bench/render_doom.render` (host pacing, one copy); when that driver
grows a `capture` option this module should call it instead of repeating the schedule.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path

import numpy as np

from ..compiler.frontend_c import compile_c
from ..compiler.kernel import compile_kernel, compile_program, kernel_outputs, loop_body
from ..lib.kernel import build_pipeline, run_pipeline_batched
from ..sim.model import Params

MUL_OPS = ("MUL", "MULP_ROW")


# ---------------------------------------------------------------------------- probes
def multiplier_cells(pl) -> list:
    return [c for c in pl.cells if c.op in MUL_OPS]


def probes_for(pl, cell) -> dict:
    """Neuron ids of the handshake events of `cell`, keyed by event name. Producers are the
    cells whose done pulse requests it; consumers the cells that request it."""
    roles = pl.net.roles
    ids = {"start": int(cell.start), "act": int(cell.act.u),
           "completion": int(cell.stage.completion.u), "creq": int(cell.creq[1].u),
           "commit": int(cell.commit_pulse), "done": int(cell.reg.done_relay), "idle": int(cell.idle[1].u)}
    actd = [(int(m.group(1)), i) for i, r in enumerate(roles)
            for m in [re.fullmatch(re.escape(cell.name) + r"\.actd\.d(\d+)", r)] if m]
    if actd:
        ids["actd"] = max(actd)[1]  # the chain's last relay: its rise is ACT^d, the sampling instant
    for src, pair in cell.reqs.items():
        ids[f"req:{src}"] = int(pair[1].u)
    by_name = {c.name: c for c in pl.cells}
    for src in cell.reqs:
        if src in by_name:
            ids[f"producer_done:{src}"] = int(by_name[src].reg.done_relay)
    for other in pl.cells:
        if cell.name in other.reqs:
            ids[f"consumer_req:{other.name}"] = int(other.reqs[cell.name][1].u)
            ids[f"consumer_start:{other.name}"] = int(other.start)
    return ids


def capture_ids(pl) -> tuple[np.ndarray, dict]:
    """All probe ids of every multiplier cell (node 0), and the per-cell probe maps."""
    per_cell = {c.name: probes_for(pl, c) for c in multiplier_cells(pl)}
    ids = sorted({v for p in per_cell.values() for v in p.values()})
    return np.asarray(ids, dtype=np.int64), per_cell


# ---------------------------------------------------------------------------- analysis
def rises(steps: np.ndarray, gap: int) -> list[int]:
    """First spike of each burst: a latch fires a ~213 Hz train while set, a pulse neuron once
    or a few times; a rise is a spike after at least `gap` silent steps (the runner's rule)."""
    steps = np.sort(np.asarray(steps, dtype=np.int64))
    if not len(steps):
        return []
    out = [int(steps[0])]
    for prev, s in zip(steps[:-1], steps[1:]):
        if s - prev > gap:
            out.append(int(s))
    return out


def _first_at_or_after(seq: list[int], t: int, limit: int | None = None) -> int | None:
    for s in seq:
        if s >= t and (limit is None or s < limit):
            return s
    return None


def _last_at_or_before(seq: list[int], t: int) -> int | None:
    out = None
    for s in seq:
        if s <= t:
            out = s
        else:
            break
    return out


def token_timeline(events: dict, dt_ms: float) -> list[dict]:
    """`events`: probe name -> rise steps. One row per START: the step of each event of that
    token and the phase durations in ms (None when an event did not happen)."""
    starts = events.get("start", [])
    rows = []
    prev_done = None
    for k, t_start in enumerate(starts):
        nxt = starts[k + 1] if k + 1 < len(starts) else None
        reqs = {n[4:]: _last_at_or_before(v, t_start) for n, v in events.items() if n.startswith("req:")}
        last_req = max((t for t in reqs.values() if t is not None), default=None)
        idle_ok = _last_at_or_before(events.get("idle", []), t_start)
        t_actd = _first_at_or_after(events.get("actd", events.get("act", [])), t_start, nxt)
        t_comp = _first_at_or_after(events.get("completion", []), t_actd or t_start, nxt)
        t_creq = _first_at_or_after(events.get("creq", []), t_comp or t_start, nxt)
        t_commit = _first_at_or_after(events.get("commit", []), t_comp or t_start, nxt)
        t_done = _first_at_or_after(events.get("done", []), t_commit or t_start, nxt)
        cons_start = {n[len("consumer_start:"):]: _first_at_or_after(v, t_done or t_start)
                      for n, v in events.items() if n.startswith("consumer_start:")}
        ms = lambda a, b: None if a is None or b is None else (b - a) * dt_ms
        row = {"token": k, "start": t_start, "last_request": last_req, "requests": reqs, "idle_restored": idle_ok,
               "actd": t_actd, "completion": t_comp, "creq": t_creq, "commit": t_commit, "done": t_done,
               "consumer_start": cons_start,
               "wait_input_ms": ms(prev_done, last_req) if prev_done is not None else None,
               "go_ms": ms(max(x for x in (last_req, idle_ok) if x is not None), t_start) if (last_req or idle_ok) else None,
               "compute_ms": ms(t_start, t_comp), "commit_wait_ms": ms(t_comp, t_commit),
               "drain_ms": ms(t_commit, t_done), "start_to_done_ms": ms(t_start, t_done),
               "period_ms": ms(t_start, nxt) if nxt is not None else None}
        rows.append(row)
        prev_done = t_done if t_done is not None else prev_done
    return rows


def summarize(rows: list[dict]) -> dict:
    keys = ("wait_input_ms", "go_ms", "compute_ms", "commit_wait_ms", "drain_ms", "start_to_done_ms", "period_ms")
    out = {"tokens": len(rows)}
    for key in keys:
        vals = [r[key] for r in rows if r.get(key) is not None]
        out[key] = {"n": len(vals), "median": statistics.median(vals) if vals else None,
                    "min": min(vals) if vals else None, "max": max(vals) if vals else None}
    return out


def analyse(captured: tuple, per_cell: dict, drive_period: int, dt_ms: float, cells_by_name: dict) -> dict:
    steps, neurons = (np.asarray(x, dtype=np.int64) for x in captured)
    gap = 3 * drive_period
    report = {"cells": {}}
    for cname, probes in per_cell.items():
        events = {ev: rises(steps[neurons == nid], gap) for ev, nid in probes.items()}
        rows = token_timeline(events, dt_ms)
        summary = summarize(rows)
        commits = [r["commit"] for r in rows if r["commit"] is not None]
        summary["inter_commit_ms"] = ([(b - a) * dt_ms for a, b in zip(commits[:-1], commits[1:])] or None)
        first_req = min((s[0] for n, s in events.items() if n.startswith("req:") and s), default=None)
        summary["first_result_ms"] = (rows[0]["done"] - first_req) * dt_ms if rows and rows[0]["done"] is not None and first_req is not None else None
        c = cells_by_name[cname]
        report["cells"][cname] = {"op": c.op, "row": c.row, "prev": c.prev, "events": events, "tokens": rows, "summary": summary}
    # pipelined rows: does token k+1 enter row 0 before token k leaves the last row?
    rows_of = {}
    for cname, info in report["cells"].items():
        if info["op"] == "MULP_ROW":
            rows_of.setdefault(cname.rsplit(".r", 1)[0] if ".r" in cname else cname, []).append((info["row"], cname))
    overlap = {}
    for base, lst in rows_of.items():
        lst.sort()
        first, last = report["cells"][lst[0][1]], report["cells"][lst[-1][1]]
        s0 = first["events"].get("start", [])
        dl = last["events"].get("done", [])
        overlapping = sum(1 for k in range(1, len(s0)) if k - 1 < len(dl) and s0[k] < dl[k - 1])
        overlap[base] = {"rows": len(lst), "tokens": len(s0), "tokens_entering_before_previous_left": overlapping}
    report["row_overlap"] = overlap
    return report


def print_report(report: dict) -> None:
    for cname, info in report["cells"].items():
        s = info["summary"]
        med = lambda k: "—" if s[k]["median"] is None else f"{s[k]['median']:8.1f}"
        print(f"{cname:28s} {info['op']:9s} tokens {s['tokens']:3d}  first result {s['first_result_ms'] or float('nan'):8.1f} ms | "
              f"wait_input {med('wait_input_ms')}  go {med('go_ms')}  compute {med('compute_ms')}  "
              f"commit_wait {med('commit_wait_ms')}  drain {med('drain_ms')}  period {med('period_ms')} ms")
    for base, o in report.get("row_overlap", {}).items():
        print(f"rows of {base}: {o['rows']} rows, {o['tokens']} tokens, {o['tokens_entering_before_previous_left']} entered before the previous token left")


# ---------------------------------------------------------------------------- workloads
def perspective_workload(mul: str, tokens: int):
    prog = compile_c(open("examples/render2.c").read())
    ks = compile_kernel(prog, loop_body(prog), "col", params={"heading": 3}, mul=mul)
    pl = build_pipeline(Params(), prog.width, ks.cells, consts=ks.consts, mems=ks.mems, outputs=ks.outputs)
    sched = [list(range(tokens))]
    ref = kernel_outputs(ks, list(range(tokens)))
    return ks, pl, sched, [len(sched[0]) * len(ks.outputs)], ref


def doom_workload(source: str, width: int, height: int, frames: int, inputs: str, mul: str,
                  pixel_stream: str = "p", tick_stream: str = "f"):
    """One copy, host pacing: the schedule `bench/render_doom.render` builds for node 0."""
    prog = compile_c(open(source).read())
    ks = compile_program(prog, params=None, pacing="host", mul=mul)
    pix_s, tick_s = pixel_stream, tick_stream
    col_streams = [s for s in ks.streams if s not in (pix_s, tick_s)]
    pix_key, tick_key = ks.stream_keys[pix_s], ks.stream_keys[tick_s]
    cell_by_name = {c["name"]: c for c in ks.cells if c["op"] != "RING"}
    store_cells = {s: [o for o in ks.outputs if cell_by_name[o]["stream"] == ks.stream_keys[s]] for s in col_streams}
    tick_cells = [o for o in ks.outputs if cell_by_name[o]["stream"] == tick_key]
    sx, sy = 160 // width, 100 // height
    cols = [x * sx for x in range(width)]
    rows = [y * sy for y in range(height)]
    ticks = [int(v) for v in inputs.split(",")][:frames]
    if len(ticks) < frames:
        raise SystemExit(f"--inputs gives {len(ticks)} tick values for {frames} frames")
    schedule, owed = [], 0
    for f in range(frames):
        for stream in col_streams:
            schedule += [(stream, c, owed) for c in cols]
            owed += len(cols) * len(store_cells[stream])
        schedule += [(pix_s, (r << 8) | c, owed) for c in cols for r in rows]
        owed += len(cols) * len(rows)
        schedule.append((tick_s, ticks[f], owed))
        owed += len(tick_cells)
    pl = build_pipeline(Params(), prog.width, ks.cells, consts=ks.consts, mems=ks.mems, outputs=ks.outputs,
                        streams=ks.streams, phases=ks.phases)
    return ks, pl, [schedule], [owed], None


def run(args) -> dict:
    t0 = time.perf_counter()
    if args.kernel == "perspective":
        ks, pl, scheds, expect, ref = perspective_workload(args.mul, args.tokens)
    else:
        ks, pl, scheds, expect, ref = doom_workload(args.source, args.width, args.height, args.frames, args.inputs, args.mul)
    ids, per_cell = capture_ids(pl)
    P = Params()
    print(f"{args.kernel}/{args.mul}: {pl.net.n} neurons, {len(per_cell)} multiplier cells, {len(ids)} probes; "
          f"{len(scheds[0])} tokens; build {time.perf_counter() - t0:.0f} s", flush=True)
    import torch
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    kw = {"backend": args.backend} if args.backend != "torch" else {}
    outs, sim, stats = run_pipeline_batched(pl, P, scheds, max_ms=args.max_ms, device=args.device, dtype=dtype,
                                            expect_outputs=expect, progress=300,
                                            capture_spikes=(0, ids), **kw)
    wall = time.perf_counter() - t0
    report = analyse(stats["captured_spikes"], per_cell, pl.drive.loop_period_steps, P.dt, {c.name: c for c in pl.cells})
    wrong = None
    if ref is not None:
        got = [v for _, v in outs[0][ks.outputs[0]]]
        wrong = sum(1 for g, r in zip(got, ref) if g != r[0]) + max(0, len(ref) - len(got))
    report["run"] = {"kernel": args.kernel, "mul": args.mul, "source": getattr(args, "source", None), "device": args.device,
                     "backend": args.backend, "dtype": args.dtype, "neurons": pl.net.n, "tokens": len(scheds[0]),
                     "neural_ms": stats["neural_ms"], "wall_s": wall, "faults": stats["faults"], "timeouts": stats["timeouts"],
                     "bad_outputs": stats["bad_outputs"], "outputs": stats["outputs"], "wrong_vs_reference": wrong,
                     "host_stalls": stats.get("host_stalls"), "truncated": stats.get("truncated")}
    print_report(report)
    print(f"run: {stats['neural_ms'] / 1000:.1f} s neural, {wall:.0f} s wall, outputs {stats['outputs']}/{expect}, "
          f"faults {stats['faults']}, timeouts {stats['timeouts']}, wrong {wrong}", flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out + ".json").write_text(json.dumps(report, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else x))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kernel", choices=("perspective", "doom"), default="perspective")
    ap.add_argument("--mul", choices=("array", "pipelined"), default="array")
    ap.add_argument("--tokens", type=int, default=16, help="perspective: column tokens")
    ap.add_argument("--source", default="examples/doom4.c")
    ap.add_argument("--width", type=int, default=8)
    ap.add_argument("--height", type=int, default=5)
    ap.add_argument("--frames", type=int, default=1)
    ap.add_argument("--inputs", default="6")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--backend", default="torch", choices=("torch", "torch-fast"))
    ap.add_argument("--dtype", default="float64", choices=("float64", "float32"))
    ap.add_argument("--max-ms", type=float, default=3600000)
    ap.add_argument("--out", default=None)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
