"""Stage D exit: the 1,000-tick canonical-state comparison on the world-update kernel
(docs/plan.md "Stage D — world-update kernel", docs/perf_campaign.md §6, docs/stage_d.md).

The program is `examples/tick2.c` in its kernel form (`compile_kernel(prog, loop_body(prog),
"i")`, the campaign's `block("tick")`): one tick per input token, the token being the velocity,
px / mx / health carried across ticks by three state cells. After every tick the neural
state is serialized canonically (`canonical_state`) and compared with the reference's.

    python -m drosophilos.bench.stage_d --ticks 1000 --seed 1 --backend torch-fast --device cuda \\
        --copies 1 --mix none --out data/stage_d/nominal_s1.json

Reference: the compiler's kernel oracle (`kernel_outputs` with the state cells as outputs),
cross-checked against the IR interpreter on every tick and the portable C reference
(`compiler.golden.run_golden`, clang with UBSan) on the first `--c-ticks` ticks: the review's
three comparison points, portable C <-> IR interpreter <-> neural execution.
"""

from __future__ import annotations

import argparse
import bisect
import dataclasses
import json
import re
import shutil
import time
from pathlib import Path

import numpy as np

from ..compiler.frontend_c import compile_c
from ..compiler.kernel import KernelSpec, compile_kernel, kernel_outputs, loop_body
from ..isa.ir import interpret
from ..lib.kernel import build_pipeline, run_pipeline, run_pipeline_batched
from ..sim.model import Params

PROGRAM = "examples/tick2.c"
FORMAT = "stage-d-canonical-v1"
# The canonical fields: the program's persistent state, in declaration order in tick2.c.
# Excluded: vel (the token itself), dist and contact (written before read every tick: no state),
# i (the loop counter of `main`; the kernel form has none). `load_kernel` checks this list
# against the compiler's state cells, so a change to the program cannot drift silently.
STATE_VARS = ("px", "mx", "health")
CAMPAIGN_TOKENS = [5, 5, 250, 3, 0, 40, 40, 40]  # kernel_campaign block("tick")
# Scripted prefix, then fresh random bytes. With the oracle's states (tick: token -> px, mx, health):
#   0-7   the campaign tokens: west-wall wrap at 7 (107 + 40 has bit 7 set -> px = 0)
#   8-12  edge values 0, 1, 127, 128, 255: 1 + 127 = 128 wraps, 128 and 255 from px = 0 wrap
#   13    66 puts px beside the monster: contact, health 100 -> 90
#   14-33 twenty zeros: the monster oscillates onto px every other tick, health 90 -> 0 -> 246
#         (an 8-bit unsigned wrap through zero)
#   34-41 2 x 8: player and monster move together, contact every tick, health 236 -> 166
#   42-46 127, 1, 126, 129, 254: wrap, creep, 1 + 126 = 127 (the largest px), wrap, 254 -> wrap
#   47-52 3 x 6: steady walk east, the monster closing from above
# The east wall (px == 200 -> 199) is unreachable at 8 bits: px < 128 after the west check.
# Its SEL cell (c4_sel's condition c3_xor) still runs every tick; the reference is exact regardless.
SCRIPTED_TOKENS = (CAMPAIGN_TOKENS + [0, 1, 127, 128, 255] + [66] + [0] * 20 + [2] * 8
                   + [127, 1, 126, 129, 254] + [3] * 6)


# --- canonical state ----------------------------------------------------------------------

def canonical_layout(width: int = 8, fields=STATE_VARS) -> list[dict]:
    """The serialized layout: one field per state variable in `fields` order, each
    ceil(width / 8) bytes, unsigned, big-endian, no padding between fields and none at the end
    (bits above `width` in a field's leading byte are zero)."""
    nb = (width + 7) // 8
    return [{"field": f, "offset": i * nb, "bytes": nb, "bits": width, "signed": False, "byteorder": "big"}
            for i, f in enumerate(fields)]


def canonical_state(values: dict, width: int = 8, fields=STATE_VARS) -> bytes:
    """Serialize `values` (field -> unsigned int) in the canonical layout. The field order is
    `fields`, never the dict's; extra keys are ignored; a missing field, a non-integer (an
    undecodable neural word is None) or a value outside [0, 2**width) raises ValueError, so a
    comparison can never pass by masking or by the host's object layout."""
    nb = (width + 7) // 8
    out = bytearray()
    for f in fields:
        if f not in values:
            raise ValueError(f"canonical_state: field {f!r} missing")
        v = values[f]
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
            raise ValueError(f"canonical_state: field {f!r} is {v!r}, not an unsigned integer")
        v = int(v)
        if not 0 <= v < (1 << width):
            raise ValueError(f"canonical_state: field {f!r} = {v} outside [0, 2**{width})")
        out += v.to_bytes(nb, "big")
    return bytes(out)


def decode_canonical(data: bytes, width: int = 8, fields=STATE_VARS) -> dict:
    nb = (width + 7) // 8
    if len(data) != nb * len(fields):
        raise ValueError(f"decode_canonical: {len(data)} bytes, expected {nb * len(fields)}")
    return {f: int.from_bytes(data[i * nb:(i + 1) * nb], "big") for i, f in enumerate(fields)}


# --- program, reference ---------------------------------------------------------------------

@dataclasses.dataclass
class Kernel:
    source: str
    prog: object
    ks: KernelSpec
    fields: tuple  # canonical order
    cells: dict  # field -> state cell
    width: int


def load_kernel(path: str = PROGRAM) -> Kernel:
    source = open(path).read()
    prog = compile_c(source)
    ks = compile_kernel(prog, loop_body(prog), "i")
    declared = sorted(ks.state_cells, key=lambda v: prog.variables[v])  # declaration order
    if tuple(declared) != STATE_VARS:
        raise RuntimeError(f"{path}: state variables {declared}, the canonical format names {STATE_VARS}")
    return Kernel(source, prog, ks, STATE_VARS, {v: ks.state_cells[v] for v in STATE_VARS}, prog.width)


def _spec_from(k: Kernel, state: dict | None = None) -> KernelSpec:
    """The kernel with the state cells as its outputs (and, given `state`, their inits)."""
    cells = k.ks.cells
    if state is not None:
        by_cell = {k.cells[f]: state[f] for f in k.fields}
        cells = [dict(c, init=by_cell[c["name"]]) if c["name"] in by_cell else c for c in cells]
    return dataclasses.replace(k.ks, cells=cells, outputs=[k.cells[f] for f in k.fields])


def reference_states(k: Kernel, tokens: list[int], start: dict | None = None) -> list[dict]:
    """The oracle's state after every tick: `kernel_outputs` (the IR semantics the compiler
    lowers) with the three state cells as outputs, from the program's inits or `start`."""
    return [dict(zip(k.fields, row)) for row in kernel_outputs(_spec_from(k, start), list(tokens))]


def tokens_for(ticks: int, seed: int) -> list[int]:
    """The scripted prefix (cut to `ticks`), then fresh random bytes from `seed`."""
    rng = np.random.default_rng(seed)
    pre = SCRIPTED_TOKENS[:ticks]
    return pre + [int(x) for x in rng.integers(0, 256, size=ticks - len(pre))]


def _chunk_source(k: Kernel, start: dict, n: int) -> str:
    """tick2.c with the state initializers set to `start`, the loop run `n` times, and px, mx,
    health emitted at the end of every tick (after the two OUTs the program already makes)."""
    src = k.source
    decl = re.search(r"^static\s+u\d+\s+[^;]*;", src, re.M)
    if decl is None:
        raise RuntimeError("no static declaration line to rewrite")
    line = decl.group(0)
    for f in k.fields:
        line, hits = re.subn(rf"\b{f}\s*=\s*\d+", f"{f} = {start[f]}", line, count=1)
        if hits != 1:
            raise RuntimeError(f"no initializer for {f} in {decl.group(0)!r}")
    src = src[:decl.start()] + line + src[decl.end():]
    src, hits = re.subn(r"\bi\s*=\s*\d+\s*;", f"i = {n};", src, count=1)
    emit = "".join(f"out_pixel({f}); " for f in k.fields)
    src, hits2 = re.subn(r"\bi\s*=\s*i\s*-\s*1\s*;", emit + "i = i - 1;", src, count=1)
    if hits != 1 or hits2 != 1:
        raise RuntimeError("tick2.c's loop counter changed shape; update _chunk_source")
    return src


def _per_tick(outs: list[int], n: int, k: Kernel) -> list[dict]:
    per = len(outs[:-1]) // n if n else 0  # the loop's OUTs per tick; one OUT (health) after it
    nf = len(k.fields)
    if per < nf or len(outs) != per * n + 1:
        raise RuntimeError(f"unexpected OUT count {len(outs)} for {n} ticks")
    return [dict(zip(k.fields, outs[t * per + per - nf:(t + 1) * per])) for t in range(n)]


CHUNK = 50  # ticks per rewritten program: the C shim holds 64 inputs and 256 outputs; i is u8


def ir_states(k: Kernel, tokens: list[int]) -> list[dict]:
    """The IR interpreter's state after every tick (the rewritten program, in chunks)."""
    state, out = {f: int(v) for f, v in reference_start(k).items()}, []
    for c0 in range(0, len(tokens), CHUNK):
        chunk = tokens[c0:c0 + CHUNK]
        prog = compile_c(_chunk_source(k, state, len(chunk)))
        budget = 200 * len(chunk) + 1000
        r = interpret(prog, list(chunk), max_steps=budget)
        if r["steps"] >= budget:
            raise RuntimeError("IR interpreter ran out of steps on a chunk")
        rows = _per_tick(r["outs"], len(chunk), k)
        out += rows
        state = rows[-1]
    return out


def c_states(k: Kernel, tokens: list[int]) -> list[dict]:
    """The portable C reference's state after every tick (clang -fsanitize=undefined)."""
    from ..compiler.golden import run_golden
    state, out = reference_start(k), []
    for c0 in range(0, len(tokens), CHUNK):
        chunk = tokens[c0:c0 + CHUNK]
        g = run_golden(_chunk_source(k, state, len(chunk)), list(chunk), list(k.fields), k.width)
        rows = _per_tick(g["outs"], len(chunk), k)
        if dict(g["state"]) != rows[-1]:
            raise RuntimeError(f"C reference: final state {g['state']} differs from its last tick {rows[-1]}")
        out += rows
        state = rows[-1]
    return out


def reference_start(k: Kernel) -> dict:
    return {f: next(c["init"] for c in k.ks.cells if c["name"] == k.cells[f]) for f in k.fields}


def three_point_check(k: Kernel, tokens: list[int], ref: list[dict], c_ticks: int) -> dict:
    """Oracle vs the IR interpreter on every tick, and vs portable C on the first `c_ticks`."""
    res = {"oracle": "compiler.kernel.kernel_outputs", "ir_ticks": len(tokens), "ir_first_disagreement": None}
    ir = ir_states(k, tokens)
    for t, (a, b) in enumerate(zip(ref, ir)):
        if a != b:
            res["ir_first_disagreement"] = {"tick": t, "oracle": a, "ir": b}
            break
    res["ir_equal"] = res["ir_first_disagreement"] is None and len(ir) == len(ref)
    n_c = min(c_ticks, len(tokens))
    res["c_ticks"] = n_c
    if not any(shutil.which(x) for x in ("clang", "gcc", "cc")):
        res.update(c_equal=None, c_status="not_run: no C compiler on PATH")
        return res
    cs = c_states(k, tokens[:n_c])
    first = next(({"tick": t, "oracle": a, "c": b} for t, (a, b) in enumerate(zip(ref, cs)) if a != b), None)
    res.update(c_equal=first is None and len(cs) == n_c, c_first_disagreement=first, c_status="run")
    return res


# --- neural run -------------------------------------------------------------------------------

def build(k: Kernel, params: Params, datapath: str = "generic"):
    """The tick kernel's pipeline with every state cell decoded by the host: the kernel's own
    outputs (px, mx) plus health's carrier, so each tick commits one word per canonical field."""
    outputs = list(k.ks.outputs) + [k.cells[f] for f in k.fields if k.cells[f] not in k.ks.outputs]
    return build_pipeline(params, k.width, k.ks.cells, consts=k.ks.consts, mems=k.ks.mems,
                          outputs=outputs, datapath=datapath)


def make_sim(pl, params: Params, copies: int, backend: str, mix: str, seed: int, device: str, dtype, n_steps: int):
    """The simulator, built here (not by the runner) so the stall watch can read its clock.
    mix "none": nominal weights, the same classes and arguments the runner would build."""
    if backend == "ref":
        if copies != 1 or mix != "none":
            raise SystemExit("--backend ref runs one nominal copy (use torch / torch-fast for copies or a mix)")
        from ..sim.ref64 import RefSim
        return RefSim(pl.net.topology(), params)
    kw = {"dtype": dtype} if dtype is not None else {}
    if mix == "none":
        if backend == "torch-fast":
            from ..sim.lif_fast import FastSim
            return FastSim(pl.net.topology(), params, n_nodes=copies, device=device, **kw)
        from ..sim.lif_torch import TorchSim
        return TorchSim(pl.net.topology(), params, n_nodes=copies, device=device, **kw)
    from ..lib.campaign import make_perturbed_sim
    from .a2_campaigns import MIXES
    return make_perturbed_sim(pl.net.topology(), params, copies, MIXES[mix], np.random.default_rng(seed), n_steps,
                              device=device, backend=backend, **kw)


def run_neural(k: Kernel, pl, params: Params, tokens: list[int], *, copies: int = 1, backend: str = "torch",
               mix: str = "none", seed: int = 0, device: str = "cpu", dtype=None, max_ms: float = 60000,
               stall_ms: float | None = None, progress=0, retry_refused: bool = True, rail_filter=None) -> dict:
    """Runs `copies` copies on the same tokens. Returns per copy the decoded state-cell lists
    (cell -> [(step, value)]), load events, refusals, the wall time of every output, and the
    run's stats. `stall_ms`: stop once every copy has finished or gone that long (neural)
    without an output (None: run to max_ms)."""
    n_steps = int(max_ms / params.dt) + 5000
    sim = make_sim(pl, params, copies, backend, mix, seed, device, dtype, n_steps)
    n_out = len(pl.outputs)
    want = len(tokens) * n_out
    got = [0] * copies
    last = [0] * copies
    wall = [[] for _ in range(copies)]
    stall_steps = None if stall_ms is None else int(stall_ms / params.dt)
    watch = {"polls": 0, "stalled": False}

    def on_output(b, cell, stp, value, wall_s):
        got[b] += 1
        last[b] = max(last[b], stp)
        wall[b].append((cell, stp, wall_s))

    def should_stop():
        watch["polls"] += 1
        if stall_steps is None or watch["polls"] % 1000:
            return False
        now = sim.step_index
        if all(got[b] >= want or now - last[b] > stall_steps for b in range(copies)) and any(got[b] < want for b in range(copies)):
            watch["stalled"] = True
            return True
        return False

    t0 = time.perf_counter()
    if backend == "ref":
        _, sim, st = run_pipeline(pl, params, list(tokens), max_ms=max_ms, sim=sim, expect_outputs=want,
                                  on_output=on_output, should_stop=should_stop, retry_refused=retry_refused,
                                  rail_filter=rail_filter)
        outs = [st.pop("outputs_by_cell")]  # run_pipeline's shapes -> run_pipeline_batched's (one node)
        st["load_events"] = [st["load_events"]]
        st["load_steps"] = [st["load_steps"]]
        st.setdefault("neural_ms", sim.step_index * params.dt)
    else:
        outs, sim, st = run_pipeline_batched(pl, params, [list(tokens) for _ in range(copies)], max_ms=max_ms,
                                             device=device, expect_outputs=[want] * copies, sim=sim, dtype=dtype,
                                             backend=backend, progress=progress, on_output=on_output,
                                             should_stop=should_stop, retry_refused=retry_refused,
                                             rail_filter=rail_filter)
    st["run_wall_s"] = time.perf_counter() - t0
    st["run_started_perf"] = t0
    st["stopped_on_stall"] = watch["stalled"]
    st["simulator"] = type(sim).__name__
    return {"outs": outs, "stats": st, "wall": wall}


# --- comparison -------------------------------------------------------------------------------

def compare_copy(k: Kernel, ref: list[dict], tokens: list[int], outs: dict, *, t_load0: int | None,
                 dt: float, load_events: list | None = None, refused: list | None = None,
                 blocked: bool = False, limit: str | None = None, run_end_step: int | None = None,
                 stall_ms: float | None = None) -> dict:
    """One copy against the per-tick reference. Tick t is complete when every canonical field's
    cell has committed exactly one word after tick t's input injection and before the next input
    injection.  The injection boundaries are important: positional matching cannot distinguish a
    missed commit followed by a duplicate while a field happens to retain its value.  Counting
    stops at the first mismatch (a diverged state makes every later tick wrong and says nothing
    more).  `faults` and `timeouts` deliberately do not appear here: kernel.py reports those for
    the whole batch, not for an individual copy."""
    fields, width = k.fields, k.width
    lists = [outs.get(k.cells[f], []) for f in fields]
    requested = len(ref)
    ref_bytes = [canonical_state(r, width, fields) for r in ref]
    first, matched, invalid = None, 0, 0

    # A retry has the same schedule_index as its original injection.  The final injection is the
    # one whose resulting state can commit, so it is the lower boundary for that tick.
    loads_by_tick = {}
    for event in load_events or []:
        index, step = event.get("schedule_index"), event.get("event_step")
        if isinstance(index, int) and isinstance(step, (int, float)) and 0 <= index < requested:
            loads_by_tick[index] = max(loads_by_tick.get(index, step), step)
    load_steps = [loads_by_tick.get(t) for t in range(requested)]
    commit_counts_checkable = requested == 0 or all(s is not None for s in load_steps)

    if commit_counts_checkable:
        # Bin each cell's commits between successive token injections.  Besides the counts, keep
        # the bin for each positional word: a word must be both the next word and in its tick's
        # injection interval.
        bins = [[[] for _ in range(requested)] for _ in fields]
        positions = [[] for _ in fields]
        out_of_window = 0
        for i, commits in enumerate(lists):
            for ordinal, item in enumerate(commits):
                step = item[0]
                t = bisect.bisect_right(load_steps, step) - 1
                positions[i].append(t)
                if 0 <= t < requested:
                    bins[i][t].append(item)
                else:
                    out_of_window += 1
        count_rows = [{f: len(bins[i][t]) for i, f in enumerate(fields)} for t in range(requested)]
        missing = sum(1 for row in count_rows if any(n == 0 for n in row))
        duplicates = out_of_window + sum(max(0, n - 1) for row in count_rows for n in row.values())
        completed = sum(1 for row in count_rows if all(n == 1 for n in row.values()))
    else:
        # Older records did not retain injections.  Preserve the former positional comparison,
        # but label the new count invariant as uncheckable rather than claiming it passed.
        completed = min((len(x) for x in lists), default=0)
        duplicates = sum(max(0, len(x) - requested) for x in lists)
        missing = requested - min(completed, requested)
        bins = [[[] for _ in range(requested)] for _ in fields]
        positions = [[] for _ in fields]
        for i, commits in enumerate(lists):
            for t, item in enumerate(commits[:requested]):
                bins[i][t].append(item)
                positions[i].append(t)
        count_rows = [{f: len(bins[i][t]) for i, f in enumerate(fields)} for t in range(requested)]

    for t in range(requested if commit_counts_checkable else min(completed, requested)):
        counts = count_rows[t]
        # The counts are the primary comparison when injection evidence is available.  Report
        # the first bad cell and both its expected and observed count, even when its value is
        # unchanged (the failure that positional matching missed).
        bad_count = next((f for f in fields if counts[f] != 1), None)
        if bad_count is not None:
            got = {f: (bins[i][t][0][1] if len(bins[i][t]) == 1 else None) for i, f in enumerate(fields)}
            step = max((item[0] for row in bins for item in row[t]), default=None)
            first = {"tick": t, "token": tokens[t], "field": bad_count, "expected": ref[t][bad_count],
                     "got": got[bad_count], "expected_state": ref[t], "got_state": got, "step": step,
                     "class": "commit count", "expected_commits": 1, "got_commits": counts[bad_count],
                     "commit_counts": counts}
            break
        # In addition to one commit in each interval, the t-th observed commit must be that
        # interval's commit.  This makes the order constraint explicit in the record.
        bad_order = next((fields[i] for i in range(len(fields))
                          if len(positions[i]) > t and positions[i][t] != t), None)
        if bad_order is not None:
            got = {f: bins[i][t][0][1] for i, f in enumerate(fields)}
            first = {"tick": t, "token": tokens[t], "field": bad_order, "expected": ref[t][bad_order],
                     "got": got[bad_order], "expected_state": ref[t], "got_state": got,
                     "step": bins[fields.index(bad_order)][t][0][0], "class": "commit order",
                     "expected_commits": 1, "got_commits": 1, "commit_counts": counts}
            break
        got = {f: bins[i][t][0][1] for i, f in enumerate(fields)}
        try:
            ok = canonical_state(got, width, fields) == ref_bytes[t]
        except ValueError:
            ok, invalid = False, 1
        if ok:
            matched += 1
            continue
        field = next(f for f in fields if got[f] != ref[t][f])
        first = {"tick": t, "token": tokens[t], "field": field, "expected": ref[t][field], "got": got[field],
                 "expected_state": ref[t], "got_state": got, "step": max(x[t][0] for x in lists),
                 "class": _mismatch_class(k, ref, tokens, t, got)}
        break
    # Per-tick timing is meaningful only for structurally complete ticks.  In the fallback it
    # retains the old positional calculation for compatibility with pre-v2 records.
    if commit_counts_checkable:
        tick_steps = [max(bins[i][t][0][0] for i in range(len(fields)))
                      for t in range(requested) if all(count_rows[t][f] == 1 for f in fields)]
    else:
        tick_steps = [max(x[t][0] for x in lists) for t in range(completed)]
    tick_ms = [round((s_ - (t_load0 or 0)) * dt, 1) for s_ in tick_steps]
    retried = sorted({e["schedule_index"] for e in (load_events or []) if e.get("retry")})
    # A missing trailing state is a liveness outcome, not a wrong state.  A missing commit
    # followed by a later complete tick (or paired with an extra commit) is a true structural
    # mismatch.  This distinction keeps a max-ms-cut copy from being mislabeled merely because
    # its uncompleted final tick has no words.
    terminal_shortfall = (first is not None and first["class"] == "commit count" and duplicates == 0 and
                          not any(all(count_rows[u][f] == 1 for f in fields)
                                  for u in range(first["tick"] + 1, requested)))
    liveness_shortfall = terminal_shortfall or (first is None and missing > 0)
    wrong = 1 if first is not None and not terminal_shortfall else 0
    last_commit_step = max((item[0] for commits in lists for item in commits), default=None)
    stopped_at_tick = completed - 1 if completed else None
    if matched == requested and missing == 0 and duplicates == 0:
        status = "matched"
    elif (liveness_shortfall and run_end_step is not None and stall_ms is not None and
          run_end_step - (last_commit_step if last_commit_step is not None else (t_load0 or 0)) > stall_ms / dt):
        status = "stalled"
    elif liveness_shortfall and limit == "max_ms":
        # A copy only earns this label when it was still making progress at the run's time cap.
        status = "truncated"
    elif liveness_shortfall and limit == "stall":
        status = "stalled"
    elif first is not None:
        status = "mismatch"
    else:
        status = "stalled" if blocked else "truncated"
    return {"status": status, "requested": requested, "completed": completed, "matched": matched,
            "wrong": wrong, "unscored": max(0, completed - matched - wrong),
            "missing": missing, "duplicates": duplicates, "invalid": invalid,
            "first_mismatch": first, "refusals": len(refused or []), "retries": len(retried),
            "retried_ticks": retried,
            "retried_ticks_matched": all(t < matched for t in retried) if retried else None,
            "applied_twice": bool(duplicates) or (first is not None and first["class"] == "previous word applied twice"),
            "commit_counts_checkable": commit_counts_checkable, "last_commit_step": last_commit_step,
            "stopped_at_tick": stopped_at_tick, "tick_ms": tick_ms}


def _mismatch_class(k: Kernel, ref, tokens, t, got) -> str:
    """Names the two known state-order failures: a lagging state (the tick's word not yet
    applied, seed 109 copy 61) and the previous word applied a second time (a retried or
    doubled commit, seed 108 copy 55)."""
    try:
        start = ref[t - 1] if t else reference_start(k)
        if got == start:
            return "state lags a tick"
        if t and got == reference_states(k, [tokens[t - 1]], start)[0]:
            return "previous word applied twice"
    except (TypeError, KeyError):
        pass
    return "other"


def tick_wall_s(k: Kernel, wall: list, t0: float) -> list[float]:
    """Wall seconds from the run's start to each tick's last field commit (one copy; every copy
    of a batched run shares the simulator's clock)."""
    per = {f: [w for c, _, w in wall if c == k.cells[f]] for f in k.fields}
    n = min(len(v) for v in per.values())
    return [round(max(per[f][t] for f in k.fields) - t0, 2) for t in range(n)]


def verdict(per_copy: list[dict], ticks: int, *, faults: int = 0, timeouts: int = 0,
            truncated: bool = False) -> str:
    """The Stage D exit.  Faults/timeouts are intentionally arguments rather than copy fields:
    kernel.py counts those latches once for the entire batched run."""
    if (per_copy and not truncated and faults == 0 and timeouts == 0 and
            all(c["status"] == "matched" and c["matched"] == ticks and c["wrong"] == 0 and
                c["missing"] == 0 and c["duplicates"] == 0 for c in per_copy)):
        return "exit met"
    problems = []
    if faults:
        problems.append(f"{faults} faults (run-level)")
    if timeouts:
        problems.append(f"{timeouts} timeouts (run-level)")
    if truncated:
        problems.append("run truncated")
    kinds = sorted({c["status"] for c in per_copy if c["status"] != "matched"})
    problems += [f"{sum(1 for c in per_copy if c['status'] == s)} {s}" for s in kinds]
    # Include retry/refusal accounting in the verdict line as well as the counters record; this
    # makes a pass with recovered input work visible without treating a retry as a failure.
    refusals = sum(c.get("refusals", 0) for c in per_copy)
    retries = sum(c.get("retries", 0) for c in per_copy)
    problems.append(f"refusals {refusals}")
    problems.append(f"retries {retries}")
    return "exit not met: " + ", ".join(problems)


# --- CLI --------------------------------------------------------------------------------------

def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ticks", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0, help="the random tokens after the scripted prefix, and mix B's draws")
    ap.add_argument("--backend", default="torch", choices=("torch", "torch-fast", "ref"),
                    help="ref: the float64 RefSim, one nominal copy (laptop checks)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--copies", type=int, default=1)
    ap.add_argument("--mix", default="none", choices=("none", "0", "B", "B+"),
                    help="none: nominal weights; B: the campaigns' perturbation (make_perturbed_sim)")
    ap.add_argument("--datapath", choices=("generic", "specialized"), default="generic")
    ap.add_argument("--max-ms", type=float, default=None,
                    help="neural-time ceiling; default: calibrated from a short nominal run x --margin")
    ap.add_argument("--calibrate-ticks", type=int, default=4)
    ap.add_argument("--margin", type=float, default=1.5, help="max_ms = (first-tick latency + ticks x per-tick) x margin")
    ap.add_argument("--stall-ticks", type=float, default=20.0,
                    help="a copy with no output for this many calibrated tick times is stalled (0: never)")
    ap.add_argument("--c-ticks", type=int, default=100, help="ticks cross-checked against the portable C reference")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--recheck", metavar="RECORD.json",
                    help="recompute a saved record's per-copy status and verdict; write it in place unless --out is given")
    return ap


def _saved_load_events(events: list) -> list[dict]:
    """The small, durable portion of runner load_events needed for tick assignment."""
    return [{key: e[key] for key in ("schedule_index", "event_step", "retry", "attempts") if key in e}
            for e in events]


def _record_commit_events(outs: list[dict]) -> list[dict]:
    """Commit evidence retained so --recheck can apply future state-order rules."""
    return [{cell: [[step, value] for step, value in commits] for cell, commits in copy.items()}
            for copy in outs]


def _old_record_copy_status(copy: dict, rec: dict, ticks: int) -> dict:
    """Reclassify an old v1 record when its raw load/commit evidence was not retained.
    Value mismatches and liveness can still be judged; the count invariant cannot."""
    c = dict(copy)
    c["commit_counts_checkable"] = False
    if c.get("first_mismatch") is not None or c.get("wrong", 0):
        c["status"] = "mismatch"
        return c
    if c.get("matched", 0) == ticks and c.get("missing", 0) == 0 and c.get("duplicates", 0) == 0:
        c["status"] = "matched"
        return c
    ms = c.get("tick_ms", [])
    if not ms:
        # v1 keeps tick_ms at the top level, and callers attach it before invoking this helper.
        ms = []
    last_ms = ms[-1] if ms else None
    end_ms = rec.get("neural_s", 0) * 1000
    stall_ms = rec.get("stall_ms")
    if last_ms is not None and stall_ms is not None and end_ms - last_ms > stall_ms:
        c["status"] = "stalled"
    elif rec.get("stop") == "stall" or rec.get("stalled"):
        c["status"] = "stalled"
    else:
        c["status"] = "truncated"
    c["stopped_at_tick"] = c.get("completed", 0) - 1 if c.get("completed", 0) else None
    return c


def recheck_record(rec: dict) -> dict:
    """Rejudge a persisted Stage D record without a neural run.

    New records contain commit_events plus load_events, allowing the exact-once-per-tick rule to
    be reapplied.  Old records remain useful: their saved counters, mismatch evidence, timings,
    and batch latches are enough for the fault/timeout and per-copy-liveness rules.
    """
    ticks = int(rec.get("ticks", len(rec.get("tokens", []))))
    raw = rec.get("commit_events")
    loads = rec.get("load_events")
    checkable = (isinstance(raw, list) and isinstance(loads, list) and len(raw) == len(loads) and
                 all({e.get("schedule_index") for e in events if isinstance(e, dict)} >= set(range(ticks))
                     for events in loads))
    old_copies = rec.get("per_copy", [])
    per_copy = []
    if checkable:
        k = load_kernel(rec.get("program", PROGRAM))
        tokens = rec.get("tokens", [])
        if len(tokens) != ticks:
            raise ValueError("record tokens do not match record ticks")
        ref = reference_states(k, tokens)
        dt = float(rec.get("dt_ms", Params().dt))
        run_end_step = rec.get("run_end_step")
        if run_end_step is None and rec.get("neural_s") is not None:
            run_end_step = round(float(rec["neural_s"]) * 1000 / dt)
        for b, outs in enumerate(raw):
            ev = loads[b]
            first_load = next((e.get("event_step") for e in ev if e.get("schedule_index") == 0), 0)
            c = compare_copy(k, ref, tokens, outs, t_load0=first_load, dt=dt, load_events=ev,
                             refused=[None] * int(old_copies[b].get("refusals", 0)) if b < len(old_copies) else [],
                             limit="max_ms" if rec.get("stop") == "max_ms" else None,
                             run_end_step=run_end_step, stall_ms=rec.get("stall_ms"))
            c["copy"] = b
            per_copy.append(c)
    else:
        for b, old in enumerate(old_copies):
            c = dict(old)
            if b < len(rec.get("tick_ms", [])):
                c["tick_ms"] = rec["tick_ms"][b]
            c = _old_record_copy_status(c, rec, ticks)
            c["copy"] = c.get("copy", b)
            per_copy.append(c)

    # Preserve the compact on-disk per_copy representation.  tick_ms remains top-level.
    rec["per_copy"] = [{x: y for x, y in c.items() if x != "tick_ms"} for c in per_copy]
    if checkable:
        rec["tick_ms"] = [c["tick_ms"] for c in per_copy]
    run_truncated = bool(rec.get("truncated") or rec.get("stop") == "max_ms")
    rec["verdict"] = verdict(per_copy, ticks, faults=int(rec.get("faults", 0)),
                             timeouts=int(rec.get("timeouts", 0)), truncated=run_truncated)
    rec["verdict_accounting"] = {
        "faults": int(rec.get("faults", 0)), "timeouts": int(rec.get("timeouts", 0)),
        "refusals": sum(c.get("refusals", 0) for c in per_copy),
        "retries": sum(c.get("retries", 0) for c in per_copy),
        "fault_timeout_scope": "run-level batch counters (kernel.py), not per-copy",
    }
    rec["recheck"] = {
        "commit_counts_checkable": checkable,
        "note": ("exactly-once commit counts rechecked from saved load events"
                 if checkable else "exactly-once commit counts not checkable: saved record lacks load steps/commit evidence"),
        "fault_timeout_scope": "run-level batch counters (kernel.py), not per-copy",
        "refusals": sum(c.get("refusals", 0) for c in per_copy),
        "retries": sum(c.get("retries", 0) for c in per_copy),
    }
    return rec


def calibrate(k: Kernel, pl, params, tokens, a, dtype) -> dict:
    """A short nominal single-copy run on the same backend: neural ms to the first tick and per tick."""
    n = max(2, min(a.calibrate_ticks, len(tokens)))
    backend = a.backend
    r = run_neural(k, pl, params, tokens[:n], copies=1, backend=backend, device=a.device, dtype=dtype,
                   max_ms=n * 30000.0)
    st = r["stats"]
    ref = reference_states(k, tokens[:n])
    c = compare_copy(k, ref, tokens[:n], r["outs"][0], t_load0=st["load_steps"][0][0] if st["load_steps"][0] else 0,
                     dt=params.dt)
    ms = c["tick_ms"]
    if len(ms) < 2:
        raise SystemExit(f"calibration: {len(ms)} of {n} ticks completed ({c['status']}); give --max-ms")
    per_tick = (ms[-1] - ms[0]) / (len(ms) - 1)
    return {"ticks": n, "first_tick_ms": ms[0], "per_tick_ms": round(per_tick, 1), "matched": c["matched"],
            "wall_s": round(st["run_wall_s"], 1), "neural_ms": st["neural_ms"],
            "wall_per_neural": round(st["run_wall_s"] / (st["neural_ms"] / 1000), 3) if st["neural_ms"] else None}


def main(argv=None):
    a = parser().parse_args(argv)
    if a.recheck:
        source = Path(a.recheck)
        rec = recheck_record(json.loads(source.read_text()))
        target = Path(a.out) if a.out else source
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w") as f:
            json.dump(rec, f, indent=1)
        print(json.dumps({"verdict": rec["verdict"], "recheck": rec["recheck"]}, indent=1), flush=True)
        return rec
    import torch
    P = Params()
    dtype = torch.float32 if a.fp32 else (torch.float64 if a.backend != "ref" else None)
    k = load_kernel()
    tokens = tokens_for(a.ticks, a.seed)
    ref = reference_states(k, tokens)
    check = three_point_check(k, tokens, ref, a.c_ticks)
    print("three-point check:", json.dumps({x: y for x, y in check.items()}), flush=True)
    if not check["ir_equal"] or check.get("c_equal") is False:
        raise SystemExit("the references disagree; not running the neural comparison")
    pl = build(k, P, a.datapath)
    calib = None
    if a.max_ms is None:
        calib = calibrate(k, pl, P, tokens, a, dtype)
        max_ms = (calib["first_tick_ms"] + a.ticks * calib["per_tick_ms"]) * a.margin
        print("calibration:", json.dumps(calib), f"-> max_ms {max_ms:.0f}", flush=True)
    else:
        max_ms = a.max_ms
    per_tick_est = calib["per_tick_ms"] if calib else max_ms / max(1, a.ticks)
    stall_ms = a.stall_ticks * per_tick_est if a.stall_ticks > 0 else None
    t0 = time.time()
    r = run_neural(k, pl, P, tokens, copies=a.copies, backend=a.backend, mix=a.mix, seed=a.seed, device=a.device,
                   dtype=dtype, max_ms=max_ms, stall_ms=stall_ms, progress=300)
    st = r["stats"]
    limit = "stall" if st["stopped_on_stall"] else ("max_ms" if st.get("host_stalls") or st.get("truncated") else None)
    run_end_step = round(float(st["neural_ms"]) / P.dt)
    refused = st.get("refused") or [[] for _ in range(a.copies)]
    per_copy = []
    for b in range(a.copies):
        loads = st["load_steps"][b]
        c = compare_copy(k, ref, tokens, r["outs"][b], t_load0=loads[0] if loads else 0, dt=P.dt,
                         load_events=st["load_events"][b], refused=refused[b],
                         blocked=b in st.get("blocked_nodes", []), limit=limit, run_end_step=run_end_step,
                         stall_ms=stall_ms)
        c["copy"] = b
        per_copy.append(c)
    n_ticks = [c["tick_ms"] for c in per_copy if len(c["tick_ms"]) > 1]
    per_tick_neural = float(np.median([np.median(np.diff(x)) for x in n_ticks])) if n_ticks else None
    totals = {x: sum(c[x] for c in per_copy) for x in ("requested", "completed", "matched", "wrong", "unscored",
                                                         "missing", "duplicates", "invalid", "refusals", "retries")}
    rec = {
        "stage": "D", "program": PROGRAM, "kernel": "compile_kernel(prog, loop_body(prog), 'i')",
        "canonical_format": FORMAT, "canonical_layout": canonical_layout(k.width, k.fields),
        "state_cells": k.cells, "width": k.width,
        "ticks": a.ticks, "seed": a.seed, "scripted_prefix": min(a.ticks, len(SCRIPTED_TOKENS)), "tokens": tokens,
        "backend": a.backend, "simulator": st["simulator"], "device": a.device, "dtype": str(dtype),
        "copies": a.copies, "mix": a.mix, "datapath": pl.datapath, "neurons": pl.net.n,
        "build_options": dict(pl.build_options),
        "max_ms": round(max_ms, 1), "max_ms_source": "calibrated" if calib else "given", "calibration": calib,
        "margin": a.margin, "stall_ms": stall_ms, "stop": limit or "complete", "dt_ms": P.dt,
        "run_end_step": run_end_step,
        "three_point_check": check,
        "reference_canonical_hex": [canonical_state(s, k.width, k.fields).hex() for s in ref],
        "counters": totals, "faults": st["faults"], "timeouts": st["timeouts"], "bad_outputs": st.get("bad_outputs"),
        "fault_timeout_scope": "run-level batch counters (kernel.py), not per-copy",
        "verdict_accounting": {"faults": st["faults"], "timeouts": st["timeouts"],
                                "refusals": totals["refusals"], "retries": totals["retries"],
                                "fault_timeout_scope": "run-level batch counters (kernel.py), not per-copy"},
        "blocked_nodes": st.get("blocked_nodes", []), "truncated": limit == "max_ms", "stalled": limit == "stall",
        "retried_commit_applied_once": (not any(c["applied_twice"] for c in per_copy)),
        "retried_commit_scope": "host refusal resend only (lib/kernel.py retry_refused): no commit log, no TMR",
        "per_copy": [{x: y for x, y in c.items() if x != "tick_ms"} for c in per_copy],
        "tick_ms": [c["tick_ms"] for c in per_copy],
        "commit_events": _record_commit_events(r["outs"]),
        "load_events": [_saved_load_events(events) for events in st["load_events"]],
        "tick_wall_s_copy0": tick_wall_s(k, r["wall"][0], st["run_started_perf"]),
        "per_tick_neural_ms": per_tick_neural, "neural_s": st["neural_ms"] / 1000, "wall_s": round(time.time() - t0, 1),
        "verdict": verdict(per_copy, a.ticks, faults=st["faults"], timeouts=st["timeouts"],
                           truncated=limit == "max_ms"),
    }
    summary = {x: rec[x] for x in ("verdict", "counters", "faults", "timeouts", "stop", "neural_s", "wall_s",
                                   "per_tick_neural_ms", "retried_commit_applied_once")}
    print(json.dumps(summary, indent=1), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(rec, open(a.out, "w"), indent=1)
    return rec


if __name__ == "__main__":
    main()
