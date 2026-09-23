"""Localize a resident-kernel stall from one campaign copy's spike dump.

The dump is read-only.  The tool rebuilds the tick kernel, verifies every recorded
``neuron -> role`` mapping against that netlist, and reconstructs request-rail intervals
and the request -> go -> START -> ACT^d -> stage -> commit -> DONE path.  It accepts both
the compact seed-107 format (``uniq``/``role_of``) and the role-filtered replay format
(``role`` alongside every spike).

Example::

    python -m drosophilos.bench.stall_diag data/a2/tick_s107_node18_compact.npz \
      --campaign data/a2/kc_tick_B90s107b.json --node 18 \
      --out docs/a2/tick_s107_node18_stall_diag.md

This is an event diagnostic, not a deterministic replay.  In particular, a copy's stray
stream depends on the campaign batch size.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..compiler.frontend_c import compile_c
from ..compiler.kernel import compile_kernel, loop_body
from ..lib.kernel import build_pipeline
from ..sim.model import Params


# Juno 413471 uses this filter.  A filtered NPZ does not store the expression, so these are
# the roles whose absence can still be interpreted as silence.  Anything outside the filter
# is rendered as unavailable rather than silently treated as a missing event.
REPLAY_ROLE_FILTER = re.compile(
    r"\.(req\.|relight|start$|actd\.|idle|creq|autocommit|commit|done|kill|Q\.comp|Q\.valid|fault|ready|go\.)"
)


@dataclass
class SpikeDump:
    step: np.ndarray
    neuron: np.ndarray
    present_ids: set[int]
    full_capture: bool
    format_name: str
    current_to_dump: dict[int, int]
    role_id_drift: int

    @property
    def end(self) -> int:
        return int(self.step[-1]) if len(self.step) else 0


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_tick_pipeline(campaign: dict[str, Any]):
    """Rebuild exactly the kernel shape used by ``kernel_campaign tick``."""
    if campaign.get("block", "tick") != "tick":
        raise ValueError(f"stall_diag currently diagnoses the tick kernel, not {campaign.get('block')!r}")
    from ..lib import control

    source = _repo_root() / "examples" / "tick2.c"
    prog = compile_c(source.read_text())
    ks = compile_kernel(prog, loop_body(prog), "i")
    params = Params()
    # the kill train's shape is part of the netlist: campaigns record it since 2026-09-20;
    # earlier ones (seeds 107, 108) were built with 3 x 0.75
    pulses, strength = campaign.get("kill_train", (3, 0.75))
    saved = control.KILL_PULSES, control.KILL_STRENGTH
    control.KILL_PULSES, control.KILL_STRENGTH = int(pulses), float(strength)
    try:
        pl = build_pipeline(
            params,
            prog.width,
            ks.cells,
            consts=ks.consts,
            mems=ks.mems,
            outputs=ks.outputs,
            datapath=campaign.get("datapath", "generic"),
            relight_requests=campaign.get("relight_requests", True),
            powerup_veto=campaign.get("powerup_veto", False),  # recorded since 2026-09-20; older builds had none
            commit_reignite=campaign.get("commit_reignite", False),
            true_guards=campaign.get("true_guards", False),  # recorded since 2026-09-23; older guards accepted dark pairs
            retry_clear=campaign.get("retry_clear", False),
            start_relight_hops=campaign.get("start_relight_hops", 0),  # recorded since 2026-09-20; older builds relit at START
            request_clear_pulses=campaign.get("request_clear_pulses", int(pulses)),  # older builds: the kill train's own count
        )
    finally:
        control.KILL_PULSES, control.KILL_STRENGTH = saved
    expected_n = campaign.get("neurons")
    if expected_n is not None and int(expected_n) != pl.net.n:
        raise AssertionError(f"campaign has {expected_n} neurons; rebuilt tick kernel has {pl.net.n}")
    return params, ks, pl


def load_dump(path: str | Path, roles: list[str]) -> SpikeDump:
    """Load arrays and validate recorded roles in vectorized chunks.

    No Python loop visits individual spikes.  The only chunked loop is the role assertion;
    it bounds the temporary string array for the new per-spike-role format.
    """
    path = Path(path)
    role_table = np.asarray(roles)
    with np.load(path, allow_pickle=False) as raw:
        if "step" not in raw or "neuron" not in raw:
            raise ValueError(f"{path} must contain step and neuron arrays")
        step = np.asarray(raw["step"], dtype=np.int64)
        neuron = np.asarray(raw["neuron"], dtype=np.int64)
        if step.ndim != 1 or neuron.ndim != 1 or len(step) != len(neuron):
            raise ValueError("step and neuron must be equal-length one-dimensional arrays")
        if len(step) and np.any(step[1:] < step[:-1]):
            raise ValueError("dump steps must be sorted")
        if len(neuron) and (int(neuron.min()) < 0 or int(neuron.max()) >= len(roles)):
            raise AssertionError("dump contains a neuron id outside the rebuilt netlist")

        if "uniq" in raw and "role_of" in raw:
            uniq = np.asarray(raw["uniq"], dtype=np.int64)
            role_of = np.asarray(raw["role_of"]).astype(str)
            if len(uniq) != len(role_of):
                raise ValueError("uniq and role_of must have equal length")
            # Track B reordered generic ALU construction without changing its role set or
            # neuron count.  The seed-107 artifact predates that reorder.  Assert every role
            # against the rebuilt netlist and translate those legacy datapath ids by role;
            # handshake ids (including the localized 11828/11970 pair) remain exact.
            role_to_current: dict[str, list[int]] = defaultdict(list)
            for current, role in enumerate(roles):
                role_to_current[role].append(current)
            missing_roles = [role for role in role_of.tolist() if role not in role_to_current]
            if missing_roles:
                raise AssertionError(f"dump roles are absent from rebuilt netlist: {missing_roles[:3]!r}")
            mismatch = role_of != role_table[uniq]
            for old, old_role, current_role in zip(uniq[mismatch].tolist(), role_of[mismatch].tolist(),
                                                   role_table[uniq[mismatch]].tolist()):
                handshake_moved = (REPLAY_ROLE_FILTER.search(str(old_role)) or
                                   REPLAY_ROLE_FILTER.search(str(current_role)))
                if handshake_moved:
                    raise AssertionError(
                        f"dump role mismatch for neuron {old}: {old_role!r} != {current_role!r}"
                    )
            current_to_dump = {}
            for old, role in zip(uniq.tolist(), role_of.tolist()):
                if roles[int(old)] == role:
                    current_to_dump[int(old)] = int(old)
                elif len(role_to_current[str(role)]) == 1:
                    current_to_dump[role_to_current[str(role)][0]] = int(old)
            drift = int(mismatch.sum())
            observed = np.unique(neuron)
            if not np.array_equal(np.sort(uniq), observed):
                raise AssertionError("compact dump uniq does not match the neuron array")
            present = set(int(x) for x in uniq.tolist())
            full_capture, format_name = True, "compact/full"
        elif "role" in raw:
            dump_roles = np.asarray(raw["role"])
            if len(dump_roles) != len(neuron):
                raise ValueError("role-filtered dumps must have one role string per spike")
            chunk = 1_000_000
            for lo in range(0, len(neuron), chunk):
                hi = min(lo + chunk, len(neuron))
                got = dump_roles[lo:hi].astype(str)
                want = role_table[neuron[lo:hi]]
                if not np.array_equal(got, want):
                    bad = int(np.flatnonzero(got != want)[0]) + lo
                    raise AssertionError(
                        f"dump role mismatch at spike {bad}, neuron {int(neuron[bad])}: "
                        f"{str(dump_roles[bad])!r} != {roles[int(neuron[bad])]!r}"
                    )
            present = set(int(x) for x in np.unique(neuron).tolist())
            full_capture, format_name = False, "role-filtered"
            current_to_dump = {neuron_: neuron_ for neuron_ in present}
            drift = 0
        else:
            raise ValueError("dump needs either uniq/role_of or a per-spike role array")

    return SpikeDump(step=step, neuron=neuron, present_ids=present,
                     full_capture=full_capture, format_name=format_name,
                     current_to_dump=current_to_dump, role_id_drift=drift)


def rises(steps: np.ndarray, gap: int) -> list[int]:
    """Return the first spike of each train separated by more than ``gap`` steps."""
    steps = np.asarray(steps, dtype=np.int64)
    if not len(steps):
        return []
    return [int(x) for x in steps[np.r_[True, np.diff(steps) > gap]].tolist()]


def intervals(steps: np.ndarray, gap: int) -> list[tuple[int, int, int]]:
    """``(first, last, spike_count)`` for every latch train."""
    steps = np.asarray(steps, dtype=np.int64)
    if not len(steps):
        return []
    cuts = np.flatnonzero(np.diff(steps) > gap) + 1
    starts = np.r_[0, cuts]
    stops = np.r_[cuts, len(steps)]
    return [(int(steps[a]), int(steps[b - 1]), int(b - a)) for a, b in zip(starts, stops)]


def _last_before(values: list[int], limit: int) -> int | None:
    a = np.asarray(values, dtype=np.int64)
    k = int(np.searchsorted(a, limit, side="right")) - 1
    return int(a[k]) if k >= 0 else None


def _first_after(values: list[int], start: int, limit: int | None = None) -> int | None:
    a = np.asarray(values, dtype=np.int64)
    k = int(np.searchsorted(a, start, side="left"))
    if k == len(a) or (limit is not None and int(a[k]) >= limit):
        return None
    return int(a[k])


def _actd_id(roles: list[str], name: str) -> int | None:
    found = []
    pat = re.compile(re.escape(name) + r"\.actd\.d(\d+)$")
    for neuron, role in enumerate(roles):
        match = pat.fullmatch(role)
        if match:
            found.append((int(match.group(1)), neuron))
    return max(found)[1] if found else None


def _probe_layout(pl) -> tuple[dict[str, dict[str, Any]], np.ndarray]:
    roles = pl.net.roles
    role_id = {role: neuron for neuron, role in enumerate(roles)}
    layout: dict[str, dict[str, Any]] = {}
    wanted: set[int] = set()
    for cell in pl.cells:
        reqs = {}
        for src, pair in cell.reqs.items():
            prefix = f"{cell.name}.req.{src}"
            req = {
                "false_u": int(pair[0].u), "false_v": int(pair[0].v),
                "true_u": int(pair[1].u), "true_v": int(pair[1].v),
                "received": role_id.get(prefix + ".received"),
                "clear": role_id.get(prefix + ".k1.inh"),
                "relight": role_id.get(f"{cell.name}.relight.{src}.edge"),
            }
            reqs[src] = req
            wanted.update(x for x in req.values() if x is not None)
        probes = {
            "start": int(cell.start),
            "actd": _actd_id(roles, cell.name),
            "stage": int(cell.stage.completion.u),
            "commit": int(cell.commit_pulse),
            "done": int(cell.reg.done_relay),
            "idle_false": int(cell.idle[0].u),
            "idle_true": int(cell.idle[1].u),
            "creq": int(cell.creq[1].u),
            "fault": int(cell.stage.fault_latch.u),
        }
        wanted.update(x for x in probes.values() if x is not None)
        go = {role: neuron for neuron, role in enumerate(roles)
              if role.startswith(cell.name + ".go.") and
              (role.endswith((".edge", ".pulse", ".u", ".v", ".veto")) or ".d" in role)}
        wanted.update(go.values())
        layout[cell.name] = {"cell": cell, "probes": probes, "requests": reqs, "go": go}

    input_probes = {}
    for stream, (reg, _producer) in pl.inputs.items():
        prefix = "IN" if stream == "input" else f"IN.{stream}"
        input_probes[stream] = {
            "ready": int(reg.stage.ready),
            "done": role_id.get(prefix + ".done.edge"),
            "fault": int(reg.stage.fault_latch.u),
        }
        wanted.update(x for x in input_probes[stream].values() if x is not None)
    wanted.update(neuron for neuron, role in enumerate(roles)
                  if role.endswith("faultL.u") or role.endswith("wd.timeout.u"))
    layout["__inputs__"] = input_probes
    return layout, np.asarray(sorted(wanted), dtype=np.int64)


def _extract_steps(dump: SpikeDump, probe_ids: np.ndarray, n_neurons: int) -> dict[int, np.ndarray]:
    """One vectorized pass over all spikes, then one stable grouping of relevant spikes."""
    selected = np.zeros(n_neurons, dtype=bool)
    translated = {dump.current_to_dump[int(current)]: int(current) for current in probe_ids
                  if int(current) in dump.current_to_dump}
    selected[np.asarray(list(translated), dtype=np.int64)] = True
    mask = selected[dump.neuron]
    neurons = dump.neuron[mask]
    steps = dump.step[mask]
    if not len(neurons):
        return {}
    order = np.argsort(neurons, kind="stable")
    neurons, steps = neurons[order], steps[order]
    cuts = np.flatnonzero(np.diff(neurons)) + 1
    starts = np.r_[0, cuts]
    stops = np.r_[cuts, len(neurons)]
    return {translated[int(neurons[a])]: steps[a:b] for a, b in zip(starts, stops)}


def _observed(dump: SpikeDump, roles: list[str], neuron: int | None) -> bool:
    if neuron is None:
        return False
    return dump.full_capture or bool(REPLAY_ROLE_FILTER.search(roles[neuron]))


def _event_list(events: dict[int, np.ndarray], neuron: int | None, gap: int) -> list[int]:
    return rises(events.get(neuron, np.empty(0, dtype=np.int64)), gap) if neuron is not None else []


def _is_live(events: dict[int, np.ndarray], neuron: int, end: int, gap: int) -> bool:
    values = events.get(neuron)
    return bool(values is not None and len(values) and int(values[-1]) >= end - gap)


def _cell_events(info: dict[str, Any], events: dict[int, np.ndarray], gap: int) -> dict[str, list[int]]:
    return {name: _event_list(events, neuron, gap) for name, neuron in info["probes"].items()}


def _healthy_timeout(layout: dict[str, dict[str, Any]], events: dict[int, np.ndarray], gap: int) -> int:
    durations = []
    for name, info in layout.items():
        if name == "__inputs__":
            continue
        ev = _cell_events(info, events, gap)
        for start in ev["start"]:
            done = _first_after(ev["done"], start)
            if done is not None:
                durations.append(done - start)
    # Tick's healthy cell cycles are about 8--12k steps.  Twice the observed maximum and a
    # 2 s floor distinguish a long-lived blockage from an in-flight tail at capture end.
    return max(20_000, 2 * max(durations, default=10_000))


def _candidate_anomalies(pl, dump: SpikeDump, layout: dict[str, dict[str, Any]],
                         events: dict[int, np.ndarray], gap: int, timeout: int) -> list[dict[str, Any]]:
    roles, end = pl.net.roles, dump.end
    candidates: list[dict[str, Any]] = []
    for name, info in layout.items():
        if name == "__inputs__":
            continue
        ev = _cell_events(info, events, gap)
        starts = ev["start"]

        # A transaction which STARTed but did not cross a later phase.
        for index, start in enumerate(starts):
            stop = starts[index + 1] if index + 1 < len(starts) else None
            stage = _first_after(ev["stage"], start, stop)
            commit = _first_after(ev["commit"], stage if stage is not None else start, stop)
            done = _first_after(ev["done"], commit if commit is not None else start, stop)
            if end - start < timeout:
                continue
            if stage is None and _observed(dump, roles, info["probes"]["stage"]):
                candidates.append({"time": start, "kind": "datapath", "cell": name,
                                   "transaction": index + 1, "start": start})
            elif commit is None and _observed(dump, roles, info["probes"]["commit"]):
                candidates.append({"time": stage, "kind": "consumer_holds", "cell": name,
                                   "transaction": index + 1, "start": start, "stage": stage})
            elif done is None and _observed(dump, roles, info["probes"]["done"]):
                candidates.append({"time": commit, "kind": "completion", "cell": name,
                                   "transaction": index + 1, "start": start, "commit": commit})

        # All request-true rails and IDLE are live, but there is no later START.  A false rail
        # live at the same time identifies the veto; its most recent received/clear sequence
        # dates the request failure earlier than the visibly blocked go.
        req_ready: dict[str, int] = {}
        stuck = []
        usable = bool(info["requests"])
        for src, req in info["requests"].items():
            if not _observed(dump, roles, req["true_u"]):
                usable = False
                break
            true_steps = events.get(req["true_u"], np.empty(0, dtype=np.int64))
            if not _is_live(events, req["true_u"], end, gap):
                usable = False
                break
            req_ready[src] = intervals(true_steps, gap)[-1][0]
            if _is_live(events, req["false_u"], end, gap):
                stuck.append(src)
        idle_n = info["probes"]["idle_true"]
        if usable and _observed(dump, roles, idle_n) and _is_live(events, idle_n, end, gap):
            idle_since = intervals(events[idle_n], gap)[-1][0]
            ready = max([idle_since, *req_ready.values()])
            if _first_after(starts, ready) is None and end - ready >= timeout:
                clear_times = []
                for src in stuck:
                    received = _event_list(events, info["requests"][src]["received"], gap)
                    if received:
                        clear_times.append(received[-1])
                candidates.append({
                    "time": min(clear_times, default=ready),
                    "ready": ready,
                    "kind": "stuck_request" if stuck else "blocked_go",
                    "cell": name,
                    "transaction": len(starts) + 1,
                    "requests_ready": req_ready,
                    "idle_since": idle_since,
                    "stuck_sources": stuck,
                })

    # Producer/watchdog: a completed input word with no later READY.  This ranks after an
    # upstream request failure when backpressure is merely propagating to the host.
    for stream, probes in layout["__inputs__"].items():
        if not (_observed(dump, roles, probes["done"]) and _observed(dump, roles, probes["ready"])):
            continue
        dones = _event_list(events, probes["done"], gap)
        ready = _event_list(events, probes["ready"], gap)
        for index, done in enumerate(dones):
            if _first_after(ready, done) is None and end - done >= timeout:
                candidates.append({"time": done, "kind": "producer", "cell": f"IN.{stream}",
                                   "transaction": index + 1, "done": done})
    return sorted(candidates, key=lambda x: (x["time"], x["kind"], x["cell"]))


def _transaction_rows(info: dict[str, Any], events: dict[int, np.ndarray], gap: int) -> list[dict[str, Any]]:
    ev = _cell_events(info, events, gap)
    rows = []
    for index, start in enumerate(ev["start"]):
        stop = ev["start"][index + 1] if index + 1 < len(ev["start"]) else None
        requests = {}
        for src, req in info["requests"].items():
            requests[src] = _last_before(_event_list(events, req["true_u"], gap), start)
        actd = _first_after(ev["actd"], start, stop)
        stage = _first_after(ev["stage"], actd if actd is not None else start, stop)
        commit = _first_after(ev["commit"], stage if stage is not None else start, stop)
        done = _first_after(ev["done"], commit if commit is not None else start, stop)
        rows.append({"transaction": index + 1, "requests": requests, "start": start, "actd": actd,
                     "stage": stage, "commit": commit, "done": done})
    return rows


def _campaign_counts(campaign: dict[str, Any], node: int) -> dict[str, Any]:
    tokens = int(campaign.get("tokens", len(campaign.get("tokens_list", []))))
    outputs = list(campaign.get("output_cells", []))
    requested_outputs = tokens * len(outputs)
    per_node = campaign.get("per_node", [])
    ok = wrong = missing = None
    if 0 <= node < len(per_node):
        ok, wrong, missing = (int(x) for x in per_node[node])
    node_outputs = campaign.get("outputs", [])
    duplicates = 0
    completed = None if ok is None else ok + wrong
    if 0 <= node < len(node_outputs):
        counts = [len(node_outputs[node].get(cell, [])) for cell in outputs]
        duplicates = sum(max(0, count - tokens) for count in counts)
        completed = sum(counts)
    per_node_refusals = campaign.get("per_node_refusals")
    refusals = None
    if per_node_refusals is not None and 0 <= node < len(per_node_refusals):
        refusals = int(per_node_refusals[node])
    return {
        "requested_transactions": tokens,
        "requested_outputs": requested_outputs,
        "completed_outputs": completed,
        "wrong": wrong,
        "duplicates": duplicates,
        "campaign_refusals": refusals,  # the runner's count (TIMEOUT rises); None for older campaigns
        "unfinished": missing,
    }


def analyze_dump(dump_path: str | Path, campaign_path: str | Path, node: int) -> dict[str, Any]:
    """Return structured evidence used by the Markdown renderer and regression test."""
    campaign_path = Path(campaign_path)
    campaign = json.loads(campaign_path.read_text())
    copies = int(campaign.get("copies", node + 1))
    if not 0 <= node < copies:
        raise ValueError(f"node must be in [0, {copies})")
    params, ks, pl = build_tick_pipeline(campaign)
    dump = load_dump(dump_path, pl.net.roles)
    layout, probe_ids = _probe_layout(pl)
    events = _extract_steps(dump, probe_ids, pl.net.n)
    gap = 3 * pl.drive.loop_period_steps
    timeout = _healthy_timeout(layout, events, gap)
    candidates = _candidate_anomalies(pl, dump, layout, events, gap, timeout)
    anomaly = candidates[0] if candidates else {
        "time": dump.end, "kind": "unresolved", "cell": None, "transaction": None,
    }

    stuck_neuron = repair_neuron = repair_step = clear_step = None
    clear_pulses: list[int] = []
    false_intervals: list[tuple[int, int, int]] = []
    source = None
    if anomaly["kind"] == "stuck_request":
        source = anomaly["stuck_sources"][0]
        req = layout[anomaly["cell"]]["requests"][source]
        stuck_neuron = req["false_u"]
        received = _event_list(events, req["received"], gap)
        clear_step = received[-1] if received else None
        if req["clear"] is not None:
            raw_clear = events.get(req["clear"], np.empty(0, dtype=np.int64))
            if clear_step is not None:
                raw_clear = raw_clear[(raw_clear >= clear_step) & (raw_clear <= clear_step + 500)]
            clear_pulses = [int(x) for x in raw_clear.tolist()]
        repairs = _event_list(events, req["relight"], gap)
        prior_repairs = [x for x in repairs if clear_step is None or x < clear_step]
        if prior_repairs:
            repair_step = prior_repairs[-1]
            repair_neuron = req["relight"]
        false_intervals = intervals(events.get(stuck_neuron, np.empty(0, dtype=np.int64)), gap)

    cell_name = anomaly.get("cell")
    neighbours: list[str] = []
    timelines: dict[str, list[dict[str, Any]]] = {}
    if cell_name in layout and cell_name != "__inputs__":
        info = layout[cell_name]
        producers = [src for src in info["requests"] if src in layout]
        consumers = [name for name, other in layout.items()
                     if name != "__inputs__" and cell_name in other.get("requests", {})]
        order = [c.name for c in pl.cells]
        neighbours = sorted(set([*producers, cell_name, *consumers]), key=order.index)
        timelines = {name: _transaction_rows(layout[name], events, gap) for name in neighbours}

    refusals = 0
    refusal_roles = []
    for neuron, role in enumerate(pl.net.roles):
        if (role.endswith("faultL.u") or role.endswith("wd.timeout.u")) and neuron in events:
            count = len(rises(events[neuron], gap))
            if count:
                refusals += count
                refusal_roles.append((role, neuron, count))

    return {
        "dump": str(dump_path), "campaign": str(campaign_path), "node": node,
        "dump_format": dump.format_name, "spikes": len(dump.step), "distinct_neurons": len(dump.present_ids),
        "role_id_drift": dump.role_id_drift,
        "first_step": int(dump.step[0]) if len(dump.step) else None, "last_step": dump.end,
        "dt_ms": params.dt, "gap_steps": gap, "stall_threshold_steps": timeout,
        "kernel_neurons": pl.net.n, "kernel_synapses": pl.net.nnz,
        "output_cells": list(ks.outputs), "counts": _campaign_counts(campaign, node),
        "detected_refusals": refusals, "refusal_roles": refusal_roles,
        "classification": anomaly["kind"], "anomaly": anomaly, "candidates": candidates,
        "first_blocked_cell": cell_name if anomaly["kind"] in ("stuck_request", "blocked_go") else None,
        "stuck_source": source, "stuck_latch_neuron": stuck_neuron,
        "repair_neuron": repair_neuron, "repair_step": repair_step,
        "clear_step": clear_step, "clear_pulses": clear_pulses, "false_intervals": false_intervals,
        "neighbours": neighbours, "timelines": timelines,
        "roles": pl.net.roles, "layout": layout,
    }


def _n(value: int | None) -> str:
    return "—" if value is None else f"{value:,}"


def _ordinal(value: int) -> str:
    if 10 <= value % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
    return f"{value}{suffix}"


def _steps(values: list[int]) -> str:
    return "—" if not values else ", ".join(f"{x:,}" for x in values)


def _request_text(requests: dict[str, int | None]) -> str:
    return "; ".join(f"{src} {_n(step)}" for src, step in requests.items()) or "—"


def render_markdown(report: dict[str, Any]) -> str:
    """Render the evidence as a self-contained Markdown handoff."""
    anomaly = report["anomaly"]
    roles = report["roles"]
    lines = [
        f"# Tick stall diagnostic: copy {report['node']}",
        "",
        "## Classification",
        "",
    ]
    if report["classification"] == "stuck_request":
        cell, src = anomaly["cell"], report["stuck_source"]
        latch, repair = report["stuck_latch_neuron"], report["repair_neuron"]
        explanation = (
            f"The first blocked cell is **`{cell}`**, at its "
            f"**{_ordinal(anomaly['transaction'])} START**. "
            f"This is a **request / repair / clear** failure, not a completion failure: false "
            f"request latch **{latch}, `{roles[latch]}`** remains lit after the next `{src}` "
            "request's clear. "
        )
        if repair is not None:
            explanation += (
                f"The preceding repair pulse **{repair}, `{roles[repair]}`**, at "
                f"step **{report['repair_step']:,}** reached an already-live false rail. "
            )
        else:
            explanation += "No preceding repair pulse is present in the captured repair roles. "
        explanation += (
            "Both request "
            "rails are consequently live; false vetoes the go chain, so START, ACT^d, stage "
            "completion, commit, and DONE do not occur."
        )
        lines.append(explanation)
    elif report["classification"] == "blocked_go":
        lines.append(
            f"The first blocked cell is **`{anomaly['cell']}`**, transaction "
            f"**{anomaly['transaction']}**. All captured request-true rails and IDLE are live, "
            "but the go chain never produces START. No stuck false rail is visible in the "
            "captured request circuitry."
        )
    elif report["classification"] == "datapath":
        lines.append(
            f"**`{anomaly['cell']}`** STARTed transaction {anomaly['transaction']} at step "
            f"{anomaly['start']:,}, but its captured stage completion never rose: **datapath**."
        )
    elif report["classification"] == "consumer_holds":
        lines.append(
            f"**`{anomaly['cell']}`** completed its stage at step {anomaly['stage']:,}, but "
            "commit never fired: **consumer holds / commit gating**."
        )
    elif report["classification"] == "completion":
        lines.append(
            f"**`{anomaly['cell']}`** committed at step {anomaly['commit']:,}, but DONE never "
            "fired: **completion / clear**."
        )
    elif report["classification"] == "producer":
        lines.append(
            f"The input register completed at step {anomaly['done']:,}, but READY never rose: "
            "**watchdog / producer**, unless upstream backpressure is identified earlier."
        )
    else:
        lines.append(
            "The captured roles do not establish a request, datapath, consumer-hold, completion, "
            "or producer anomaly beyond the workload-derived stall threshold."
        )

    lines += [
        "",
        "## Artifact and threshold",
        "",
        f"- Dump: `{report['dump']}` ({report['dump_format']}; {report['spikes']:,} spikes, "
        f"{report['distinct_neurons']:,} firing neuron ids; steps {_n(report['first_step'])}–{_n(report['last_step'])}).",
        f"- Rebuilt kernel: {report['kernel_neurons']:,} neurons / {report['kernel_synapses']:,} synapses. "
        "Every recorded role string was asserted against the rebuilt netlist" +
        ("; the legacy compact artifact has " + f"{report['role_id_drift']:,} datapath role ids reordered "
         "by the later Track B build; needed moved ids are translated by unique role, while all handshake ids remain exact."
         if report["role_id_drift"] else " at the recorded neuron id."),
        f"- A latch train is continuous across gaps ≤ {report['gap_steps']:,} steps "
        f"(three nominal loop periods). A phase is called stalled only after "
        f"{report['stall_threshold_steps']:,} steps ({report['stall_threshold_steps'] * report['dt_ms'] / 1000:g} s): "
        "twice the longest completed START→DONE in this workload, with a 2 s floor.",
        "- The analysis is read-only. It does not attempt a single-copy replay; the campaign's "
        "device-drawn stray stream depends on the original batch size.",
    ]

    if report["classification"] == "stuck_request":
        cell, src = anomaly["cell"], report["stuck_source"]
        req = report["layout"][cell]["requests"][src]
        probes = report["layout"][cell]["probes"]
        lines += [
            "",
            "## Request / repair / clear evidence",
            "",
            "| Signal | Neuron | Observed step(s) |",
            "|---|---:|---:|",
            f"`{cell}.req.{src}` false u | {req['false_u']} | live through {_n(report['last_step'])} |",
            f"`{cell}.req.{src}` false v | {req['false_v']} | partner of the stuck latch |",
            f"`{cell}.req.{src}` true u | {req['true_u']} | {_n(anomaly['requests_ready'][src])} onward |",
            f"repair edge | {_n(req['relight'])} | **{_n(report['repair_step'])}** |",
            f"received / clear trigger | {req['received']} | {_n(report['clear_step'])} |",
            f"clear inhibitory train | {req['clear']} | {_steps(report['clear_pulses'])} |",
            f"START | {probes['start']} | no {_ordinal(anomaly['transaction'])} pulse |",
            f"ACT^d | {probes['actd']} | no pulse after the blockage |",
            f"stage completion | {probes['stage']} | no completion after the blockage |",
            f"commit | {probes['commit']} | no commit after the blockage |",
            f"DONE | {probes['done']} | no DONE after the blockage |",
            "",
            "False-u train intervals are `first–last (spikes)`: " +
            "; ".join(f"{a:,}–{b:,} ({count})" for a, b, count in report["false_intervals"][-3:]) + ".",
        ]

    if report["timelines"]:
        lines += [
            "",
            "## Blocked cell and neighbour timeline",
            "",
            "The rows retain the transaction before the failure, the failed transaction, and "
            "the immediately adjacent producer/consumer work visible in the dump.",
            "",
            "| Cell / transaction | request-true rises | START | ACT^d | stage | commit | DONE |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for cell in report["neighbours"]:
            for row in report["timelines"][cell][-3:]:
                lines.append(
                    f"| `{cell}` / {row['transaction']} | {_request_text(row['requests'])} | "
                    f"{_n(row['start'])} | {_n(row['actd'])} | {_n(row['stage'])} | "
                    f"{_n(row['commit'])} | {_n(row['done'])} |"
                )
            if cell == anomaly.get("cell") and anomaly["kind"] in ("stuck_request", "blocked_go"):
                lines.append(
                    f"| **`{cell}` / {anomaly['transaction']} (blocked)** | "
                    f"{_request_text(anomaly['requests_ready'])} | **—** | — | — | — | — |"
                )

    counts = report["counts"]
    lines += [
        "",
        "## Campaign accounting for this copy",
        "",
        "| requested transactions | requested outputs | completed outputs | wrong | duplicates | detected refusals | unfinished |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {counts['requested_transactions']} | {counts['requested_outputs']} | "
        f"{_n(counts['completed_outputs'])} | {_n(counts['wrong'])} | {counts['duplicates']} | "
        f"{report['detected_refusals']}"
        + ("" if counts.get("campaign_refusals") is None else f" (runner: {counts['campaign_refusals']})")
        + f" | {_n(counts['unfinished'])} |",
        "",
        "Detected refusals are counted from the dump's FAULT and `wd.timeout` latches (capture "
        "`wd\\.timeout` in `--dump-roles` to see them) and, in parentheses, from the runner's "
        "own count in the campaign record. A refused input word that the host does not resend "
        "shifts every later output by one and scores as wrong values (copy 77, seed 108); the "
        "runner resends it since 2026-09-20.",
        "",
        "A capture or campaign stopped by its neural-time or wall-time resource limit is "
        "**truncated**, not automatically stalled. The stall label above rests on a specific "
        "handshake phase remaining blocked past the stated workload threshold; unfinished "
        "counts alone are not that evidence.",
        "",
    ]
    return "\n".join(lines)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump", help="one copy's .npz spike dump")
    ap.add_argument("--campaign", required=True, help="kernel_campaign JSON for topology/options/accounting")
    ap.add_argument("--node", type=int, required=True, help="copy index in the campaign")
    ap.add_argument("--out", default=None, help="write Markdown here (stdout when omitted)")
    return ap


def main(argv=None) -> dict[str, Any]:
    args = parser().parse_args(argv)
    report = analyze_dump(args.dump, args.campaign, args.node)
    markdown = render_markdown(report)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown)
    else:
        print(markdown)
    return report


if __name__ == "__main__":
    main()
