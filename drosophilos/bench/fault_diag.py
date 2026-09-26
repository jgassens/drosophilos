"""Read-only fault-gate diagnostic for a role-filtered kernel spike dump.

The campaign record is used to rebuild the exact tick pipeline.  Every captured neuron is
checked against that netlist by :func:`stall_diag.load_dump`; the analysis then reports each
stage fault-gate rise, the surrounding cell transaction, and the stage valid latches at the
fault step.  Spike selection and grouping are vectorized -- no Python loop visits individual
spikes.

Example::

    python -m drosophilos.bench.fault_diag data/a2/tick_s109_node73_spec.npz \
      --campaign data/a2/kc_tick_B90s109_n73_spec.json --node 73
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from .stall_diag import REPLAY_ROLE_FILTER, build_tick_pipeline, load_dump, rises


def _actd_id(roles: list[str], name: str) -> int | None:
    matches = []
    pattern = re.compile(re.escape(name) + r"\.actd\.d(\d+)$")
    for neuron, role in enumerate(roles):
        match = pattern.fullmatch(role)
        if match:
            matches.append((int(match.group(1)), neuron))
    return max(matches)[1] if matches else None


def _group_probe_steps(dump, probe_ids: np.ndarray, n_neurons: int) -> dict[int, np.ndarray]:
    """Select and group probe spikes without iterating over the dump's individual spikes."""
    translated = {
        dump.current_to_dump[int(current)]: int(current)
        for current in probe_ids
        if int(current) in dump.current_to_dump
    }
    if not translated:
        return {}
    selected = np.zeros(n_neurons, dtype=bool)
    selected[np.fromiter(translated, dtype=np.int64)] = True
    mask = selected[dump.neuron]
    neurons, steps = dump.neuron[mask], dump.step[mask]
    if not len(neurons):
        return {}
    order = np.argsort(neurons, kind="stable")
    neurons, steps = neurons[order], steps[order]
    cuts = np.flatnonzero(np.diff(neurons)) + 1
    starts, stops = np.r_[0, cuts], np.r_[cuts, len(neurons)]
    return {
        translated[int(neurons[start])]: steps[start:stop]
        for start, stop in zip(starts, stops)
    }


def _event_rises(events: dict[int, np.ndarray], neuron: int | None, gap: int) -> list[int]:
    if neuron is None:
        return []
    return rises(events.get(neuron, np.empty(0, dtype=np.int64)), gap)


def _first_between(values: list[int], start: int, stop: int | None) -> int | None:
    array = np.asarray(values, dtype=np.int64)
    index = int(np.searchsorted(array, start, side="left"))
    if index == len(array) or (stop is not None and int(array[index]) >= stop):
        return None
    return int(array[index])


def _observed(dump, roles: list[str], neuron: int) -> bool:
    """Whether silence is meaningful for a standard role-filtered replay."""
    return (
        dump.full_capture
        or neuron in dump.current_to_dump
        or bool(REPLAY_ROLE_FILTER.search(roles[neuron]))
    )


def _valid_state(
    dump,
    roles: list[str],
    events: dict[int, np.ndarray],
    valid,
    bit: int,
    step: int,
    start: int | None,
    gap: int,
) -> dict[str, Any]:
    neuron = int(valid.u)
    spikes = events.get(neuron, np.empty(0, dtype=np.int64))
    index = int(np.searchsorted(spikes, step, side="right"))
    before = int(spikes[index - 1]) if index else None
    after = int(spikes[index]) if index < len(spikes) else None
    valid_rises = rises(spikes, gap)
    rise_index = int(np.searchsorted(valid_rises, step, side="right")) - 1
    rise = int(valid_rises[rise_index]) if rise_index >= 0 else None
    rose_in_transaction = rise is not None and (start is None or rise >= start)
    available = _observed(dump, roles, neuron)
    active = bool(available and before is not None and step - before <= gap)
    return {
        "bit": bit,
        "neuron": neuron,
        "role": roles[neuron],
        "state": "active" if active else "inactive" if available else "unavailable",
        "rose_in_transaction": rose_in_transaction,
        "rise_step": rise if rose_in_transaction else None,
        "last_spike": before,
        "next_spike": after,
    }


def _subjects(pl) -> list[dict[str, Any]]:
    out = []
    for cell in pl.cells:
        out.append(
            {
                "name": cell.name,
                "kind": "cell",
                "cell": cell,
                "stage": cell.stage,
                "start": int(cell.start),
                "actd": _actd_id(pl.net.roles, cell.name),
                "completion": int(cell.stage.completion.u),
                "commit": int(cell.commit_pulse),
                "done": int(cell.reg.done_relay),
            }
        )
    for stream, (reg, _producer) in pl.inputs.items():
        prefix = "IN" if stream == "input" else f"IN.{stream}"
        out.append(
            {
                "name": prefix,
                "kind": "input",
                "cell": None,
                "stage": reg.stage,
                "start": None,
                "actd": None,
                "completion": int(reg.stage.completion.u),
                "commit": int(reg.commit_in),
                "done": int(reg.done_relay),
            }
        )
    return out


def analyze_faults(dump_path: str | Path, campaign_path: str | Path, node: int) -> dict[str, Any]:
    """Rebuild the campaign pipeline and return all captured stage fault-gate rises."""
    campaign_path = Path(campaign_path)
    campaign = json.loads(campaign_path.read_text())
    copies = int(campaign.get("copies", node + 1))
    if not 0 <= node < copies:
        raise ValueError(f"node must be in [0, {copies})")

    _params, _kernel, pipeline = build_tick_pipeline(campaign)
    roles = pipeline.net.roles
    dump = load_dump(dump_path, roles)
    subjects = _subjects(pipeline)

    probes: set[int] = set()
    for subject in subjects:
        stage = subject["stage"]
        probes.update(int(neuron) for neuron in stage.fault)
        probes.update(int(valid.u) for valid in stage.valid)
        probes.add(int(stage.fault_latch.u))
        probes.update(
            int(neuron)
            for key in ("start", "actd", "completion", "commit", "done")
            if (neuron := subject[key]) is not None
        )
    probe_ids = np.asarray(sorted(probes), dtype=np.int64)
    events = _group_probe_steps(dump, probe_ids, pipeline.net.n)
    gap = 3 * pipeline.drive.loop_period_steps

    faults = []
    for subject in subjects:
        starts = _event_rises(events, subject["start"], gap)
        phase_rises = {
            key: _event_rises(events, subject[key], gap)
            for key in ("actd", "completion", "commit", "done")
        }
        fault_latch_rises = _event_rises(events, subject["stage"].fault_latch.u, gap)
        for bit, gate in enumerate(subject["stage"].fault):
            gate = int(gate)
            for fault_step in _event_rises(events, gate, gap):
                transaction_index = int(np.searchsorted(starts, fault_step, side="right")) - 1
                if transaction_index >= 0:
                    transaction = transaction_index + 1
                    start = starts[transaction_index]
                    stop = starts[transaction_index + 1] if transaction_index + 1 < len(starts) else None
                else:
                    transaction, start, stop = None, None, starts[0] if starts else None
                phase_start = start if start is not None else 0
                transaction_row = {
                    "number": transaction,
                    "start": start,
                    **{
                        key: _first_between(values, phase_start, stop)
                        for key, values in phase_rises.items()
                    },
                }
                latch_index = int(np.searchsorted(fault_latch_rises, fault_step, side="left"))
                fault_latch = (
                    fault_latch_rises[latch_index]
                    if latch_index < len(fault_latch_rises)
                    and (stop is None or fault_latch_rises[latch_index] < stop)
                    else None
                )
                valid = [
                    _valid_state(dump, roles, events, latch, valid_bit, fault_step, start, gap)
                    for valid_bit, latch in enumerate(subject["stage"].valid)
                ]
                cell = subject["cell"]
                faults.append(
                    {
                        "step": fault_step,
                        "cell": subject["name"],
                        "subject_kind": subject["kind"],
                        "operation": getattr(cell, "op", None),
                        "datapath": getattr(cell, "datapath", None),
                        "bit": bit,
                        "gate_neuron": gate,
                        "gate_role": roles[gate],
                        "fault_latch_step": fault_latch,
                        "transaction": transaction_row,
                        "valid_latches": valid,
                        "valid_bits_active": [item["bit"] for item in valid if item["state"] == "active"],
                        "valid_bits_rose": [item["bit"] for item in valid if item["rose_in_transaction"]],
                    }
                )

    mapping = [
        {"neuron": int(dump_neuron), "role": roles[int(current)]}
        for current, dump_neuron in sorted(dump.current_to_dump.items(), key=lambda item: item[1])
    ]
    return {
        "dump": str(dump_path),
        "campaign": str(campaign_path),
        "node": node,
        "dump_format": dump.format_name,
        "dump_end": dump.end,
        "captured_neurons": len(mapping),
        "neuron_roles": mapping,
        "role_id_drift": dump.role_id_drift,
        "netlist": {"neurons": pipeline.net.n, "synapses": pipeline.net.nnz},
        "build_options": dict(pipeline.build_options),
        "gap_steps": gap,
        "faults": sorted(faults, key=lambda item: (item["step"], item["gate_neuron"])),
    }


def render_text(report: dict[str, Any]) -> str:
    lines = [
        f"node {report['node']}: {len(report['faults'])} stage fault-gate rise(s); "
        f"{report['captured_neurons']} captured neuron ids mapped",
    ]
    for fault in report["faults"]:
        transaction = fault["transaction"]
        active = ",".join(str(bit) for bit in fault["valid_bits_active"]) or "none"
        rose = ",".join(str(bit) for bit in fault["valid_bits_rose"]) or "none"
        lines.append(
            f"step {fault['step']}: {fault['gate_role']} (neuron {fault['gate_neuron']}, "
            f"bit {fault['bit']}), {fault['cell']} transaction {transaction['number']} "
            f"START={transaction['start']} ACT^d={transaction['actd']} "
            f"completion={transaction['completion']} commit={transaction['commit']} "
            f"DONE={transaction['done']}; valid active=[{active}] rose=[{rose}]; "
            f"fault latch={fault['fault_latch_step']}"
        )
    return "\n".join(lines)


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("dump", help="one copy's role-filtered .npz spike dump")
    argument_parser.add_argument("--campaign", required=True, help="campaign JSON with build options")
    argument_parser.add_argument("--node", type=int, required=True, help="copy index in the campaign")
    argument_parser.add_argument("--json", action="store_true", help="emit the complete structured report")
    return argument_parser


def main(argv=None) -> dict[str, Any]:
    args = parser().parse_args(argv)
    report = analyze_faults(args.dump, args.campaign, args.node)
    print(json.dumps(report, indent=2) if args.json else render_text(report))
    return report


if __name__ == "__main__":
    main()
