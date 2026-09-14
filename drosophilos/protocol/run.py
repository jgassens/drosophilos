"""Drive a channel through repeated transactions, with asynchronous operand arrival and
runtime fault injection, and decode each transaction.

The host's only actions are transduction: inject the next word into the producer when the
consumer has signalled READY (or at t=0), inject the requested fault spikes, and read
spikes. Everything else is neural.

Fault specs (per transaction, list of tuples):
    ("late_opposite", bit, delay_steps)   a pulse on the consumer's opposite rail of `bit`,
                                          `delay_steps` after ACCEPT
    ("duplicate", bit, delay_steps)       an extra pulse on the consumer's same rail,
                                          `delay_steps` after load
    ("corrupt", bit)                      both rails of `bit` ignited at load
    ("stale", bit, rail, delay_steps)     a pulse on consumer rail (bit, rail) `delay_steps`
                                          after the consumer's reset trigger fires
    ("stale_internal", role, delay_steps) an ignition pulse into the consumer neuron with
                                          that role `delay_steps` after the reset trigger
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..sim.model import Params
from ..sim.ref64 import RefSim
from .handshake import Channel
from .token import decode_at, rails_for


@dataclass
class TxRecord:
    word: int
    load_step: int
    accept_step: int | None
    cleared_step: int | None
    ready_step: int | None
    decoded: int | None
    status: str  # valid | fault | incomplete
    fault_spikes: int
    reset_step: int | None = None
    injected: list = field(default_factory=list)
    timeout_spikes: int = 0
    stale_retry_spikes: int = 0


def run_transactions(ch: Channel, params: Params, words: list[int], *, max_steps_per_tx: int = 8000,
                     gap_steps: int = 0, bit_offsets: list[list[int]] | None = None,
                     faults: dict[int, list[tuple]] | None = None, sim=None,
                     expected: list[int] | None = None) -> tuple[list[TxRecord], RefSim, dict]:
    """`expected[k]` is the value the consumer should decode for word k (defaults to the word
    itself; an adder channel expects the sum)."""
    net, drive = ch.net, ch.drive
    sim = sim or RefSim(net.topology(), params)
    P, Q = ch.producer, ch.consumer
    accept_n, cleared_n, ready_n, qreset_n = Q.completion.u, P.ready, Q.ready, Q.reset_trigger
    fault_set = set(Q.fault)
    if P.watchdog is not None:
        fault_set.add(P.watchdog.timeout.u)  # a neural TIMEOUT is a FAULT-ACCEPT source too
    rail_taps = Q.rail_taps
    window = 2 * drive.loop_period_steps
    faults = faults or {}
    records: list[TxRecord] = []
    step = sim.step_index
    for k, w in enumerate(words):
        load_step = max(step, sim.step_index)
        offsets = bit_offsets[k] if bit_offsets else [0] * ch.width
        for i, r in rails_for(w, ch.width):
            sim.add_events(0, [load_step + offsets[i]], [P.rails[i][r].u], [drive.ignite])
        specs = faults.get(k, [])
        injected = []
        for spec in specs:
            if spec[0] == "corrupt":
                i = spec[1]
                r = 1 - ((w >> i) & 1)
                t = load_step + offsets[i] + 100  # 10 ms after the real bit
                sim.add_events(0, [t], [Q.rails[i][r].u], [drive.ignite])
                injected.append(("corrupt", i, t))
            elif spec[0] == "duplicate":
                i, d = spec[1], spec[2]
                r = (w >> i) & 1
                t = load_step + offsets[i] + d
                sim.add_events(0, [t], [Q.rails[i][r].u], [drive.ignite])
                injected.append(("duplicate", i, t))
        accept_step = cleared_step = ready_step = reset_step = None
        first_fault = None
        decoded, status = None, "incomplete"
        pending_late = [s for s in specs if s[0] == "late_opposite"]
        pending_stale = [s for s in specs if s[0] in ("stale", "stale_internal")]
        deadline = load_step + max(offsets) + max_steps_per_tx
        while sim.step_index < deadline:
            sim.step()
            s = sim.step_index - 1
            if not sim._spk_step or sim._spk_step[-1][0] != s:
                continue
            fired = set(sim._spk_neuron[-1].tolist())
            if first_fault is None and fired & fault_set:
                first_fault = s
            if accept_step is None and accept_n in fired:
                accept_step = s
                decoded, status = decode_at(sim.trace, rail_taps, s, window)
                for spec in pending_late:
                    i, d = spec[1], spec[2]
                    r = 1 - ((w >> i) & 1)
                    sim.add_events(0, [s + d], [Q.rails[i][r].u], [drive.ignite])
                    injected.append(("late_opposite", i, s + d))
                pending_late = []
            if accept_step is None and first_fault is not None and cleared_n in fired:
                # FAULT-ACCEPT: the consumer never completed; the producer was cleared by the fault path
                status = "fault"
            if (accept_step is not None or first_fault is not None) and cleared_step is None and cleared_n in fired:
                cleared_step = s
            if reset_step is None and qreset_n in fired:
                reset_step = s
                for spec in pending_stale:
                    if spec[0] == "stale":
                        i, r, d = spec[1], spec[2], spec[3]
                        sim.add_events(0, [s + d], [Q.rails[i][r].u], [drive.ignite])
                        injected.append(("stale", i, r, s + d))
                    else:
                        role, d = spec[1], spec[2]
                        sim.add_events(0, [s + d], [net.roles.index(role)], [drive.ignite])
                        injected.append(("stale_internal", role, s + d))
                pending_stale = []
            if cleared_step is not None and ready_n in fired:
                ready_step = s
                break
        tr = sim.trace
        fault_spikes = int(sum(((tr.events["neuron"] == f) & (tr.events["step"] >= load_step)).sum() for f in Q.fault))
        if status == "valid" and fault_spikes and accept_step is not None:
            # a fault after acceptance is flagged, the value stands (spec.md: flag only)
            pass
        exp = expected[k] if expected is not None else w
        rec = TxRecord(exp, load_step, accept_step, cleared_step, ready_step, decoded, status,
                       fault_spikes, reset_step, injected)
        end = ready_step if ready_step is not None else sim.step_index
        if P.watchdog is not None:
            tu = P.watchdog.timeout.u
            rec.timeout_spikes = int(((tr.events["neuron"] == tu) & (tr.events["step"] >= load_step) & (tr.events["step"] <= end)).sum())
        for reg in (P, Q):
            if reg.monitor is not None:
                su = reg.monitor.stale.u
                rec.stale_retry_spikes += int(((tr.events["neuron"] == su) & (tr.events["step"] >= load_step) & (tr.events["step"] <= end)).sum())
        if rec.timeout_spikes and rec.status != "valid":
            rec.status = "timeout"
        records.append(rec)
        step = (ready_step if ready_step is not None else sim.step_index) + gap_steps
        if ready_step is None:
            break
    tr = sim.trace
    per_tx = np.array([r.ready_step - r.load_step for r in records if r.ready_step is not None], dtype=float)
    stats = {
        "transactions": len(records),
        "correct": sum(1 for r in records if r.decoded == r.word and r.status == "valid"),
        "completed": sum(1 for r in records if r.ready_step is not None),
        "accept_latency_ms": [((r.accept_step - r.load_step) * params.dt) if r.accept_step is not None else None for r in records],
        "cycle_ms": (per_tx * params.dt).tolist(),
        "total_spikes": len(tr),
        "spikes_per_tx": len(tr) / max(1, len(records)),
        "neurons": net.n,
        "synapses": net.nnz,
    }
    return records, sim, stats
