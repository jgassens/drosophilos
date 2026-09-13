"""Drive a channel through repeated transactions and decode each one.

The host's only actions are transduction: inject the next word into the producer when the
consumer has signalled READY (or at t=0), and read spikes. Everything else is neural.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    status: str
    fault_spikes: int


def run_transactions(ch: Channel, params: Params, words: list[int], *, max_steps_per_tx: int = 6000,
                     gap_steps: int = 0, sim=None) -> tuple[list[TxRecord], RefSim, dict]:
    net, drive = ch.net, ch.drive
    topo = net.topology()
    sim = sim or RefSim(topo, params)
    P, Q = ch.producer, ch.consumer
    accept_n, cleared_n, ready_n = Q.completion.u, P.ready, Q.ready
    rail_taps = Q.rail_taps
    window = 2 * drive.loop_period_steps
    records: list[TxRecord] = []
    step = sim.step_index
    for w in words:
        # load the producer: one pulse per active rail
        load_step = max(step, sim.step_index)
        for i, r in rails_for(w, ch.width):
            sim.add_events(0, [load_step], [P.rails[i][r].u], [drive.ignite])
        accept_step = cleared_step = ready_step = None
        decoded, status = None, "incomplete"
        deadline = load_step + max_steps_per_tx
        while sim.step_index < deadline:
            sim.step()
            s = sim.step_index - 1
            if not sim._spk_step:
                continue
            last_neurons = sim._spk_neuron[-1] if sim._spk_step[-1][0] == s else None
            if last_neurons is None:
                continue
            fired = set(last_neurons.tolist())
            if accept_step is None and accept_n in fired:
                accept_step = s
                decoded, status = decode_at(sim.trace, rail_taps, s, window)
            if accept_step is not None and cleared_step is None and cleared_n in fired:
                cleared_step = s
            if cleared_step is not None and ready_n in fired:
                ready_step = s
                break
        tr = sim.trace
        fault_spikes = int(sum(len(tr.neuron_steps(f)) for f in Q.fault))
        records.append(TxRecord(w, load_step, accept_step, cleared_step, ready_step, decoded, status, fault_spikes))
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
