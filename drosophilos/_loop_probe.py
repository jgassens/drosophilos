"""Probe the steady period of an isolated two-neuron latch loop (used to set rate-mode targets)."""

from __future__ import annotations

import numpy as np

from .sim.model import Params, Topology
from .sim.ref64 import RefSim


def probe_loop(params: Params, loop_quanta: int, n_steps: int = 3000) -> tuple[int, float]:
    d = params.default_delay_steps
    topo = Topology.from_edges(2, [0, 1], [1, 0], [loop_quanta, loop_quanta], [d, d])
    sim = RefSim(topo, params)
    sim.add_events(0, [50], [0], [loop_quanta])
    sim.run(n_steps)
    st = sim.trace.neuron_steps(0)
    if len(st) < 3:
        raise RuntimeError("loop did not self-sustain at this drive")
    isi = np.diff(st)
    return int(isi[-1]), float((st[0] - 50) * params.dt)
