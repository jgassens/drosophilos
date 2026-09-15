"""Dual-rail encoding and spike-level decoding of words."""

from __future__ import annotations

import numpy as np

from ..sim.trace import SpikeTrace


def rails_for(word: int, width: int) -> list[tuple[int, int]]:
    """(bit index, rail) pairs that carry a `width`-bit unsigned word."""
    return [(i, (word >> i) & 1) for i in range(width)]


def decode_at(trace: SpikeTrace, rail_neurons: list[list[int]], step: int, window: int, node: int = 0):
    """Read a word from latch activity in [step-window, step].

    rail_neurons[i][r] is the tap neuron of bit i, rail r. Returns (value, status) where
    status is "valid", "fault" (both rails active on some bit), or "incomplete".
    """
    ev = trace.events
    m = (ev["node"] == node) & (ev["step"] > step - window) & (ev["step"] <= step)
    active = set(ev["neuron"][m].tolist())
    value = 0
    status = "valid"
    for i, (r0, r1) in enumerate(rail_neurons):
        a0, a1 = r0 in active, r1 in active
        if a0 and a1:
            return None, "fault"
        if not (a0 or a1):
            status = "incomplete"
        elif a1:
            value |= 1 << i
    return (value if status == "valid" else None), status


def recent_active(sim, step: int, window: int) -> set:
    """Neurons that spiked in (step - window, step], read from the simulator's per-step spike
    lists. For polling a live simulator: `sim.trace` rebuilds and sorts the whole event array
    on every access, which made a runner quadratic in the run length (measured: a 40 s run of
    13k neurons took 50+ minutes in the sort and 70 s without it)."""
    active = set()
    for st, nr in zip(reversed(sim._spk_step), reversed(sim._spk_neuron)):
        s_ = int(st[0])
        if s_ <= step - window:
            break
        if s_ <= step:
            active.update(nr.tolist())
    return active


def decode_recent(sim, rail_neurons: list[list[int]], step: int, window: int):
    """`decode_at` on a live simulator without building the trace (same result)."""
    active = recent_active(sim, step, window)
    value, status = 0, "valid"
    for i, (r0, r1) in enumerate(rail_neurons):
        a0, a1 = r0 in active, r1 in active
        if a0 and a1:
            status = "fault"
        elif not (a0 or a1):
            status = "incomplete"
        elif a1:
            value |= 1 << i
    return value, status
