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


def recent_active(sim, step: int, window: int, node: int | None = None) -> set:
    """Neurons that spiked in (step - window, step], read from the simulator's per-step spike
    lists (of one node of a batched simulator when `node` is given). For polling a live
    simulator: `sim.trace` rebuilds and sorts the whole event array on every access, which
    made a runner quadratic in the run length (measured: a 40 s run of 13k neurons took 50+
    minutes in the sort and 70 s without it). An observer (`sim/observe.py`) is read through
    its `active_in`, which refuses a window that is not observed yet or already trimmed."""
    if hasattr(sim, "active_in"):
        return sim.active_in(step, window, node)
    active = set()
    nodes = getattr(sim, "_spk_node", None) if node is not None else None
    for k in range(len(sim._spk_step) - 1, -1, -1):
        st, nr = sim._spk_step[k], sim._spk_neuron[k]
        s_ = int(st[0])
        if s_ <= step - window:
            break
        if s_ <= step:
            if nodes is not None:
                nr = nr[nodes[k] == node]
            active.update(nr.tolist())
    return active


def decode_recent(sim, rail_neurons: list[list[int]], step: int, window: int, node: int | None = None):
    """`decode_at` on a live simulator without building the trace (same result)."""
    active = recent_active(sim, step, window, node)
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


def recent_spikes(sim, neuron: int, since_step: int) -> list:
    """Steps in (since_step, now] at which `neuron` spiked, from the per-step spike lists
    (node 0 of a batched simulator)."""
    out = []
    for st, nr in zip(reversed(sim._spk_step), reversed(sim._spk_neuron)):
        s_ = int(st[0])
        if s_ <= since_step:
            break
        if neuron in nr:
            out.append(s_)
    return out[::-1]
