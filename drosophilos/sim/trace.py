"""Spike-event traces: the spike-level comparison object (schedule.md §7)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_DTYPE = np.dtype([("step", np.int64), ("node", np.int64), ("neuron", np.int64)])


@dataclass
class SpikeTrace:
    events: np.ndarray  # structured, sorted by (step, node, neuron)

    @classmethod
    def from_arrays(cls, step, node, neuron) -> "SpikeTrace":
        ev = np.empty(len(step), dtype=_DTYPE)
        ev["step"] = step
        ev["node"] = node
        ev["neuron"] = neuron
        ev.sort(order=["step", "node", "neuron"])
        return cls(ev)

    @classmethod
    def empty(cls) -> "SpikeTrace":
        return cls(np.empty(0, dtype=_DTYPE))

    def __len__(self) -> int:
        return int(len(self.events))

    def __eq__(self, other) -> bool:
        if not isinstance(other, SpikeTrace):
            return NotImplemented
        return len(self) == len(other) and bool(np.array_equal(self.events, other.events))

    def node(self, b: int) -> "SpikeTrace":
        """Events of one node, renumbered to node 0 (for batch-independence checks)."""
        sel = self.events[self.events["node"] == b].copy()
        sel["node"] = 0
        return SpikeTrace(sel)

    def neuron_steps(self, neuron: int, node: int = 0) -> np.ndarray:
        m = (self.events["neuron"] == neuron) & (self.events["node"] == node)
        return self.events["step"][m]

    def after(self, step: int) -> "SpikeTrace":
        return SpikeTrace(self.events[self.events["step"] >= step].copy())

    def before(self, step: int) -> "SpikeTrace":
        return SpikeTrace(self.events[self.events["step"] < step].copy())

    def concat(self, other: "SpikeTrace") -> "SpikeTrace":
        ev = np.concatenate([self.events, other.events])
        ev.sort(order=["step", "node", "neuron"])
        return SpikeTrace(ev)

    def diff(self, other: "SpikeTrace") -> tuple[np.ndarray, np.ndarray]:
        """(events only in self, events only in other)."""
        a = set(map(tuple, self.events.tolist()))
        b = set(map(tuple, other.events.tolist()))
        only_a = np.array(sorted(a - b), dtype=np.int64).reshape(-1, 3)
        only_b = np.array(sorted(b - a), dtype=np.int64).reshape(-1, 3)
        return only_a, only_b

    def counts(self, n_nodes: int, n: int) -> np.ndarray:
        out = np.zeros((n_nodes, n), dtype=np.int64)
        np.add.at(out, (self.events["node"], self.events["neuron"]), 1)
        return out
