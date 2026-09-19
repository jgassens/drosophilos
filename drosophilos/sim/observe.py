"""Filtered, buffered spike observation shared by simulator backends.

Simulation consumes the full dense spike tensor.  Observation is deliberately separate:
only selected neuron columns are copied to a fixed-shape device buffer and transferred to
the host in blocks.  The three ``_spk_*`` lists are compatibility views for the live token
decoders; new code should prefer :meth:`fired_at` and :meth:`active_in`.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch

from .trace import SpikeTrace


class Observer:
    """Retain spikes for a declared set of neurons.

    ``observe_every`` is both the transfer interval and the device-buffer capacity.  A
    simulator must call :meth:`write_dense` exactly once per observed step.  ``full_trace``
    is an explicit promise that ``watch_ids`` covers every neuron; requesting a full trace
    from a filtered observer raises instead of silently returning an incomplete trace.
    """

    def __init__(
        self,
        watch_ids: Iterable[int],
        n_nodes: int,
        *,
        n_neurons: int,
        device: str | torch.device = "cpu",
        observe_every: int = 1,
        full_trace: bool = False,
        max_watch_ids: int | None = None,
    ):
        ids = np.asarray(sorted(set(int(i) for i in watch_ids)), dtype=np.int64)
        if len(ids) and (ids[0] < 0 or ids[-1] >= int(n_neurons)):
            raise ValueError("observer neuron id out of range")
        if max_watch_ids is not None and len(ids) > int(max_watch_ids):
            raise OverflowError(
                f"observer needs {len(ids)} watched neurons but capacity is {max_watch_ids}"
            )
        if full_trace and not np.array_equal(ids, np.arange(int(n_neurons), dtype=np.int64)):
            raise OverflowError(
                "full_trace requires every neuron in watch_ids; refusing an incomplete trace"
            )
        if int(observe_every) < 1:
            raise ValueError("observe_every must be at least one")

        self.watch_ids = ids
        self.B = int(n_nodes)
        self.n = int(n_neurons)
        self.device = torch.device(device)
        self.observe_every = int(observe_every)
        self.full_trace = bool(full_trace)
        self._watch_device = torch.as_tensor(ids, device=self.device, dtype=torch.int64)
        self._device_ring = torch.empty(
            (self.observe_every, self.B, len(ids)), device=self.device, dtype=torch.bool
        )
        self._pending_steps: list[int] = []
        self._events_by_step: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._spk_step: list[np.ndarray] = []
        self._spk_node: list[np.ndarray] = []
        self._spk_neuron: list[np.ndarray] = []
        self._legacy_seen = 0
        self.available_through = -1

    @property
    def pending(self) -> int:
        return len(self._pending_steps)

    def write_dense(self, step: int, spikes: torch.Tensor, *, slot: int | None = None) -> None:
        """Gather a dense ``(B, n)`` spike mask into the fixed observation buffer."""
        if spikes.shape != (self.B, self.n) or spikes.dtype != torch.bool:
            raise ValueError(f"spikes must be bool shape ({self.B}, {self.n})")
        index = len(self._pending_steps) if slot is None else int(slot)
        if index < 0 or index >= self.observe_every:
            raise OverflowError(
                f"observation block exceeded its {self.observe_every}-step device buffer"
            )
        if index != len(self._pending_steps):
            raise RuntimeError("observation slots must be written consecutively")
        if len(self.watch_ids):
            torch.index_select(spikes, 1, self._watch_device, out=self._device_ring[index])
        self._pending_steps.append(int(step))
        if len(self._pending_steps) == self.observe_every:
            self.flush()

    def flush(self) -> None:
        """Copy the pending fixed-shape block once and unpack it on the host."""
        count = len(self._pending_steps)
        if not count:
            return
        # One static-shape device-to-host transfer per observation block.  Slicing is only
        # used for the final partial block (for example at the end of ``run``).
        host = self._device_ring[:count].detach().to(device="cpu").numpy()
        for row, step in zip(host, self._pending_steps):
            nodes, columns = np.nonzero(row)
            neurons = self.watch_ids[columns]
            nodes = nodes.astype(np.int64, copy=False)
            neurons = neurons.astype(np.int64, copy=False)
            self._events_by_step[step] = (nodes, neurons)
            if len(nodes):
                self._spk_step.append(np.full(len(nodes), step, dtype=np.int64))
                self._spk_node.append(nodes.copy())
                self._spk_neuron.append(neurons.copy())
            self.available_through = max(self.available_through, step)
        self._pending_steps.clear()

    def feed_legacy(self, sim) -> None:
        """Ingest new chunks from RefSim/TorchSim without exposing them to a runner.

        Those backends already copied full spike indices to the host.  Filtering here gives
        them the same public observation interface while leaving their comparison behavior
        untouched; FastSim writes the compact device buffer directly.
        """
        total = len(sim._spk_step)
        if total < self._legacy_seen:
            # A restored or explicitly trimmed legacy simulator starts a new trace epoch.
            self._legacy_seen = 0
        watch = self.watch_ids
        for steps, nodes, neurons in zip(
            sim._spk_step[self._legacy_seen :],
            sim._spk_node[self._legacy_seen :],
            sim._spk_neuron[self._legacy_seen :],
        ):
            if len(watch):
                keep = np.isin(neurons, watch, assume_unique=False)
                steps, nodes, neurons = steps[keep], nodes[keep], neurons[keep]
            else:
                steps = steps[:0]
                nodes = nodes[:0]
                neurons = neurons[:0]
            if len(steps):
                for step in np.unique(steps):
                    at = steps == step
                    self._append_host(int(step), nodes[at], neurons[at])
        self._legacy_seen = total
        # A step with no spikes has no legacy chunk, but it is still available for polling.
        self.available_through = max(self.available_through, int(sim.step_index) - 1)

    def _append_host(self, step: int, nodes: np.ndarray, neurons: np.ndarray) -> None:
        nodes = np.asarray(nodes, dtype=np.int64)
        neurons = np.asarray(neurons, dtype=np.int64)
        old = self._events_by_step.get(step)
        if old is not None:
            nodes = np.concatenate((old[0], nodes))
            neurons = np.concatenate((old[1], neurons))
        order = np.lexsort((neurons, nodes))
        nodes, neurons = nodes[order], neurons[order]
        self._events_by_step[step] = (nodes, neurons)
        self._spk_step.append(np.full(len(nodes), step, dtype=np.int64))
        self._spk_node.append(nodes.copy())
        self._spk_neuron.append(neurons.copy())

    def fired_at(self, step: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(node_ids, neuron_ids)`` for watched neurons at ``step``."""
        step = int(step)
        if step > self.available_through:
            self.flush()
        empty = np.empty(0, dtype=np.int64)
        return self._events_by_step.get(step, (empty, empty))

    def active_in(self, step: int, window: int, node: int | None = None) -> set[int]:
        """Watched neurons active in ``(step-window, step]``."""
        step, window = int(step), int(window)
        if step > self.available_through:
            self.flush()
        active: set[int] = set()
        for at in range(step - window + 1, step + 1):
            nodes, neurons = self._events_by_step.get(
                at, (np.empty(0, np.int64), np.empty(0, np.int64))
            )
            if node is not None:
                neurons = neurons[nodes == int(node)]
            active.update(neurons.tolist())
        return active

    def capture_spikes(self, node: int, neuron_ids: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
        """Return retained ``(steps, neurons)`` for a diagnostic subset."""
        self.flush()
        wanted = np.asarray(sorted(set(int(i) for i in neuron_ids)), dtype=np.int64)
        if len(wanted) and not np.isin(wanted, self.watch_ids).all():
            raise OverflowError("capture requested neurons that were not allocated in the observer")
        steps_out: list[np.ndarray] = []
        neurons_out: list[np.ndarray] = []
        for step in sorted(self._events_by_step):
            nodes, neurons = self._events_by_step[step]
            keep = (nodes == int(node)) & np.isin(neurons, wanted)
            if np.any(keep):
                neurons_out.append(neurons[keep])
                steps_out.append(np.full(int(keep.sum()), step, dtype=np.int64))
        if not steps_out:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty.copy()
        return np.concatenate(steps_out), np.concatenate(neurons_out)

    @property
    def trace(self) -> SpikeTrace:
        if not self.full_trace:
            raise RuntimeError("full spike trace unavailable from a filtered Observer")
        self.flush()
        if not self._spk_step:
            return SpikeTrace.empty()
        return SpikeTrace.from_arrays(
            np.concatenate(self._spk_step),
            np.concatenate(self._spk_node),
            np.concatenate(self._spk_neuron),
        )

    def clear(self) -> None:
        """Start a new observation epoch (used by simulator restore)."""
        self._pending_steps.clear()
        self._events_by_step.clear()
        self._spk_step.clear()
        self._spk_node.clear()
        self._spk_neuron.clear()
        self._legacy_seen = 0
        self.available_through = -1
