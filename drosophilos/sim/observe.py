"""Spike observation, separated from spike propagation (docs/perf_campaign.md §4, Track A1/A2).

A simulator propagates every spike internally; what reaches the host is decided here. An
``Observer`` is built with the neuron ids a runner needs (READY, completion, output rail
taps, fault and timeout latches, diagnostic captures) and retains only those. Two feeding
paths exist:

* ``FastSim`` writes ``spk[:, watch_ids]`` — a static-shape gather — into a preallocated
  device buffer ``_obs[K, B, W]`` each step (:meth:`gather`) and, every K steps, one
  ``.cpu()`` copies the block and :meth:`flush` unpacks it. No per-step ``nonzero`` and no
  per-step host copy.
* ``RefSim``/``TorchSim`` already copy full spike index lists to the host; :meth:`feed_legacy`
  filters those lists into the same interface so the runners have a single read path.

``full_trace=True`` watches every neuron and retains everything (the reference/debug mode,
:attr:`trace`). Otherwise the watched spikes are retained for ``retain_steps`` steps and
``capture`` ids are retained for the whole run. Overflow never drops spikes silently: writing
past the device buffer, asking for a step that was never observed or was already trimmed,
asking for a full trace from a filtered observer, and capturing an unwatched neuron all raise.

Compatibility: the per-step chunk lists ``_spk_step``/``_spk_node``/``_spk_neuron`` are kept
(for the watched set only, in step order, non-empty chunks) so ``protocol.token.recent_active``,
``decode_recent`` and ``recent_spikes`` work unchanged when handed an ``Observer`` instead of a
simulator. New code should use :meth:`fired_at` and :meth:`active_in`.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch

from .trace import SpikeTrace

_EMPTY = np.empty(0, dtype=np.int64)


class Observer:
    def __init__(
        self,
        watch_ids: Iterable[int],
        n_nodes: int,
        *,
        n_neurons: int,
        device: str | torch.device = "cpu",
        observe_every: int = 1,
        full_trace: bool = False,
        retain_steps: int | None = None,
        capture: tuple[int, Iterable[int]] | None = None,
    ):
        """``watch_ids``: neurons whose spikes reach the host. ``observe_every`` (K): device
        buffer capacity and transfer interval for ``FastSim``. ``retain_steps``: how many
        recent steps of watched spikes stay on the host (None: all). ``capture=(node, ids)``:
        a diagnostic subset (must be watched) retained for the whole run regardless of
        ``retain_steps``. ``full_trace`` watches every neuron and retains everything."""
        n_neurons = int(n_neurons)
        if full_trace:
            ids = np.arange(n_neurons, dtype=np.int64)
        else:
            ids = np.asarray(sorted(set(int(i) for i in watch_ids)), dtype=np.int64)
        if len(ids) and (ids[0] < 0 or ids[-1] >= n_neurons):
            raise ValueError("observer neuron id out of range")
        if int(observe_every) < 1:
            raise ValueError("observe_every must be at least one")
        if full_trace:
            retain_steps = None
        if retain_steps is not None and int(retain_steps) < int(observe_every):
            raise ValueError("retain_steps must cover at least one observation block")

        self.watch_ids = ids
        self.B = int(n_nodes)
        self.n = n_neurons
        self.device = torch.device(device)
        self.observe_every = int(observe_every)
        self.full_trace = bool(full_trace)
        self.retain_steps = None if retain_steps is None else int(retain_steps)
        self._watch_device = torch.as_tensor(ids, device=self.device, dtype=torch.int64)
        self._obs = torch.zeros((self.observe_every, self.B, len(ids)), device=self.device, dtype=torch.bool)
        self._pending_first: int | None = None  # step held in device slot 0
        self._pending = 0  # device slots written since the last flush

        self.capture_node: int | None = None
        self.capture_ids = _EMPTY
        if capture is not None:
            node, cap = capture
            cap = np.asarray(sorted(set(int(i) for i in cap)), dtype=np.int64)
            if len(cap) and not np.isin(cap, ids).all():
                raise ValueError("capture ids must be a subset of the watched neurons")
            if not (0 <= int(node) < self.B):
                raise ValueError("capture node out of range")
            self.capture_node, self.capture_ids = int(node), cap
        self._cap_steps: list[np.ndarray] = []
        self._cap_neurons: list[np.ndarray] = []

        # host side: one entry per step with at least one watched spike, in step order
        self._events_by_step: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._spk_step: list[np.ndarray] = []
        self._spk_node: list[np.ndarray] = []
        self._spk_neuron: list[np.ndarray] = []
        self._legacy_seen = 0
        self.available_through = -1  # last step whose watched spikes are on the host
        self.retained_from = 0  # oldest step still on the host

    # ---- device path (FastSim) ---------------------------------------------------------
    @property
    def pending(self) -> int:
        return self._pending

    def gather(self, slot: int, spikes: torch.Tensor) -> None:
        """Device-only: copy the watched columns of a dense ``(B, n)`` bool mask into slot
        ``slot`` of the device buffer. No host bookkeeping, so it can sit inside a captured
        block; pair it with :meth:`commit_block`."""
        if len(self.watch_ids):
            torch.index_select(spikes, 1, self._watch_device, out=self._obs[slot])

    def write_dense(self, step: int, spikes: torch.Tensor) -> None:
        """One eager step's observation; the block is transferred as soon as it is full, so
        with ``observe_every=1`` every step is on the host before the simulator returns."""
        step = int(step)
        if self._pending == 0:
            self._pending_first = step
        elif step != self._pending_first + self._pending:
            raise RuntimeError("observation steps must be written consecutively")
        self.gather(self._pending, spikes)
        self._pending += 1
        if self._pending == self.observe_every:
            self.flush()

    def commit_block(self, first_step: int, count: int) -> None:
        """Declare that slots ``0..count-1`` hold steps ``first_step..`` and copy them out."""
        if self._pending:
            raise RuntimeError("commit_block on a buffer with un-flushed eager steps")
        if count < 0 or count > self.observe_every:
            raise OverflowError(
                f"observation block of {count} steps exceeds the {self.observe_every}-step device buffer"
            )
        self._pending_first, self._pending = int(first_step), int(count)
        self.flush()

    def flush(self) -> None:
        """One static-shape device-to-host copy of the pending block, then unpack."""
        count = self._pending
        if not count:
            return
        first = self._pending_first
        host = self._obs[:count].to("cpu", copy=True).numpy()
        for k in range(count):
            step = first + k
            nodes, cols = np.nonzero(host[k])
            if len(nodes):
                self._append_host(step, nodes.astype(np.int64), self.watch_ids[cols])
            self.available_through = step
        self._pending = 0
        self._pending_first = None
        self._trim()

    # ---- legacy path (RefSim / TorchSim) --------------------------------------------------
    def feed_legacy(self, sim, trim_chunks: int | None = None) -> None:
        """Ingest the chunks a RefSim/TorchSim appended since the last call, filtered to the
        watched set. ``trim_chunks``: keep only that many recent chunks on the simulator (the
        runners' memory bound; a 128-copy run retaining everything was OOM-killed at 96 GB).
        A simulator that cleared its own lists starts a new epoch (its cleared chunks were
        never seen here, which is the caller's choice, not a silent drop by this class)."""
        total = len(sim._spk_step)
        if total < self._legacy_seen:
            self._legacy_seen = 0
        watch = self.watch_ids
        for steps, nodes, neurons in zip(
            sim._spk_step[self._legacy_seen:], sim._spk_node[self._legacy_seen:], sim._spk_neuron[self._legacy_seen:]
        ):
            keep = np.isin(neurons, watch) if len(watch) else np.zeros(len(neurons), dtype=bool)
            if keep.any():
                for step in np.unique(steps[keep]):
                    at = keep & (steps == step)
                    self._append_host(int(step), nodes[at].astype(np.int64), neurons[at].astype(np.int64))
        self._legacy_seen = total
        # a step without spikes has no chunk but is observed all the same
        self.available_through = max(self.available_through, int(sim.step_index) - 1)
        if trim_chunks is not None and total > 2 * trim_chunks:
            del sim._spk_step[:-trim_chunks]
            del sim._spk_neuron[:-trim_chunks]
            if hasattr(sim, "_spk_node"):
                del sim._spk_node[:-trim_chunks]
            self._legacy_seen = len(sim._spk_step)
        self._trim()

    # ---- host bookkeeping ---------------------------------------------------------------
    def _append_host(self, step: int, nodes: np.ndarray, neurons: np.ndarray) -> None:
        if step in self._events_by_step:  # a second chunk for the same step (legacy feed)
            old_n, old_u = self._events_by_step[step]
            nodes = np.concatenate((old_n, nodes))
            neurons = np.concatenate((old_u, neurons))
            k = len(self._spk_step) - 1
            while k >= 0 and int(self._spk_step[k][0]) != step:
                k -= 1
            if k >= 0:
                del self._spk_step[k], self._spk_node[k], self._spk_neuron[k]
        order = np.lexsort((neurons, nodes))
        nodes, neurons = nodes[order], neurons[order]
        self._events_by_step[step] = (nodes, neurons)
        self._spk_step.append(np.full(len(nodes), step, dtype=np.int64))
        self._spk_node.append(nodes)
        self._spk_neuron.append(neurons)
        if self.capture_node is not None and len(self.capture_ids):
            keep = (nodes == self.capture_node) & np.isin(neurons, self.capture_ids)
            if keep.any():
                self._cap_steps.append(np.full(int(keep.sum()), step, dtype=np.int64))
                self._cap_neurons.append(neurons[keep])

    def _trim(self) -> None:
        if self.retain_steps is None:
            return
        floor = self.available_through - self.retain_steps + 1
        if floor <= self.retained_from:
            return
        self.retained_from = floor
        k = 0
        while k < len(self._spk_step) and int(self._spk_step[k][0]) < floor:
            del self._events_by_step[int(self._spk_step[k][0])]
            k += 1
        if k:
            del self._spk_step[:k], self._spk_node[:k], self._spk_neuron[:k]

    def _check_available(self, step: int) -> None:
        if step > self.available_through:
            raise RuntimeError(
                f"step {step} is not observed yet (available through {self.available_through}); "
                "flush the observer or wait for the block to complete"
            )
        if step < self.retained_from:
            raise RuntimeError(
                f"step {step} was trimmed from the observer (retains {self.retain_steps} steps, "
                f"oldest {self.retained_from})"
            )

    # ---- what the runners read ----------------------------------------------------------
    def fired_at(self, step: int) -> tuple[np.ndarray, np.ndarray]:
        """``(node_ids, neuron_ids)`` of the watched neurons that spiked at ``step``."""
        step = int(step)
        self._check_available(step)
        return self._events_by_step.get(step, (_EMPTY, _EMPTY))

    def active_in(self, step: int, window: int, node: int | None = None) -> set[int]:
        """Watched neurons that spiked in ``(step - window, step]`` (of one node if given):
        what ``decode_recent`` reads."""
        step, window = int(step), int(window)
        self._check_available(step)
        start = max(step - window + 1, 0)
        if start < self.retained_from:
            raise RuntimeError(
                f"window ({step - window}, {step}] reaches into trimmed steps (oldest retained {self.retained_from}); "
                "a partial read would decode silently"
            )
        active: set[int] = set()
        for at in range(start, step + 1):
            ev = self._events_by_step.get(at)
            if ev is None:
                continue
            nodes, neurons = ev
            if node is not None:
                neurons = neurons[nodes == int(node)]
            active.update(neurons.tolist())
        return active

    def capture_spikes(self) -> tuple[np.ndarray, np.ndarray]:
        """``(steps, neurons)`` retained for the ``capture`` subset over the whole run."""
        if self.capture_node is None:
            raise RuntimeError("no capture subset was configured on this observer")
        self.flush()
        if not self._cap_steps:
            return _EMPTY.copy(), _EMPTY.copy()
        return np.concatenate(self._cap_steps), np.concatenate(self._cap_neurons)

    @property
    def trace(self) -> SpikeTrace:
        """The complete spike trace; only a ``full_trace`` observer has one."""
        if not self.full_trace:
            raise RuntimeError(
                "this observer watches a filtered set; build it with full_trace=True for a "
                "spike trace, or read the watched subset through watched_trace()"
            )
        return self.watched_trace()

    def watched_trace(self) -> SpikeTrace:
        """Trace of the retained watched spikes (partial unless ``full_trace``)."""
        self.flush()
        if not self._spk_step:
            return SpikeTrace.empty()
        return SpikeTrace.from_arrays(
            np.concatenate(self._spk_step), np.concatenate(self._spk_node), np.concatenate(self._spk_neuron)
        )

    def clear(self) -> None:
        """Start a new observation epoch (a simulator's ``restore``)."""
        self._pending, self._pending_first = 0, None
        self._events_by_step.clear()
        self._spk_step.clear()
        self._spk_node.clear()
        self._spk_neuron.clear()
        self._cap_steps.clear()
        self._cap_neurons.clear()
        self._legacy_seen = 0
        self.available_through = -1
        self.retained_from = 0
