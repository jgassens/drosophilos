"""Float64 NumPy reference simulator: the executable form of schedule.md.

Every other backend is certified against this one. Clarity beats speed here; the
production backend is `lif_torch.py`.
"""

from __future__ import annotations

import copy
from collections import defaultdict

import numpy as np

from .model import D_MAX, Params, Topology, broadcast_param
from .trace import SpikeTrace


class RefSim:
    def __init__(
        self,
        topo: Topology,
        params: Params = Params(),
        n_nodes: int = 1,
        *,
        V_th=None,
        bias=None,
        gain=1.0,
        silenced=False,
        quanta=None,
        record: list[tuple[int, int]] | None = None,
    ):
        self.topo = topo
        self.params = params
        self.B = int(n_nodes)
        self.n = int(topo.n)
        self.L = D_MAX + 1
        a, c, k = params.constants()
        self.a, self.c, self.k = float(a), float(c), float(k)
        self.w_unit = float(params.w_unit)
        self.n_ref = int(params.n_ref)

        B, n = self.B, self.n
        self.V_th = broadcast_param(params.V_th if V_th is None else V_th, B, n, np.float64)
        self.bias = broadcast_param(topo.sim_bias(bias), B, n, np.float64)  # None: the topology's own biases (flip-flops), else 0
        self.gain = broadcast_param(gain, B, n, np.float64)
        self.silenced = broadcast_param(silenced, B, n, bool)
        if quanta is None:
            self.quanta = None  # shared topology quanta
        else:
            q = np.asarray(quanta, dtype=np.int32)
            if q.shape != (B, topo.nnz):
                raise ValueError(f"per-node quanta must have shape ({B}, {topo.nnz})")
            self.quanta = q.copy()

        self.V = np.full((B, n), params.E_L, dtype=np.float64)
        self.g = np.zeros((B, n), dtype=np.float64)
        self.r = np.zeros((B, n), dtype=np.int32)
        self.ring = np.zeros((self.L, B, n), dtype=np.int64)
        self.step_index = 0

        self._events: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(list)
        self._spk_step: list[np.ndarray] = []
        self._spk_node: list[np.ndarray] = []
        self._spk_neuron: list[np.ndarray] = []
        self.record = list(record) if record else []
        self.rec_V: list[np.ndarray] = []
        self.rec_g: list[np.ndarray] = []

    # ---- external port input -------------------------------------------------------
    def add_events(self, node: int, steps, neurons, quanta) -> None:
        """Schedule external events (schedule.md §6): delivered at `steps`, felt one step later."""
        steps = np.asarray(steps, dtype=np.int64).ravel()
        neurons = np.asarray(neurons, dtype=np.int64).ravel()
        quanta = np.asarray(quanta, dtype=np.int64).ravel()
        if not (len(steps) == len(neurons) == len(quanta)):
            raise ValueError("steps, neurons, quanta must have equal length")
        if len(steps) and (steps.min() < self.step_index):
            raise ValueError("cannot schedule events in the past")
        for s in np.unique(steps):
            m = steps == s
            self._events[int(s)].append((np.full(m.sum(), node, np.int64), neurons[m], quanta[m]))

    # ---- one step (schedule.md §5) -------------------------------------------------
    def step(self) -> None:
        s = self.step_index
        P = self.params
        V, g, r = self.V, self.g, self.r

        # 1. integrate
        held = r > 0
        active = (r == 0) & ~self.silenced
        Vn = P.E_L + self.bias + (V - P.E_L - self.bias) * self.a + g * self.k
        V = np.where(active, Vn, V)
        V[held] = P.V_reset
        g = g * self.c
        V[self.silenced] = P.E_L
        g[self.silenced] = 0.0
        r = r.copy()
        r[held] -= 1

        # 2. threshold
        spk = active & (V > self.V_th)
        b_idx, n_idx = np.nonzero(spk)
        if len(b_idx):
            self._spk_step.append(np.full(len(b_idx), s, np.int64))
            self._spk_node.append(b_idx.astype(np.int64))
            self._spk_neuron.append(n_idx.astype(np.int64))

        # 3. deliver
        slot = s % self.L
        if len(b_idx):
            b_rep, syn = self._gather_out(b_idx, n_idx)
            if len(syn):
                q = self.topo.quanta[syn] if self.quanta is None else self.quanta[b_rep, syn]
                at = (s + self.topo.delay[syn]) % self.L
                np.add.at(self.ring, (at, b_rep, self.topo.dst[syn]), q.astype(np.int64))
        for node_arr, neur_arr, q_arr in self._events.pop(s, ()):
            np.add.at(self.ring, (slot, node_arr, neur_arr), q_arr)
        due = self.ring[slot]
        g = g + self.w_unit * self.gain * due.astype(np.float64)
        self.ring[slot] = 0
        g[self.silenced] = 0.0

        # 4. reset
        V[spk] = P.V_reset
        r[spk] = self.n_ref - 1

        # 5. observe
        self.V, self.g, self.r = V, g, r
        if self.record:
            self.rec_V.append(np.array([V[b, i] for b, i in self.record]))
            self.rec_g.append(np.array([g[b, i] for b, i in self.record]))
        self.step_index = s + 1

    def _gather_out(self, b_idx: np.ndarray, n_idx: np.ndarray):
        starts = self.topo.indptr[n_idx]
        ends = self.topo.indptr[n_idx + 1]
        lens = ends - starts
        total = int(lens.sum())
        if total == 0:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        b_rep = np.repeat(b_idx.astype(np.int64), lens)
        offsets = np.repeat(starts - np.concatenate([[0], np.cumsum(lens)[:-1]]), lens)
        syn = np.arange(total, dtype=np.int64) + offsets
        return b_rep, syn

    def run(self, n_steps: int) -> None:
        for _ in range(int(n_steps)):
            self.step()

    # ---- results -------------------------------------------------------------------
    @property
    def trace(self) -> SpikeTrace:
        if not self._spk_step:
            return SpikeTrace.empty()
        return SpikeTrace.from_arrays(
            np.concatenate(self._spk_step),
            np.concatenate(self._spk_node),
            np.concatenate(self._spk_neuron),
        )

    def recorded(self) -> tuple[np.ndarray, np.ndarray]:
        """(V, g) arrays of shape (steps, len(record))."""
        return np.array(self.rec_V), np.array(self.rec_g)

    # ---- snapshots (schedule.md §8) --------------------------------------------------
    def snapshot(self) -> dict:
        return {
            "step_index": self.step_index,
            "V": self.V.copy(),
            "g": self.g.copy(),
            "r": self.r.copy(),
            "ring": self.ring.copy(),
            "events": copy.deepcopy(dict(self._events)),
            "quanta": None if self.quanta is None else self.quanta.copy(),
            "V_th": self.V_th.copy(),
            "bias": self.bias.copy(),
            "gain": self.gain.copy(),
            "silenced": self.silenced.copy(),
        }

    def restore(self, snap: dict) -> None:
        self.step_index = int(snap["step_index"])
        self.V = snap["V"].copy()
        self.g = snap["g"].copy()
        self.r = snap["r"].copy()
        self.ring = snap["ring"].copy()
        self._events = defaultdict(list, copy.deepcopy(snap["events"]))
        self.quanta = None if snap["quanta"] is None else snap["quanta"].copy()
        self.V_th = snap["V_th"].copy()
        self.bias = snap["bias"].copy()
        self.gain = snap["gain"].copy()
        self.silenced = snap["silenced"].copy()
        self._spk_step, self._spk_node, self._spk_neuron = [], [], []
        self.rec_V, self.rec_g = [], []
