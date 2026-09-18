"""Production simulator: PyTorch, batched over nodes, CPU / CUDA / MPS.

Same algorithm as ref64.py, expression for expression (schedule.md §5). Integer quanta
make synaptic delivery order-independent, so spike traces are bit-identical to the
reference in float64 on CPU; other devices/dtypes are certified at the transaction level.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from .model import D_MAX, Params, Topology, broadcast_param
from .profile import Profiler
from .trace import SpikeTrace


class TorchSim:
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
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        record: list[tuple[int, int]] | None = None,
        stray_rate_hz: float = 0.0,
        stray_quanta: int = 0,
        stray_seed: int | None = None,
        profiler: Profiler | None = None,
    ):
        self.topo = topo
        self.params = params
        self.B = int(n_nodes)
        self.n = int(topo.n)
        self.L = D_MAX + 1
        self.device = torch.device(device)
        self.dtype = dtype
        a, c, k = params.constants()
        self.a, self.c, self.k = float(a), float(c), float(k)
        self.w_unit = float(params.w_unit)
        self.n_ref = int(params.n_ref)
        B, n = self.B, self.n
        dev = self.device

        def T(arr, dt=None):
            return torch.as_tensor(np.ascontiguousarray(arr), device=dev, dtype=dt)

        self.V_th = T(broadcast_param(params.V_th if V_th is None else V_th, B, n, np.float64), dtype)
        self.bias = T(broadcast_param(topo.sim_bias(bias), B, n, np.float64), dtype)  # None: the topology's own biases
        self.gain = T(broadcast_param(gain, B, n, np.float64), dtype)
        self.silenced = T(broadcast_param(silenced, B, n, bool), torch.bool)
        self.E_L = float(params.E_L)
        self.V_reset = float(params.V_reset)

        self.t_src = T(topo.src, torch.int64)
        self.t_dst = T(topo.dst, torch.int64)
        self.t_delay = T(topo.delay, torch.int64)
        self.t_indptr = T(topo.indptr, torch.int64)
        self.t_quanta_shared = T(topo.quanta, torch.int64)
        if quanta is None:
            self.t_quanta = None
        else:
            q = np.asarray(quanta, dtype=np.int64)
            if q.shape != (B, topo.nnz):
                raise ValueError(f"per-node quanta must have shape ({B}, {topo.nnz})")
            self.t_quanta = T(q, torch.int64)

        self.V = torch.full((B, n), self.E_L, device=dev, dtype=dtype)
        self.g = torch.zeros((B, n), device=dev, dtype=dtype)
        self.r = torch.zeros((B, n), device=dev, dtype=torch.int32)
        self.ring = torch.zeros((self.L, B, n), device=dev, dtype=torch.int64)
        # stray background input: every neuron of every node receives `stray_quanta` with
        # probability rate * dt each step, drawn on the device as the step runs (the campaigns'
        # mix; pre-drawing the events as lists for 100 copies over a minute of neural time was
        # 10^8-10^9 Python objects and hours before the first step)
        self.stray_p = float(stray_rate_hz) * params.dt / 1000.0
        self.stray_q = int(stray_quanta)
        self._stray_gen = None
        if self.stray_p > 0.0:
            self._stray_gen = torch.Generator(device=dev)
            # no seed given: a fresh one, so two unseeded runs never share a stray stream
            self._stray_gen.manual_seed(int(stray_seed if stray_seed is not None else np.random.default_rng().integers(2**31 - 1)))
        self.step_index = 0
        self.profiler = profiler if profiler is not None else Profiler(False)

        self._events: dict[int, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = defaultdict(list)
        self._spk_step: list[np.ndarray] = []
        self._spk_node: list[np.ndarray] = []
        self._spk_neuron: list[np.ndarray] = []
        self.record = list(record) if record else []
        self.rec_V: list[np.ndarray] = []
        self.rec_g: list[np.ndarray] = []
        if self.record:
            self._rec_b = torch.tensor([b for b, _ in self.record], device=dev, dtype=torch.int64)
            self._rec_i = torch.tensor([i for _, i in self.record], device=dev, dtype=torch.int64)

    # ---- external port input -------------------------------------------------------
    def add_events(self, node: int, steps, neurons, quanta) -> None:
        steps = np.asarray(steps, dtype=np.int64).ravel()
        neurons = np.asarray(neurons, dtype=np.int64).ravel()
        quanta = np.asarray(quanta, dtype=np.int64).ravel()
        if not (len(steps) == len(neurons) == len(quanta)):
            raise ValueError("steps, neurons, quanta must have equal length")
        if len(steps) and (steps.min() < self.step_index):
            raise ValueError("cannot schedule events in the past")
        dev = self.device
        for s in np.unique(steps):
            m = steps == s
            self._events[int(s)].append(
                (
                    torch.full((int(m.sum()),), node, device=dev, dtype=torch.int64),
                    torch.as_tensor(neurons[m], device=dev),
                    torch.as_tensor(quanta[m], device=dev),
                )
            )

    # ---- one step (schedule.md §5) -------------------------------------------------
    def _timed(self, name, fn, *args):
        if not self.profiler.enabled:
            return fn(*args)
        with self.profiler.region(name):
            return fn(*args)

    def _integrate(self, V, g, r):
        held = r > 0
        active = (r == 0) & ~self.silenced
        Vn = self.E_L + self.bias + (V - self.E_L - self.bias) * self.a + g * self.k
        V = torch.where(active, Vn, V)
        V = torch.where(held, torch.full_like(V, self.V_reset), V)
        g = g * self.c
        V = torch.where(self.silenced, torch.full_like(V, self.E_L), V)
        g = torch.where(self.silenced, torch.zeros_like(g), g)
        r = torch.where(held, r - 1, r)
        return V, g, r, held, active

    def _threshold(self, V, active):
        spk = active & (V > self.V_th)
        idx = torch.nonzero(spk)
        return spk, idx

    def _observe(self, s, idx):
        b_idx, n_idx = idx[:, 0], idx[:, 1]
        if len(b_idx):
            self._spk_step.append(np.full(len(b_idx), s, np.int64))
            self._spk_node.append(b_idx.cpu().numpy().astype(np.int64))
            self._spk_neuron.append(n_idx.cpu().numpy().astype(np.int64))

    def _deliver(self, s, g, idx):
        b_idx, n_idx = idx[:, 0], idx[:, 1]
        slot = s % self.L
        if len(b_idx):
            starts = self.t_indptr[n_idx]
            lens = self.t_indptr[n_idx + 1] - starts
            total = int(lens.sum())
            if total:
                b_rep = torch.repeat_interleave(b_idx, lens)
                base = torch.repeat_interleave(starts - torch.cumsum(lens, 0) + lens, lens)
                syn = torch.arange(total, device=self.device, dtype=torch.int64) + base
                q = self.t_quanta_shared[syn] if self.t_quanta is None else self.t_quanta[b_rep, syn]
                at = (s + self.t_delay[syn]) % self.L
                self.ring.index_put_((at, b_rep, self.t_dst[syn]), q, accumulate=True)
        for node_t, neur_t, q_t in self._events.pop(s, ()):
            slot_t = torch.full_like(node_t, slot)
            self.ring.index_put_((slot_t, node_t, neur_t), q_t, accumulate=True)
        due = self.ring[slot]
        if self.stray_p > 0.0:
            hit = torch.rand((self.B, self.n), device=self.device, generator=self._stray_gen) < self.stray_p
            due = due + hit.to(due.dtype) * self.stray_q
        g = g + self.w_unit * self.gain * due.to(self.dtype)
        self.ring[slot] = 0
        g = torch.where(self.silenced, torch.zeros_like(g), g)
        return g

    def _reset(self, V, r, spk):
        V = torch.where(spk, torch.full_like(V, self.V_reset), V)
        r = torch.where(spk, torch.full_like(r, self.n_ref - 1), r)
        return V, r

    def _record(self, V, g):
        if self.record:
            self.rec_V.append(V[self._rec_b, self._rec_i].cpu().numpy())
            self.rec_g.append(g[self._rec_b, self._rec_i].cpu().numpy())

    @torch.no_grad()
    def step(self) -> None:
        s = self.step_index
        V, g, r = self.V, self.g, self.r
        V, g, r, held, active = self._timed("integrate", self._integrate, V, g, r)
        spk, idx = self._timed("threshold", self._threshold, V, active)
        self._timed("observe", self._observe, s, idx)
        g = self._timed("deliver", self._deliver, s, g, idx)
        V, r = self._timed("reset", self._reset, V, r, spk)
        self.V, self.g, self.r = V, g, r
        self._timed("record", self._record, V, g)
        self.step_index = s + 1

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
        return np.array(self.rec_V), np.array(self.rec_g)

    # ---- snapshots -----------------------------------------------------------------
    def snapshot(self) -> dict:
        return {
            "step_index": self.step_index,
            "V": self.V.clone(),
            "g": self.g.clone(),
            "r": self.r.clone(),
            "ring": self.ring.clone(),
            "events": {k: [(a.clone(), b.clone(), c.clone()) for a, b, c in v] for k, v in self._events.items()},
            "quanta": None if self.t_quanta is None else self.t_quanta.clone(),
            "V_th": self.V_th.clone(),
            "bias": self.bias.clone(),
            "gain": self.gain.clone(),
            "silenced": self.silenced.clone(),
        }

    def restore(self, snap: dict) -> None:
        self.step_index = int(snap["step_index"])
        self.V = snap["V"].clone()
        self.g = snap["g"].clone()
        self.r = snap["r"].clone()
        self.ring = snap["ring"].clone()
        self._events = defaultdict(list, {k: [(a.clone(), b.clone(), c.clone()) for a, b, c in v] for k, v in snap["events"].items()})
        self.t_quanta = None if snap["quanta"] is None else snap["quanta"].clone()
        self.V_th = snap["V_th"].clone()
        self.bias = snap["bias"].clone()
        self.gain = snap["gain"].clone()
        self.silenced = snap["silenced"].clone()
        self._spk_step, self._spk_node, self._spk_neuron = [], [], []
        self.rec_V, self.rec_g = [], []
