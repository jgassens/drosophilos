"""Static-shape PyTorch LIF backend for CUDA/MPS/CPU.

Unlike :class:`lif_torch.TorchSim`, thresholding never extracts a variable-length index
array.  Spikes stay as a dense bool tensor, delivery is one sparse-dense product per delay
group (a static scatter reduction on MPS/per-node weights), and only watched spike columns
leave the device through :class:`observe.Observer`.
"""

from __future__ import annotations

from collections import defaultdict
import logging

import numpy as np
import torch

from .model import D_MAX, Params, Topology, broadcast_param
from .observe import Observer
from .profile import Profiler


LOG = logging.getLogger(__name__)


class FastSim:
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
        observer: Observer | None = None,
        observe_every: int = 1,
        graph_steps: int = 0,
    ):
        self.topo = topo
        self.params = params
        self.B = int(n_nodes)
        self.n = int(topo.n)
        self.L = D_MAX + 1
        self.device = torch.device(device)
        self.dtype = dtype
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("FastSim supports float32 or float64")
        if self.device.type == "mps" and dtype != torch.float32:
            raise ValueError("MPS FastSim supports float32 only")
        self.graph_steps = int(graph_steps)
        if self.graph_steps < 0 or self.graph_steps > self.L:
            raise ValueError(f"graph_steps must lie in [0, {self.L}]")

        a, c, k = params.constants()
        self.a, self.c, self.k = float(a), float(c), float(k)
        self.w_unit = float(params.w_unit)
        self.n_ref = int(params.n_ref)
        B, n, dev = self.B, self.n, self.device

        def T(arr, dt=None):
            return torch.as_tensor(np.ascontiguousarray(arr), device=dev, dtype=dt)

        self.V_th = T(broadcast_param(params.V_th if V_th is None else V_th, B, n, np.float64), dtype)
        self.bias = T(broadcast_param(topo.sim_bias(bias), B, n, np.float64), dtype)
        self.gain = T(broadcast_param(gain, B, n, np.float64), dtype)
        self.silenced = T(broadcast_param(silenced, B, n, bool), torch.bool)
        self._enabled = ~self.silenced
        self.E_L = float(params.E_L)
        self.V_reset = float(params.V_reset)
        self._rest = self.bias + self.E_L

        q_override = None
        if quanta is not None:
            q_override = np.asarray(quanta, dtype=np.int64)
            if q_override.shape != (B, topo.nnz):
                raise ValueError(f"per-node quanta must have shape ({B}, {topo.nnz})")
        self._check_exact_delivery(q_override)
        self._per_node_quanta = None if q_override is None else T(q_override, dtype)

        self.V = torch.full((B, n), self.E_L, device=dev, dtype=dtype)
        self.g = torch.zeros((B, n), device=dev, dtype=dtype)
        self.r = torch.zeros((B, n), device=dev, dtype=torch.int32)
        # Quanta remain exactly represented in this float ring after the constructor bound.
        self.ring = torch.zeros((self.L, B, n), device=dev, dtype=dtype)
        self._slot = torch.zeros(1, device=dev, dtype=torch.int64)
        self._due = torch.empty((1, B, n), device=dev, dtype=dtype)
        self._Vn = torch.empty_like(self.V)
        self._scaled = torch.empty_like(self.g)
        self._held = torch.empty((B, n), device=dev, dtype=torch.bool)
        self._active = torch.empty((B, n), device=dev, dtype=torch.bool)
        self._spk = torch.empty((B, n), device=dev, dtype=torch.bool)

        self._delay_groups = self._make_delay_groups(q_override)
        distinct = tuple(group[0] for group in self._delay_groups)
        if q_override is not None:
            self.delivery_path = f"static scatter, per-node quanta, delays={distinct}"
        elif self.device.type == "mps":
            self.delivery_path = f"static scatter (MPS sparse CSR unavailable), delays={distinct}"
        else:
            self.delivery_path = f"sparse CSR, delays={distinct}"
        if len(distinct) > 1:
            LOG.info("FastSim grouped delivery across delays %s", distinct)

        self.stray_p = float(stray_rate_hz) * params.dt / 1000.0
        self.stray_q = int(stray_quanta)
        self._stray_gen = None
        if self.stray_p > 0.0:
            self._stray_gen = torch.Generator(device=dev)
            seed = int(stray_seed if stray_seed is not None else np.random.default_rng().integers(2**31 - 1))
            self._stray_gen.manual_seed(seed)

        self.step_index = 0
        self.profiler = profiler if profiler is not None else Profiler(False)
        self._events: dict[int, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = defaultdict(list)
        self.observer = observer or Observer(
            range(n), B, n_neurons=n, device=dev, observe_every=observe_every, full_trace=True
        )
        if self.observer.B != B or self.observer.n != n or self.observer.device != dev:
            raise ValueError("observer shape/device does not match FastSim")
        # Compatibility for token.py and existing diagnostics.  The lists belong to Observer.
        self._spk_step = self.observer._spk_step
        self._spk_node = self.observer._spk_node
        self._spk_neuron = self.observer._spk_neuron

        self.record = list(record) if record else []
        self.rec_V: list[np.ndarray] = []
        self.rec_g: list[np.ndarray] = []
        if self.record:
            self._rec_b = torch.tensor([b for b, _ in self.record], device=dev, dtype=torch.int64)
            self._rec_i = torch.tensor([i for _, i in self.record], device=dev, dtype=torch.int64)
        self._compiled_block = None
        self.graph_enabled = False
        self.graph_fallback_reason: str | None = None

    def _check_exact_delivery(self, per_node: np.ndarray | None) -> None:
        limit = 2**24 if self.dtype == torch.float32 else 2**53
        if not self.topo.nnz:
            return
        if per_node is None:
            inbound = np.zeros(self.n, dtype=np.int64)
            np.add.at(inbound, self.topo.dst, np.abs(self.topo.quanta.astype(np.int64)))
            bound = int(inbound.max(initial=0))
        else:
            bound = 0
            for row in per_node:
                inbound = np.zeros(self.n, dtype=np.int64)
                np.add.at(inbound, self.topo.dst, np.abs(row))
                bound = max(bound, int(inbound.max(initial=0)))
        if bound >= limit:
            raise ValueError(
                f"FastSim {self.dtype} delivery could exceed exact integer range: "
                f"max absolute inbound quanta {bound} >= {limit}"
            )

    def _make_delay_groups(self, per_node: np.ndarray | None):
        groups = []
        for delay in np.unique(self.topo.delay).tolist():
            edge_ids = np.flatnonzero(self.topo.delay == delay).astype(np.int64)
            src_np = self.topo.src[edge_ids].astype(np.int64)
            dst_np = self.topo.dst[edge_ids].astype(np.int64)
            src = torch.as_tensor(src_np, device=self.device, dtype=torch.int64)
            dst = torch.as_tensor(dst_np, device=self.device, dtype=torch.int64)
            sparse = None
            if per_node is None and self.device.type != "mps":
                order = np.lexsort((src_np, dst_np))
                rows = dst_np[order]
                cols = src_np[order]
                crow = np.zeros(self.n + 1, dtype=np.int64)
                np.cumsum(np.bincount(rows, minlength=self.n), out=crow[1:])
                values = self.topo.quanta[edge_ids][order].astype(np.float64)
                sparse = torch.sparse_csr_tensor(
                    torch.as_tensor(crow, device=self.device),
                    torch.as_tensor(cols, device=self.device),
                    torch.as_tensor(values, device=self.device, dtype=self.dtype),
                    size=(self.n, self.n), device=self.device, dtype=self.dtype,
                )
            groups.append((int(delay), edge_ids, src, dst, sparse))
        return groups

    def add_events(self, node: int, steps, neurons, quanta) -> None:
        steps = np.asarray(steps, dtype=np.int64).ravel()
        neurons = np.asarray(neurons, dtype=np.int64).ravel()
        quanta = np.asarray(quanta, dtype=np.int64).ravel()
        if not (len(steps) == len(neurons) == len(quanta)):
            raise ValueError("steps, neurons, quanta must have equal length")
        if len(steps) and steps.min() < self.step_index:
            raise ValueError("cannot schedule events in the past")
        if len(neurons) and (neurons.min() < 0 or neurons.max() >= self.n):
            raise ValueError("event neuron id out of range")
        exact_limit = 2**24 if self.dtype == torch.float32 else 2**53
        if len(quanta) and np.abs(quanta).max() >= exact_limit:
            raise ValueError("external event quanta exceed the dtype's exact integer range")
        for step in np.unique(steps):
            at = steps == step
            self._events[int(step)].append((
                torch.full((int(at.sum()),), int(node), device=self.device, dtype=torch.int64),
                torch.as_tensor(neurons[at], device=self.device, dtype=torch.int64),
                torch.as_tensor(quanta[at], device=self.device, dtype=self.dtype),
            ))

    def _stage_events(self, first: int, count: int) -> None:
        """Put host events in ring slots before the static numerical block starts."""
        for step in range(int(first), int(first) + int(count)):
            slot = step % self.L
            for nodes, neurons, quanta in self._events.pop(step, ()):
                slots = torch.full_like(nodes, slot)
                self.ring.index_put_((slots, nodes, neurons), quanta, accumulate=True)

    def _timed(self, name, fn, *args):
        if not self.profiler.enabled:
            return fn(*args)
        with self.profiler.region(name):
            return fn(*args)

    def _integrate(self) -> None:
        torch.gt(self.r, 0, out=self._held)
        torch.eq(self.r, 0, out=self._active)
        self._active.logical_and_(self._enabled)
        self._Vn.copy_(self.V).sub_(self._rest).mul_(self.a).add_(self._rest).add_(self.g, alpha=self.k)
        torch.where(self._active, self._Vn, self.V, out=self.V)
        self.V.masked_fill_(self._held, self.V_reset)
        self.g.mul_(self.c)
        self.V.masked_fill_(self.silenced, self.E_L)
        self.g.masked_fill_(self.silenced, 0.0)
        torch.sub(self.r, self._held, out=self.r)

    def _threshold(self) -> None:
        torch.gt(self.V, self.V_th, out=self._spk)
        self._spk.logical_and_(self._active)

    def _group_sum(self, group) -> torch.Tensor:
        _delay, edge_ids, src, dst, sparse = group
        if sparse is not None:
            return torch.sparse.mm(sparse, self._spk.to(self.dtype).T).T
        # MPS has no usable sparse CSR matmul.  This is still static-shape and never extracts
        # spike indices; per-node quanta naturally uses the same edge-wise representation.
        values = self._spk.index_select(1, src).to(self.dtype)
        if self._per_node_quanta is None:
            q = torch.as_tensor(
                self.topo.quanta[edge_ids], device=self.device, dtype=self.dtype
            ).unsqueeze(0)
        else:
            ids = torch.as_tensor(edge_ids, device=self.device, dtype=torch.int64)
            q = self._per_node_quanta.index_select(1, ids)
        values.mul_(q)
        out = torch.zeros_like(self.g)
        out.scatter_add_(1, dst.unsqueeze(0).expand(self.B, -1), values)
        return out

    def _deliver(self) -> None:
        for group in self._delay_groups:
            delay = group[0]
            delivery = self._group_sum(group)
            at = torch.remainder(self._slot + delay, self.L)
            self.ring.index_add_(0, at, delivery.unsqueeze(0))
        torch.index_select(self.ring, 0, self._slot, out=self._due)
        if self.stray_p > 0.0:
            hit = torch.rand(
                (self.B, self.n), device=self.device, generator=self._stray_gen
            ) < self.stray_p
            self._due[0].add_(hit.to(self.dtype), alpha=self.stray_q)
        self._scaled.copy_(self.gain).mul_(self.w_unit).mul_(self._due[0])
        self.g.add_(self._scaled)
        self.ring.index_fill_(0, self._slot, 0.0)
        self.g.masked_fill_(self.silenced, 0.0)

    def _reset(self) -> None:
        self.V.masked_fill_(self._spk, self.V_reset)
        self.r.masked_fill_(self._spk, self.n_ref - 1)
        self._slot.add_(1).remainder_(self.L)

    def _record(self) -> None:
        if self.record:
            self.rec_V.append(self.V[self._rec_b, self._rec_i].cpu().numpy())
            self.rec_g.append(self.g[self._rec_b, self._rec_i].cpu().numpy())

    @torch.no_grad()
    def step(self) -> None:
        s = self.step_index
        self._stage_events(s, 1)
        self._timed("integrate", self._integrate)
        self._timed("threshold", self._threshold)
        self._timed("observe", self.observer.write_dense, s, self._spk)
        self._timed("deliver", self._deliver)
        self._timed("reset", self._reset)
        self._timed("record", self._record)
        self.step_index = s + 1

    def _run_eager_block(self, count: int) -> None:
        for _ in range(int(count)):
            self.step()

    def _configure_compiled_block(self, count: int) -> None:
        """Best-effort CUDA low-overhead compilation; eager remains the safe fallback.

        PyTorch's sparse CSR capture support is version/device dependent.  Compilation is
        deliberately lazy and failures are reported through ``graph_fallback_reason``.
        """
        if self.device.type != "cuda":
            self.graph_fallback_reason = "CUDA graph/compile unavailable on this device; eager K-step loop used"
            return
        if self.stray_p or self.record:
            self.graph_fallback_reason = "stray RNG or host trajectory recording requires eager steps"
            return
        # ``step`` includes observation bookkeeping and step labels, so compile the fixed
        # numerical sequence while the outer block owns event staging and host observation.
        # Cluster validation decides whether the installed torch sparse backend keeps this
        # loop in one CUDA graph; any graph break remains correct and is visible in profiling.
        try:
            self._compiled_block = torch.compile(self._run_eager_block, mode="reduce-overhead")
            self.graph_enabled = True
        except Exception as exc:  # pragma: no cover - CUDA-only construction
            self.graph_fallback_reason = f"torch.compile unavailable: {type(exc).__name__}: {exc}"
            LOG.warning("FastSim graph fallback: %s", self.graph_fallback_reason)

    @torch.no_grad()
    def run(self, n_steps: int) -> None:
        remaining = int(n_steps)
        if remaining < 0:
            raise ValueError("n_steps must be non-negative")
        K = self.graph_steps
        if not K:
            self._run_eager_block(remaining)
            self.observer.flush()
            return
        if self._compiled_block is None and self.graph_fallback_reason is None:
            self._configure_compiled_block(K)
        while remaining:
            count = min(K, remaining)
            # Partial tail blocks and every non-CUDA run use the same eager function tested
            # on CPU/MPS.  Event injection remains inside each step, preserving exact timing.
            if count == K and self._compiled_block is not None:
                self._compiled_block(count)
            else:
                self._run_eager_block(count)
            remaining -= count
        self.observer.flush()

    @property
    def trace(self):
        return self.observer.trace

    def recorded(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array(self.rec_V), np.array(self.rec_g)

    def snapshot(self) -> dict:
        return {
            "step_index": self.step_index,
            "V": self.V.clone(), "g": self.g.clone(), "r": self.r.clone(),
            "ring": self.ring.clone(), "slot": self._slot.clone(),
            "events": {k: [(a.clone(), b.clone(), c.clone()) for a, b, c in v]
                       for k, v in self._events.items()},
            "quanta": None if self._per_node_quanta is None else self._per_node_quanta.clone(),
            "V_th": self.V_th.clone(), "bias": self.bias.clone(), "gain": self.gain.clone(),
            "silenced": self.silenced.clone(),
            "stray_rng": None if self._stray_gen is None else self._stray_gen.get_state(),
        }

    def restore(self, snap: dict) -> None:
        self.step_index = int(snap["step_index"])
        self.V.copy_(snap["V"]); self.g.copy_(snap["g"]); self.r.copy_(snap["r"])
        self.ring.copy_(snap["ring"]); self._slot.copy_(snap["slot"])
        self._events = defaultdict(
            list, {k: [(a.clone(), b.clone(), c.clone()) for a, b, c in v]
                   for k, v in snap["events"].items()}
        )
        if self._per_node_quanta is not None and snap["quanta"] is not None:
            self._per_node_quanta.copy_(snap["quanta"])
        self.V_th.copy_(snap["V_th"]); self.bias.copy_(snap["bias"])
        self.gain.copy_(snap["gain"]); self.silenced.copy_(snap["silenced"])
        self._enabled.copy_(~self.silenced)
        self._rest.copy_(self.bias + self.E_L)
        if self._stray_gen is not None and snap["stray_rng"] is not None:
            self._stray_gen.set_state(snap["stray_rng"])
        self.observer.clear()
        self.rec_V.clear(); self.rec_g.clear()
