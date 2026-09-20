"""Static-shape PyTorch LIF backend (docs/perf_campaign.md §4, Track A3/A4).

The same model as ``lif_torch.TorchSim`` (schedule.md §5), expression for expression and in
the same floating-point order, but the step has no data-dependent shapes and no host
synchronisation:

* **threshold** keeps the spikes as a dense ``(B, n)`` bool mask — no ``torch.nonzero``;
* **deliver** is one sparse-dense product per distinct delay, ``W_dᵀ (n×n CSR of quanta)
  @ spkᵀ`` — no ``repeat_interleave``/``cumsum``/gathers/``index_put_``. Quanta are
  integers and the product is exact (sums stay below 2^53 in float64, 2^24 in float32, checked
  at construction), so delivery order does not matter and the ring stays int64 as in
  ``TorchSim``: the spike traces are bit-identical to ``RefSim``/``TorchSim``;
* **observe** gathers the watched columns into a device buffer (``observe.Observer``) and
  copies a block to the host every K steps;
* **integrate** and **reset** are in-place ops on preallocated buffers.

Devices: CUDA uses the CSR product (cuSPARSE); CPU (whose CSR kernel is slower than the
alternative) and MPS (no sparse support), and per-node quanta on any device (perturbed
campaigns have no shared matrix), use an edge-wise gather/``scatter_add_`` in int64 — still
static-shape and exact. ``delivery="sparse"|"scatter"`` overrides the choice. ``graph_steps=K`` runs K steps as one block: on
CUDA the block is captured once as a ``torch.cuda.CUDAGraph`` and replayed; elsewhere (and
for partial tail blocks) the same function runs eagerly, so the block semantics — events
pre-staged into the ring before the block, observation committed after it — are testable on
this laptop. See docs/track_a.md.
"""

from __future__ import annotations

import logging
import warnings
from collections import defaultdict

import numpy as np
import torch

from .model import D_MAX, Params, Topology, broadcast_param
from .observe import Observer
from .profile import Profiler

LOG = logging.getLogger(__name__)

_EXACT_LIMIT = {torch.float32: 2**24, torch.float64: 2**53}


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
        delivery: str = "auto",
    ):
        """``TorchSim``'s constructor plus: ``observer`` (default: a full-trace observer
        transferring every ``observe_every`` steps), ``graph_steps`` (K-step blocks; 0 = one
        eager step at a time), ``delivery`` ("auto" | "sparse" | "scatter")."""
        self.topo = topo
        self.params = params
        self.B = int(n_nodes)
        self.n = int(topo.n)
        self.L = D_MAX + 1
        self.device = torch.device(device)
        self.dtype = dtype
        if dtype not in _EXACT_LIMIT:
            raise ValueError("FastSim supports float32 or float64")
        if self.device.type == "mps" and dtype != torch.float32:
            raise ValueError("FastSim on MPS supports float32 only")
        self.graph_steps = int(graph_steps)
        if self.graph_steps < 0 or self.graph_steps > self.L:
            raise ValueError(f"graph_steps must lie in [0, {self.L}]: events are staged one ring length ahead")
        if delivery not in ("auto", "sparse", "scatter"):
            raise ValueError("delivery must be 'auto', 'sparse' or 'scatter'")

        a, c, k = params.constants()
        self.a, self.c, self.k = float(a), float(c), float(k)
        self.w_unit = float(params.w_unit)
        self.n_ref = int(params.n_ref)
        self.E_L = float(params.E_L)
        self.V_reset = float(params.V_reset)
        B, n, dev = self.B, self.n, self.device

        def T(arr, dt=None):
            return torch.as_tensor(np.ascontiguousarray(arr), device=dev, dtype=dt)

        self.V_th = T(broadcast_param(params.V_th if V_th is None else V_th, B, n, np.float64), dtype)
        self.bias = T(broadcast_param(topo.sim_bias(bias), B, n, np.float64), dtype)  # None: the topology's own biases
        self.gain = T(broadcast_param(gain, B, n, np.float64), dtype)
        self.silenced = T(broadcast_param(silenced, B, n, bool), torch.bool)
        self._enabled = ~self.silenced
        # the two per-step constants TorchSim recomputes: E_L + bias and w_unit * gain, in
        # the same expressions so the roundings match
        self._rest = self.E_L + self.bias
        self._wg = self.w_unit * self.gain

        q_override = None
        if quanta is not None:
            q_override = np.asarray(quanta, dtype=np.int64)
            if q_override.shape != (B, topo.nnz):
                raise ValueError(f"per-node quanta must have shape ({B}, {topo.nnz})")
        self.t_quanta = None if q_override is None else T(q_override, torch.int64)

        # state (TorchSim's, same dtypes) and preallocated scratch
        self.V = torch.full((B, n), self.E_L, device=dev, dtype=dtype)
        self.g = torch.zeros((B, n), device=dev, dtype=dtype)
        self.r = torch.zeros((B, n), device=dev, dtype=torch.int32)
        self.ring = torch.zeros((self.L, B, n), device=dev, dtype=torch.int64)
        self._slot = torch.zeros(1, device=dev, dtype=torch.int64)  # step_index % L, on the device
        self._at = torch.zeros(1, device=dev, dtype=torch.int64)
        self._Vn = torch.empty_like(self.V)
        self._tmp = torch.empty_like(self.V)
        self._held = torch.empty((B, n), device=dev, dtype=torch.bool)
        self._active = torch.empty((B, n), device=dev, dtype=torch.bool)
        self._spk = torch.zeros((B, n), device=dev, dtype=torch.bool)
        self._due = torch.empty((1, B, n), device=dev, dtype=torch.int64)
        self._due_f = torch.empty((B, n), device=dev, dtype=dtype)
        self._spkT = torch.empty((n, B), device=dev, dtype=dtype)
        self._acc = torch.empty((B, n), device=dev, dtype=torch.int64)

        self._groups = self._build_delivery(q_override, delivery)
        self.delays = tuple(g["delay"] for g in self._groups)
        self.delivery_path = self._groups[0]["path"] if self._groups else "none (no synapses)"
        if len(self.delays) > 1:
            LOG.info("FastSim: %d distinct delays %s -> one sparse product per delay per step", len(self.delays), self.delays)

        self.stray_p = float(stray_rate_hz) * params.dt / 1000.0
        self.stray_q = int(stray_quanta)
        self._stray_gen = None
        if self.stray_p > 0.0:
            self._stray_gen = torch.Generator(device=dev)
            self._stray_gen.manual_seed(int(stray_seed if stray_seed is not None else np.random.default_rng().integers(2**31 - 1)))

        self.step_index = 0
        self.profiler = profiler if profiler is not None else Profiler(False)
        self._events: dict[int, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = defaultdict(list)

        self.record = list(record) if record else []
        self.rec_V: list[np.ndarray] = []
        self.rec_g: list[np.ndarray] = []
        if self.record:
            self._rec_b = torch.tensor([b for b, _ in self.record], device=dev, dtype=torch.int64)
            self._rec_i = torch.tensor([i for _, i in self.record], device=dev, dtype=torch.int64)

        self._graph = None
        self.graph_active = False
        self.graph_fallback_reason: str | None = None
        if observer is None:  # the default: everything, transferred every observe_every steps (or per block)
            observer = Observer(range(n), B, n_neurons=n, device=dev, full_trace=True,
                                observe_every=max(1, int(observe_every), self.graph_steps))
        self.set_observer(observer)

    # ---- construction helpers ---------------------------------------------------------
    def set_observer(self, observer: Observer) -> None:
        """Install the observer the host reads through (a runner replaces the default)."""
        if observer.B != self.B or observer.n != self.n or observer.device != self.device:
            raise ValueError("observer shape/device does not match this FastSim")
        if self.graph_steps and observer.observe_every < self.graph_steps:
            raise ValueError(f"observer buffer ({observer.observe_every}) is smaller than graph_steps ({self.graph_steps})")
        if self._graph is not None:
            # a captured block gathers into the buffer of the observer it was captured with (its
            # address is baked into the graph): a new observer would see silent blocks with no
            # error (review finding). Drop the graph; the next whole block re-captures.
            self._graph = None
            self.graph_active = False
        self.observer = observer
        # compatibility with protocol.token.recent_active & co. when handed the simulator
        self._spk_step = observer._spk_step
        self._spk_node = observer._spk_node
        self._spk_neuron = observer._spk_neuron

    def _build_delivery(self, per_node: np.ndarray | None, delivery: str) -> list[dict]:
        topo, dev, n = self.topo, self.device, self.n
        if not topo.nnz:
            return []
        groups = []
        for delay in np.unique(topo.delay).tolist():
            edges = np.flatnonzero(topo.delay == delay).astype(np.int64)
            src = topo.src[edges].astype(np.int64)
            dst = topo.dst[edges].astype(np.int64)
            q = topo.quanta[edges].astype(np.int64) if per_node is None else per_node[:, edges]
            # the largest sum one step's product can form at one target: each source spikes at
            # most once per step, so this bounds every partial sum of the float product
            inbound = np.zeros(n, dtype=np.int64)
            np.add.at(inbound, dst, np.abs(q).max(axis=0) if per_node is not None else np.abs(q))
            bound = int(inbound.max(initial=0))
            path = delivery
            if path == "auto":
                # the CSR product is the CUDA path (cuSPARSE SpMM, one kernel per delay). The
                # CPU CSR kernel is single-threaded and ~10 ns per synapse (measured: 190 us
                # for 19k edges, 10x the edge-wise scatter), and MPS has no sparse CSR at all,
                # so both take the edge-wise int64 scatter, which is exact for any dtype.
                if per_node is not None:
                    path, why = "scatter", "per-node quanta"
                elif dev.type != "cuda":
                    path, why = "scatter", f"{dev.type} device"
                elif bound >= _EXACT_LIMIT[self.dtype]:
                    path, why = "scatter", f"inbound quanta bound {bound} >= {_EXACT_LIMIT[self.dtype]} would round in {self.dtype}"
                else:
                    path, why = "sparse", ""
                if why:
                    LOG.debug("FastSim: delay %d uses edge-wise int64 scatter delivery (%s)", delay, why)
            if path == "sparse":
                if per_node is not None:
                    raise ValueError("delivery='sparse' needs shared quanta; per-node quanta use 'scatter'")
                if bound >= _EXACT_LIMIT[self.dtype]:
                    raise ValueError(
                        f"delivery='sparse' in {self.dtype} is not exact here: inbound quanta bound {bound} >= "
                        f"{_EXACT_LIMIT[self.dtype]}; use float64 or delivery='scatter'"
                    )
                # W_dᵀ: rows = targets, columns = sources; duplicate (src, dst) pairs summed
                pair = dst * n + src
                uniq, inv = np.unique(pair, return_inverse=True)
                vals = np.zeros(len(uniq), dtype=np.int64)
                np.add.at(vals, inv, q)
                rows, cols = uniq // n, uniq % n
                crow = np.zeros(n + 1, dtype=np.int64)
                np.cumsum(np.bincount(rows, minlength=n), out=crow[1:])
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta")
                    W = torch.sparse_csr_tensor(
                        torch.as_tensor(crow, device=dev), torch.as_tensor(cols, device=dev),
                        torch.as_tensor(vals, device=dev, dtype=self.dtype), size=(n, n), device=dev, dtype=self.dtype,
                    )
                groups.append({"delay": int(delay), "W": W, "path": f"sparse CSR ({self.dtype})"})
            else:
                q_t = torch.as_tensor(np.ascontiguousarray(q), device=dev, dtype=torch.int64)
                groups.append({
                    "delay": int(delay),
                    "src": torch.as_tensor(src, device=dev), "q": q_t,
                    "dst": torch.as_tensor(dst, device=dev).unsqueeze(0).expand(self.B, -1),
                    "path": "edge-wise int64 scatter" + (" (per-node quanta)" if per_node is not None else ""),
                })
        return groups

    # ---- external port input ------------------------------------------------------------
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
        dev = self.device
        for s in np.unique(steps):
            m = steps == s
            self._events[int(s)].append((
                torch.full((int(m.sum()),), int(node), device=dev, dtype=torch.int64),
                torch.as_tensor(neurons[m], device=dev),
                torch.as_tensor(quanta[m], device=dev),
            ))

    def _stage_events(self, first: int, count: int) -> None:
        """Host injection: put the events of steps ``first .. first+count-1`` into their ring
        slots before the block runs. Safe for ``count <= L``: slot ``s % L`` was last cleared
        at step ``s - L``, before the block, and the block's own deliveries only add."""
        for step in range(int(first), int(first) + int(count)):
            slot = step % self.L
            for node_t, neur_t, q_t in self._events.pop(step, ()):
                self.ring.index_put_((torch.full_like(node_t, slot), node_t, neur_t), q_t, accumulate=True)

    # ---- one step, tensor ops only (schedule.md §5) ---------------------------------------
    def _integrate(self) -> None:
        V, g, r = self.V, self.g, self.r
        torch.gt(r, 0, out=self._held)
        torch.eq(r, 0, out=self._active)
        self._active.logical_and_(self._enabled)
        # Vn = E_L + bias + (V - E_L - bias) * a + g * k, in TorchSim's evaluation order
        self._Vn.copy_(V).sub_(self.E_L).sub_(self.bias).mul_(self.a).add_(self._rest)
        torch.mul(g, self.k, out=self._tmp)
        self._Vn.add_(self._tmp)
        torch.where(self._active, self._Vn, V, out=self._tmp)  # not out=V: V is also an input (review)
        V.copy_(self._tmp)
        V.masked_fill_(self._held, self.V_reset)
        g.mul_(self.c)
        V.masked_fill_(self.silenced, self.E_L)
        g.masked_fill_(self.silenced, 0.0)
        r.sub_(self._held.to(torch.int32))  # r = where(held, r - 1, r)

    def _threshold(self) -> None:
        torch.gt(self.V, self.V_th, out=self._spk)
        self._spk.logical_and_(self._active)

    def _deliver(self) -> None:
        for grp in self._groups:
            if "W" in grp:
                self._spkT.copy_(self._spk.t())
                prod = torch.sparse.mm(grp["W"], self._spkT)  # (n_dst, B), exact integers
                self._acc.copy_(prod.t())
            else:
                vals = self._spk.index_select(1, grp["src"]).to(torch.int64)
                vals.mul_(grp["q"])
                self._acc.zero_()
                self._acc.scatter_add_(1, grp["dst"], vals)
            torch.add(self._slot, grp["delay"], out=self._at)
            self._at.remainder_(self.L)
            self.ring.index_add_(0, self._at, self._acc.unsqueeze(0))
        torch.index_select(self.ring, 0, self._slot, out=self._due)
        due = self._due[0]
        if self.stray_p > 0.0:
            hit = torch.rand((self.B, self.n), device=self.device, generator=self._stray_gen) < self.stray_p
            due.add_(hit.to(torch.int64) * self.stray_q)
        self._due_f.copy_(due)
        torch.mul(self._wg, self._due_f, out=self._tmp)
        self.g.add_(self._tmp)
        self.ring.index_fill_(0, self._slot, 0)
        self.g.masked_fill_(self.silenced, 0.0)

    def _reset(self) -> None:
        self.V.masked_fill_(self._spk, self.V_reset)
        self.r.masked_fill_(self._spk, self.n_ref - 1)
        self._slot.add_(1).remainder_(self.L)

    def _record(self) -> None:
        if self.record:
            self.rec_V.append(self.V[self._rec_b, self._rec_i].cpu().numpy())
            self.rec_g.append(self.g[self._rec_b, self._rec_i].cpu().numpy())

    def _timed(self, name, fn, *args):
        if not self.profiler.enabled:
            return fn(*args)
        with self.profiler.region(name):
            return fn(*args)

    def _step_core(self, obs_slot: int | None) -> None:
        """Integrate, threshold, observe, deliver, reset. ``obs_slot`` None: the observer's
        own eager bookkeeping; an int: device-only gather into that slot (block mode)."""
        self._timed("integrate", self._integrate)
        self._timed("threshold", self._threshold)
        if obs_slot is None:
            self._timed("observe", self.observer.write_dense, self.step_index, self._spk)
        else:
            self._timed("observe", self.observer.gather, obs_slot, self._spk)
        self._timed("deliver", self._deliver)
        self._timed("reset", self._reset)

    @torch.no_grad()
    def step(self) -> None:
        """One eager step (events staged for this step only)."""
        s = self.step_index
        self._stage_events(s, 1)
        self._step_core(None)
        self._timed("record", self._record)
        self.step_index = s + 1

    # ---- K-step blocks ----------------------------------------------------------------------
    def _block_eager(self, count: int) -> None:
        for k in range(count):
            self._step_core(k)
            if self.record:
                self._record()

    def _run_block(self, count: int) -> None:
        s0 = self.step_index
        self.observer.flush()
        self._stage_events(s0, count)
        if count == self.graph_steps and self._graph_ready():
            self._timed("graph_replay", self._graph.replay)
        else:
            self._block_eager(count)
        self._timed("observe_commit", self.observer.commit_block, s0, count)
        self.step_index = s0 + count

    def _graph_ready(self) -> bool:
        if self._graph is not None:
            return True
        if self.graph_fallback_reason is not None:
            return False
        if self.profiler.enabled:  # per-step regions are being timed: eager for now, capture later
            return False
        reason = None
        if self.device.type != "cuda":
            reason = f"CUDA graphs need a CUDA device (this is {self.device.type}); the K-step block runs eagerly"
        elif self.stray_p > 0.0:
            reason = "stray input draws from a torch.Generator each step; capturing it needs the graph-safe RNG registration (TODO(cluster))"
        elif self.record:
            reason = "record= copies V/g to the host every step; a captured block cannot"
        if reason is None:
            try:
                self._capture_graph()
            except Exception as exc:  # pragma: no cover - CUDA only
                reason = f"CUDA graph capture failed: {type(exc).__name__}: {exc}"
                self._graph = None
        if reason is not None:
            self.graph_fallback_reason = reason
            (LOG.info if self.device.type != "cuda" else LOG.warning)("FastSim graph_steps=%d: %s", self.graph_steps, reason)
            return False
        self.graph_active = True
        return True

    def _capture_graph(self) -> None:  # pragma: no cover - CUDA only
        """Capture ``_block_eager(K)`` once. Warm-up and capture advance the device state, so
        it is saved before and copied back after; the observer buffer is overwritten before
        the next commit anyway."""
        K = self.graph_steps
        saved = self._state_clone()
        try:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    self._block_eager(K)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._block_eager(K)
            torch.cuda.synchronize()
            self._graph = graph
        finally:
            # the warm-up and the capture advance V/g/r/ring/slot and consume the staged
            # events; a failure must not leave the eager fallback running from that state
            # (review finding): restore on every exit, success or exception
            try:
                torch.cuda.synchronize()
            except Exception:  # pragma: no cover - a device error is the reason we are here
                pass
            self._state_restore(saved)

    @torch.no_grad()
    def run(self, n_steps: int) -> None:
        remaining = int(n_steps)
        if remaining < 0:
            raise ValueError("n_steps must be non-negative")
        if self.graph_steps == 0:
            for _ in range(remaining):
                self.step()
        else:
            while remaining:
                count = min(self.graph_steps, remaining)
                self._run_block(count)
                remaining -= count
        self.observer.flush()

    # ---- results -----------------------------------------------------------------------------
    @property
    def trace(self):
        return self.observer.trace

    def recorded(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array(self.rec_V), np.array(self.rec_g)

    # ---- snapshots (schedule.md §8) ------------------------------------------------------------
    def _state_clone(self) -> dict:
        return {
            "V": self.V.clone(), "g": self.g.clone(), "r": self.r.clone(),
            "ring": self.ring.clone(), "slot": self._slot.clone(),
            "stray_rng": None if self._stray_gen is None else self._stray_gen.get_state(),
        }

    def _state_restore(self, st: dict) -> None:
        self.V.copy_(st["V"]); self.g.copy_(st["g"]); self.r.copy_(st["r"])
        self.ring.copy_(st["ring"]); self._slot.copy_(st["slot"])
        if self._stray_gen is not None and st["stray_rng"] is not None:
            self._stray_gen.set_state(st["stray_rng"])

    def snapshot(self) -> dict:
        snap = self._state_clone()
        snap.update({
            "step_index": self.step_index,
            "events": {k: [(a.clone(), b.clone(), c.clone()) for a, b, c in v] for k, v in self._events.items()},
            "quanta": None if self.t_quanta is None else self.t_quanta.clone(),
            "V_th": self.V_th.clone(), "bias": self.bias.clone(), "gain": self.gain.clone(),
            "silenced": self.silenced.clone(),
        })
        return snap

    def restore(self, snap: dict) -> None:
        self.step_index = int(snap["step_index"])
        self._state_restore(snap)
        self._events = defaultdict(list, {k: [(a.clone(), b.clone(), c.clone()) for a, b, c in v] for k, v in snap["events"].items()})
        if snap["quanta"] is not None:
            if self.t_quanta is None or not torch.equal(self.t_quanta, snap["quanta"]):
                raise ValueError("FastSim bakes per-node quanta into its delivery tables; restore cannot change them")
        self.V_th.copy_(snap["V_th"]); self.bias.copy_(snap["bias"]); self.gain.copy_(snap["gain"])
        self.silenced.copy_(snap["silenced"])
        # into the existing buffers: a captured graph holds their addresses
        self._enabled.copy_(~self.silenced)
        self._rest.copy_(self.E_L + self.bias)
        self._wg.copy_(self.w_unit * self.gain)
        self.observer.clear()
        self.rec_V, self.rec_g = [], []
