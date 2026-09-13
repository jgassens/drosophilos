"""Model parameters and topology. See schedule.md for the normative definitions."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np

#: Longest synaptic delay, in steps (schedule.md §5).
D_MAX = 100
#: One anatomical synapse expressed in integer weight quanta (schedule.md §4).
QUANTA_PER_SYNAPSE = 16


@dataclass(frozen=True)
class Params:
    """Global model parameters (one versioned set per run). Times in ms, voltages in mV."""

    dt: float = 0.1
    E_L: float = -52.0
    V_th: float = -45.0
    V_reset: float = -52.0
    tau_m: float = 20.0
    tau_s: float = 5.0
    t_ref: float = 2.2
    w_syn: float = 0.275
    default_delay_ms: float = 1.8

    @property
    def w_unit(self) -> float:
        """Voltage jump per weight quantum, mV."""
        return self.w_syn / QUANTA_PER_SYNAPSE

    @property
    def n_ref(self) -> int:
        """Refractory length in steps; integration resumes `n_ref` steps after the spike step."""
        return int(round(self.t_ref / self.dt))

    @property
    def default_delay_steps(self) -> int:
        return int(round(self.default_delay_ms / self.dt))

    def constants(self) -> tuple[float, float, float]:
        """(a, c, k) of the exact one-step solution (schedule.md §3)."""
        a = math.exp(-self.dt / self.tau_m)
        c = math.exp(-self.dt / self.tau_s)
        if self.tau_s != self.tau_m:
            k = self.tau_s / (self.tau_s - self.tau_m) * (c - a)
        else:
            k = (self.dt / self.tau_m) * a
        return a, c, k

    def with_dt(self, dt: float) -> "Params":
        return dataclasses.replace(self, dt=dt)


@dataclass
class Topology:
    """Shared synapse table in CSR-by-source order.

    Arrays are aligned per synapse: `src[k] -> dst[k]` carries `quanta[k]` quanta with a
    delay of `delay[k]` steps. Sorted by (src, dst, delay); `indptr` indexes by source.
    """

    n: int
    src: np.ndarray
    dst: np.ndarray
    quanta: np.ndarray
    delay: np.ndarray
    indptr: np.ndarray

    @classmethod
    def from_edges(
        cls,
        n: int,
        src,
        dst,
        quanta,
        delay,
    ) -> "Topology":
        src = np.asarray(src, dtype=np.int32)
        dst = np.asarray(dst, dtype=np.int32)
        quanta = np.asarray(quanta, dtype=np.int32)
        delay = np.asarray(delay, dtype=np.int32)
        if not (len(src) == len(dst) == len(quanta) == len(delay)):
            raise ValueError("edge arrays must have equal length")
        if len(src):
            if src.min() < 0 or src.max() >= n or dst.min() < 0 or dst.max() >= n:
                raise ValueError("neuron index out of range")
            if delay.min() < 0 or delay.max() > D_MAX:
                raise ValueError(f"delays must lie in [0, {D_MAX}] steps")
        order = np.lexsort((delay, dst, src))
        src, dst, quanta, delay = src[order], dst[order], quanta[order], delay[order]
        counts = np.bincount(src, minlength=n).astype(np.int64)
        indptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(counts, out=indptr[1:])
        return cls(n=n, src=src, dst=dst, quanta=quanta, delay=delay, indptr=indptr)

    @classmethod
    def empty(cls, n: int) -> "Topology":
        return cls.from_edges(n, [], [], [], [])

    @property
    def nnz(self) -> int:
        return int(len(self.src))

    def out_slice(self, j: int) -> slice:
        return slice(int(self.indptr[j]), int(self.indptr[j + 1]))

    def out_degree(self) -> np.ndarray:
        return np.diff(self.indptr)

    def with_added(self, src, dst, quanta, delay) -> "Topology":
        """Profile 3 only: a new topology with extra synapses (logged by the caller)."""
        return Topology.from_edges(
            self.n,
            np.concatenate([self.src, np.asarray(src, np.int32)]),
            np.concatenate([self.dst, np.asarray(dst, np.int32)]),
            np.concatenate([self.quanta, np.asarray(quanta, np.int32)]),
            np.concatenate([self.delay, np.asarray(delay, np.int32)]),
        )


def broadcast_param(value, n_nodes: int, n: int, dtype) -> np.ndarray:
    """Broadcast a scalar / (n,) / (B, n) parameter to a (B, n) array of `dtype`."""
    arr = np.asarray(value, dtype=dtype)
    if arr.ndim == 0:
        return np.full((n_nodes, n), arr, dtype=dtype)
    if arr.ndim == 1:
        if arr.shape[0] != n:
            raise ValueError(f"expected shape ({n},), got {arr.shape}")
        return np.broadcast_to(arr, (n_nodes, n)).copy()
    if arr.shape != (n_nodes, n):
        raise ValueError(f"expected shape ({n_nodes}, {n}), got {arr.shape}")
    return arr.copy()
