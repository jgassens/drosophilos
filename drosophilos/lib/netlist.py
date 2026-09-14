"""Netlist builder for Profile 3 circuits, and the drive constants every primitive uses.

All drives derive from the model's measured physics (see connectome/embed_h0.py and the
H0 report), not from assumptions:
  single-pulse need  : one synchronous input that just reaches threshold
  loop               : 1.4x that, regenerates a circulating spike (period ~4.7 ms)
  pulse              : same as loop; "each spike alone fires the target"
  and_in             : 0.65x the sustained-train need per input: one live input at 65 % of
                       the gap, two at 1.3x. Measured window with doublet-free ignition
                       (latch rates 195-236 Hz): 0.75 leaks one-input false positives in
                       long-exposure gates (adder 4 %), 0.70/0.68 still leak a few, 0.65 is
                       clean on 2,000 perturbed additions and transports; 0.55 (1.1x) fails
                       to fire under noise. Earlier, 0.65 failed only because 2x ignition
                       doublets pushed latch rates to +22 %.
  or_in              : 2x the sustained-train need; one train suffices
  reset              : -1.5x loop, delivered to BOTH latch members by a single spike
  ignite             : 1.8x need (~1.29x loop), the largest doublet-free pulse. It was 2x loop
                       to beat the post-reset hangover at READY, but that made ignition a
                       doublet, a latch then carries two spikes for a while, and every
                       rate-mode gate reading it sees up to +22 %: one-input ANDs fired
                       (~14 % of perturbed 4-bit additions). With the 15-hop READY chain the
                       hangover at READY is negligible and 1.8x need suffices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..sim.model import Params, Topology


@dataclass(frozen=True)
class Drive:
    single_need: float
    rate_need: float
    loop: int
    pulse: int
    and_in: int
    or_in: int
    reset: int
    loop_period_steps: int
    ignite: int = 0  # 1.8x need: the largest doublet-free ignition pulse. 2x loop produced a doublet
                     # and a latch briefly carrying two spikes read as +22 % rate to its gates
    relay_in: int = 0  # 1.8x need (~1.29x loop): edge relays fire ~1.8 ms after the source, >= 2 ms before their inhibitor lands; doublet-free below ~1.9x

    @classmethod
    def from_params(cls, params: Params, loop_margin=1.4, and_fraction=0.65, or_margin=2.0, reset_factor=1.5) -> "Drive":
        from ..connectome.embed_h0 import loop_period_steps, needed_quanta

        nq = needed_quanta(params)
        loop = int(math.ceil(loop_margin * nq))
        period, _ = loop_period_steps(params, loop)
        gap = params.V_th - params.E_L
        rate_need = gap * (period * params.dt) / (params.w_unit * params.tau_s)
        return cls(nq, rate_need, loop, loop, int(math.ceil(and_fraction * rate_need)),
                   int(math.ceil(or_margin * rate_need)), -int(math.ceil(reset_factor * loop)), period,
                   ignite=int(math.ceil(1.8 * nq)), relay_in=int(math.ceil(1.8 * nq)))


@dataclass
class Netlist:
    params: Params
    roles: list = field(default_factory=list)
    src: list = field(default_factory=list)
    dst: list = field(default_factory=list)
    quanta: list = field(default_factory=list)
    delay: list = field(default_factory=list)
    groups: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.roles)

    @property
    def nnz(self) -> int:
        return len(self.src)

    def neuron(self, role: str) -> int:
        self.roles.append(role)
        return len(self.roles) - 1

    def synapse(self, pre: int, post: int, quanta: int, delay_steps: int | None = None) -> None:
        self.src.append(int(pre))
        self.dst.append(int(post))
        self.quanta.append(int(quanta))
        self.delay.append(self.params.default_delay_steps if delay_steps is None else int(delay_steps))

    def group(self, name: str, neurons) -> None:
        self.groups[name] = [int(x) for x in neurons]

    def topology(self) -> Topology:
        return Topology.from_edges(self.n, self.src, self.dst, self.quanta, self.delay)

    def summary(self) -> dict:
        q = np.asarray(self.quanta)
        return {
            "neurons": self.n,
            "synapses": self.nnz,
            "excitatory_synapses": int((q > 0).sum()),
            "inhibitory_synapses": int((q < 0).sum()),
            "total_abs_quanta": int(np.abs(q).sum()),
        }
