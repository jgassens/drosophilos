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

    # ------------------------------------------------------------------------------------
    # broadcast neurons -> trees of identical copies (opt-in; nothing else calls this)
    # ------------------------------------------------------------------------------------
    def out_degrees(self) -> np.ndarray:
        return np.bincount(np.asarray(self.src, dtype=np.int64), minlength=self.n)

    def split_hubs(self, max_fanout: int, roles=None) -> dict:
        """Bound every out-degree by `max_fanout` without changing any spike time.

        A neuron with more than `max_fanout` outgoing synapses keeps its first `max_fanout`
        (in synapse order) and hands the rest, in slices of `max_fanout`, to new neurons
        `<role>.c1`, `.c2`, ... Each copy has the original's physics (the Netlist's neurons
        share `params`; `roles` is the only per-neuron field) and a duplicate of every
        incoming synapse (same pre, quanta, delay), so it receives exactly what the original
        receives and spikes exactly when the original spikes: no hop is added, and every
        target still sees its input at the designed time. The original synapses keep their
        indices (only `src` is rewritten); the duplicated inputs are appended.

        Duplicating the inputs raises each source's out-degree by the number of copies, so
        sources pushed past the bound are split too, and so on upstream, until no neuron in
        scope exceeds the bound. `roles` (None, an iterable of exact role strings, or a
        predicate on the role) restricts the neurons split on their own account; a source
        whose out-degree the transform itself raised past the bound is always in scope.

        A neuron that feeds itself through the tree (a self-loop it keeps, or a cycle of
        splits X -> Y -> X) can go on for ever: each split adds a synapse to a source that is
        X's own tree, which then splits again. The guard is conservative: as soon as a split
        of X has, through the chain of splits it caused, pushed X or one of its copies past
        the bound again, ValueError is raised and the netlist is left untouched.

        Returns {original: [copies]}, keyed by the neuron the copies descend from (a copy
        that is itself split later contributes its copies to the same key). Groups are
        left as they are (copies do not join them)."""
        if max_fanout < 1:
            raise ValueError("max_fanout must be >= 1")
        if roles is None:
            in_scope = lambda r: True  # noqa: E731
        elif callable(roles):
            in_scope = roles
        else:
            wanted = set(roles)
            in_scope = lambda r: r in wanted  # noqa: E731

        # dry-run on copies of the lists so a rejected cycle leaves the netlist untouched
        src, dst, quanta, delay = list(self.src), list(self.dst), list(self.quanta), list(self.delay)
        new_roles = list(self.roles)
        root: dict[int, int] = {}  # copy -> the neuron it descends from
        copies: dict[int, list[int]] = {}
        made: dict[int, int] = {}  # neuron -> copies it has spawned so far (a neuron can be split
        #   more than once: after its own split, a split of one of its targets grows it again)
        out: list[list[int]] = [[] for _ in range(len(new_roles))]
        inc: list[list[int]] = [[] for _ in range(len(new_roles))]
        for e, (s, d) in enumerate(zip(src, dst)):
            out[s].append(e)
            inc[d].append(e)

        # pending: neuron -> roots whose splits transitively caused it to exceed the bound.
        # Downstream first: a neuron is split only once none of its targets is still pending,
        # so a source collects all the growth its hubs cause before it is split itself (a
        # smaller tree than splitting sources as soon as they cross the bound). Only a cycle
        # leaves no such neuron, and the cause sets catch it.
        pending: dict[int, frozenset] = {x: frozenset() for x in range(len(new_roles))
                                         if len(out[x]) > max_fanout and in_scope(new_roles[x])}
        while pending:
            ready = [x for x in pending if not any(dst[e] in pending for e in out[x])]
            x = ready[0] if ready else next(iter(pending))
            cause = pending.pop(x)
            if len(out[x]) <= max_fanout:
                continue
            rx = root.get(x, x)
            if rx in cause:
                raise ValueError(f"split_hubs: cycle through {self.roles[rx]!r}: splitting it raises its own out-degree")
            k = -(-len(out[x]) // max_fanout) - 1
            base_role = new_roles[x]
            new = []
            for i in range(k):
                c = len(new_roles)
                made[x] = made.get(x, 0) + 1
                new_roles.append(f"{base_role}.c{made[x]}")
                out.append([])
                inc.append([])
                root[c] = rx
                new.append(c)
            copies.setdefault(rx, []).extend(new)
            # hand the outgoing synapses past the first max_fanout over, in slices
            extra = out[x][max_fanout:]
            out[x] = out[x][:max_fanout]
            for i, c in enumerate(new):
                for e in extra[i * max_fanout:(i + 1) * max_fanout]:
                    src[e] = c
                    out[c].append(e)
            # every copy gets the original's inputs; the sources grow by k
            grown = []
            for e in list(inc[x]):
                s = src[e]
                for c in new:
                    ne = len(src)
                    src.append(s); dst.append(c); quanta.append(quanta[e]); delay.append(delay[e])
                    out[s].append(ne)
                    inc[c].append(ne)
                grown.append(s)
            next_cause = cause | {rx}
            for s in dict.fromkeys(grown):
                if len(out[s]) > max_fanout:
                    pending[s] = pending.get(s, frozenset()) | next_cause
        if not copies:
            return {}
        self.roles, self.src, self.dst, self.quanta, self.delay = new_roles, src, dst, quanta, delay
        return copies
