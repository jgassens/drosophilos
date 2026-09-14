"""Dual-rail gates on latch trains (rate mode). Every gate reads only latch taps (standard
213 Hz trains) and every gate output that feeds another gate is latched through an edge
relay, so weights never depend on an upstream gate's firing rate.

    NOT  : rail swap, no neurons
    AND  : y1 = a1 AND b1 ; y0 = a0 OR b0
    OR   : y1 = a1 OR b1  ; y0 = a0 AND b0
    XOR  : y1 = (a1 AND b0) OR (a0 AND b1) ; y0 = (a1 AND b1) OR (a0 AND b0)
    MAJ3 : y1 = maj(a1,b1,c1) ; y0 = maj(a0,b0,c0)     (2-of-3 threshold, rate mode)
"""

from __future__ import annotations

from dataclasses import dataclass

from ..protocol.celement import _ignite_from, add_and_latched, add_delay_chain, add_or_latched, add_veto_relay
from ..protocol.latch import Latch, add_latch
from .netlist import Drive, Netlist


@dataclass(frozen=True)
class Tap:
    """A neuron whose train stands in for a rail (a delayed rail); usable wherever only `.u` is read."""
    u: int


@dataclass(frozen=True)
class Rail2:
    r0: Latch | Tap
    r1: Latch | Tap

    @property
    def taps(self) -> tuple[int, int]:
        return (self.r0.u, self.r1.u)

    @property
    def latches(self) -> list[Latch]:
        return [self.r0, self.r1]


def swap(x: Rail2) -> Rail2:  # NOT
    return Rail2(x.r1, x.r0)


def add_maj_latched(net: Netlist, drive: Drive, name: str, inputs: list[int]) -> tuple[int, Latch]:
    """2-of-3 majority: each input at and_in (0.65x need): one input 4.6 mV, two 9.1 mV,
    three 13.7 mV against the 7 mV gap."""
    assert len(inputs) == 3
    g = net.neuron(f"{name}.maj")
    for x in inputs:
        net.synapse(x, g, drive.and_in)
    l = add_latch(net, drive, f"{name}.L")
    _ignite_from(net, drive, name, g, l)
    return g, l


class Gates:
    """Collects every internal latch and gate neuron so a reset domain can claim them."""

    def __init__(self, net: Netlist, drive: Drive):
        self.net, self.drive = net, drive
        self.latches: list[Latch] = []
        self.gates: list[int] = []

    def _and(self, name, a, b) -> Latch:
        g, l = add_and_latched(self.net, self.drive, name, [a, b])
        self.gates.append(g); self.latches.append(l)
        return l

    def _or(self, name, inputs) -> Latch:
        g, l = add_or_latched(self.net, self.drive, name, inputs)
        self.gates.append(g); self.latches.append(l)
        return l

    def _maj(self, name, a, b, c) -> Latch:
        g, l = add_maj_latched(self.net, self.drive, name, [a, b, c])
        self.gates.append(g); self.latches.append(l)
        return l

    # --- ordered (veto-relay) gates: every output is ignited by a one-shot relay driven by a
    # rail of LATE, vetoed by a rail of EARLY. EARLY must be valid >= ~6 ms before LATE rises
    # (see add_veto_relay). No rate-mode gate, no exposure window, ~5 ms after LATE.
    def latch(self, name: str) -> Latch:
        l = add_latch(self.net, self.drive, name)
        self.latches.append(l)
        return l

    def veto(self, name: str, driver: int, vetoes: list[int], target: Latch) -> int:
        return add_veto_relay(self.net, self.drive, name, driver, vetoes, target)

    def and2_ordered(self, name: str, EARLY: Rail2, LATE: Rail2) -> Rail2:
        y1, y0 = self.latch(f"{name}.y1"), self.latch(f"{name}.y0")
        self.veto(f"{name}.y1.a", LATE.r1.u, [EARLY.r0.u], y1)  # E1 and L1
        self.veto(f"{name}.y0.a", LATE.r0.u, [], y0)  # L0
        self.veto(f"{name}.y0.b", LATE.r1.u, [EARLY.r1.u], y0)  # E0 and L1  (so y0 = L0 or E0 once L is valid)
        return Rail2(y0, y1)

    def or2_ordered(self, name: str, EARLY: Rail2, LATE: Rail2) -> Rail2:
        y1, y0 = self.latch(f"{name}.y1"), self.latch(f"{name}.y0")
        self.veto(f"{name}.y1.a", LATE.r1.u, [], y1)  # L1
        self.veto(f"{name}.y1.b", LATE.r0.u, [EARLY.r0.u], y1)  # E1 and L0
        self.veto(f"{name}.y0.a", LATE.r0.u, [EARLY.r1.u], y0)  # E0 and L0
        return Rail2(y0, y1)

    def xor2_ordered(self, name: str, EARLY: Rail2, LATE: Rail2) -> Rail2:
        y1, y0 = self.latch(f"{name}.y1"), self.latch(f"{name}.y0")
        self.veto(f"{name}.y1.a", LATE.r0.u, [EARLY.r0.u], y1)  # E1 and L0
        self.veto(f"{name}.y1.b", LATE.r1.u, [EARLY.r1.u], y1)  # E0 and L1
        self.veto(f"{name}.y0.a", LATE.r1.u, [EARLY.r0.u], y0)  # E1 and L1
        self.veto(f"{name}.y0.b", LATE.r0.u, [EARLY.r1.u], y0)  # E0 and L0
        return Rail2(y0, y1)

    # --- timing staging for the ordered datapath (docs/a2_alu_register.md section 4) ------
    def delayed(self, name: str, X: Rail2, hops: int) -> Rail2:
        """Both rails of X delayed by `hops` (~5.3 ms each); the result is a pair of Taps."""
        return Rail2(Tap(add_delay_chain(self.net, self.drive, f"{name}r0", X.r0.u, hops)),
                     Tap(add_delay_chain(self.net, self.drive, f"{name}r1", X.r1.u, hops)))

    def operand_gate(self, name: str, A: list[Rail2], act_taps: list[int], act_domain_inh: int,
                     act_hops: int = 11) -> tuple[list[Rail2], Latch, int]:
        """Turns levels (a master's rails, or producer rails) into this transaction's operand
        tokens at a fixed time: ACTIVE = OR-latch over `act_taps` (rails that are live for the
        whole transaction, e.g. the unit select), living in the producer's reset domain;
        ACTIVE delayed `act_hops` drives one veto relay per rail, vetoed by the source's other
        rail. Returns (A^d rails, ACTIVE, ACTIVE^d tap)."""
        net, drive = self.net, self.drive
        act_gate, ACT = add_or_latched(net, drive, f"{name}.active", act_taps)
        q = -int(round(0.75 * drive.loop))
        for x in list(ACT.members) + [act_gate]:
            net.synapse(act_domain_inh, x, q)
        act_d = add_delay_chain(net, drive, f"{name}.actd", ACT.u, act_hops)
        out = []
        for i, a in enumerate(A):
            a0, a1 = self.latch(f"{name}.a{i}r0"), self.latch(f"{name}.a{i}r1")
            self.veto(f"{name}.a{i}r0.g", act_d, [a.r1.u], a0)
            self.veto(f"{name}.a{i}r1.g", act_d, [a.r0.u], a1)
            out.append(Rail2(a0, a1))
        return out, ACT, act_d

    def ripple_adder_ordered(self, name: str, Ad: list[Rail2], bx: list[Rail2], Cin: Rail2, carry_hops: int = 5):
        """Ripple-carry adder on veto relays only. Ordering by construction: bx (EARLY) is
        valid >= 15 ms before Ad (LATE); Cin is valid before everything (a producer rail); the
        carry into bit i >= 1 is c_{i-1} delayed `carry_hops`, so it rises >= 20 ms after
        x_i = bx_i xor Ad_i. Per bit: x, generate/kill relays (driver Ad, veto bx), propagate
        relays (driver delayed carry-in, veto x), sum = x xor carry-in (ordered).
        Returns (sums, carries, x, delayed carry-ins)."""
        n = len(Ad)
        x = [self.xor2_ordered(f"{name}.fa{i}.x", bx[i], Ad[i]) for i in range(n)]
        sums, carries, cds = [], [], []
        for i in range(n):
            c1, c0 = self.latch(f"{name}.fa{i}.c1"), self.latch(f"{name}.fa{i}.c0")
            self.veto(f"{name}.fa{i}.g", Ad[i].r1.u, [bx[i].r0.u], c1)  # generate: A=1 and bx=1
            self.veto(f"{name}.fa{i}.k", Ad[i].r0.u, [bx[i].r1.u], c0)  # kill: A=0 and bx=0
            if i == 0:
                self.veto(f"{name}.fa0.p1", x[0].r1.u, [Cin.r0.u], c1)  # propagate with cin=1
                self.veto(f"{name}.fa0.p0", x[0].r1.u, [Cin.r1.u], c0)  # propagate with cin=0
                s_ = self.xor2_ordered(f"{name}.fa0.s", Cin, x[0])
            else:
                cd = self.delayed(f"{name}.fa{i}.cin", carries[-1], carry_hops)
                self.veto(f"{name}.fa{i}.p1", cd.r1.u, [x[i].r0.u], c1)
                self.veto(f"{name}.fa{i}.p0", cd.r0.u, [x[i].r0.u], c0)
                s_ = self.xor2_ordered(f"{name}.fa{i}.s", x[i], cd)
                cds.append(cd)
            sums.append(s_)
            carries.append(Rail2(c0, c1))
        return sums, carries, x, cds

    def overflow_ordered(self, name: str, Ad: list[Rail2], bx: list[Rail2], Cin: Rail2, x, carries, cds) -> Rail2:
        """V = cout xor (carry into the top stage), without ordering cout against that carry:
        if p_top = 1 then V = 0; if generate, V = not c; if kill, V = c (c = the delayed
        carry-in of the top stage, or Cin when n = 1)."""
        n = len(Ad)
        v1, v0 = self.latch(f"{name}1"), self.latch(f"{name}0")
        At, bt = Ad[n - 1], bx[n - 1]
        self.veto(f"{name}0.p", x[n - 1].r1.u, [], v0)
        if n >= 2:
            cd = cds[-1]
            self.veto(f"{name}1.k", cd.r1.u, [At.r1.u, bt.r1.u], v1)  # kill and c=1
            self.veto(f"{name}1.g", cd.r0.u, [At.r0.u, bt.r0.u], v1)  # generate and c=0
            self.veto(f"{name}0.k", cd.r0.u, [At.r1.u, bt.r1.u], v0)
            self.veto(f"{name}0.g", cd.r1.u, [At.r0.u, bt.r0.u], v0)
        else:  # Cin is EARLY: drive from Ad instead
            self.veto(f"{name}1.k", At.r0.u, [bt.r1.u, Cin.r0.u], v1)
            self.veto(f"{name}1.g", At.r1.u, [bt.r0.u, Cin.r1.u], v1)
            self.veto(f"{name}0.k", At.r0.u, [bt.r1.u, Cin.r1.u], v0)
            self.veto(f"{name}0.g", At.r1.u, [bt.r0.u, Cin.r0.u], v0)
        return Rail2(v0, v1)

    def and2(self, name: str, A: Rail2, B: Rail2) -> Rail2:
        y1 = self._and(f"{name}.y1", A.r1.u, B.r1.u)
        y0 = self._or(f"{name}.y0", [A.r0.u, B.r0.u])
        return Rail2(y0, y1)

    def or2(self, name: str, A: Rail2, B: Rail2) -> Rail2:
        y1 = self._or(f"{name}.y1", [A.r1.u, B.r1.u])
        y0 = self._and(f"{name}.y0", A.r0.u, B.r0.u)
        return Rail2(y0, y1)

    def xor2(self, name: str, A: Rail2, B: Rail2) -> Rail2:
        p = self._and(f"{name}.p", A.r1.u, B.r0.u)
        q = self._and(f"{name}.q", A.r0.u, B.r1.u)
        y1 = self._or(f"{name}.y1", [p.u, q.u])
        r = self._and(f"{name}.r", A.r1.u, B.r1.u)
        s = self._and(f"{name}.s", A.r0.u, B.r0.u)
        y0 = self._or(f"{name}.y0", [r.u, s.u])
        return Rail2(y0, y1)

    def maj3(self, name: str, A: Rail2, B: Rail2, C: Rail2) -> Rail2:
        y1 = self._maj(f"{name}.y1", A.r1.u, B.r1.u, C.r1.u)
        y0 = self._maj(f"{name}.y0", A.r0.u, B.r0.u, C.r0.u)
        return Rail2(y0, y1)

    def full_adder(self, name: str, A: Rail2, B: Rail2, C: Rail2) -> tuple[Rail2, Rail2]:
        x = self.xor2(f"{name}.x", A, B)
        s = self.xor2(f"{name}.s", x, C)
        c = self.maj3(f"{name}.c", A, B, C)
        return s, c

    def ripple_adder(self, name: str, A: list[Rail2], B: list[Rail2], Cin: Rail2) -> tuple[list[Rail2], Rail2]:
        c = Cin
        sums = []
        for i, (a, b) in enumerate(zip(A, B)):
            s, c = self.full_adder(f"{name}.fa{i}", a, b, c)
            sums.append(s)
        return sums, c
