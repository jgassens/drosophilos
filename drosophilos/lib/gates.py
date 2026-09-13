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

from ..protocol.celement import _ignite_from, add_and_latched, add_or_latched
from ..protocol.latch import Latch, add_latch
from .netlist import Drive, Netlist


@dataclass(frozen=True)
class Rail2:
    r0: Latch
    r1: Latch

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
