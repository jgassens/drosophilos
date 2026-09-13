"""Latched completion elements.

Every gate here reads only latch trains (a standard 213 Hz rate), and every gate output
that feeds another gate is latched, so rate-mode weights never depend on an upstream
gate's firing rate. A completion element therefore is: gate neuron -> latch.
  bit_valid(i)  : OR of the bit's two rails, latched      (rate-mode OR, or_in per rail)
  fault(i)      : AND of the bit's two rails              (rate-mode AND, and_in per rail)
  c2(a, b)      : AND of two latched valids, latched      (rate-mode AND, and_in each)
A word's completion is a binary tree of c2 over its bit_valid latches; its root latch is
the state-holding element: set when all inputs agree (all valid), cleared only by RESET.
"""

from __future__ import annotations

from ..lib.netlist import Drive, Netlist
from .latch import Latch, add_edge_relay, add_latch


def _ignite_from(net: Netlist, drive: Drive, name: str, gate: int, latch: Latch) -> None:
    """Gate -> latch through an edge relay: one ignition pulse per gate activation. A gate
    that kept pulsing the latch would add a second drive on top of the loop and let the
    latch ride through the reset train (observed on valid and tree latches)."""
    relay = add_edge_relay(net, drive, f"{name}.ign", gate)
    net.synapse(relay, latch.u, drive.ignite)


def add_or_latched(net: Netlist, drive: Drive, name: str, inputs: list[int]) -> tuple[int, Latch]:
    g = net.neuron(f"{name}.or")
    for x in inputs:
        net.synapse(x, g, drive.or_in)
    l = add_latch(net, drive, f"{name}.L")
    _ignite_from(net, drive, name, g, l)
    return g, l


def add_and_latched(net: Netlist, drive: Drive, name: str, inputs: list[int]) -> tuple[int, Latch]:
    assert len(inputs) == 2, "rate-mode AND with margin is a 2-input gate; build a tree"
    g = net.neuron(f"{name}.and")
    for x in inputs:
        net.synapse(x, g, drive.and_in)
    l = add_latch(net, drive, f"{name}.L")
    _ignite_from(net, drive, name, g, l)
    return g, l


def add_and_gate(net: Netlist, drive: Drive, name: str, inputs: list[int]) -> int:
    assert len(inputs) == 2
    g = net.neuron(f"{name}.and")
    for x in inputs:
        net.synapse(x, g, drive.and_in)
    return g


def add_completion_tree(net: Netlist, drive: Drive, name: str, valid_latches: list[Latch]) -> tuple[Latch, list[Latch]]:
    """Binary tree of latched 2-input ANDs. Returns (root latch, all internal latches)."""
    level = list(valid_latches)
    internal: list[Latch] = []
    depth = 0
    while len(level) > 1:
        nxt = []
        for k in range(0, len(level) - 1, 2):
            _, l = add_and_latched(net, drive, f"{name}.c{depth}_{k // 2}", [level[k].u, level[k + 1].u])
            internal.append(l)
            nxt.append(l)
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
        depth += 1
    return level[0], internal
