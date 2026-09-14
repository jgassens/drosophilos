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
    latch ride through the reset train (observed on valid and tree latches). The ignited
    latch's train holds the relay down (hold_from): a gate source is too slow to do that
    itself, and a re-firing relay re-ignites the latch every ~43 ms (see add_edge_relay)."""
    relay = add_edge_relay(net, drive, f"{name}.ign", gate, hold_from=[latch.u])
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


def add_and_gate(net: Netlist, drive: Drive, name: str, inputs: list[int], fraction: float | None = None) -> int:
    """Unlatched rate-mode AND. `fraction` overrides drive.and_in (as a fraction of the
    sustained-train need); the fault gates use 0.6 so that one rail plus noise never trips
    them, while completion ANDs use 0.75 so that two inputs fire under -10 % weights."""
    assert len(inputs) == 2
    g = net.neuron(f"{name}.and")
    q = drive.and_in if fraction is None else int(round(fraction * drive.rate_need))
    for x in inputs:
        net.synapse(x, g, q)
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


def add_veto_relay(net: Netlist, drive: Drive, name: str, driver: int, vetoes: list[int], target: Latch,
                   veto_strength: float = 0.5) -> int:
    """driver AND NOT(any veto), evaluated once at the driver's rise, as a one-shot ignition of
    `target`. The driver is a latch train (so the relay's own feed-forward inhibition and the
    target's train keep it to one pulse); each veto is a latch train that, while live, holds
    the relay ~146 mV below rest through one inhibitory interneuron, against which the 12.6 mV
    driver pulse cannot fire it. There is no exposure window: unlike a rate-mode AND, the
    relay never integrates one input towards threshold.

    Ordering assumption (bounded delay, stated per user): every veto rail must be established
    >= ~15 ms before the driver rises (0.5x loop per veto spike: ~33 mV sustained, ~3 spikes
    to block; it recovers in ~55 ms once the veto rail dies, so a vetoed relay may be driven
    again 55 ms later. 2.2x would block on one spike but paralyse the relay for ~90 ms).
    Dual-rail supplies NOT for free: to compute a AND b with b the earlier operand, veto with
    b's other rail. No hold from the target is needed: the driver is a latch train, which the
    relay's own source inhibition already turns into one shot."""
    relay = add_edge_relay(net, drive, name, driver)
    if vetoes:
        v = net.neuron(f"{name}.veto")
        for t in vetoes:
            net.synapse(t, v, drive.pulse)
        net.synapse(v, relay, -int(round(veto_strength * drive.loop)))
    net.synapse(relay, target.u, drive.ignite)
    return relay
