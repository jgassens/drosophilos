"""A2 liveness primitives: a neural watchdog (timeout) and a stale-state monitor.

Watchdog: a delay chain started when the producer's register becomes active and cancelled
by ACCEPT or FAULT-ACCEPT. If it runs to the end it ignites a TIMEOUT latch that raises
FAULT-ACCEPT itself, so a transaction that never completes is refused by the machine
instead of hanging; the chain is a bounded-delay timer (~5.3 ms per hop).

Stale-state monitor: an ENABLE latch is ignited by the last pulse of a register's reset
train and cleared by that register's READY. While ENABLE holds, a rate-mode AND between it
and every latch tap of the register fires if any latch survived the train; its output is
latched (STALE), which (a) holds the READY generator down and (b) re-triggers the reset,
so READY is only issued once the register is really empty. Both latches are neural events
the harness can read, which turns "the harness noticed" into "the machine detected".
"""

from __future__ import annotations

from dataclasses import dataclass

from ..lib.netlist import Drive, Netlist
from .celement import _ignite_from
from .latch import Latch, add_edge_relay, add_latch, connect_trigger


@dataclass
class Watchdog:
    start_gate: int
    hops: list
    cancel_inh: int
    timeout: Latch


def add_watchdog(net: Netlist, drive: Drive, name: str, start_taps: list[int], cancel_taps: list[int],
                 hops: int, trigger: int, trigger_edge: int, reset_inh: int) -> Watchdog:
    """Timer started by any of `start_taps` (latch trains), cancelled by any of `cancel_taps`
    (latch trains), that raises FAULT-ACCEPT through (`trigger`, `trigger_edge`) after
    `hops` relays. `reset_inh` clears the TIMEOUT latch with the register's reset train."""
    g = net.neuron(f"{name}.start")
    for t in start_taps:
        net.synapse(t, g, drive.or_in)
    relay = add_edge_relay(net, drive, f"{name}.start", g)  # one start pulse per activation
    chain = []
    prev = relay
    for k in range(hops):
        h = net.neuron(f"{name}.hop{k}")
        net.synapse(prev, h, drive.pulse)
        chain.append(h)
        prev = h
    T = add_latch(net, drive, f"{name}.timeout")
    net.synapse(prev, T.u, drive.ignite)
    cancel = net.neuron(f"{name}.cancel_inh")
    for t in cancel_taps:
        net.synapse(t, cancel, drive.pulse)
    q = -int(round(1.5 * drive.loop))
    for h in chain:
        net.synapse(cancel, h, q)
    for x in T.members:
        net.synapse(cancel, x, q)
    connect_trigger(net, drive, T.u, trigger, trigger_edge)  # TIMEOUT is a FAULT-ACCEPT source
    for x in T.members:
        net.synapse(reset_inh, x, -int(round(0.75 * drive.loop)))
    return Watchdog(g, chain, cancel, T)


@dataclass
class StaleMonitor:
    enable: Latch
    gate: int
    stale: Latch
    block_inh: int


def add_stale_monitor(net: Netlist, drive: Drive, name: str, latch_taps: list[int], last_reset_relay: int,
                      ready_out: int, ready_chain_tail: list[int], trigger: int, trigger_edge: int,
                      reset_inh: int) -> StaleMonitor:
    E = add_latch(net, drive, f"{name}.enable")
    # two hops after the last relay: the reset's inhibitory neuron emits a trailing doublet
    # (~+27 ms) that killed ENABLE when it was ignited one hop after the last relay (observed)
    arm0 = net.neuron(f"{name}.arm0")
    arm = net.neuron(f"{name}.arm")
    net.synapse(last_reset_relay, arm0, drive.pulse)
    net.synapse(arm0, arm, drive.pulse)
    net.synapse(arm, E.u, drive.ignite)
    # READY disarms the monitor with a two-pulse 1.5x train: one pulse killed a nominal loop
    # at every phase but not always a +10 % loop, and an ENABLE that survives into the next
    # transaction flags its legitimately live latches as stale (observed: false retries,
    # READY hangs and, through a mid-transaction reset, wrong values)
    clear = net.neuron(f"{name}.enable_clear")
    clear_hop = net.neuron(f"{name}.enable_clear_hop")
    net.synapse(ready_out, clear, drive.pulse)
    net.synapse(ready_out, clear_hop, drive.pulse)
    net.synapse(clear_hop, clear, drive.pulse)
    for x in E.members:
        net.synapse(clear, x, drive.reset)
    # one two-input rate-mode AND per latch: fires only if ENABLE holds AND that latch is alive.
    # (A single AND summing every tap fires during normal operation, when several latches
    # are alive at once, and re-triggered the reset mid-transaction: observed.)
    gate = net.neuron(f"{name}.any")  # OR over the per-latch detectors
    for k, t in enumerate(latch_taps):
        d = net.neuron(f"{name}.d{k}")
        net.synapse(E.u, d, drive.and_in)
        net.synapse(t, d, drive.and_in)
        net.synapse(d, gate, drive.pulse)
    S = add_latch(net, drive, f"{name}.stale")
    _ignite_from(net, drive, f"{name}.stale", gate, S)
    block = net.neuron(f"{name}.block_inh")  # STALE holds READY and the chain tail down
    net.synapse(S.u, block, drive.pulse)
    q = -int(round(1.5 * drive.loop))
    net.synapse(block, ready_out, q)
    for h in ready_chain_tail:
        net.synapse(block, h, q)
    connect_trigger(net, drive, S.u, trigger, trigger_edge)  # ...and re-runs the reset
    for x in list(S.members) + list(E.members):
        net.synapse(reset_inh, x, -int(round(0.75 * drive.loop)))  # cleared by the (re)reset train
    return StaleMonitor(E, gate, S, block)
