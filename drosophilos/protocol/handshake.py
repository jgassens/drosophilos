"""One four-phase dual-rail channel: producer register -> consumer register.

    DATA     producer rail latch trains drive consumer rail latches (pulse-mode ignition)
    VALIDATE consumer bit_valid latches -> completion tree -> root latch W (state-holding)
    ACCEPT   W drives the producer's reset trigger
    CLEARED  producer's READY generator (vetoed by its rail latches) drives the consumer's reset
    READY    consumer's READY generator (vetoed by its rail latches and W) -> output port

The harness loads the producer by injecting one pulse into each active rail's `u`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..lib.netlist import Drive, Netlist
from ..sim.model import Params
from .celement import add_and_gate, add_completion_tree, add_or_latched
from .latch import Latch, add_edge_relay, add_latch, add_ready, add_reset, connect_trigger
from .watchdog import StaleMonitor, Watchdog, add_stale_monitor, add_watchdog


@dataclass
class Register:
    rails: list  # rails[i][r] -> Latch
    reset_trigger: int
    reset_inh: int
    ready: int
    valid: list = field(default_factory=list)  # bit_valid latches (consumer only)
    fault: list = field(default_factory=list)  # fault gate neurons (consumer only)
    completion: Latch | None = None
    internal: list = field(default_factory=list)
    reset_edge: int = -1
    fault_latch: Latch | None = None
    last_reset_relay: int = -1
    ready_chain: list = field(default_factory=list)
    monitor: StaleMonitor | None = None
    watchdog: Watchdog | None = None

    @property
    def rail_taps(self) -> list[list[int]]:
        return [[self.rails[i][0].u, self.rails[i][1].u] for i in range(len(self.rails))]

    def all_latches(self) -> list[Latch]:
        out = [l for pair in self.rails for l in pair] + list(self.valid) + list(self.internal)
        if self.completion is not None:
            out.append(self.completion)
        return out


def add_register(net: Netlist, drive: Drive, name: str, width: int, with_completion: bool) -> Register:
    rails = [[add_latch(net, drive, f"{name}.b{i}r{r}") for r in (0, 1)] for i in range(width)]
    valid, fault, internal, completion, gates = [], [], [], None, []
    if with_completion:
        for i in range(width):
            g, lv = add_or_latched(net, drive, f"{name}.valid{i}", [rails[i][0].u, rails[i][1].u])
            valid.append(lv)
            gates.append(g)
            fault.append(add_and_gate(net, drive, f"{name}.fault{i}", [rails[i][0].u, rails[i][1].u], fraction=0.55))
        completion, internal = add_completion_tree(net, drive, f"{name}.comp", valid)
        gates += [x for x, role in enumerate(net.roles) if role.startswith(f"{name}.comp.") and role.endswith(".and")]
        if completion in valid:  # width 1: the valid latch is the completion latch
            internal = []
    reg = Register(rails, -1, -1, -1, valid, fault, completion, internal)
    latches = reg.all_latches()
    trig, inh, edge = add_reset(net, drive, name, latches, gates)
    ready = add_ready(net, drive, name, trig, hops=15)  # ~80 ms: leaves the stale monitor ~19 ms to land its block
    reg.reset_trigger, reg.reset_inh, reg.ready, reg.reset_edge = trig, inh, ready, edge
    reg.last_reset_relay = max(x for x, r in enumerate(net.roles) if r.startswith(f"{name}.reset_relay"))
    reg.ready_chain = [x for x, r in enumerate(net.roles) if r.startswith(f"{name}.ready_delay")]
    net.group(f"{name}.rail_taps", [t for pair in reg.rail_taps for t in pair])
    return reg


@dataclass
class Channel:
    net: Netlist
    drive: Drive
    width: int
    producer: Register
    consumer: Register

    @property
    def accept(self) -> int:
        return self.consumer.completion.u

    @property
    def cleared(self) -> int:
        return self.producer.ready

    @property
    def ready(self) -> int:
        return self.consumer.ready


def wire_fault_path(net: Netlist, drive: Drive, P: Register, Q: Register) -> Latch:
    """FAULT as a state (spec.md §3): any fault gate ignites one shared fault latch F, which
    (a) holds the completion latch, its AND gate and its ignition relay down so the word is
    never consumed, and (b) raises FAULT-ACCEPT once through the producer's edge-detected
    reset trigger so the four phases still complete. F is reset with the consumer.
    Unlatched fault-gate spikes (~every 36 ms while both rails hold) re-armed the edge
    detector and fired a second reset that wiped the next word (observed)."""
    F = add_latch(net, drive, "Q.faultL")
    for f in Q.fault:
        net.synapse(f, F.u, drive.ignite)
    root = net.roles[Q.completion.u][: -len(".L.u")]
    held = list(Q.completion.members) + [x for x, role in enumerate(net.roles) if role in (f"{root}.and", f"{root}.ign.edge")]
    for x in held:
        net.synapse(F.u, x, drive.reset)
    connect_trigger(net, drive, F.u, P.reset_trigger, P.reset_edge)
    q = -int(round(0.75 * drive.loop))
    for x in F.members:
        net.synapse(Q.reset_inh, x, q)
    Q.fault_latch = F
    return F


def add_liveness(net: Netlist, drive: Drive, P: Register, Q: Register, watchdog_hops: int, monitor: bool = False) -> None:
    """A2: watchdog on the producer (started by its rails, cancelled by ACCEPT / FAULT-ACCEPT).
    The stale-state monitor is available but off by default: rejected on measurement. Its
    per-latch detector (a two-input rate-mode AND, tap + ENABLE) sits at 75 % of threshold on
    the tap alone for ~100 ms per transaction across a dozen latches, and one spurious spike
    ignites STALE; at mix B that produced false retries in ~2 % of transactions, READY hangs,
    and, through mid-transaction resets, wrong values. A safe fraction (0.55 + 0.55) detects
    too slowly to beat READY. Stale state stays harness-observed (and rare: 8 per 10^6)."""
    P.watchdog = add_watchdog(net, drive, "P.wd", [l.u for pair in P.rails for l in pair],
                              [Q.completion.u, Q.fault_latch.u], watchdog_hops, P.reset_trigger, P.reset_edge, P.reset_inh)
    if monitor:
        for name, reg in (("P", P), ("Q", Q)):
            taps = [l.u for l in reg.all_latches()] + ([reg.fault_latch.u] if reg.fault_latch is not None else [])
            reg.monitor = add_stale_monitor(net, drive, f"{name}.mon", taps, reg.last_reset_relay, reg.ready,
                                            reg.ready_chain[-3:], reg.reset_trigger, reg.reset_edge, reg.reset_inh)


def build_channel(params: Params, width: int, drive: Drive | None = None, liveness: bool = True,
                  watchdog_hops: int = 40, monitor: bool = False) -> Channel:
    drive = drive or Drive.from_params(params)
    net = Netlist(params)
    P = add_register(net, drive, "P", width, with_completion=False)
    Q = add_register(net, drive, "Q", width, with_completion=True)
    for i in range(width):
        for r in (0, 1):
            relay = add_edge_relay(net, drive, f"data.b{i}r{r}", P.rails[i][r].u)
            net.synapse(relay, Q.rails[i][r].u, drive.ignite)  # DATA: one ignition pulse per rail
    connect_trigger(net, drive, Q.completion.u, P.reset_trigger, P.reset_edge)  # ACCEPT (edge)
    connect_trigger(net, drive, P.ready, Q.reset_trigger, Q.reset_edge)  # CLEARED (edge)
    # FAULT (both rails of a bit active): block the completion latch so the word is never
    # consumed, and raise FAULT-ACCEPT so the four phases still complete and the channel
    # does not deadlock (spec.md §3). A fault after completion is a flag only.
    wire_fault_path(net, drive, P, Q)
    if liveness:
        add_liveness(net, drive, P, Q, watchdog_hops, monitor)
    net.group("accept", [Q.completion.u])
    net.group("cleared", [P.ready])
    net.group("ready", [Q.ready])
    return Channel(net, drive, width, P, Q)
