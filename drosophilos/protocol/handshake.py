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
            fault.append(add_and_gate(net, drive, f"{name}.fault{i}", [rails[i][0].u, rails[i][1].u]))
        completion, internal = add_completion_tree(net, drive, f"{name}.comp", valid)
        gates += [x for x, role in enumerate(net.roles) if role.startswith(f"{name}.comp.") and role.endswith(".and")]
        if completion in valid:  # width 1: the valid latch is the completion latch
            internal = []
    reg = Register(rails, -1, -1, -1, valid, fault, completion, internal)
    latches = reg.all_latches()
    trig, inh, edge = add_reset(net, drive, name, latches, gates)
    ready = add_ready(net, drive, name, trig)
    reg.reset_trigger, reg.reset_inh, reg.ready, reg.reset_edge = trig, inh, ready, edge
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


def build_channel(params: Params, width: int, drive: Drive | None = None) -> Channel:
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
    net.group("accept", [Q.completion.u])
    net.group("cleared", [P.ready])
    net.group("ready", [Q.ready])
    return Channel(net, drive, width, P, Q)
