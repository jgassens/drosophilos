"""A ripple-carry adder as a four-phase channel: producer register (A, B, carry-in) ->
combinational dual-rail adder (latched stages) -> consumer register (n+1 bits) with
completion tree. Adder-internal latches and gates belong to the consumer's reset domain."""

from __future__ import annotations

from ..protocol.handshake import Channel, add_register
from ..protocol.latch import add_edge_relay, connect_trigger
from ..sim.model import Params
from .gates import Gates, Rail2
from .netlist import Drive, Netlist


def extend_reset(net: Netlist, drive: Drive, reg, latches, gates, strength: float = 0.5) -> None:
    q = -int(round(strength * drive.loop))
    for l in latches:
        for x in l.members:
            net.synapse(reg.reset_inh, x, q)
    for g in gates:
        net.synapse(reg.reset_inh, g, q)


def operand_word(a: int, b: int, cin: int, width: int) -> int:
    """Producer word layout: bits [0,n) = A, [n,2n) = B, bit 2n = carry-in."""
    return (a & ((1 << width) - 1)) | ((b & ((1 << width) - 1)) << width) | ((cin & 1) << (2 * width))


def build_adder_channel(params: Params, width: int, drive: Drive | None = None) -> Channel:
    drive = drive or Drive.from_params(params)
    net = Netlist(params)
    P = add_register(net, drive, "P", 2 * width + 1, with_completion=False)
    Q = add_register(net, drive, "Q", width + 1, with_completion=True)
    A = [Rail2(P.rails[i][0], P.rails[i][1]) for i in range(width)]
    B = [Rail2(P.rails[width + i][0], P.rails[width + i][1]) for i in range(width)]
    Cin = Rail2(P.rails[2 * width][0], P.rails[2 * width][1])
    G = Gates(net, drive)
    sums, cout = G.ripple_adder("add", A, B, Cin)
    outputs = sums + [cout]
    for i, s in enumerate(outputs):
        for r, latch in enumerate(s.latches):
            relay = add_edge_relay(net, drive, f"out.b{i}r{r}", latch.u)
            net.synapse(relay, Q.rails[i][r].u, drive.ignite)
    extend_reset(net, drive, Q, G.latches, G.gates)
    connect_trigger(net, drive, Q.completion.u, P.reset_trigger, P.reset_edge)  # ACCEPT
    connect_trigger(net, drive, P.ready, Q.reset_trigger, Q.reset_edge)  # CLEARED
    for f in Q.fault:
        for x in Q.completion.members:
            net.synapse(f, x, drive.reset)
        connect_trigger(net, drive, f, P.reset_trigger, P.reset_edge)
    net.group("adder_latches", [x for l in G.latches for x in l.members])
    return Channel(net, drive, 2 * width + 1, P, Q)
