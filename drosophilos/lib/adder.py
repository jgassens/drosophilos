"""A ripple-carry adder as a four-phase channel: producer register (A, B, carry-in) ->
combinational dual-rail adder (latched stages) -> consumer register (n+1 bits) with
completion tree. Adder-internal latches and gates belong to the consumer's reset domain."""

from __future__ import annotations

from ..protocol.handshake import Channel, add_liveness, add_register, wire_fault_path
from ..protocol.latch import add_edge_relay, connect_trigger
from ..sim.model import Params
from .gates import Gates, Rail2
from .netlist import Drive, Netlist


def extend_reset(net: Netlist, drive: Drive, reg, latches, gates, strength: float = 0.75) -> None:
    """Adder-internal latches join the consumer's reset domain at the same 0.75x per pulse as
    every other latch (0.5x, the old default, let +10 % loops survive: stale carries and XOR
    stages, wrong sums and both-rail faults in ~25 % of perturbed additions)."""
    q = -int(round(strength * drive.loop))
    for l in latches:
        for x in l.members:
            net.synapse(reg.reset_inh, x, q)
    for g in gates:
        net.synapse(reg.reset_inh, g, q)


def operand_word(a: int, b: int, cin: int, width: int) -> int:
    """Producer word layout: bits [0,n) = A, [n,2n) = B, bit 2n = carry-in."""
    return (a & ((1 << width) - 1)) | ((b & ((1 << width) - 1)) << width) | ((cin & 1) << (2 * width))


def build_adder_channel(params: Params, width: int, drive: Drive | None = None, liveness: bool = True,
                        watchdog_hops: int = 80, ordered: bool = False) -> Channel:
    """`ordered=True`: the veto-relay adder (operand gate on A driven by the carry-in rails,
    B delayed 6 hops, carries delayed 5 hops per stage); `False`: the rate-mode M1 adder."""
    drive = drive or Drive.from_params(params)
    net = Netlist(params)
    P = add_register(net, drive, "P", 2 * width + 1, with_completion=False)
    Q = add_register(net, drive, "Q", width + 1, with_completion=True)
    A = [Rail2(P.rails[i][0], P.rails[i][1]) for i in range(width)]
    B = [Rail2(P.rails[width + i][0], P.rails[width + i][1]) for i in range(width)]
    Cin = Rail2(P.rails[2 * width][0], P.rails[2 * width][1])
    G = Gates(net, drive)
    if ordered:
        Ad, ACT, _ = G.operand_gate("add", A, [Cin.r0.u, Cin.r1.u], P.reset_inh)
        Bd = [G.delayed(f"add.b{i}", B[i], 6) for i in range(width)]
        sums, carries, _, _ = G.ripple_adder_ordered("add", Ad, Bd, Cin)
        cout = carries[-1]
    else:
        sums, cout = G.ripple_adder("add", A, B, Cin)
    outputs = sums + [cout]
    for i, s in enumerate(outputs):
        for r, latch in enumerate(s.latches):
            relay = add_edge_relay(net, drive, f"out.b{i}r{r}", latch.u)
            net.synapse(relay, Q.rails[i][r].u, drive.ignite)
    extend_reset(net, drive, Q, G.latches, G.gates)
    connect_trigger(net, drive, Q.completion.u, P.reset_trigger, P.reset_edge)  # ACCEPT
    connect_trigger(net, drive, P.ready, Q.reset_trigger, Q.reset_edge)  # CLEARED
    wire_fault_path(net, drive, P, Q)
    if liveness:
        add_liveness(net, drive, P, Q, watchdog_hops)
    net.group("adder_latches", [x for l in G.latches for x in l.members])
    return Channel(net, drive, 2 * width + 1, P, Q)
