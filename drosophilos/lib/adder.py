"""A ripple-carry adder as a four-phase channel: producer register (A, B, carry-in) ->
combinational dual-rail adder (latched stages) -> consumer register (n+1 bits) with
completion tree. Adder-internal latches and gates belong to the consumer's reset domain."""

from __future__ import annotations

from dataclasses import replace

from ..protocol.handshake import Channel, add_liveness, add_register, wire_fault_path
from ..protocol.latch import (DALE_FEEDBACK_VERSION, EXPERIMENTAL_AUTAPSE_VERSION,
                              add_dale_latch_feedback, add_edge_relay,
                              add_experimental_autapses, connect_trigger,
                              validate_latch_feedback)
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
                        watchdog_hops: int = 80, ordered: bool = False, storage: str = "latch", *,
                        rate_robust: bool = False, experimental_autapse: bool = False,
                        ordered_xor_oneshot: bool = False, latch_feedback: str = "autapse",
                        dale_feedback_version: int = DALE_FEEDBACK_VERSION) -> Channel:
    """`ordered=True`: the veto-relay adder (operand gate on A driven by the carry-in rails,
    B delayed 6 hops, carries delayed 5 hops per stage); `False`: the rate-mode M1 adder.
    `storage` is the output register's rail storage ("latch" or "flipflop", see add_register;
    the sum relays then fire each rail's set chain through `Q.rail_inputs`). The operand
    register stays a latch register: the harness loads it with one pulse per rail.
    Configuration F enables rate-robust readers, experimental latch feedback and
    one-shot sum XORs together. One-shot XORs require ordered mode and the kernel's
    default physics/drive and widths 1..8; the adder retains its own reset timing.
    `latch_feedback='dale_private'` replaces only the experimental latch self-edges
    with versioned private inhibitory relays; the one-shot self-edge remains."""
    validate_latch_feedback(latch_feedback, experimental_autapse, dale_feedback_version)
    if ordered_xor_oneshot and not (rate_robust and experimental_autapse):
        raise ValueError("ordered_xor_oneshot requires rate_robust=True and experimental_autapse=True")
    if ordered_xor_oneshot and not ordered:
        raise ValueError("ordered_xor_oneshot requires ordered=True")
    if ordered_xor_oneshot and (params != Params() or not 1 <= width <= 8 or
                                drive not in (None, Drive.from_params(params))):
        raise ValueError("ordered_xor_oneshot supports only default physics/drive and widths 1..8")
    if experimental_autapse and not rate_robust:
        raise ValueError("experimental_autapse requires rate_robust=True")
    drive = drive or Drive.from_params(params)
    if rate_robust:
        drive = replace(drive, rate_robust=True)
    net = Netlist(params)
    P = add_register(net, drive, "P", 2 * width + 1, with_completion=False)
    Q = add_register(net, drive, "Q", width + 1, with_completion=True, storage=storage)
    A = [Rail2(P.rails[i][0], P.rails[i][1]) for i in range(width)]
    B = [Rail2(P.rails[width + i][0], P.rails[width + i][1]) for i in range(width)]
    Cin = Rail2(P.rails[2 * width][0], P.rails[2 * width][1])
    G = Gates(net, drive, ordered_xor_oneshot=ordered_xor_oneshot)
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
            net.synapse(relay, Q.rail_inputs[i][r], drive.ignite)  # a latch's u; a flip-flop's set-chain trigger
    extend_reset(net, drive, Q, G.latches, G.gates)
    connect_trigger(net, drive, Q.completion.u, P.reset_trigger, P.reset_edge)  # ACCEPT
    connect_trigger(net, drive, P.ready, Q.reset_trigger, Q.reset_edge)  # CLEARED
    wire_fault_path(net, drive, P, Q)
    if liveness:
        add_liveness(net, drive, P, Q, watchdog_hops)
    net.group("adder_latches", [x for l in G.latches for x in l.members])
    if experimental_autapse:
        if latch_feedback == "autapse":
            add_experimental_autapses(net, drive)
        else:
            add_dale_latch_feedback(net, drive)
    channel = Channel(net, drive, 2 * width + 1, P, Q)
    channel.build_options = dict(rate_robust=rate_robust,
                                 experimental_autapse=experimental_autapse,
                                 experimental_autapse_version=EXPERIMENTAL_AUTAPSE_VERSION,
                                 ordered_xor_oneshot=ordered_xor_oneshot)
    if latch_feedback != "autapse":
        channel.build_options.update(latch_feedback=latch_feedback,
                                     dale_feedback_version=DALE_FEEDBACK_VERSION)
    return channel
