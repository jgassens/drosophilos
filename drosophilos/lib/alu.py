"""Dual-rail ALU on the rate-mode gate library, as a four-phase channel (plan §A2).

Operand word (producer P, 2n+6 bits):
    A[n] | B[n] | U[6] one-hot unit select (ADDER, AND, OR, XOR, PASSB, MUL) | SUB
Result word (consumer Q, n+3 bits):
    R[n] | C | Z | V
        ADDER : R = A + (B xor SUB) + SUB, i.e. A+B when SUB=0 and A-B (mod 2^n) when SUB=1.
                C = carry out (for SUB: C=1 means no borrow, A >= B unsigned).
                V = signed overflow = C xor (carry into the top stage).
        AND / OR / XOR : bitwise. PASSB : R = B (MOV). MUL : low n bits of A x B (optional unit).
        C = V = 0 for all but the adder.
        Z = 1 iff R == 0, for every unit.

Structure. Every unit computes on every transaction (all operands are always valid
tokens, so every unit completes); a one-hot mux picks the result: per result rail, one
veto relay per unit (driven by that unit's rail, vetoed by the unit's deselect rail) into
one result latch. An unselected unit's outputs therefore never reach the result rails.
The whole datapath is veto relays (protocol/celement.py::add_veto_relay) whose input order
is fixed by delay chains: B is delayed before bx = SUB xor B, A enters through an operand
gate ~70 ms after the load, and each carry is delayed before the next stage reads it. No
rate-mode gate remains between the operands and the result (see add_alu_logic).

ISA mapping (isa/semantics.md): ADD = ADD.WRAP, SUB = SUB.WRAP with OVF = V; the ISA's
compare results derive from SUB's flags (unsigned: lt = not C, eq = Z; signed: lt = Z==0
and (N xor V), N = R's top bit). Saturating ops, shifts and multiply are later macros.
"""

from __future__ import annotations

from ..protocol.handshake import Channel, add_liveness, add_register, wire_fault_path
from ..protocol.celement import add_delay_chain, add_veto_relay
from ..protocol.latch import Latch
from ..protocol.latch import add_edge_relay, connect_trigger
from ..sim.model import Params
from .adder import extend_reset
from .gates import Gates, Rail2
from .netlist import Drive, Netlist

UNITS = ("ADDER", "AND", "OR", "XOR", "PASSB", "MUL")
OPS = {"ADD": (0, 0), "SUB": (0, 1), "AND": (1, 0), "OR": (2, 0), "XOR": (3, 0), "MOV": (4, 0), "MUL": (5, 0)}  # op -> (unit, sub)
N_UNITS = len(UNITS)


def alu_reference(a: int, b: int, op: str, width: int) -> dict:
    """Bit-exact reference for one ALU operation on unsigned n-bit patterns. Returns the
    result bits and flags and the packed consumer word (R | C<<n | Z<<n+1 | V<<n+2)."""
    mask = (1 << width) - 1
    a &= mask
    b &= mask
    c = v = 0
    if op in ("ADD", "SUB"):
        b_eff = b if op == "ADD" else (~b & mask)
        full = a + b_eff + (1 if op == "SUB" else 0)
        r = full & mask
        c = (full >> width) & 1
        sa, sb, sr = (a >> (width - 1)) & 1, (b_eff >> (width - 1)) & 1, (r >> (width - 1)) & 1
        v = int(sa == sb and sr != sa)
    elif op == "AND":
        r = a & b
    elif op == "OR":
        r = a | b
    elif op == "XOR":
        r = a ^ b
    elif op == "MOV":
        r = b
    elif op == "MUL":
        r = (a * b) & mask  # MUL.WRAP: the low n bits
    else:
        raise ValueError(op)
    z = int(r == 0)
    return {"r": r, "c": c, "z": z, "v": v, "word": r | (c << width) | (z << (width + 1)) | (v << (width + 2))}


def alu_word(a: int, b: int, op: str, width: int) -> int:
    """Producer word layout: bits [0,n) = A, [n,2n) = B, [2n,2n+5) = one-hot unit, bit 2n+5 = SUB."""
    unit, sub = OPS[op]
    mask = (1 << width) - 1
    return (a & mask) | ((b & mask) << width) | (1 << (2 * width + unit)) | (sub << (2 * width + N_UNITS))


def alu_operand_word(b: int, op: str, width: int) -> int:
    """Accumulator form (A comes from the master register): bits [0,n) = B, [n,n+5) = unit, bit n+5 = SUB."""
    unit, sub = OPS[op]
    return (b & ((1 << width) - 1)) | (1 << (width + unit)) | (sub << (width + N_UNITS))


def decode_alu_word(w: int, width: int, with_a: bool = True) -> tuple[int | None, int, str]:
    """Inverse of alu_word / alu_operand_word: (a, b, op)."""
    mask = (1 << width) - 1
    off = 0
    a = None
    if with_a:
        a = w & mask
        off = width
    b = (w >> off) & mask
    onehot = (w >> (off + width)) & ((1 << N_UNITS) - 1)
    sub = (w >> (off + width + N_UNITS)) & 1
    unit = onehot.bit_length() - 1
    assert onehot == 1 << unit, f"not one-hot: {onehot:b}"
    op = next(k for k, (u, s) in OPS.items() if u == unit and s == sub)
    return a, b, op


def add_alu_logic(G: Gates, name: str, A: list[Rail2], B: list[Rail2], U: list[Rail2], SUB: Rail2,
                  act_domain_inh: int, *, act_hops: int = 11, b_hops: int = 6, carry_hops: int = 5, mul: bool = False):
    """Combinational dual-rail ALU on veto relays. Returns (R, C, Z, V).

    A and B may be levels (a master's rails) or producer rails; U and SUB are producer rails.
    Timeline (docs/a2_alu_register.md section 4): U, SUB, B at the load; B^d = B delayed
    `b_hops` (~32 ms); bx = SUB xor B^d (ordered, ~36-46 ms); A^d = the operand gate driven by
    ACTIVE delayed `act_hops` (~70 ms), so A^d is the later operand against bx by >= 24 ms;
    the carry chain is ordered by `carry_hops` per stage. Z is not computed here: see
    add_zero_flag (it needs the consumer's completion over the R bits). No rate-mode gate
    remains in the datapath (the consumer's completion tree and fault gates are the only
    ones downstream)."""
    n = len(A)
    assert len(B) == n and len(U) == N_UNITS
    Ad, ACT, act_d = G.operand_gate(name, A, [U[k].r1.u for k in range(N_UNITS)], act_domain_inh, act_hops)
    Bd = [G.delayed(f"{name}.b{i}", B[i], b_hops) for i in range(n)]
    bx = [G.xor2_ordered(f"{name}.bx{i}", SUB, Bd[i]) for i in range(n)]
    sums, carries, x, cds = G.ripple_adder_ordered(f"{name}", Ad, bx, SUB, carry_hops)
    cout = carries[-1]
    vraw = G.overflow_ordered(f"{name}.vx", Ad, bx, SUB, x, carries, cds)
    f_and = [G.and2_ordered(f"{name}.and{i}", bx[i], Ad[i]) for i in range(n)]
    f_or = [G.or2_ordered(f"{name}.or{i}", bx[i], Ad[i]) for i in range(n)]
    f_xor = [G.xor2_ordered(f"{name}.xor{i}", bx[i], Ad[i]) for i in range(n)]
    units = [sums, f_and, f_or, f_xor, bx]  # PASSB passes bx (= B when SUB = 0, not-B when SUB = 1)
    if mul:  # MUL unit: low n bits of A x bx; a constant-0 rail (lit by ACTIVE's delayed rise) feeds the carry-ins
        zero = Rail2(G.latch(f"{name}.zero0"), G.latch(f"{name}.zero1"))
        G.veto(f"{name}.zero0.g", act_d, [], zero.r0)
        units.append(G.multiplier_ordered(f"{name}.mul", Ad, bx, zero, carry_hops))
    else:
        units.append(None)  # a MUL op selects nothing: the transaction cannot complete (watchdog refuses it)
    R = []
    for i in range(n):
        rails = []
        for r in (0, 1):
            t = G.latch(f"{name}.mux{i}r{r}")
            for k, unit in enumerate(units):  # one-hot select: unit k's rail ignites R unless k is deselected
                if unit is not None:
                    G.veto(f"{name}.mux{i}r{r}u{k}", unit[i].latches[r].u, [U[k].r0.u], t)
            rails.append(t)
        R.append(Rail2(rails[0], rails[1]))
    others = [U[k].r1.u for k in range(1, N_UNITS)]

    def gated_flag(fname: str, f: Rail2) -> Rail2:  # the adder's flag, or 0 for any other unit
        y1, y0 = G.latch(f"{fname}1"), G.latch(f"{fname}0")
        G.veto(f"{fname}1.g", f.r1.u, [U[0].r0.u], y1)
        G.veto(f"{fname}0.g", f.r0.u, [U[0].r0.u], y0)
        for k, t in enumerate(others):
            G.veto(f"{fname}0.u{k + 1}", t, [], y0)
        return Rail2(y0, y1)

    C = gated_flag(f"{name}.c", cout)
    V = gated_flag(f"{name}.v", vraw)
    return R, C, V


def add_zero_flag(net: Netlist, drive: Drive, Q, n: int, name: str = "alu.z", hops: int = 3) -> None:
    """Z on the consumer. No single result bit is reliably the last to arrive (in a ripple
    adder a lower sum can wait on a long propagate chain while the top carry was decided
    early by a kill), so "every result bit is valid" must come from a completion, not a
    delay. The consumer's tree already has that node: the subtree over bits [0, n) (bit 0's
    valid latch for n = 1, `comp.c{log2 n - 1}_0` otherwise, n a power of two). Its train,
    delayed `hops` so the R rails are established >= 15 ms earlier, drives two veto relays
    into Q's Z rail 1 unless any R rail 1 is live; Z rail 0 is an OR of the R rail-1 latches
    (one relay each, no ordering needed). Latency: Z rises ~20 ms after that node, and the
    word completes one tree level later; the register's fault gate on the Z bit covers a
    double rail."""
    assert n & (n - 1) == 0, "the first n bits must form a complete subtree of the completion tree"
    roles = net.roles
    prefix = roles[Q.completion.u].split(".")[0]
    if n == 1:
        node = Q.valid[0]
    else:
        k = n.bit_length() - 2
        u = roles.index(f"{prefix}.comp.c{k}_0.L.u")
        node = Latch(u, u + 1)
    d = add_delay_chain(net, drive, f"{name}d", node.u, hops)
    z0, z1 = Q.rails[n + 1][0], Q.rails[n + 1][1]
    add_veto_relay(net, drive, f"{name}1", d, [Q.rails[i][1].u for i in range(n)], z1)  # zero: no rail 1 anywhere
    for i in range(n):  # not zero: any rail 1 (an OR needs no ordering)
        add_veto_relay(net, drive, f"{name}0.r{i}", Q.rails[i][1].u, [], z0)


def wire_outputs(net: Netlist, drive: Drive, outputs: list[Rail2], Q, bits: list[int] | None = None) -> None:
    """Each latched output rail ignites the consumer's rail latch once, through an edge relay.
    `bits[k]` is the consumer bit for outputs[k] (default: k)."""
    bits = bits or list(range(len(outputs)))
    for s, i in zip(outputs, bits):
        for r, latch in enumerate(s.latches):
            relay = add_edge_relay(net, drive, f"out.b{i}r{r}", latch.u)
            net.synapse(relay, Q.rails[i][r].u, drive.ignite)


def wire_alu(net: Netlist, drive: Drive, R, C, V, Q) -> None:
    """R -> bits [0, n), C -> n, V -> n+2 through edge relays; Z (bit n+1) from add_zero_flag."""
    n = len(R)
    wire_outputs(net, drive, R + [C, V], Q, list(range(n)) + [n, n + 2])
    add_zero_flag(net, drive, Q, n)


def build_alu_channel(params: Params, width: int, drive: Drive | None = None, liveness: bool = True,
                      watchdog_hops: int = 150, mul: bool = False) -> Channel:
    """P(A, B, U, SUB) -> ALU -> Q(R, C, Z, V) with the standard four-phase wiring."""
    drive = drive or Drive.from_params(params)
    net = Netlist(params)
    pw = 2 * width + N_UNITS + 1
    P = add_register(net, drive, "P", pw, with_completion=False)
    Q = add_register(net, drive, "Q", width + 3, with_completion=True)
    A = [Rail2(*P.rails[i]) for i in range(width)]
    B = [Rail2(*P.rails[width + i]) for i in range(width)]
    U = [Rail2(*P.rails[2 * width + k]) for k in range(N_UNITS)]
    SUB = Rail2(*P.rails[2 * width + N_UNITS])
    G = Gates(net, drive)
    R, C, V = add_alu_logic(G, "alu", A, B, U, SUB, P.reset_inh, mul=mul)
    wire_alu(net, drive, R, C, V, Q)
    extend_reset(net, drive, Q, G.latches, G.gates)
    connect_trigger(net, drive, Q.completion.u, P.reset_trigger, P.reset_edge)  # ACCEPT
    connect_trigger(net, drive, P.ready, Q.reset_trigger, Q.reset_edge)  # CLEARED
    wire_fault_path(net, drive, P, Q)
    if liveness:
        add_liveness(net, drive, P, Q, watchdog_hops)
    net.group("alu_latches", [x for l in G.latches for x in l.members])
    net.group("alu_gates", list(G.gates))
    return Channel(net, drive, pw, P, Q)
