"""Dual-rail ALU on the rate-mode gate library, as a four-phase channel (plan §A2).

Operand word (producer P, 2n+6 bits):
    A[n] | B[n] | U[5] one-hot unit select (ADDER, AND, OR, XOR, PASSB) | SUB
Result word (consumer Q, n+3 bits):
    R[n] | C | Z | V
        ADDER : R = A + (B xor SUB) + SUB, i.e. A+B when SUB=0 and A-B (mod 2^n) when SUB=1.
                C = carry out (for SUB: C=1 means no borrow, A >= B unsigned).
                V = signed overflow = C xor (carry into the top stage).
        AND / OR / XOR : bitwise. PASSB : R = B (MOV). C = V = 0 for these four.
        Z = 1 iff R == 0, for every unit.

Structure. Every unit computes on every transaction (all operands are always valid
tokens, so every unit completes); a one-hot mux picks the result: per result rail, one
veto relay per unit (driven by that unit's rail, vetoed by the unit's deselect rail) into
one result latch. An unselected unit's outputs therefore never reach the result rails.
B enters every unit through bx = B xor SUB, a rate-mode stage, which makes bx the later
operand by >= 50 ms against A and the select rails; that ordering is what lets the logic
units, the adder's first XOR, the mux and the flag gating use veto relays (no exposure
window) instead of rate-mode ANDs (see protocol/celement.py::add_veto_relay).

ISA mapping (isa/semantics.md): ADD = ADD.WRAP, SUB = SUB.WRAP with OVF = V; the ISA's
compare results derive from SUB's flags (unsigned: lt = not C, eq = Z; signed: lt = Z==0
and (N xor V), N = R's top bit). Saturating ops, shifts and multiply are later macros.
"""

from __future__ import annotations

from ..protocol.handshake import Channel, add_liveness, add_register, wire_fault_path
from ..protocol.latch import add_edge_relay, connect_trigger
from ..sim.model import Params
from .adder import extend_reset
from .gates import Gates, Rail2
from .netlist import Drive, Netlist

UNITS = ("ADDER", "AND", "OR", "XOR", "PASSB")
OPS = {"ADD": (0, 0), "SUB": (0, 1), "AND": (1, 0), "OR": (2, 0), "XOR": (3, 0), "MOV": (4, 0)}  # op -> (unit, sub)
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


def add_alu_logic(G: Gates, name: str, A: list[Rail2], B: list[Rail2], U: list[Rail2], SUB: Rail2):
    """Combinational dual-rail ALU. Returns (R, C, Z, V).

    Timing structure (the ordering the veto relays rely on): U, SUB, B and A arrive with the
    load (A ~20 ms later in the accumulator); bx = B xor SUB is a rate-mode stage (~70 ms), so
    bx is reliably the LATER operand against A and U. Everything downstream of bx that pairs it
    with A or with a select rail is an ordered (veto-relay) gate; the adder's sum and carry
    and the zero tree, whose inputs have no fixed order, stay rate-mode."""
    n = len(A)
    assert len(B) == n and len(U) == N_UNITS
    bx = [G.xor2(f"{name}.bx{i}", B[i], SUB) for i in range(n)]  # rate-mode: B and SUB arrive together
    # adder unit: A + bx + SUB
    c = SUB
    sums, carries = [], []
    for i in range(n):
        x = G.xor2_ordered(f"{name}.fa{i}.x", A[i], bx[i])
        s_ = G.xor2(f"{name}.fa{i}.s", x, c)  # x and the carry have no fixed order
        c = G.maj3(f"{name}.fa{i}.c", A[i], bx[i], c)
        sums.append(s_)
        carries.append(c)
    cout = carries[-1]
    c_top = carries[-2] if n >= 2 else SUB
    vraw = G.xor2(f"{name}.vx", cout, c_top)  # rate-mode: cout can precede c_top (a3 == b3)
    f_and = [G.and2_ordered(f"{name}.and{i}", A[i], bx[i]) for i in range(n)]
    f_or = [G.or2_ordered(f"{name}.or{i}", A[i], bx[i]) for i in range(n)]
    f_xor = [G.xor2_ordered(f"{name}.xor{i}", A[i], bx[i]) for i in range(n)]
    units = [sums, f_and, f_or, f_xor, bx]  # PASSB passes bx (= B when SUB = 0, not-B when SUB = 1)
    R = []
    for i in range(n):
        rails = []
        for r in (0, 1):
            t = G.latch(f"{name}.mux{i}r{r}")
            for k, unit in enumerate(units):  # one-hot select: unit k's rail ignites R unless k is deselected
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
    level = [R[i].r0 for i in range(n)]  # Z = 1 iff every result bit is 0: AND tree over rail 0
    d = 0
    while len(level) > 1:
        nxt = [G._and(f"{name}.z{d}_{k // 2}", level[k].u, level[k + 1].u) for k in range(0, len(level) - 1, 2)]
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
        d += 1
    z0 = G._or(f"{name}.z0", [R[i].r1.u for i in range(n)])
    Z = Rail2(z0, level[0])
    return R, C, Z, V


def wire_outputs(net: Netlist, drive: Drive, outputs: list[Rail2], Q) -> None:
    """Each latched output rail ignites the consumer's rail latch once, through an edge relay."""
    for i, s in enumerate(outputs):
        for r, latch in enumerate(s.latches):
            relay = add_edge_relay(net, drive, f"out.b{i}r{r}", latch.u)
            net.synapse(relay, Q.rails[i][r].u, drive.ignite)


def build_alu_channel(params: Params, width: int, drive: Drive | None = None, liveness: bool = True,
                      watchdog_hops: int = 150) -> Channel:
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
    R, C, Z, V = add_alu_logic(G, "alu", A, B, U, SUB)
    wire_outputs(net, drive, R + [C, Z, V], Q)
    extend_reset(net, drive, Q, G.latches, G.gates)
    connect_trigger(net, drive, Q.completion.u, P.reset_trigger, P.reset_edge)  # ACCEPT
    connect_trigger(net, drive, P.ready, Q.reset_trigger, Q.reset_edge)  # CLEARED
    wire_fault_path(net, drive, P, Q)
    if liveness:
        add_liveness(net, drive, P, Q, watchdog_hops)
    net.group("alu_latches", [x for l in G.latches for x in l.members])
    net.group("alu_gates", list(G.gates))
    return Channel(net, drive, pw, P, Q)
