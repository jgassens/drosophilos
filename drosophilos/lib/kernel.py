"""Resident kernels: a loop body as a spatial dataflow pipeline (plan: the default execution
model; `docs/capacity_doom.md` §3).

A pipeline is a chain of cells. Every cell owns a master register (its output value, a
level) and is triggered by the upstream master's commit-done pulse:

    trigger pulse -> ACT latch -> ACT delayed -> operand gates: A^d, B^d tokens from the
    source masters (or a constant) -> the ALU (or a RAM read for a load cell) -> stage ->
    automatic COMMIT at completion -> master rewritten -> done pulse -> next cell

Sources are levels (masters or constants), so a cell may read a master that its producer
is about to rewrite: the operand gate snapshots the level ~60 ms after the trigger and the
producer's next commit is a full cell latency away (>= 400 ms), so no back-pressure is
needed at one token per cell latency. The input cell is a staged register whose producer
the host loads (transduced input), one token at a time after READY; the output cell's
done pulse is the pixel record, decoded by the host from the master's rails.

No fetch, no decode, no PC: the kernel is the program.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..protocol.celement import add_delay_chain, add_or_latched, add_veto_relay
from ..protocol.handshake import Register, add_liveness, add_register, wire_fault_path
from ..protocol.latch import Latch, add_edge_relay, add_latch, connect_trigger
from ..protocol.token import decode_at, rails_for
from ..sim.model import Params
from ..sim.ref64 import RefSim
from .adder import extend_reset
from .alu import N_UNITS, OPS as ALU_OPS, add_alu_logic, alu_reference, wire_alu
from .gates import Gates, Rail2
from .netlist import Drive, Netlist
from .ram import Memory, add_memory, add_read_port
from .staged import StagedRegister, add_staged_commit


@dataclass
class Cell:
    name: str
    op: str  # ALU op name (ADD, SUB, AND, OR, XOR, MOV, MUL) or "LOAD" (RAM read at address = A)
    a: object  # source: a Cell, "input", or ("const", value)
    b: object = None  # source for B (ALU cells): a Cell, ("const", value) or None for LOAD
    mem: Memory | None = None  # for LOAD cells
    reg: StagedRegister | None = None
    master: Register | None = None
    act: Latch | None = None
    trigger_in: int = -1  # neuron to pulse to start the cell (driven by the upstream done relay)


@dataclass
class Pipeline:
    net: Netlist
    drive: Drive
    n: int
    input_reg: StagedRegister
    input_producer: Register
    cells: list
    consts: dict = field(default_factory=dict)  # (name) -> Register-like of latches
    mems: dict = field(default_factory=dict)

    @property
    def output(self) -> Cell:
        return self.cells[-1]


def _const_rails(net: Netlist, drive: Drive, name: str, n: int) -> list:
    """n dual-rail bit latches; the host lights them once (a constant is a level)."""
    return [[add_latch(net, drive, f"{name}.b{i}r{r}") for r in (0, 1)] for i in range(n)]


def build_pipeline(params: Params, n: int, spec: list[dict], consts: dict | None = None, mems: dict | None = None,
                   drive: Drive | None = None, act_hops: int = 11, watchdog_hops: int = 170) -> Pipeline:
    """`spec`: cells in order, each {"name", "op", "a": source, "b": source or None, "mem": name}
    where a source is "input", a cell name, or ("const", name). `consts`: name -> value (bits n).
    `mems`: name -> (n_words, contents dict)."""
    drive = drive or Drive.from_params(params)
    net = Netlist(params)
    # input register: producer P (host-loaded) -> stage -> master, auto-commit at completion
    P = add_register(net, drive, "IN.P", n, with_completion=False)
    S = add_register(net, drive, "IN.Q", n, with_completion=True)
    for i in range(n):
        for r in (0, 1):
            relay = add_edge_relay(net, drive, f"IN.data.b{i}r{r}", P.rails[i][r].u)
            net.synapse(relay, S.rails[i][r].u, drive.ignite)
    connect_trigger(net, drive, S.completion.u, P.reset_trigger, P.reset_edge)
    wire_fault_path(net, drive, P, S)
    add_liveness(net, drive, P, S, 55)
    in_reg = add_staged_commit(net, drive, "IN", S, P, ordered_grant=True)
    rl = add_edge_relay(net, drive, "IN.autocommit", S.completion.u, fast_inhibitor=True)
    net.synapse(rl, in_reg.commit_in, drive.ignite)
    const_rails = {name: _const_rails(net, drive, f"K.{name}", n) for name in (consts or {})}
    mem_objs = {name: add_memory(net, drive, f"MEM.{name}", nw, n) for name, (nw, _) in (mems or {}).items()}
    cells: dict[str, Cell] = {}
    order = []

    def rails_of(src):
        if src == "input":
            return [[in_reg.master.rails[i][0], in_reg.master.rails[i][1]] for i in range(n)], in_reg.done_relay
        if isinstance(src, tuple) and src[0] == "const":
            return const_rails[src[1]], None
        c = cells[src]
        return [[c.master.rails[i][0], c.master.rails[i][1]] for i in range(n)], c.reg.done_relay

    for cs in spec:
        name, op = cs["name"], cs["op"]
        A_rails, a_done = rails_of(cs["a"])
        Ccell = Cell(name, op, cs["a"], cs.get("b"), mem_objs.get(cs.get("mem")))
        # trigger: the upstream master's done pulse lights ACT
        act = add_latch(net, drive, f"{name}.act")
        trig = net.neuron(f"{name}.trigger")
        net.synapse(trig, act.u, drive.ignite)
        if a_done is not None:
            net.synapse(a_done, trig, drive.relay_in)
        act_d = add_delay_chain(net, drive, f"{name}.actd", act.u, act_hops)
        Sc = add_register(net, drive, f"{name}.Q", n + 3, with_completion=True)
        G = Gates(net, drive)
        if op == "LOAD":  # address = A (a master's level): read port driven by ACT^d into P-like rails
            Bt = [[G.latch(f"{name}.b{i}r0"), G.latch(f"{name}.b{i}r1")] for i in range(n)]
            addr_taps = [[A_rails[j][0].u, A_rails[j][1].u] for j in range(max(1, (Ccell.mem.n_words - 1).bit_length()))]
            add_read_port(net, drive, f"{name}.rd", Ccell.mem, act_d, addr_taps, [[l for l in pair] for pair in Bt])
            # PASSB through the ALU: U = PASSB constant rails, SUB = 0, A = don't care (use the read as B)
            U = _unit_rails(net, drive, f"{name}.u", "MOV", G)
            SUB = Rail2(G.latch(f"{name}.sub0"), G.latch(f"{name}.sub1"))
            G.veto(f"{name}.sub0.g", act_d, [], SUB.r0)
            A_tok = [Rail2(G.latch(f"{name}.a{i}r0"), G.latch(f"{name}.a{i}r1")) for i in range(n)]
            for i in range(n):  # A tokens = 0 (unused by PASSB)
                G.veto(f"{name}.a{i}r0.g", act_d, [], A_tok[i].r0)
            R, C, V = add_alu_logic_tokens(G, name, A_tok, [Rail2(*pair) for pair in Bt], U, SUB, act_d)
        else:
            unit, sub = ALU_OPS[op]
            B_rails, _ = rails_of(cs["b"])
            U = _unit_rails(net, drive, f"{name}.u", op, G)
            SUB = Rail2(G.latch(f"{name}.sub0"), G.latch(f"{name}.sub1"))
            G.veto(f"{name}.sub.g", act_d, [], SUB.r1 if sub else SUB.r0)
            A_tok, B_tok = [], []
            for i in range(n):  # operand gates: tokens from the source levels at ACT^d
                a0, a1 = G.latch(f"{name}.a{i}r0"), G.latch(f"{name}.a{i}r1")
                G.veto(f"{name}.a{i}r0.g", act_d, [A_rails[i][1].u], a0)
                G.veto(f"{name}.a{i}r1.g", act_d, [A_rails[i][0].u], a1)
                b0, b1 = G.latch(f"{name}.b{i}r0"), G.latch(f"{name}.b{i}r1")
                G.veto(f"{name}.b{i}r0.g", act_d, [B_rails[i][1].u], b0)
                G.veto(f"{name}.b{i}r1.g", act_d, [B_rails[i][0].u], b1)
                A_tok.append(Rail2(a0, a1)); B_tok.append(Rail2(b0, b1))
            R, C, V = add_alu_logic_tokens(G, name, A_tok, B_tok, U, SUB, act_d, mul=(op == "MUL"))
        wire_alu(net, drive, R, C, V, Sc)
        extend_reset(net, drive, Sc, G.latches + [act], G.gates)
        # the stage has no producer: a fault latch and a commit path
        F = _fault_latch(net, drive, name, Sc)
        reg = add_staged_commit(net, drive, name, Sc, _NoProducer(), ordered_grant=True)
        rl = add_edge_relay(net, drive, f"{name}.autocommit", Sc.completion.u, fast_inhibitor=True)
        net.synapse(rl, reg.commit_in, drive.ignite)
        Ccell.reg, Ccell.master, Ccell.act, Ccell.trigger_in = reg, reg.master, act, trig
        cells[name] = Ccell
        order.append(Ccell)
    pl = Pipeline(net, drive, n, in_reg, P, order, const_rails, mem_objs)
    pl.const_values = dict(consts or {})
    pl.mem_contents = {k: v for k, (_, v) in (mems or {}).items()}
    return pl


class _NoProducer:
    """Stands in for the producer in add_staged_commit for a cell that has none: the stage is
    fed by the ALU inside the cell, so there is no producer watchdog to wire."""
    watchdog = None


def _fault_latch(net, drive, name, S):
    """A fault latch for a stage without a producer (wire_fault_path needs one): holds the
    completion down and is reset with the stage."""
    F = add_latch(net, drive, f"{name}.faultL")
    for f in S.fault:
        net.synapse(f, F.u, drive.ignite)
    root = net.roles[S.completion.u][: -len(".L.u")]
    held = list(S.completion.members) + [x for x, role in enumerate(net.roles) if role in (f"{root}.and", f"{root}.ign.edge")]
    for x in held:
        net.synapse(F.u, x, drive.reset)
    for x in F.members:
        net.synapse(S.reset_inh, x, -int(round(0.75 * drive.loop)))
    S.fault_latch = F
    return F


def _unit_rails(net, drive, name, op, G: Gates):
    """Constant one-hot unit-select rails for a fixed op: latches lit by ACT^d? No: levels lit at
    power-up would sit forever; the mux vetoes need the *deselect* rails live, so they are
    plain latches the host lights once (part of the image)."""
    unit, _ = ALU_OPS[op]
    rails = []
    for k in range(N_UNITS):
        r0, r1 = add_latch(net, drive, f"{name}{k}r0"), add_latch(net, drive, f"{name}{k}r1")
        rails.append(Rail2(r0, r1))
    G.unit_rails = getattr(G, "unit_rails", []) + [(rails, unit)]
    return rails


def add_alu_logic_tokens(G: Gates, name: str, A_tok, B_tok, U, SUB, act_d, mul: bool = False):
    """The ALU datapath on already-tokenised operands (no operand gate inside): B is delayed
    as in add_alu_logic so bx is the later operand against... here A and B tokens rise
    together at ACT^d + 4, so B is delayed 6 hops to make bx the EARLY operand and A^d the
    LATE one by delaying A a further 11 hops."""
    n = len(A_tok)
    Bd = [G.delayed(f"{name}.b{i}d", B_tok[i], 6) for i in range(n)]
    bx = [G.xor2_ordered(f"{name}.bx{i}", SUB, Bd[i]) for i in range(n)]
    Ad = [G.delayed(f"{name}.a{i}d", A_tok[i], 17) for i in range(n)]
    sums, carries, x, cds = G.ripple_adder_ordered(f"{name}", Ad, bx, SUB, 5)
    cout = carries[-1]
    vraw = G.overflow_ordered(f"{name}.vx", Ad, bx, SUB, x, carries, cds)
    f_and = [G.and2_ordered(f"{name}.and{i}", bx[i], Ad[i]) for i in range(n)]
    f_or = [G.or2_ordered(f"{name}.or{i}", bx[i], Ad[i]) for i in range(n)]
    f_xor = [G.xor2_ordered(f"{name}.xor{i}", bx[i], Ad[i]) for i in range(n)]
    units = [sums, f_and, f_or, f_xor, bx]
    if mul:
        zero = Rail2(G.latch(f"{name}.zero0"), G.latch(f"{name}.zero1"))
        G.veto(f"{name}.zero0.g", act_d, [], zero.r0)
        units.append(G.multiplier_ordered(f"{name}.mul", Ad, bx, zero, 5))
    else:
        units.append(None)
    R = []
    for i in range(n):
        rails = []
        for r in (0, 1):
            t = G.latch(f"{name}.mux{i}r{r}")
            for k, unit in enumerate(units):
                if unit is not None:
                    G.veto(f"{name}.mux{i}r{r}u{k}", unit[i].latches[r].u, [U[k].r0.u], t)
            rails.append(t)
        R.append(Rail2(rails[0], rails[1]))
    others = [U[k].r1.u for k in range(1, N_UNITS)]

    def gated_flag(fname, f):
        y1, y0 = G.latch(f"{fname}1"), G.latch(f"{fname}0")
        G.veto(f"{fname}1.g", f.r1.u, [U[0].r0.u], y1)
        G.veto(f"{fname}0.g", f.r0.u, [U[0].r0.u], y0)
        for k, t in enumerate(others):
            G.veto(f"{fname}0.u{k + 1}", t, [], y0)
        return Rail2(y0, y1)

    return R, gated_flag(f"{name}.c", cout), gated_flag(f"{name}.v", vraw)


# ------------------------------------------------------------------------------ running
def load_pipeline_image(sim, pl: Pipeline, node: int = 0, step: int = 1) -> None:
    """Constants, memories and every cell's unit-select rails are levels the host lights once."""
    for name, rails in pl.consts.items():
        for i, r in rails_for(pl.const_values[name], pl.n):
            sim.add_events(node, [step], [rails[i][r].u], [pl.drive.ignite])
    for name, mem in pl.mems.items():
        for addr, v in pl.mem_contents[name].items():
            for i, r in rails_for(v, pl.n):
                sim.add_events(node, [step], [mem.words[addr].rails[i][r].u], [pl.drive.ignite])
    for role_idx, role in enumerate(pl.net.roles):
        pass
    for c in pl.cells:
        unit, _ = ALU_OPS["MOV" if c.op == "LOAD" else c.op]
        for k in range(N_UNITS):
            r = 1 if k == unit else 0
            sim.add_events(node, [step], [pl.net.roles.index(f"{c.name}.u{k}r{r}.u")], [pl.drive.ignite])


def run_pipeline(pl: Pipeline, params: Params, tokens: list[int], *, max_ms: float = 60000, gap_ms: float = 0,
                 sim=None) -> tuple[list, RefSim, dict]:
    """Streams `tokens` into the input producer (each after the input stage's READY) and
    decodes every commit of the output cell's master in order."""
    net, drive = pl.net, pl.drive
    sim = sim or RefSim(net.topology(), params)
    load_pipeline_image(sim, pl)
    sim.run(3000)  # the image's completions settle
    P, S = pl.input_producer, pl.input_reg.stage
    out = pl.output.master
    window, period = 2 * drive.loop_period_steps, drive.loop_period_steps
    loads, outs, last_wm = [], [], None
    k = 0
    while sim.step_index < int(max_ms / params.dt):
        ev = sim.trace.events
        n_ready = int((ev["neuron"] == S.ready).sum())  # the input stage's READY count so far
        if k < len(tokens) and n_ready >= k:  # token k goes in after the k-th READY (the first at once)
            t = sim.step_index + 5 + int(gap_ms / params.dt)
            for i, r in rails_for(tokens[k], pl.n):
                sim.add_events(0, [t], [P.rails[i][r].u], [drive.ignite])
            loads.append(t)
            k += 1
        sim.run(200)
        ev = sim.trace.events
        recent = ev["step"] >= sim.step_index - 200
        st, fired = ev["step"][recent], ev["neuron"][recent]
        for s_ in st[fired == out.completion.u]:
            if last_wm is None or s_ - last_wm > 3 * period:
                outs.append((int(s_), decode_at(sim.trace, out.rail_taps, int(s_), window)[0]))
            last_wm = int(s_)
        if len(outs) >= len(tokens):
            break
    stats = {"neurons": net.n, "tokens": len(tokens), "outputs": len(outs),
             "first_output_ms": (outs[0][0] - loads[0]) * params.dt if outs else None,
             "per_token_ms": ((outs[-1][0] - outs[0][0]) / max(1, len(outs) - 1)) * params.dt if len(outs) > 1 else None}
    return outs, sim, stats
