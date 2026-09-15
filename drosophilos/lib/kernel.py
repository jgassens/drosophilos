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
from ..protocol.token import decode_recent, rails_for
from ..sim.model import Params
from ..sim.ref64 import RefSim
from .adder import extend_reset
from .control import add_kill_pair, add_kill_train
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
    trigger_in: int = -1  # neuron to pulse to request the cell (driven by the upstream done relay)
    req: list = None  # kill pair [no request, request pending]
    idle: list = None  # kill pair [busy, idle]
    creq: list = None  # kill pair [nothing to commit, commit pending]
    commit_pulse: int = -1


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
    image_latches: list = field(default_factory=list)  # levels the host lights once (idle, no-request, ...)
    in_creq: list = None

    @property
    def output(self) -> Cell:
        return self.cells[-1]


def _const_rails(net: Netlist, drive: Drive, name: str, n: int) -> list:
    """n dual-rail bit latches; the host lights them once (a constant is a level)."""
    return [[add_latch(net, drive, f"{name}.b{i}r{r}") for r in (0, 1)] for i in range(n)]


class _N:
    def __init__(self, u):
        self.u = u


def guarded_pulse(net: Netlist, drive: Drive, name: str, A: list, B: list, target: int, d1: int = 12, d2: int = 20) -> None:
    """One pulse on `target` when A and B are both true, issued at the later of their rises: A
    and B are kill pairs [r_false, r_true]. Two relays cover the two orders: A's rise (delayed
    d1 hops) vetoed by "B false", and B's rise (delayed d2 hops) vetoed by "A false". A veto
    rail that died less than ~55 ms before the driver still blocks it, so the delays differ by
    ~45 ms (8 hops) and the two windows overlap: whichever rail flipped second, one relay sees
    its veto long dead (measured rule, `celement.add_veto_relay`). Both may fire when the rises
    are within the overlap; the target's consumers take a doublet as one event."""
    a_d = add_delay_chain(net, drive, f"{name}.ad", A[1].u, d1)
    b_d = add_delay_chain(net, drive, f"{name}.bd", B[1].u, d2)
    add_veto_relay(net, drive, f"{name}.pa", a_d, [B[0].u], _N(target))
    add_veto_relay(net, drive, f"{name}.pb", b_d, [A[0].u], _N(target))


def build_pipeline(params: Params, n: int, spec: list[dict], consts: dict | None = None, mems: dict | None = None,
                   drive: Drive | None = None, act_hops: int = 11, watchdog_hops: int = 170, idle_hops: int = 20) -> Pipeline:
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
    in_creq = add_kill_pair(net, drive, "IN.creq")  # [nothing to commit, commit pending]
    rl = add_edge_relay(net, drive, "IN.autocommit", S.completion.u, fast_inhibitor=True)
    net.synapse(rl, in_creq[1].u, drive.ignite)
    image = [in_creq[0]]
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
        # Back-pressure (the dataflow handshake): the upstream done pulse requests the cell
        # (REQ); the cell starts when it is idle and a request is pending, in either order
        # (guarded_pulse); starting clears REQ and IDLE. Without it a cell re-triggered in its
        # reset tail lost the token (measured: a 990 ms run followed by an 873 ms upstream run).
        act = add_latch(net, drive, f"{name}.act")
        trig = net.neuron(f"{name}.trigger")
        req = add_kill_pair(net, drive, f"{name}.req")
        idle = add_kill_pair(net, drive, f"{name}.idle")
        net.synapse(trig, req[1].u, drive.ignite)
        if a_done is not None:
            net.synapse(a_done, trig, drive.relay_in)
        start = net.neuron(f"{name}.start")
        guarded_pulse(net, drive, f"{name}.go", req, idle, start)
        for l in (act, req[0], idle[0]):
            net.synapse(start, l.u, drive.ignite)
        # The pair's own kill (r0's rise kills r1) is a relay driven by r0's train, and r0 was
        # silent for only ~60 ms (killed by the request, re-lit by the start), inside its ~86 ms
        # recovery: the request rail survived (measured). The start pulse kills them itself.
        add_kill_train(net, drive, f"{name}.start.kill", start, [req[1], idle[1]])
        image += [req[0], idle[1]]
        # ACT^d is a chain of pulses from the start pulse, not from the ACT latch's train: relays
        # driven by a train need ~86 ms of source silence before they fire again, and a chain
        # fed by ACT keeps firing ~85 ms after ACT is cleared, so a cell restarted less than
        # ~170 ms after its done pulse sampled nothing (measured). The ACT latch stays as the
        # busy level for the reset domain.
        act_d = add_delay_chain(net, drive, f"{name}.actd", start, act_hops)
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
        creq = add_kill_pair(net, drive, f"{name}.creq")  # the commit waits for the consumer (wired below)
        rl = add_edge_relay(net, drive, f"{name}.autocommit", Sc.completion.u, fast_inhibitor=True)
        net.synapse(rl, creq[1].u, drive.ignite)
        image.append(creq[0])
        # idle again ~106 ms after the done pulse: the reset train's paralysis (~80 ms) is over
        idle_d = add_delay_chain(net, drive, f"{name}.idled", reg.done_relay, idle_hops)
        net.synapse(idle_d, idle[1].u, drive.ignite)
        Ccell.reg, Ccell.master, Ccell.act, Ccell.trigger_in = reg, reg.master, act, trig
        Ccell.req, Ccell.idle, Ccell.creq = req, idle, creq
        cells[name] = Ccell
        order.append(Ccell)
    # Commit gating: a producer rewrites its master only once every consumer of it has started
    # on the previous value (its REQ cleared); otherwise the consumer's operand gates could
    # sample a value in the middle of its rewrite, or miss one. Consumers of a master: the
    # cells that read it as A or B. The output cell's master is read by the host: free.
    always = add_kill_pair(net, drive, "FREE")  # [never, always]: a constant "consumer is free"
    image.append(always[1])

    def gate_commit(pname: str, creq, commit_in: int, consumers: list):
        pulse = net.neuron(f"{pname}.commit_pulse")
        net.synapse(pulse, commit_in, drive.ignite)
        net.synapse(pulse, creq[0].u, drive.ignite)
        add_kill_train(net, drive, f"{pname}.commit.kill", pulse, [creq[1]])  # see the start pulse's kill
        if not consumers:
            guarded_pulse(net, drive, f"{pname}.cg", creq, always, pulse)
        for k, c in enumerate(consumers):  # every consumer must be free: chain the guards
            free = [c.req[1], c.req[0]]  # true when the consumer has no pending request
            if len(consumers) == 1:
                guarded_pulse(net, drive, f"{pname}.cg", creq, free, pulse)
            else:
                raise NotImplementedError("a master with several consumers needs a joined free rail")
        return pulse

    readers = {c.name: [d for d in order if d.a == c.name or d.b == c.name] for c in order}
    for c in order:
        c.commit_pulse = gate_commit(c.name, c.creq, c.reg.commit_in, readers[c.name])
    in_readers = [d for d in order if d.a == "input" or d.b == "input"]
    gate_commit("IN", in_creq, in_reg.commit_in, in_readers)
    pl = Pipeline(net, drive, n, in_reg, P, order, const_rails, mem_objs, image, in_creq)
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
    def gated_flag(fname, f):
        # C and V: the adder's own flags when the adder is selected; otherwise a 0 token per run.
        # The unit-select rails are levels here (lit once by the image), so the "not the adder"
        # zero must be driven by ACT^d, a rise per run, not by the select rail's rise as in the
        # sequencer's ALU (measured: the second token through an AND cell never completed its
        # stage, the flag latches having been cleared by the first commit and never re-driven).
        y1, y0 = G.latch(f"{fname}1"), G.latch(f"{fname}0")
        G.veto(f"{fname}1.g", f.r1.u, [U[0].r0.u], y1)
        G.veto(f"{fname}0.g", f.r0.u, [U[0].r0.u], y0)
        G.veto(f"{fname}0.na", act_d, [U[0].r1.u], y0)
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
    for l in pl.image_latches:  # "no request", "idle", "nothing to commit", "always free"
        sim.add_events(node, [step], [l.u], [pl.drive.ignite])
    for c in pl.cells:
        unit, _ = ALU_OPS["MOV" if c.op == "LOAD" else c.op]
        for k in range(N_UNITS):
            r = 1 if k == unit else 0
            sim.add_events(node, [step], [pl.net.roles.index(f"{c.name}.u{k}r{r}.u")], [pl.drive.ignite])


CELL_LATENCY_MS = 1000.0  # trigger -> done of an ALU or LOAD cell is ~880 ms (measured); MUL adds ~600


def run_pipeline(pl: Pipeline, params: Params, tokens: list[int], *, max_ms: float = 60000, gap_ms: float | None = None,
                 sim=None) -> tuple[list, RefSim, dict]:
    """Streams `tokens` into the input producer (each after the input stage's READY, and no
    sooner than `gap_ms` after the previous one) and decodes every commit of the output cell's
    master in order. Steps the simulator one step at a time and reads the per-step spike
    lists (a trace rebuild per poll is quadratic).

    The gap is the host's rate limit: a cell re-triggered while busy drops the token (its ACT
    latch is already lit, so the operand gates never sample again; measured with the input
    register cycling at 680 ms against cells of 880 ms). Until the cells carry their own
    back-pressure (a request latch per cell and commit gating), the transducer must not
    offer tokens faster than the slowest cell."""
    net, drive = pl.net, pl.drive
    if gap_ms is None:
        gap_ms = 0.0  # the cells carry their own back-pressure; the input stage's READY paces the host
    gap = int(gap_ms / params.dt)
    sim = sim or RefSim(net.topology(), params)
    load_pipeline_image(sim, pl)
    sim.run(3000)  # the image's completions settle
    P, S = pl.input_producer, pl.input_reg.stage
    out = pl.output.master
    window, period = 2 * drive.loop_period_steps, drive.loop_period_steps
    ready_n, comp_n = S.ready, out.completion.u
    loads, outs, last_wm, last_ready = [], [], None, None
    n_ready, k = 0, 0
    pending = []  # (step, decode window end) for output completions to decode once the word is stable
    while sim.step_index < int(max_ms / params.dt):
        if k < len(tokens) and n_ready >= k and (not loads or sim.step_index >= loads[-1] + gap):  # after the k-th READY
            t = sim.step_index + 5
            for i, r in rails_for(tokens[k], pl.n):
                sim.add_events(0, [t], [P.rails[i][r].u], [drive.ignite])
            loads.append(t)
            k += 1
        sim.step()
        s_ = sim.step_index - 1
        if pending and s_ >= pending[0] + window:
            st = pending.pop(0)
            outs.append((st, decode_recent(sim, out.rail_taps, st + window, window)[0]))
            if len(outs) >= len(tokens):
                break
        if not sim._spk_step or int(sim._spk_step[-1][0]) != s_:
            continue
        fired = sim._spk_neuron[-1]
        if ready_n in fired and (last_ready is None or s_ - last_ready > 3 * period):
            n_ready += 1
        if ready_n in fired:
            last_ready = s_
        if comp_n in fired:
            if last_wm is None or s_ - last_wm > 3 * period:
                pending.append(s_)
            last_wm = s_
    stats = {"neurons": net.n, "tokens": len(tokens), "outputs": len(outs),
             "first_output_ms": (outs[0][0] - loads[0]) * params.dt if outs else None,
             "per_token_ms": ((outs[-1][0] - outs[0][0]) / max(1, len(outs) - 1)) * params.dt if len(outs) > 1 else None}
    return outs, sim, stats
