"""Resident kernels: a loop body as a spatial dataflow pipeline (plan: the default execution
model; `docs/capacity_doom.md` §3).

A pipeline is a graph of cells. Every cell owns a master register (its output value, a
level) and a datapath on the levels of its sources:

    requests (one kill pair per source: its done pulse) + IDLE -> start pulse -> ACT, ACT^d
    -> operand gates sample the sources -> ALU / RAM read / select -> stage -> commit
    request -> commit once every consumer has started on the previous value -> master
    rewritten -> done pulse -> the consumers' requests

Cell kinds: ALU ops (ADD SUB AND OR XOR MOV MUL) on sources a, b; LOAD (RAM read at
address a); SEL (c != 0 ? a : b, c a cell whose Z flag decides). Sources: "input" (the
input register the host loads, one token after each READY), ("const", name), or a cell
name, "input:NAME" (another input stream: a second host-loaded register), or
("param", cell) — a level read without a request and without holding the producer's commit,
for a value that changes only between the reader's tokens (a renderer's heading from the
world update); the host must pace the streams so the producer never commits while a reader
of the parameter is in flight. A source built later than its reader is a feedback edge (loop-carried state): the
reader's request for it is asserted by the image, and the cell carries `init`, the state's
value at power-up. A cell listing "input" in `trigger` is requested by every input token
even if it does not read it (a state update paced by the tick). `outputs` names the cells
the host decodes at each completion (default: the last cell).

No fetch, no decode, no PC: the kernel is the program. `docs/a3_kernels.md`.
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
from .ram import address_vetoes
from ..protocol.celement import add_veto_neuron
from .staged import StagedRegister, add_staged_commit


@dataclass
class Cell:
    name: str
    op: str  # ADD SUB AND OR XOR MOV MUL | LOAD | SEL | SHL SHR (imm)
    a: object  # source: a Cell name, "input", or ("const", name)
    b: object = None  # ALU second operand / SEL's "c == 0" arm; None for LOAD
    c: object = None  # SEL's condition cell (its Z flag)
    mem: tuple | None = None  # for LOAD cells: (n_words, contents) of the ROM
    init: int | None = None  # state cells: the master's value at power-up
    imm: int | None = None  # SHL/SHR: the shift count
    trigger: tuple = ()  # extra request sources ("input")
    reg: StagedRegister | None = None
    master: Register | None = None
    act: Latch | None = None
    stage: Register | None = None
    reqs: dict = field(default_factory=dict)  # source name -> kill pair [no request, pending]
    idle: list = None
    creq: list = None
    start: int = -1
    commit_pulse: int = -1
    feedback: set = field(default_factory=set)  # sources that are feedback edges

    @property
    def sources(self) -> list:
        out = []
        for src in (self.a, self.b, self.c):
            if src is not None and src not in out:
                out.append(src)
        return out

    @property
    def request_sources(self) -> list:
        """Sources whose done pulse requests the cell: cells and input streams, not params or constants."""
        return [x for x in self.sources if isinstance(x, str)] + [t for t in self.trigger if t not in self.sources]


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
    outputs: list = field(default_factory=list)  # output cells
    inputs: dict = field(default_factory=dict)  # stream name -> (StagedRegister, producer P)

    @property
    def output(self) -> Cell:
        return self.outputs[-1]


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


def _chain_true(net: Netlist, drive: Drive, name: str, pairs: list, target: int, image: list, reset_pulse: int) -> None:
    """`target` pulses once when every pair in `pairs` is true (see _all_true_pulse)."""
    cur = pairs[0]
    for k, pr in enumerate(pairs[1:]):
        last = k == len(pairs) - 2
        if last:
            guarded_pulse(net, drive, f"{name}.g{k}", cur, pr, target)
            return
        passed = add_kill_pair(net, drive, f"{name}.p{k}")
        pk = net.neuron(f"{name}.p{k}.pulse")
        guarded_pulse(net, drive, f"{name}.g{k}", cur, pr, pk)
        net.synapse(pk, passed[1].u, drive.ignite)
        add_kill_train(net, drive, f"{name}.p{k}.kill0", pk, [passed[0]])
        net.synapse(reset_pulse, passed[0].u, drive.ignite)
        add_kill_train(net, drive, f"{name}.p{k}.kill1", reset_pulse, [passed[1]])
        image.append(passed[0])
        cur = passed
    always = add_kill_pair(net, drive, f"{name}.always")  # one pair: guard it against a constant true
    image.append(always[1])
    guarded_pulse(net, drive, f"{name}.g", cur, always, target)


def build_pipeline(params: Params, n: int, spec: list[dict], consts: dict | None = None, mems: dict | None = None,
                   drive: Drive | None = None, act_hops: int = 11, watchdog_hops: int = 170, idle_hops: int = 20,
                   outputs: list | None = None, in_watchdog_hops: int | None = None, streams: list | None = None) -> Pipeline:
    """`spec`: cells in order, each {"name", "op", "a", "b", "c", "mem", "init", "trigger"} (see
    the module docstring). `consts`: name -> value. `mems`: name -> (n_words, contents dict).
    `outputs`: names of the cells the host decodes (default: the last). `streams`: the input
    stream names (default ["input"]; a stream NAME is the source "input:NAME")."""
    drive = drive or Drive.from_params(params)
    net = Netlist(params)
    image: list = []
    streams = list(streams or ["input"])

    def input_register(stream: str):
        """Producer P (host-loaded) -> stage -> master, the commit gated by its readers."""
        pre = "IN" if stream == "input" else f"IN.{stream}"
        P = add_register(net, drive, f"{pre}.P", n, with_completion=False)
        S = add_register(net, drive, f"{pre}.Q", n, with_completion=True)
        for i in range(n):
            for r in (0, 1):
                relay = add_edge_relay(net, drive, f"{pre}.data.b{i}r{r}", P.rails[i][r].u)
                net.synapse(relay, S.rails[i][r].u, drive.ignite)
        connect_trigger(net, drive, S.completion.u, P.reset_trigger, P.reset_edge)
        wire_fault_path(net, drive, P, S)
        # the producer's watchdog must outlast the stage's completion, which grows with the
        # width (the completion tree is one level deeper per doubling and the rail events are
        # more): 148 ms after the load at 8 bits, 308 ms at 32. A fixed 55 hops (~290 ms) timed
        # out the 32-bit input register before its stage completed, cleared the stage, and the
        # commit then copied an empty stage as double rails (measured).
        add_liveness(net, drive, P, S, in_watchdog_hops if in_watchdog_hops is not None else 60 + 4 * n)
        reg = add_staged_commit(net, drive, pre, S, P, ordered_grant=True)
        creq = add_kill_pair(net, drive, f"{pre}.creq")  # [nothing to commit, commit pending]
        rl = add_edge_relay(net, drive, f"{pre}.autocommit", S.completion.u, fast_inhibitor=True)
        net.synapse(rl, creq[1].u, drive.ignite)
        image.append(creq[0])
        return reg, P, creq

    inputs = {st: input_register(st) for st in streams}
    in_reg, P, in_creq = inputs[streams[0]]
    const_rails = {name: _const_rails(net, drive, f"K.{name}", n) for name in (consts or {})}
    # Memories a kernel reads are ROMs: the contents are wired into the read relays (a rail
    # relay per set bit and per word, vetoed by "address is not w"), ~25 neurons per 8-bit
    # word against ~250 for a RAM master with its completion and reset. A kernel never
    # writes memory (STOREX is rejected), so nothing is lost.
    mem_objs = {name: (nw, dict(contents)) for name, (nw, contents) in (mems or {}).items()}

    # ---- pass 1: every cell's stage, master, handshake pairs (sources may be built later: feedback)
    cells: dict[str, Cell] = {}
    order: list[Cell] = []
    for cs in spec:
        c = Cell(cs["name"], cs["op"], cs["a"], b=cs.get("b"), c=cs.get("c"), mem=mem_objs.get(cs.get("mem")), init=cs.get("init"),
                 trigger=tuple(cs.get("trigger", ())), imm=cs.get("imm"))
        c.stage = add_register(net, drive, f"{c.name}.Q", n + 3, with_completion=True)
        c.act = add_latch(net, drive, f"{c.name}.act")
        c.idle = add_kill_pair(net, drive, f"{c.name}.idle")
        image.append(c.idle[1])
        _fault_latch(net, drive, c.name, c.stage)
        c.reg = add_staged_commit(net, drive, c.name, c.stage, _NoProducer(), ordered_grant=True)
        c.master = c.reg.master
        c.creq = add_kill_pair(net, drive, f"{c.name}.creq")
        rl = add_edge_relay(net, drive, f"{c.name}.autocommit", c.stage.completion.u, fast_inhibitor=True)
        net.synapse(rl, c.creq[1].u, drive.ignite)
        image.append(c.creq[0])
        idle_d = add_delay_chain(net, drive, f"{c.name}.idled", c.reg.done_relay, idle_hops)
        net.synapse(idle_d, c.idle[1].u, drive.ignite)  # idle again ~106 ms after done: the reset's paralysis is over
        c.start = net.neuron(f"{c.name}.start")
        cells[c.name] = c
        order.append(c)
    for c in order:
        if c.init is not None:
            for i, r in rails_for(c.init, n):
                image.append(c.master.rails[i][r])

    def stream_of(src):
        if src == "input":
            return streams[0]
        return src[len("input:"):] if isinstance(src, str) and src.startswith("input:") else None

    def rails_of(src):
        st = stream_of(src)
        if st is not None:
            m = inputs[st][0].master
            return [[m.rails[i][0], m.rails[i][1]] for i in range(n)]
        if isinstance(src, tuple) and src[0] == "const":
            return const_rails[src[1]]
        if isinstance(src, tuple) and src[0] == "param":
            src = src[1]
        return [[cells[src].master.rails[i][0], cells[src].master.rails[i][1]] for i in range(n)]

    def done_of(src):
        st = stream_of(src)
        return inputs[st][0].done_relay if st is not None else cells[src].reg.done_relay

    # ---- pass 2: requests, start, datapath
    built = set()
    for c in order:
        for src in c.request_sources:
            pair = add_kill_pair(net, drive, f"{c.name}.req.{src}")
            c.reqs[src] = pair
            trig = net.neuron(f"{c.name}.trigger.{src}")
            net.synapse(done_of(src), trig, drive.relay_in)
            net.synapse(trig, pair[1].u, drive.ignite)
            if stream_of(src) is None and src not in built:  # feedback: the state is there at power-up
                c.feedback.add(src)
                image.append(pair[1])
                assert cells[src].init is not None, f"{c.name} reads {src} before it is written: it needs an init"
            else:
                image.append(pair[0])
        _chain_true(net, drive, f"{c.name}.go", list(c.reqs.values()) + [c.idle], c.start, image, c.start)
        for l in [c.act, c.idle[0]] + [pr[0] for pr in c.reqs.values()]:
            net.synapse(c.start, l.u, drive.ignite)
        # The pair's own kill (r0's rise kills r1) is a relay driven by r0's train, and r0 was
        # silent for only ~60 ms (killed by the request, re-lit by the start), inside its ~86 ms
        # recovery: the request rail survived (measured). The start pulse kills them itself.
        add_kill_train(net, drive, f"{c.name}.start.kill", c.start, [c.idle[1]] + [pr[1] for pr in c.reqs.values()])
        # ACT^d: a chain of pulses from the start pulse (a chain fed by the ACT latch's train
        # keeps firing ~85 ms after ACT is cleared, and relays need ~86 ms of source silence)
        act_d = add_delay_chain(net, drive, f"{c.name}.actd", c.start, act_hops)
        G = Gates(net, drive)
        Sc = c.stage
        name = c.name
        if c.op == "LOAD":  # address = A (a level): ROM read driven by ACT^d into token latches, PASSB through the ALU
            A_rails = rails_of(c.a)
            Bt = [[G.latch(f"{name}.b{i}r0"), G.latch(f"{name}.b{i}r1")] for i in range(n)]
            n_words, contents = c.mem
            addr_taps = [[A_rails[j][0].u, A_rails[j][1].u] for j in range(max(1, (n_words - 1).bit_length()))]
            for w, value in contents.items():  # an unwritten address reads nothing: the cell stalls (fail-stop)
                vn = add_veto_neuron(net, drive, f"{name}.rom.w{w}.notw", address_vetoes(addr_taps, w))
                for i in range(n):
                    r = (value >> i) & 1
                    add_veto_relay(net, drive, f"{name}.rom.w{w}.b{i}", act_d, [], Bt[i][r], veto_neurons=[vn])
            U = _unit_rails(net, drive, f"{name}.u", "MOV", G)
            SUB = Rail2(G.latch(f"{name}.sub0"), G.latch(f"{name}.sub1"))
            G.veto(f"{name}.sub0.g", act_d, [], SUB.r0)
            A_tok = [Rail2(G.latch(f"{name}.a{i}r0"), G.latch(f"{name}.a{i}r1")) for i in range(n)]
            for i in range(n):
                G.veto(f"{name}.a{i}r0.g", act_d, [], A_tok[i].r0)
            R, C, V = add_alu_logic_tokens(G, name, A_tok, [Rail2(*pair) for pair in Bt], U, SUB, act_d)
        elif c.op in ("SHL", "SHR"):  # a shift by a constant is wiring: bit i <- bit i -/+ k, zeros shifted in
            A_rails = rails_of(c.a)
            k = c.imm
            R = []
            for i in range(n):
                r0, r1 = G.latch(f"{name}.s{i}r0"), G.latch(f"{name}.s{i}r1")
                j = i - k if c.op == "SHL" else i + k
                if 0 <= j < n:
                    G.veto(f"{name}.s{i}r1.g", act_d, [A_rails[j][0].u], r1)
                    G.veto(f"{name}.s{i}r0.g", act_d, [A_rails[j][1].u], r0)
                else:
                    G.veto(f"{name}.s{i}r0.z", act_d, [], r0)
                R.append(Rail2(r0, r1))
            C = Rail2(G.latch(f"{name}.c0"), G.latch(f"{name}.c1"))
            V = Rail2(G.latch(f"{name}.v0"), G.latch(f"{name}.v1"))
            G.veto(f"{name}.c0.g", act_d, [], C.r0)
            G.veto(f"{name}.v0.g", act_d, [], V.r0)
        elif c.op == "SEL":  # c != 0 ? a : b, per bit from the condition's Z rails and the arms' levels
            A_rails, B_rails = rails_of(c.a), rails_of(c.b)
            zr = cells[c.c].master.rails[n + 1]  # [Z0 = c != 0, Z1 = c == 0]
            R = []
            for i in range(n):
                r0, r1 = G.latch(f"{name}.s{i}r0"), G.latch(f"{name}.s{i}r1")
                G.veto(f"{name}.s{i}r1.a", act_d, [zr[1].u, A_rails[i][0].u], r1)  # c != 0 and a = 1
                G.veto(f"{name}.s{i}r0.a", act_d, [zr[1].u, A_rails[i][1].u], r0)
                G.veto(f"{name}.s{i}r1.b", act_d, [zr[0].u, B_rails[i][0].u], r1)  # c == 0 and b = 1
                G.veto(f"{name}.s{i}r0.b", act_d, [zr[0].u, B_rails[i][1].u], r0)
                R.append(Rail2(r0, r1))
            C = Rail2(G.latch(f"{name}.c0"), G.latch(f"{name}.c1"))
            V = Rail2(G.latch(f"{name}.v0"), G.latch(f"{name}.v1"))
            G.veto(f"{name}.c0.g", act_d, [], C.r0)
            G.veto(f"{name}.v0.g", act_d, [], V.r0)
        else:
            unit, sub = ALU_OPS[c.op]
            A_rails, B_rails = rails_of(c.a), rails_of(c.b)
            U = _unit_rails(net, drive, f"{name}.u", c.op, G)
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
            R, C, V = add_alu_logic_tokens(G, name, A_tok, B_tok, U, SUB, act_d, mul=(c.op == "MUL"))
        wire_alu(net, drive, R, C, V, Sc)
        extend_reset(net, drive, Sc, G.latches + [c.act], G.gates)
        built.add(c.name)

    # ---- commit gating: a producer rewrites its master only once every reader of it has started
    # on the previous value (its request for this source cleared), so a reader's operand gates
    # never sample a value mid-rewrite and never miss one. The host reads the outputs: free.
    def gate_commit(pname: str, creq, commit_in: int, readers: list):
        pulse = net.neuron(f"{pname}.commit_pulse")
        net.synapse(pulse, commit_in, drive.ignite)
        net.synapse(pulse, creq[0].u, drive.ignite)
        add_kill_train(net, drive, f"{pname}.commit.kill", pulse, [creq[1]])
        frees = [[r.reqs[pname][1], r.reqs[pname][0]] for r in readers]  # true when the reader has no pending request
        _chain_true(net, drive, f"{pname}.cg", [creq] + frees, pulse, image, pulse)
        return pulse

    for c in order:
        c.commit_pulse = gate_commit(c.name, c.creq, c.reg.commit_in, [r for r in order if c.name in r.reqs])
    for st, (reg, _, creq) in inputs.items():
        key = "input" if st == streams[0] else f"input:{st}"
        gate_commit(key, creq, reg.commit_in, [r for r in order if key in r.reqs])
    outs = [cells[o] for o in (outputs or [order[-1].name])]
    pl = Pipeline(net, drive, n, in_reg, P, order, const_rails, mem_objs, image, in_creq, outs,
                  {st: (reg, P_) for st, (reg, P_, _) in inputs.items()})
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
    """Constants, memories, state inits, every cell's unit-select rails and the handshake's
    resting levels are lit once by the host."""
    for name, rails in pl.consts.items():
        for i, r in rails_for(pl.const_values[name], pl.n):
            sim.add_events(node, [step], [rails[i][r].u], [pl.drive.ignite])
    for l in pl.image_latches:  # "no request", "idle", "nothing to commit", feedback requests, state values
        sim.add_events(node, [step], [l.u], [pl.drive.ignite])
    for c in pl.cells:
        if c.op in ("SEL", "SHL", "SHR"):
            continue
        unit, _ = ALU_OPS["MOV" if c.op == "LOAD" else c.op]
        for k in range(N_UNITS):
            r = 1 if k == unit else 0
            sim.add_events(node, [step], [pl.net.roles.index(f"{c.name}.u{k}r{r}.u")], [pl.drive.ignite])


def run_pipeline(pl: Pipeline, params: Params, tokens: list, *, max_ms: float = 60000, gap_ms: float = 0.0,
                 sim=None, per_token: int | None = None, expect_outputs: int | None = None) -> tuple[list, RefSim, dict]:
    """Streams `tokens` into the input producers and decodes every completion of each output
    cell's master in order. A token is a value (the first stream), a pair (stream, value), or
    a triple (stream, value, min_outputs): the host schedule; a triple is loaded only once the
    outputs so far (all cells) number at least min_outputs, which is how the host paces a
    parameter's producer (a world-update tick) behind the readers in flight (a frame's
    columns). Every token goes in after its stream's READY and no sooner than `gap_ms` after
    the previous load. Steps the simulator one step at a time and reads the per-step spike
    lists (a trace rebuild per poll is quadratic). Returns the first output's list of (step,
    value); `stats["outputs_by_cell"]` holds every output's list. The run ends when the outputs
    number `expect_outputs` in total (default: `per_token` per token, default one) or at `max_ms`."""
    net, drive = pl.net, pl.drive
    sim = sim or RefSim(net.topology(), params)
    load_pipeline_image(sim, pl)
    sim.run(3000)  # the image's completions settle
    first_stream = next(iter(pl.inputs))
    sched = []
    for t in tokens:
        if isinstance(t, tuple):
            sched.append((t[0], t[1], t[2] if len(t) > 2 else 0))
        else:
            sched.append((first_stream, t, 0))
    window, period = 2 * drive.loop_period_steps, drive.loop_period_steps
    gap = int(gap_ms / params.dt)
    ready_of = {st: reg.stage.ready for st, (reg, _) in pl.inputs.items()}
    n_ready = {st: 0 for st in pl.inputs}
    loaded = {st: 0 for st in pl.inputs}
    last_ready = {st: None for st in pl.inputs}
    comp = {c.completion.u: c for c in (o.master for o in pl.outputs)}
    outs = {c.name: [] for c in pl.outputs}
    last_wm = {u: None for u in comp}
    loads, k = [], 0
    pending = []  # (completion step, output cell)
    want = expect_outputs or (per_token * len(sched) if per_token else len(sched))  # total outputs over all cells
    while sim.step_index < int(max_ms / params.dt):
        if k < len(sched):
            st, value, min_outs = sched[k]
            n_out = sum(len(v) for v in outs.values())
            if n_ready[st] >= loaded[st] and n_out >= min_outs and (not loads or sim.step_index >= loads[-1] + gap):
                t = sim.step_index + 5
                Pst = pl.inputs[st][1]
                for i, r in rails_for(value, pl.n):
                    sim.add_events(0, [t], [Pst.rails[i][r].u], [drive.ignite])
                loads.append(t)
                loaded[st] += 1
                k += 1
        sim.step()
        s_ = sim.step_index - 1
        while pending and s_ >= pending[0][0] + window:
            stp, cell = pending.pop(0)
            outs[cell.name].append((stp, decode_recent(sim, cell.master.rail_taps[: pl.n], stp + window, window)[0]))  # R bits; C Z V follow
        if k >= len(sched) and sum(len(v) for v in outs.values()) >= want:
            break
        if not sim._spk_step or int(sim._spk_step[-1][0]) != s_:
            continue
        fired = sim._spk_neuron[-1]
        for st, rn in ready_of.items():
            if rn in fired:
                if last_ready[st] is None or s_ - last_ready[st] > 3 * period:
                    n_ready[st] += 1
                last_ready[st] = s_
        for u, master in comp.items():
            if u in fired:
                if last_wm[u] is None or s_ - last_wm[u] > 3 * period:
                    pending.append((s_, next(o for o in pl.outputs if o.master is master)))
                last_wm[u] = s_
    first = outs[pl.outputs[0].name]
    fault_n = {c.stage.fault_latch.u for c in pl.cells} | {reg.stage.fault_latch.u for reg, _ in pl.inputs.values()}
    timeout_n = {P_.watchdog.timeout.u for _, P_ in pl.inputs.values() if P_.watchdog is not None}
    n_fault = n_timeout = 0
    seen_f, seen_t = set(), set()
    for st_, nr in zip(sim._spk_step, sim._spk_neuron):  # first spike of each fault / timeout latch
        for x in nr.tolist():
            if x in fault_n and x not in seen_f:
                seen_f.add(x); n_fault += 1
            elif x in timeout_n and x not in seen_t:
                seen_t.add(x); n_timeout += 1
    stats = {"neurons": net.n, "tokens": len(sched), "outputs": len(first), "outputs_by_cell": outs, "loaded": dict(loaded),
             "ready": dict(n_ready), "load_steps": loads,
             "faults": n_fault, "timeouts": n_timeout,
             "first_output_ms": (first[0][0] - loads[0]) * params.dt if first else None,
             "per_token_ms": ((first[-1][0] - first[0][0]) / max(1, len(first) - 1)) * params.dt if len(first) > 1 else None}
    return first, sim, stats


def run_pipeline_batched(pl: Pipeline, params: Params, schedules: list, *, max_ms: float = 60000, device: str = "cpu",
                         expect_outputs: list | None = None, dtype=None) -> tuple[list, object, dict]:
    """`run_pipeline` on B copies of the kernel at once (the batched torch simulator: one
    node per copy, the cluster's "many brains running the same kernel on different tokens").
    `schedules[b]` is node b's host schedule (see run_pipeline); `expect_outputs[b]` the
    outputs node b owes (default: one per token). Returns per node the dict of output lists
    (cell -> [(step, value)]), the simulator, and stats."""
    import torch
    from ..sim.lif_torch import TorchSim
    net, drive = pl.net, pl.drive
    B = len(schedules)
    kw = {"dtype": dtype} if dtype is not None else {}
    sim = TorchSim(net.topology(), params, n_nodes=B, device=device, **kw)
    for b in range(B):
        load_pipeline_image(sim, pl, node=b)
    sim.run(3000)
    first_stream = next(iter(pl.inputs))
    scheds = []
    for sc in schedules:
        out = []
        for t in sc:
            out.append((t[0], t[1], t[2] if len(t) > 2 else 0) if isinstance(t, tuple) else (first_stream, t, 0))
        scheds.append(out)
    window, period = 2 * drive.loop_period_steps, drive.loop_period_steps
    ready_of = {st: reg.stage.ready for st, (reg, _) in pl.inputs.items()}
    watch = {}  # neuron -> ("ready", stream) | ("out", cell)
    for st, rn in ready_of.items():
        watch[rn] = ("ready", st)
    for o in pl.outputs:
        watch[o.master.completion.u] = ("out", o)
    watch_ids = np.array(sorted(watch), dtype=np.int64)
    n_ready = [{st: 0 for st in pl.inputs} for _ in range(B)]
    loaded = [{st: 0 for st in pl.inputs} for _ in range(B)]
    last_ready = [{st: None for st in pl.inputs} for _ in range(B)]
    last_wm = [{o.name: None for o in pl.outputs} for _ in range(B)]
    outs = [{o.name: [] for o in pl.outputs} for _ in range(B)]
    loads = [[] for _ in range(B)]
    k = [0] * B
    pending = []  # (step, node, cell)
    want = expect_outputs or [len(sc) for sc in scheds]
    done_nodes = [False] * B
    while sim.step_index < int(max_ms / params.dt):
        for b in range(B):
            if k[b] < len(scheds[b]):
                st, value, min_outs = scheds[b][k[b]]
                n_out = sum(len(v) for v in outs[b].values())
                if n_ready[b][st] >= loaded[b][st] and n_out >= min_outs:
                    t = sim.step_index + 5
                    Pst = pl.inputs[st][1]
                    ev_n = [Pst.rails[i][r].u for i, r in rails_for(value, pl.n)]
                    sim.add_events(b, [t] * len(ev_n), ev_n, [drive.ignite] * len(ev_n))
                    loads[b].append(t)
                    loaded[b][st] += 1
                    k[b] += 1
        sim.step()
        s_ = sim.step_index - 1
        while pending and s_ >= pending[0][0] + window:
            stp, b, cell = pending.pop(0)
            outs[b][cell.name].append((stp, decode_recent(sim, cell.master.rail_taps[: pl.n], stp + window, window, node=b)[0]))
        for b in range(B):
            if not done_nodes[b] and k[b] >= len(scheds[b]) and sum(len(v) for v in outs[b].values()) >= want[b]:
                done_nodes[b] = True
        if all(done_nodes) and not pending:
            break
        if not sim._spk_step or int(sim._spk_step[-1][0]) != s_:
            continue
        nr, nd = sim._spk_neuron[-1], sim._spk_node[-1]
        m = np.isin(nr, watch_ids)
        for u, b in zip(nr[m].tolist(), nd[m].tolist()):
            kind, what = watch[u]
            if kind == "ready":
                if last_ready[b][what] is None or s_ - last_ready[b][what] > 3 * period:
                    n_ready[b][what] += 1
                last_ready[b][what] = s_
            else:
                if last_wm[b][what.name] is None or s_ - last_wm[b][what.name] > 3 * period:
                    pending.append((s_, b, what))
                last_wm[b][what.name] = s_
    stats = {"neurons": net.n, "nodes": B, "tokens": [len(sc) for sc in scheds], "loaded": loaded,
             "outputs": [sum(len(v) for v in o.values()) for o in outs], "neural_ms": sim.step_index * params.dt}
    return outs, sim, stats
