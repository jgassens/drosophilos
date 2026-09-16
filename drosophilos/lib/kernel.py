"""Resident kernels: a loop body as a spatial dataflow pipeline (plan: the default execution
model; `docs/capacity_doom.md` §3).

A pipeline is a graph of cells. Every cell owns a master register (its output value, a
level) and a datapath on the levels of its sources:

    requests (one kill pair per source: its done pulse) + IDLE -> start pulse -> ACT, ACT^d
    -> operand gates sample the sources -> ALU / RAM read / select -> stage -> commit
    request -> commit once every consumer has started on the previous value -> master
    rewritten -> done pulse -> the consumers' requests

Cell kinds: ALU ops (ADD SUB AND OR XOR MOV MUL) on sources a, b; LOAD (a read at address a
from a ROM, or from a RAM: word masters with the machine's read port); STORE (a write of b
at address a into a RAM: the machine's write port, driven from the cell's sampled tokens;
the cell's own master carries the data written, so the host sees the write land); SEL
(c != 0 ? a : b, c a cell whose Z flag decides); SHL/SHR by a constant; MULP (a pipelined
multiplier). A RAM is shared by every kernel of the pipeline; a read of a word being
written is not protected by the handshake — the host paces the passes (a column pass
writes the buffer, the pixel pass reads it), as it paces parameters. Several STORE cells
may write one RAM: each write port marks the word it is writing until the write lands, and
the mark vetoes the other ports' copies (`_share_write_ports`); the program must not issue
two stores to the same word within one write's window (the select to the word's completion
plus ~50 ms of the completion's kill train: measured 254 + 50 ms after the select, ~340 ms
after the store's ACT^d at 4 bits, more at wider words) — inside one pass two cells of the
same kernel store disjoint words, and the paced passes guarantee it between passes.
Sources: "input" (the
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

import time
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
from .alu import N_UNITS, OPS as ALU_OPS, add_alu_logic, alu_reference, wire_alu, wire_outputs
from .gates import Gates, Rail2
from .netlist import Drive, Netlist
from .ram import Memory, add_memory, add_read_port, add_write_port, address_vetoes
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
    row: int | None = None  # MULP rows: this row's index (0 .. n-1); the last row is the cell itself
    prev: str | None = None  # MULP rows: the previous row's name (its master carries acc | flags | A | B)
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
    wport: object = None  # STORE cells: the write port (its selects, copies and copy relays: shared-RAM marks)

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


def guarded_pulse(net: Netlist, drive: Drive, name: str, A: list, B: list, target: int, d1: int = 12, d2: int = 20,
                  extra_vetoes: tuple[int, ...] = ()) -> None:
    """One pulse on `target` when A and B are both true, issued at the later of their rises: A
    and B are kill pairs [r_false, r_true]. Two relays cover the two orders: A's rise (delayed
    d1 hops) vetoed by "B false", and B's rise (delayed d2 hops) vetoed by "A false". A veto
    rail that died less than ~55 ms before the driver still blocks it, so the delays differ by
    ~45 ms (8 hops) and the two windows overlap: whichever rail flipped second, one relay sees
    its veto long dead (measured rule, `celement.add_veto_relay`). Both may fire when the rises
    are within the overlap; the target's consumers take a doublet as one event. `extra_vetoes`
    recheck source-false rails hidden behind a chained guard's cached passed pair."""
    a_d = add_delay_chain(net, drive, f"{name}.ad", A[1].u, d1)
    b_d = add_delay_chain(net, drive, f"{name}.bd", B[1].u, d2)
    add_veto_relay(net, drive, f"{name}.pa", a_d, [B[0].u, *extra_vetoes], _N(target))
    add_veto_relay(net, drive, f"{name}.pb", b_d, [A[0].u, *extra_vetoes], _N(target))


def _chain_true(net: Netlist, drive: Drive, name: str, pairs: list, target: int, image: list, reset_pulse: int) -> None:
    """`target` pulses once when every pair in `pairs` is true (see _all_true_pulse).

    Intermediate passed pairs are only a cache: the final guard also vetoes on every original
    pair that cache represents. A guard doublet can otherwise perturb a passed pair around its
    reset and leave it true; when the last pair (a cell's IDLE, or a reader-free condition)
    rises later, the stale cache would issue a second start or commit with no new requests.
    """
    cur = pairs[0]
    for k, pr in enumerate(pairs[1:]):
        last = k == len(pairs) - 2
        if last:
            # `cur` is a passed pair once three or more inputs are chained. Recheck the
            # original inputs it summarises, whose false rails have been stable for the whole
            # completed run when a stale passed pair meets a later rise of `pr`.
            recheck = tuple(p[0].u for p in pairs[: k + 1]) if k else ()
            guarded_pulse(net, drive, f"{name}.g{k}", cur, pr, target, extra_vetoes=recheck)
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


def add_pacing_ring(net: Netlist, drive: Drive, name: str, K: int, advance_pulses: list[int], image: list) -> tuple[list, int]:
    """A one-hot ring counter of K lines per advance source, and one wrap pulse.

    Neural pacing counts a phase's tokens; the count was a state cell fed back through an
    adder and a compare (three handshakes per token), and as a *reader* of the pass's output
    cells it held their commits until it had started, so every token waited for its loop
    (measured: ~7 s of neural time per token against ~1.5-3 s host-paced). A ring holds no
    datapath and no commit: the count is which line is lit. Each line is a kill pair [dark,
    lit]; line 0 is lit at power-up, the others dark (the image). One advance pulse (a done
    relay's spike) drives K veto relays at once, `inc{k}` vetoed by line k's dark rail, so only
    the lit line's relay passes: it ignites line k+1's lit rail and line k's dark rail (each
    pair's own kill train clears the other rail). `inc{K-1}` also fires the wrap. A line's
    dark rail is a level established >= ~300 ms before the next advance (a cell's cycle), which
    is the veto relay's ordering assumption; a relay vetoed at one advance is driven again
    >= 300 ms later, past its 55 ms recovery.

    Every source gets a ring of its own: the sources (a pass's last output and every STORE
    of the pass) fire once per token each but can run several tokens apart in a deep
    dataflow, and a ring shared through one pulse neuron would lose two dones that fall within
    one hop. With one ring per source no two advances of a ring are closer than the source
    cell's own cycle. The phase's wrap is the join of the rings' wraps: each sets a kill pair
    [not wrapped, wrapped], a chained guard pulses the wrap when all are set, and that pulse
    clears them. A ring cannot lap another before the join fires: the phase's gate closes at
    the wrap and the host deals the next frame's tokens of this stream only after the other
    phases' tokens. Returns (lines, wrap): `lines[j][k]` is the k-th pair of source j's ring."""
    rings, wraps = [], []
    for j, src in enumerate(advance_pulses):
        pre = f"{name}.s{j}"
        lines = [add_kill_pair(net, drive, f"{pre}.l{k}") for k in range(K)]
        image.append(lines[0][1])
        image.extend(l[0] for l in lines[1:])
        adv = net.neuron(f"{pre}.adv")
        net.synapse(src, adv, drive.relay_in)
        wrap = net.neuron(f"{pre}.wrap")
        for k, line in enumerate(lines):
            nxt = lines[(k + 1) % K]
            relay = add_veto_relay(net, drive, f"{pre}.inc{k}", adv, [line[0].u], nxt[1])
            if K > 1:  # K == 1: the line stays lit (re-ignition is harmless) and every advance wraps
                net.synapse(relay, line[0].u, drive.ignite)
            if k == K - 1:
                net.synapse(relay, wrap, drive.ignite)
        rings.append(lines)
        wraps.append(wrap)
    if len(wraps) == 1:
        return rings, wraps[0]
    wrapped = []
    for j, w in enumerate(wraps):
        pr = add_kill_pair(net, drive, f"{name}.s{j}.wrapped")  # [not wrapped this frame, wrapped]
        net.synapse(w, pr[1].u, drive.ignite)
        image.append(pr[0])
        wrapped.append(pr)
    pulse = net.neuron(f"{name}.join.pulse")
    _chain_true(net, drive, f"{name}.join", wrapped, pulse, image, pulse)
    for pr in wrapped:
        net.synapse(pulse, pr[0].u, drive.ignite)
    add_kill_train(net, drive, f"{name}.join.kill", pulse, [pr[1] for pr in wrapped])
    # the chained guard's two paths (and a passed pair's) can each fire: measured, three sources
    # 100 ms apart gave three pulses over 18 ms; the relay makes the wrap one pulse (the pair
    # resets above take the repeats as one event, as a commit pulse's consumers do)
    return rings, add_edge_relay(net, drive, f"{name}.wrap", pulse, fast_inhibitor=True)


def build_pipeline(params: Params, n: int, spec: list[dict], consts: dict | None = None, mems: dict | None = None,
                   drive: Drive | None = None, act_hops: int = 11, watchdog_hops: int = 170, idle_hops: int = 20,
                   outputs: list | None = None, in_watchdog_hops: int | None = None, streams: list | None = None,
                   phases: list | None = None) -> Pipeline:
    """`spec`: cells in order, each {"name", "op", "a", "b", "c", "mem", "init", "trigger"} (see
    the module docstring). `consts`: name -> value. `mems`: name -> (n_words, contents dict).
    `outputs`: names of the cells the host decodes (default: the last). `streams`: the input
    stream names (default ["input"]; a stream NAME is the source "input:NAME"). `phases`:
    neural pacing (Stage F2's first step) — [(stream, name, mode), ...] in the order the phases
    run each frame: a stream's register commits only while its phase's OK pair is true; the
    pair is set by the previous phase's end and cleared by this phase's end. Mode "wrap": the
    name is a RING pseudo-cell of `spec` ({"name", "op": "RING", "k", "trigger", "stream"}, the
    compiler's per-stream token count K and the cells whose dones count a token) and the phase
    ends when its ring (`add_pacing_ring`) wraps, every K tokens; mode "each": the phase ends at
    every done of the named cell (None: of the stream's register). The host then only deals
    tokens in program order; the phase order is the substrate's."""
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
    # word against ~250 for a RAM master with its completion and reset. A memory some STORE
    # writes is a RAM: word masters with the machine's write and read ports.
    mem_objs = {}
    for name, spec_m in (mems or {}).items():
        nw, contents = spec_m[0], dict(spec_m[1])
        if len(spec_m) > 2 and spec_m[2] == "ram":  # word masters, written by STORE cells
            mem_objs[name] = ("ram", add_memory(net, drive, f"MEM.{name}", nw, n), contents)
        else:
            mem_objs[name] = (nw, contents)

    # ---- pass 1: every cell's stage, master, handshake pairs (sources may be built later: feedback)
    cells: dict[str, Cell] = {}
    order: list[Cell] = []
    expanded = []
    ring_specs = {cs["name"]: cs for cs in spec if cs["op"] == "RING"}  # pacing rings: no cell, built with the phases
    for cs in spec:
        if cs["op"] == "RING":
            continue
        if cs["op"] == "MULP":  # a pipelined multiplier: n row cells, the last one named as the cell
            assert cs.get("init") is None, "a MULP cell cannot carry state"
            for j in range(n):
                nm = cs["name"] if j == n - 1 else f"{cs['name']}.r{j}"
                prev = None if j == 0 else (f"{cs['name']}.r{j - 1}")
                expanded.append({"name": nm, "op": "MULP_ROW", "a": cs["a"] if j == 0 else prev, "b": cs.get("b") if j == 0 else None,
                                 "trigger": cs.get("trigger", ()) if j == 0 else (), "row": j, "prev": prev})
        else:
            expanded.append(cs)
    for cs in expanded:
        c = Cell(cs["name"], cs["op"], cs["a"], b=cs.get("b"), c=cs.get("c"), mem=mem_objs.get(cs.get("mem")), init=cs.get("init"),
                 trigger=tuple(cs.get("trigger", ())), imm=cs.get("imm"), row=cs.get("row"), prev=cs.get("prev"))
        width = 3 * n + 3 if c.op == "MULP_ROW" else n + 3  # a row's word: acc | C Z V | A | B
        c.stage = add_register(net, drive, f"{c.name}.Q", width, with_completion=True)
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
        if c.init is not None:  # the value and its Z flag (a SEL on a state condition reads the Z
            for i, r in rails_for(c.init, n):  # rails; dark rails fired both arms: review finding). C and V
                image.append(c.master.rails[i][r])  # stay dark on purpose: a complete master would fire its done
            image.append(c.master.rails[n + 1][1 if c.init == 0 else 0])  # pulse at power-up and request every reader

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
        # §10.5: the start pulse re-lights a request's false rail ~55 ms after that rail's own
        # kill train; at a 3 sigma corner of the pair the loop does not catch, the pair is dark,
        # and the row's next IDLE starts it with no request and replays the next word (the
        # perspective duplicate). An unconditional re-light from ACT^d fixed that and stalled
        # copies under mix B (a DONE landing between the start and ACT^d had REQ true lit when
        # the re-light landed). This one is gated: a veto relay driven ~74 ms after the start
        # (ACT^d + 3 hops, past the veto's 55 ms recovery from the start's kill of REQ true)
        # re-ignites REQ false only while REQ true is dark -- the (dark, dark) failure -- and is
        # vetoed by a pending request. Ordering: the earliest legitimate re-rise of REQ true is
        # the producer's free -> commit guard -> master reset -> done, >= ~150 ms after the
        # start; one drive per start, >= a cell cycle apart. (Kimi review run-20260916-150155.)
        relight_in = add_delay_chain(net, drive, f"{c.name}.actd2", act_d, 3)
        for src, pr in c.reqs.items():
            add_veto_relay(net, drive, f"{c.name}.relight.{src}", relight_in, [pr[1].u], pr[0])
        G = Gates(net, drive)
        Sc = c.stage
        name = c.name
        if c.op == "MULP_ROW":
            # Row j of a pipelined multiplier: acc' = acc + ((A if b_j else 0) << j), with A and B
            # carried along in the row's word so the next row reads them from this row's master
            # (the sources' masters may be rewritten by then). Operands are sampled at ACT^d like
            # any cell; B is delayed 6 hops (EARLY), A 17 (LATE) as in the ALU, and acc 23 so the
            # running sum is the LATE operand of the ordered ripple adder against the partial
            # product (EARLY, valid at A^d + 4). Throughput one token per cell latency; latency n
            # cells: the array multiplier's n^2 x 29 ms latency at the same throughput as an ADD.
            j = c.row
            if j == 0:
                A_rails, B_rails = rails_of(c.a), rails_of(c.b)
                acc_rails = None
            else:
                pm = cells[c.prev].master
                acc_rails = [[pm.rails[i][0], pm.rails[i][1]] for i in range(n)]
                A_rails = [[pm.rails[n + 3 + i][0], pm.rails[n + 3 + i][1]] for i in range(n)]
                B_rails = [[pm.rails[2 * n + 3 + i][0], pm.rails[2 * n + 3 + i][1]] for i in range(n)]
            A_tok, B_tok, acc_tok = [], [], []
            for i in range(n):
                a0, a1 = G.latch(f"{name}.a{i}r0"), G.latch(f"{name}.a{i}r1")
                G.veto(f"{name}.a{i}r0.g", act_d, [A_rails[i][1].u], a0)
                G.veto(f"{name}.a{i}r1.g", act_d, [A_rails[i][0].u], a1)
                b0, b1 = G.latch(f"{name}.b{i}r0"), G.latch(f"{name}.b{i}r1")
                G.veto(f"{name}.b{i}r0.g", act_d, [B_rails[i][1].u], b0)
                G.veto(f"{name}.b{i}r1.g", act_d, [B_rails[i][0].u], b1)
                A_tok.append(Rail2(a0, a1)); B_tok.append(Rail2(b0, b1))
                s0, s1 = G.latch(f"{name}.s{i}r0"), G.latch(f"{name}.s{i}r1")
                if acc_rails is None:
                    G.veto(f"{name}.s{i}r0.z", act_d, [], s0)  # acc = 0 into the first row
                else:
                    G.veto(f"{name}.s{i}r0.g", act_d, [acc_rails[i][1].u], s0)
                    G.veto(f"{name}.s{i}r1.g", act_d, [acc_rails[i][0].u], s1)
                acc_tok.append(Rail2(s0, s1))
            Bd = [G.delayed(f"{name}.b{i}d", B_tok[i], 6) for i in range(n)]
            Ad = [G.delayed(f"{name}.a{i}d", A_tok[i], 17) for i in range(n)]
            accd = [G.delayed(f"{name}.s{i}d", acc_tok[i], 23) for i in range(n)]
            bj = Bd[j]
            zero = Rail2(G.latch(f"{name}.zero0"), G.latch(f"{name}.zero1"))
            G.veto(f"{name}.zero0.g", act_d, [], zero.r0)
            pp = []  # partial product bit i: A_i AND b_j (A LATE, b_j EARLY), valid at Ad + 4
            for i in range(n):
                l1, l0 = G.latch(f"{name}.pp{i}r1"), G.latch(f"{name}.pp{i}r0")
                G.veto(f"{name}.pp{i}.g", Ad[i].r1.u, [bj.r0.u], l1)
                G.veto(f"{name}.pp{i}.a", Ad[i].r0.u, [], l0)
                G.veto(f"{name}.pp{i}.b", Ad[i].r1.u, [bj.r1.u], l0)
                pp.append(Rail2(l0, l1))
            addend = [zero] * j + [pp[m - j] for m in range(j, n)]  # A * b_j << j, low n bits
            sums, carries, x, cds = G.ripple_adder_ordered(f"{name}", accd, addend, zero, 5)
            R = sums
            C = Rail2(G.latch(f"{name}.c0"), G.latch(f"{name}.c1"))
            V = Rail2(G.latch(f"{name}.v0"), G.latch(f"{name}.v1"))
            G.veto(f"{name}.c0.g", act_d, [], C.r0)
            G.veto(f"{name}.v0.g", act_d, [], V.r0)
            wire_alu(net, drive, R, C, V, Sc)  # acc' -> bits 0..n-1, C, Z, V
            wire_outputs(net, drive, Ad + Bd, Sc, list(range(n + 3, 3 * n + 3)))  # A, B carried along (delayed tokens)
            extend_reset(net, drive, Sc, G.latches + [c.act], G.gates)
            built.add(c.name)
            continue
        if c.op == "STORE":  # RAM write of b at address a; the data also passes to the cell's master
            A_rails, B_rails = rails_of(c.a), rails_of(c.b)
            kind, mem, _ = c.mem
            assert kind == "ram", f"{name}: STORE into a ROM"
            At = [[G.latch(f"{name}.x{i}r0"), G.latch(f"{name}.x{i}r1")] for i in range(n)]
            Bt = [[G.latch(f"{name}.b{i}r0"), G.latch(f"{name}.b{i}r1")] for i in range(n)]
            for i in range(n):  # address and data tokens at ACT^d (levels for the port: valid before its trigger)
                G.veto(f"{name}.x{i}r0.g", act_d, [A_rails[i][1].u], At[i][0])
                G.veto(f"{name}.x{i}r1.g", act_d, [A_rails[i][0].u], At[i][1])
                G.veto(f"{name}.b{i}r0.g", act_d, [B_rails[i][1].u], Bt[i][0])
                G.veto(f"{name}.b{i}r1.g", act_d, [B_rails[i][0].u], Bt[i][1])
            a_bits = max(1, (mem.n_words - 1).bit_length())
            trig = add_delay_chain(net, drive, f"{name}.wtrig", act_d, 6)  # ~32 ms after the tokens: the port's vetoes are set up
            high = [At[j][1].u for j in range(a_bits, n)]  # an address beyond the buffer selects no word (fail-stop, as the oracle raises)
            wp = add_write_port(net, drive, f"{name}.wr", mem, trig, [[At[j][0].u, At[j][1].u] for j in range(a_bits)],
                                [[Bt[i][0].u, Bt[i][1].u] for i in range(n)], extra_vetoes=high)
            c.wport = wp
            for l in wp.domain_latches:  # the port's COPY latches clear with the cell's stage
                for x in l.members:
                    net.synapse(Sc.reset_inh, x, -int(round(0.75 * drive.loop)))
            # the data token to the master through the ALU's PASSB (a MOV), so the cell has the
            # standard stage, commit, done; the write (~130 ms after ACT^d) lands long before the
            # done pulse (~500 ms): readers requested by it find the word written
            U = _unit_rails(net, drive, f"{name}.u", "MOV", G)
            SUB = Rail2(G.latch(f"{name}.sub0"), G.latch(f"{name}.sub1"))
            G.veto(f"{name}.sub0.g", act_d, [], SUB.r0)
            A_tok = [Rail2(G.latch(f"{name}.a{i}r0"), G.latch(f"{name}.a{i}r1")) for i in range(n)]
            for i in range(n):
                G.veto(f"{name}.a{i}r0.g", act_d, [], A_tok[i].r0)
            R, C, V = add_alu_logic_tokens(G, name, A_tok, [Rail2(*pair) for pair in Bt], U, SUB, act_d)
        elif c.op == "LOAD" and c.mem[0] == "ram":  # RAM read: the machine's read port from the word masters
            A_rails = rails_of(c.a)
            _, mem, _ = c.mem
            Bt = [[G.latch(f"{name}.b{i}r0"), G.latch(f"{name}.b{i}r1")] for i in range(n)]
            a_bits = max(1, (mem.n_words - 1).bit_length())
            add_read_port(net, drive, f"{name}.rd", mem, act_d, [[A_rails[j][0].u, A_rails[j][1].u] for j in range(a_bits)],
                          [[l for l in pair] for pair in Bt], extra_vetoes=[A_rails[j][1].u for j in range(a_bits, n)])
            U = _unit_rails(net, drive, f"{name}.u", "MOV", G)
            SUB = Rail2(G.latch(f"{name}.sub0"), G.latch(f"{name}.sub1"))
            G.veto(f"{name}.sub0.g", act_d, [], SUB.r0)
            A_tok = [Rail2(G.latch(f"{name}.a{i}r0"), G.latch(f"{name}.a{i}r1")) for i in range(n)]
            for i in range(n):
                G.veto(f"{name}.a{i}r0.g", act_d, [], A_tok[i].r0)
            R, C, V = add_alu_logic_tokens(G, name, A_tok, [Rail2(*pair) for pair in Bt], U, SUB, act_d)
        elif c.op == "LOAD":  # address = A (a level): ROM read driven by ACT^d into token latches, PASSB through the ALU
            A_rails = rails_of(c.a)
            Bt = [[G.latch(f"{name}.b{i}r0"), G.latch(f"{name}.b{i}r1")] for i in range(n)]
            n_words, contents = c.mem
            addr_taps = [[A_rails[j][0].u, A_rails[j][1].u] for j in range(max(1, (n_words - 1).bit_length()))]
            # a bit that is the same in every word needs one relay, not one per word (a 256-entry
            # one-bit map cost sixteen bits' worth of relays before this: 12k neurons per read)
            same = {i: ((next(iter(contents.values())) >> i) & 1) for i in range(n)
                    if contents and all(((v >> i) & 1) == ((next(iter(contents.values())) >> i) & 1) for v in contents.values())}
            for i, r in same.items():
                add_veto_relay(net, drive, f"{name}.rom.b{i}", act_d, [], Bt[i][r])
            high = [A_rails[j][1].u for j in range(len(addr_taps), n)]  # an address beyond the table matches no word
            for w, value in contents.items():  # an unwritten address reads nothing: the cell stalls (fail-stop)
                vn = add_veto_neuron(net, drive, f"{name}.rom.w{w}.notw", address_vetoes(addr_taps, w) + high)
                for i in range(n):
                    if i in same:
                        continue
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

    # ---- several STORE cells into one RAM: the ports' marks keep them off each other's writes
    ram_names = {id(v[1]): k for k, v in mem_objs.items() if isinstance(v[0], str) and v[0] == "ram"}
    by_ram: dict[int, list] = {}
    for c in order:
        if c.op == "STORE":
            by_ram.setdefault(id(c.mem[1]), []).append(c)
    for key, group in by_ram.items():
        if len(group) > 1:
            _share_write_ports(net, drive, f"MEM.{ram_names[key]}", group[0].mem[1], group)

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
    ok_pairs, rings, phase_ends = {}, {}, {}
    if phases:
        for ph in phases:
            ok_pairs[ph[0]] = add_kill_pair(net, drive, f"PHASE.{ph[0]}.ok")  # [not this phase, this phase]
        image.append(ok_pairs[phases[0][0]][1])  # the first phase is open at power-up
        for k, ph in enumerate(phases):
            st, name_, mode = (ph + ("wrap",))[:3]
            nxt = phases[(k + 1) % len(phases)][0]
            # the phase's end pulse: "wrap" — the ring's wrap, every K tokens, the ring advanced by
            # the trigger cells' done relays (which hold no commit: the pass pipelines at full
            # speed; the cells' own readers are unchanged); "each" — every done of the named cell
            # (a tick phase ends when the tick kernel's state has landed, not when the tick token
            # did: measured, the next frame's columns read the old heading otherwise)
            if mode == "wrap":
                rs = ring_specs[name_]
                advs = [cells[t].reg.done_relay for t in rs["trigger"]]
                rings[st], end = add_pacing_ring(net, drive, f"PHASE.{st}.ring", rs["k"], advs, image)
            else:
                src_done = cells[name_].reg.done_relay if name_ is not None else inputs[st][0].done_relay
                end = add_delay_chain(net, drive, f"PHASE.{st}.endd", src_done, 6)
            phase_ends[st] = end
            for pr, r in ((ok_pairs[st], 0), (ok_pairs[nxt], 1)):  # this phase closes, the next opens
                net.synapse(end, pr[r].u, drive.ignite)
            add_kill_train(net, drive, f"PHASE.{st}.kill", end, [ok_pairs[st][1], ok_pairs[nxt][0]])
        for st in ok_pairs:
            if st != phases[0][0]:
                image.append(ok_pairs[st][0])
    for st, (reg, _, creq) in inputs.items():
        key = "input" if st == streams[0] else f"input:{st}"
        readers = [r for r in order if key in r.reqs]
        if st in ok_pairs:
            pulse = net.neuron(f"{key}.commit_pulse")
            net.synapse(pulse, reg.commit_in, drive.ignite)
            net.synapse(pulse, creq[0].u, drive.ignite)
            add_kill_train(net, drive, f"{key}.commit.kill", pulse, [creq[1]])
            frees = [[r.reqs[key][1], r.reqs[key][0]] for r in readers]
            _chain_true(net, drive, f"{key}.cg", [creq] + frees + [ok_pairs[st]], pulse, image, pulse)
        else:
            gate_commit(key, creq, reg.commit_in, readers)
    outs = [cells[o] for o in (outputs or [order[-1].name])]
    pl = Pipeline(net, drive, n, in_reg, P, order, const_rails, mem_objs, image, in_creq, outs,
                  {st: (reg, P_) for st, (reg, P_, _) in inputs.items()})
    pl.const_values = dict(consts or {})
    pl.mem_contents = {k: dict(v[1]) for k, v in (mems or {}).items()}
    pl.phase_ok, pl.rings, pl.phase_ends = ok_pairs, rings, phase_ends  # neural pacing: stream -> OK pair / ring lines / end pulse
    return pl


def _share_write_ports(net: Netlist, drive: Drive, name: str, mem: Memory, stores: list) -> None:
    """Several STORE cells write one RAM through their own write ports (`ram.add_write_port`),
    and every port's COPY latch for a word lights at that word's READY: with two ports and no
    guard, a write by either resets the word, READY lights both COPY latches, and both ports
    copy their own data into the word, double-railing it (the control machine's timer met the
    same on its status word, `control.py` "Two ports share the status word"; measured there).
    The same fix: per (port, word) a mark latch, lit by the port's select relay (its write is
    in flight) and killed by the word's completion (the write landed), vetoes the OTHER
    ports' copy relays for that word (0.5 x loop, the veto relay's standard, established >=
    40 ms before READY: READY comes after the word's reset and hold recovery). The completion
    also kills every port's COPY latch for the word: a bystander port's COPY, lit by READY
    and vetoed, would otherwise stay lit until that cell's own stage reset, and a lit COPY has
    no rise to fire its relays at that port's next write to the word (its cell may not run
    between the two — a store in another kernel's pass).

    Rule for the program: two stores to one word must not be in flight together — no second
    store to a word within one write's window: the select to the word's completion (254 ms at
    4 bits, deeper completion trees at wider words) plus ~50 ms while the completion's kill
    train paralyses the other port's mark. Inside the window the second port's mark cannot
    light, the new READY raises both COPY latches while the first port's data is still lit,
    and every differing bit double-rails (a later LOAD faults: fail-stop, not silent; measured
    at +30 and +45 ms after the completion, correct at +60 ms). Two STORE cells of one kernel store
    disjoint words in a pass; between passes the host's pacing (or the phase gates) keeps the
    order. A store whose word never completes (a faulted data token) leaves its mark lit and
    the word closed to the other ports: fail-stop again."""
    q = -int(round(0.5 * drive.loop))
    for w in range(mem.n_words):
        marks = [add_latch(net, drive, f"{c.name}.wr.w{w}.sel") for c in stores]
        for c, m in zip(stores, marks):
            net.synapse(c.wport.selects[w], m.u, drive.ignite)
            for other in stores:
                if other is c:
                    continue
                for pair in other.wport.copy_relays[w]:
                    for relay in pair:
                        net.synapse(m.u, relay, q)
        add_kill_train(net, drive, f"{name}.w{w}.sel.kill", stores[0].wport.done[w], marks + [c.wport.copies[w] for c in stores])


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
    for name, m in pl.mems.items():  # RAM initial contents
        if isinstance(m, tuple) and m[0] == "ram":
            for addr, v in m[2].items():
                for i, r in rails_for(v, pl.n):
                    sim.add_events(node, [step], [m[1].words[addr].rails[i][r].u], [pl.drive.ignite])
    for l in pl.image_latches:  # "no request", "idle", "nothing to commit", feedback requests, state values
        sim.add_events(node, [step], [l.u], [pl.drive.ignite])
    for c in pl.cells:
        if c.op in ("SEL", "SHL", "SHR", "MULP_ROW"):
            continue
        unit, _ = ALU_OPS["MOV" if c.op in ("LOAD", "STORE") else c.op]
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
    loads, k, bad = [], 0, 0
    pending = []  # (completion step, output cell)
    fault_n = {c.stage.fault_latch.u for c in pl.cells} | {reg.stage.fault_latch.u for reg, _ in pl.inputs.values()}
    timeout_n = {P_.watchdog.timeout.u for _, P_ in pl.inputs.values() if P_.watchdog is not None}
    seen_f, seen_t = set(), set()
    want = expect_outputs or (per_token or len(pl.outputs)) * len(sched)  # total outputs over all cells: one per output cell per token
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
        if len(sim._spk_step) > 8 * window:  # keep the recent spikes only (memory); faults and timeouts are counted as they happen
            for st_, nr in zip(sim._spk_step[: -4 * window], sim._spk_neuron[: -4 * window]):
                for x in nr.tolist():
                    if x in fault_n and x not in seen_f:
                        seen_f.add(x)
                    elif x in timeout_n and x not in seen_t:
                        seen_t.add(x)
            del sim._spk_step[: -4 * window]; del sim._spk_neuron[: -4 * window]
            if hasattr(sim, "_spk_node"): del sim._spk_node[: -4 * window]
        while pending and s_ >= pending[0][0] + window:
            stp, cell = pending.pop(0)
            v, status = decode_recent(sim, cell.master.rail_taps[: pl.n], stp + window, window)  # R bits; C Z V follow
            outs[cell.name].append((stp, v if status == "valid" else None))  # a faulted or partial word is None
            bad += status != "valid"
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
    for st_, nr in zip(sim._spk_step, sim._spk_neuron):  # first spike of each fault / timeout latch (the rest was counted on trimming)
        for x in nr.tolist():
            if x in fault_n and x not in seen_f:
                seen_f.add(x)
            elif x in timeout_n and x not in seen_t:
                seen_t.add(x)
    n_fault, n_timeout = len(seen_f), len(seen_t)
    stats = {"neurons": net.n, "tokens": len(sched), "outputs": len(first), "outputs_by_cell": outs, "loaded": dict(loaded),
             "ready": dict(n_ready), "load_steps": loads,
             "faults": n_fault, "timeouts": n_timeout, "bad_outputs": bad,
             "first_output_ms": (first[0][0] - loads[0]) * params.dt if first else None,
             "per_token_ms": ((first[-1][0] - first[0][0]) / max(1, len(first) - 1)) * params.dt if len(first) > 1 else None}
    return first, sim, stats


def run_pipeline_batched(pl: Pipeline, params: Params, schedules: list, *, max_ms: float = 60000, device: str = "cpu",
                         expect_outputs: list | None = None, dtype=None, sim=None,
                         progress: "float | callable | None" = 300,
                         capture_spikes: "tuple[int, object] | None" = None, gap_ms: float = 0.0) -> tuple[list, object, dict]:
    """`run_pipeline` on B copies of the kernel at once (the batched torch simulator: one
    node per copy, the cluster's "many brains running the same kernel on different tokens").
    `schedules[b]` is node b's host schedule (see run_pipeline); `expect_outputs[b]` the
    outputs node b owes (default: one per token). `gap_ms` is run_pipeline's: no load sooner
    than that after the node's previous load. Returns per node the dict of output lists
    (cell -> [(step, value)]), the simulator, and stats.

    `progress`: the interval in wall seconds between calls (default 300; None or 0 disables,
    uses the default printer), a callable(dict) (called at the default 300 s interval), or a
    (seconds, callable) pair for a custom interval with a custom callable. The default
    callable prints one flushed line per call: elapsed wall time, neural ms simulated, steps/s
    over the interval, outputs collected so far / expected, nodes that have finished their
    schedule, and faults/timeouts so far. The report dict passed to any callable carries the
    same fields under `elapsed_s`, `neural_ms`, `steps_per_s`, `outputs`, `expected_outputs`,
    `nodes_done`, `total_nodes`, `faults`, `timeouts`, plus `outs` — the per-node dict of
    output lists collected so far (the same live object the run fills in, for a caller that
    wants to assemble and write a partial result while the run continues). Wall time is
    polled only every ~1000 steps, so the per-step cost of the run loop is unchanged.

    `capture_spikes=(node, neuron_ids)` retains every matching `(step, neuron)` before the
    runner trims the simulator's trace.  The arrays are returned in
    `stats["captured_spikes"]`; campaigns use this for a narrow role-filtered handshake dump
    without retaining every spike of a 100-copy run."""
    import torch
    from ..sim.lif_torch import TorchSim
    net, drive = pl.net, pl.drive
    B = len(schedules)
    kw = {"dtype": dtype} if dtype is not None else {}
    sim = sim or TorchSim(net.topology(), params, n_nodes=B, device=device, **kw)  # a perturbed simulator may be passed (campaigns)
    for b in range(B):
        load_pipeline_image(sim, pl, node=b)
    sim.run(3000)
    captured_steps, captured_neurons = [], []
    if capture_spikes is not None:
        capture_node, capture_ids = capture_spikes
        capture_ids = np.asarray(sorted(set(int(x) for x in capture_ids)), dtype=np.int64)

        def capture_chunk(stp, nr, nd) -> None:
            mask = (nd == capture_node) & np.isin(nr, capture_ids)
            if np.any(mask):
                captured_steps.extend(stp[mask].tolist())
                captured_neurons.extend(nr[mask].tolist())

        for stp, nr, nd in zip(sim._spk_step, sim._spk_neuron, sim._spk_node):
            capture_chunk(stp, nr, nd)
    first_stream = next(iter(pl.inputs))
    scheds = []
    for sc in schedules:
        out = []
        for t in sc:
            out.append((t[0], t[1], t[2] if len(t) > 2 else 0) if isinstance(t, tuple) else (first_stream, t, 0))
        scheds.append(out)
    window, period = 2 * drive.loop_period_steps, drive.loop_period_steps
    gap = int(gap_ms / params.dt)
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
    fault_n = {c.stage.fault_latch.u for c in pl.cells} | {reg.stage.fault_latch.u for reg, _ in pl.inputs.values()}
    timeout_n = {P_.watchdog.timeout.u for _, P_ in pl.inputs.values() if P_.watchdog is not None}
    fault_ids = np.array(sorted(fault_n | timeout_n), dtype=np.int64)
    seen_f, seen_t, bad = set(), set(), 0  # (node, latch) first spikes; non-valid output words
    want = expect_outputs or [len(sc) for sc in scheds]
    done_nodes = [False] * B

    def default_progress(report: dict) -> None:
        print(f"progress: {report['elapsed_s']:.0f}s elapsed, {report['neural_ms']:.0f} ms neural, "
              f"{report['steps_per_s']:.0f} steps/s, outputs {report['outputs']}/{report['expected_outputs']}, "
              f"nodes done {report['nodes_done']}/{report['total_nodes']}, "
              f"faults {report['faults']}, timeouts {report['timeouts']}", flush=True)

    if isinstance(progress, tuple):
        progress_interval, progress_fn = progress
    elif callable(progress):
        progress_interval, progress_fn = 300, progress
    else:
        progress_interval, progress_fn = progress, default_progress
    t_start = time.time()
    t_last_report = t_start
    step_last_report = sim.step_index
    STEP_CHECK = 1000
    steps_since_check = 0

    while sim.step_index < int(max_ms / params.dt):
        for b in range(B):
            if k[b] < len(scheds[b]):
                st, value, min_outs = scheds[b][k[b]]
                n_out = sum(len(v) for v in outs[b].values())
                if n_ready[b][st] >= loaded[b][st] and n_out >= min_outs and (not loads[b] or sim.step_index >= loads[b][-1] + gap):
                    t = sim.step_index + 5
                    Pst = pl.inputs[st][1]
                    ev_n = [Pst.rails[i][r].u for i, r in rails_for(value, pl.n)]
                    sim.add_events(b, [t] * len(ev_n), ev_n, [drive.ignite] * len(ev_n))
                    loads[b].append(t)
                    loaded[b][st] += 1
                    k[b] += 1
        n_spike_chunks = len(sim._spk_step)
        sim.step()
        if capture_spikes is not None:
            for stp, nr, nd in zip(sim._spk_step[n_spike_chunks:], sim._spk_neuron[n_spike_chunks:],
                                   sim._spk_node[n_spike_chunks:]):
                capture_chunk(stp, nr, nd)
        s_ = sim.step_index - 1
        steps_since_check += 1
        if progress_interval and steps_since_check >= STEP_CHECK:
            steps_since_check = 0
            now = time.time()
            if now - t_last_report >= progress_interval:
                steps_per_s = (sim.step_index - step_last_report) / (now - t_last_report)
                progress_fn({
                    "elapsed_s": now - t_start, "neural_ms": sim.step_index * params.dt, "steps_per_s": steps_per_s,
                    "outputs": sum(sum(len(v) for v in o.values()) for o in outs), "expected_outputs": sum(want),
                    "nodes_done": sum(done_nodes), "total_nodes": B,
                    "faults": len(seen_f), "timeouts": len(seen_t), "outs": outs,
                })
                t_last_report, step_last_report = now, sim.step_index
        if len(sim._spk_step) > 8 * window:  # keep the recent spikes only: 128 nodes' whole run was OOM-killed at 96 GB
            del sim._spk_step[: -4 * window]; del sim._spk_neuron[: -4 * window]; del sim._spk_node[: -4 * window]
        while pending and s_ >= pending[0][0] + window:
            stp, b, cell = pending.pop(0)
            v, status = decode_recent(sim, cell.master.rail_taps[: pl.n], stp + window, window, node=b)
            outs[b][cell.name].append((stp, v if status == "valid" else None))
            bad += status != "valid"
        for b in range(B):
            if not done_nodes[b] and k[b] >= len(scheds[b]) and sum(len(v) for v in outs[b].values()) >= want[b]:
                done_nodes[b] = True
        if all(done_nodes) and not pending:
            break
        if not sim._spk_step or int(sim._spk_step[-1][0]) != s_:
            continue
        nr, nd = sim._spk_neuron[-1], sim._spk_node[-1]
        mf = np.isin(nr, fault_ids)
        for u, b in zip(nr[mf].tolist(), nd[mf].tolist()):
            (seen_t if u in timeout_n else seen_f).add((b, u))
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
             "outputs": [sum(len(v) for v in o.values()) for o in outs], "neural_ms": sim.step_index * params.dt,
             "faults": len(seen_f), "timeouts": len(seen_t), "bad_outputs": bad}
    if capture_spikes is not None:
        stats["captured_spikes"] = (np.asarray(captured_steps, dtype=np.int64),
                                    np.asarray(captured_neurons, dtype=np.int64))
    return outs, sim, stats
