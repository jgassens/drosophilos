"""Control machine v0: a one-hot sequencer that executes a program held in neural memory
(plan §A2/B: control FSM, program image).

    IMEM   n_prog words x 17 dual-rail bits, latch-only (the host loads the program image once)
    PC     one-hot ring of n_prog lines; a newly lit line's rise kills the others
    FSM    one-hot ring FETCH -> COMMIT -> NEXT -> FETCH; a state's rise kills the others
    ACC    the accumulator (ALU + stage + master) with an ordered grant
    IR     the instruction's control fields (LOAD, STORE, JZ, JNZ, ADDR) as kill pairs (each
           bit's two rail latches kill each other), re-loaded from IMEM 32 ms after FETCH
    DMEM   data RAM (lib/ram.py) with a read port into the ALU's B rails and a write port
           from the accumulator

Instruction word (IMEM bit order): B[n] | U[6] one-hot | SUB | flags... | ADDR[a]
    MOV/ADD/SUB/AND/OR/XOR imm     the ALU op on (acc, imm)
    ADDM/SUBM/ANDM/ORM/XORM [a]     the ALU op on (acc, DMEM[a])  (the LOAD flag routes B)
    LOAD [a]                        acc <- DMEM[a]   (U = PASSB, B from the read port)
    JMP t                           PC <- t  (JZ and JNZ both set)
    STORE [a]                       DMEM[a] <- acc   (U = OR, B = 0: acc unchanged, Z updated)
    JZ t / JNZ t                    PC <- t if Z / if not Z  (U = OR, B = 0)
    IRET                            PC <- the link ring (return from the interrupt handler)
    CLR [a]                         DMEM[a] emptied (STORE with the copy vetoed; NEXT on the word's READY)
    LOADI / ADDI.. / STOREI         acc <- DMEM[X] / op with DMEM[X] / DMEM[X] <- acc  (X = the index word)
    HALT                            JZ self and JNZ self
Every instruction runs the ALU and commits the accumulator.

Cycle (times from FETCH's rise, clean model):
    +0    FETCH lit by the PC line's rise; kill train on JT; the old PC line dies at ~+8
    +82   IR <- IMEM[PC] control fields   (fetch relays driven by FETCH delayed 15 hops: the
          old PC line's "not this word" veto needs ~55 ms to decay before a fetch relay can
          fire; measured: at +34 the IR never loaded and every flag stayed 0)
    +166  P  <- IMEM[PC] B/U/SUB          (FETCH delayed 31 hops; B vetoed by IR.LOAD, whose
          other rail may have died at +95: a relay must be driven >= 55 ms after any veto
          rail dies, or the veto's residual blocks it)
    +187  P.B <- DMEM[ADDR] if LOAD; SP set if STORE   (FETCH delayed 35 hops)
    ~+560 stage complete (W_S) -> COMMIT lit -> grant -> master rewritten -> W_M
    W_M's pulse: stage cleared; STORE write started; NEXT lit (unless a store is pending,
    in which case the word's "written" pulse lights NEXT)
    NEXT: JT (jump taken) from IR and the master's Z; 64 ms later the PC advances or jumps;
    the new PC line's rise lights FETCH. A jump to the lit line makes no rise: HALT.

Faults: a refused instruction (stage fault or watchdog) clears P and the stage and leaves the
FSM where it was (FETCH for a fault before completion, COMMIT for one after it): the machine
halts (fail-stop); recovery is later work. The master is not
preloaded (its W_M rise would advance the PC); programs start with MOV or LOAD.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..protocol.celement import add_delay_chain, add_veto_neuron, add_veto_relay
from ..protocol.handshake import Register
from ..protocol.latch import connect_trigger, Latch, add_edge_relay, add_latch
from ..protocol.token import decode_at, decode_recent, rails_for
from ..sim.model import Params
from ..sim.ref64 import RefSim
from .alu import N_UNITS, OPS as ALU_OPS, alu_reference
from .netlist import Drive, Netlist
from .ram import Memory, add_memory, add_read_port, add_write_port, address_vetoes
from .staged import StagedChannel, build_accumulator

# ------------------------------------------------------------------------------ ISA
FLAGS = ("LOAD", "STORE", "JZ", "JNZ", "IRET", "CLR", "LOADI", "STOREI")


def encode(instr: tuple, n: int, a: int) -> int:
    """(op, arg) -> IMEM word. op in MOV ADD SUB AND OR XOR LOAD STORE JZ JNZ HALT; HALT takes
    its own address as arg."""
    op, arg = instr
    b = unit = sub = 0
    flags = {f: 0 for f in FLAGS}
    addr = 0
    if op in ALU_OPS:
        unit, sub = ALU_OPS[op]
        b = arg & ((1 << n) - 1)
    elif op.endswith("M") and op[:-1] in ALU_OPS:  # ADDM [a] etc.: B from data memory (the LOAD flag)
        unit, sub = ALU_OPS[op[:-1]]
        flags["LOAD"], addr = 1, arg
    elif op == "LOAD":
        unit, sub = ALU_OPS["MOV"]
        flags["LOAD"], addr = 1, arg
    elif op == "JMP":
        unit, sub = ALU_OPS["OR"]
        flags["JZ"] = flags["JNZ"] = 1
        addr = arg
    elif op == "IRET":
        unit, sub = ALU_OPS["OR"]
        flags["IRET"] = 1
    elif op == "CLR":  # empty a data word (a port): the write port resets it and the copy is vetoed
        unit, sub = ALU_OPS["OR"]
        flags["STORE"] = flags["CLR"] = 1
        addr = arg
    elif op == "LOADI":  # acc <- DMEM[X]  (X = the index word; arg unused)
        unit, sub = ALU_OPS["MOV"]
        flags["LOADI"] = 1
    elif op.endswith("I") and op[:-1] in ALU_OPS:  # ADDI etc.: the ALU op on (acc, DMEM[X])
        unit, sub = ALU_OPS[op[:-1]]
        flags["LOADI"] = 1
    elif op == "STOREI":  # DMEM[X] <- acc
        unit, sub = ALU_OPS["OR"]
        flags["STOREI"] = 1
    elif op == "STORE":
        unit, sub = ALU_OPS["OR"]
        flags["STORE"], addr = 1, arg
    elif op in ("JZ", "JNZ"):
        unit, sub = ALU_OPS["OR"]
        flags[op], addr = 1, arg
    elif op == "HALT":
        unit, sub = ALU_OPS["OR"]
        flags["JZ"] = flags["JNZ"] = 1
        addr = arg
    else:
        raise ValueError(op)
    w = b | (1 << (n + unit)) | (sub << (n + N_UNITS))
    for k, f in enumerate(FLAGS):
        w |= flags[f] << (n + N_UNITS + 1 + k)
    w |= (addr & ((1 << a) - 1)) << (n + N_UNITS + 1 + len(FLAGS))
    return w


def word_bits(n: int, a: int) -> int:
    return n + N_UNITS + 1 + len(FLAGS) + a


def reference_run(program: list[tuple], n: int, a: int, dmem: dict | None = None, max_steps: int = 64,
                  interrupts_after: list[int] = (), handler_pc: int | None = None, x_word: int | None = None) -> dict:
    """Executes the program in Python: returns the trace of (pc, acc word) after each committed
    instruction, the final DMEM, and whether it halted. `interrupts_after`: execution indices
    after which an interrupt is taken at the safe point (PC saved, jump to `handler_pc`; the
    handler's IRET returns); nested interrupts are masked until IRET."""
    dmem = dict(dmem or {})
    acc = None  # not preloaded
    pc, trace, halted = 0, [], False
    link, masked = None, False
    pending = sorted(interrupts_after)
    for _ in range(max_steps):
        op, arg = program[pc]
        a_val = 0 if acc is None else acc & ((1 << n) - 1)
        if op in ALU_OPS:
            ref = alu_reference(a_val, arg, op, n)
        elif op == "LOAD" or (op.endswith("M") and op[:-1] in ALU_OPS):
            if arg not in dmem:
                return {"trace": trace, "dmem": dmem, "halted": False, "fault": ("unwritten read", pc)}
            ref = alu_reference(a_val, dmem[arg], "MOV" if op == "LOAD" else op[:-1], n)
        elif op == "LOADI" or (op.endswith("I") and op[:-1] in ALU_OPS and op != "STOREI"):
            xa = dmem.get(x_word)
            if xa is None or xa not in dmem:
                return {"trace": trace, "dmem": dmem, "halted": False, "fault": ("unwritten indexed read", pc)}
            ref = alu_reference(a_val, dmem[xa], "MOV" if op == "LOADI" else op[:-1], n)
        else:  # STORE, JZ, JNZ, JMP, HALT: OR 0
            ref = alu_reference(a_val, 0, "OR", n)
        acc = ref["word"]
        trace.append((pc, acc))
        z = ref["z"]
        if op == "STORE":
            dmem[arg] = ref["r"]
        elif op == "STOREI":
            dmem[dmem[x_word]] = ref["r"]
        elif op == "CLR":
            dmem.pop(arg, None)
        taken = op in ("HALT", "JMP") or (op == "JZ" and z) or (op == "JNZ" and not z)
        if taken and arg == pc:
            halted = True
            break
        if op == "IRET":
            pc, masked = link, False
        else:
            pc = arg if taken else (pc + 1) % len(program)
        if pending and len(trace) - 1 >= pending[0] and not masked and handler_pc is not None:
            pending.pop(0)
            link, masked, pc = pc, True, handler_pc
    return {"trace": trace, "dmem": dmem, "halted": halted, "fault": None}


# ------------------------------------------------------------------------------ rings
@dataclass
class Ring:
    lines: list  # Latch per line
    killers: list  # inhibitory interneuron per line (fires a 3-pulse train at that line's rise, on the other lines)


def add_onehot_ring(net: Netlist, drive: Drive, name: str, n: int) -> Ring:
    """n latches; a line's *rise* (one relay) fires a short kill train on every other line.
    The kill must be edge-triggered: a lit line that inhibited the others continuously would
    hold its own successor down and no transition could ever happen (measured: the FSM never
    left FETCH). After a kill train the killed lines recover in ~80 ms; a line is re-lit at
    least one instruction (>= 600 ms) later."""
    lines = [add_latch(net, drive, f"{name}.{k}") for k in range(n)]
    killers = []
    for k, l in enumerate(lines):
        others = [o for j, o in enumerate(lines) if j != k]
        killers.append(add_kill_train(net, drive, f"{name}.{k}.kill", l.u, others))
    return Ring(lines, killers)


def add_kill_pair(net: Netlist, drive: Drive, name: str) -> list[Latch]:
    """A dual-rail bit whose rail latches kill each other at their rise: loading a value ignites
    the right rail, whose rise fires a kill train on the other rail. No reset train, no
    continuous inhibition; a reload to the other rail works ~20 ms after the previous load
    (a reload to the same rail is a harmless re-ignition). Both rails can be live for ~10 ms
    at a reload; readers sample later."""
    r0, r1 = add_latch(net, drive, f"{name}r0"), add_latch(net, drive, f"{name}r1")
    add_kill_train(net, drive, f"{name}.k0", r0.u, [r1])
    add_kill_train(net, drive, f"{name}.k1", r1.u, [r0])
    return [r0, r1]


# The kill train's shape: 3 pulses x 0.75 loop, on both members. Known margin problem
# (docs/tick_stalls.md, seed-108 copy 8): a latch whose loop came out fast under mix-B noise
# (+20 % weights / -0.8 mV threshold: 38 steps instead of 47) survives this train at every
# phase, and one such request rail stalled a copy. Stronger trains were measured on
# 2026-09-20: 3 x 1.5 kills down to 33 steps at every phase in isolation, but in the 100-copy
# mix-B tick campaign it stalled 85 of 100 copies (Juno 413672) — the extra
# after-hyperpolarisation makes the rails' relights, ~190 ms after a kill, fail under noise —
# and 4 x 1.5 stops the control machine outright. The train is unchanged; a kill that beats a
# fast latch without slowing the relight is a separate experiment (tests/test_kill_margin.py
# keeps the measurement). Campaign records carry the train as `kill_train`.
KILL_PULSES = 3  # the kernel's request-rail clear alone uses four (lib/kernel.py REQUEST_CLEAR_PULSES): a fourth pulse everywhere lost the control machine's 45 ms interrupt reload (tests/test_machine.py)
KILL_STRENGTH = 0.75


def add_kill_train(net: Netlist, drive: Drive, name: str, source: int, latches: list[Latch], pulses: int | None = None,
                   strength: float | None = None) -> int:
    """`source`'s rise (one relay) starts a short train of `pulses` inhibitory pulses on every
    member of `latches` (like a reset train, without an edge detector: the source is a relay
    pulse or a fresh latch's rise). Defaults: KILL_PULSES x KILL_STRENGTH (see above)."""
    pulses = KILL_PULSES if pulses is None else pulses
    strength = KILL_STRENGTH if strength is None else strength
    relay = add_edge_relay(net, drive, f"{name}.start", source, fast_inhibitor=True)
    inh = net.neuron(f"{name}.inh")
    prev = relay
    net.synapse(prev, inh, drive.pulse)
    for k in range(1, pulses):
        h = net.neuron(f"{name}.h{k}")
        net.synapse(prev, h, drive.pulse)
        net.synapse(h, inh, drive.pulse)
        prev = h
    q = -int(round(strength * drive.loop))
    for l in latches:
        for x in l.members:
            net.synapse(inh, x, q)
    return inh


# ------------------------------------------------------------------------------ machine
@dataclass
class Machine:
    net: Netlist
    drive: Drive
    n: int
    a: int
    n_prog: int
    acc: StagedChannel
    imem: list  # per word: list of Latch pairs [ [r0, r1], ... ] (word_bits entries)
    pc: Ring
    fsm: Ring  # lines: FETCH, COMMIT, NEXT
    ir: list  # [ [r0, r1] ... ] for LOAD, STORE, JZ, JNZ, ADDR bits
    dmem: Memory
    jt: Latch
    sp: Latch
    wdone: list = field(default_factory=list)
    intp: list | None = None  # interrupt pending kill pair [r0, r1]; the host ignites r1
    intq: list | None = None  # one-deep interrupt queue [empty, queued]
    ints: list | None = None  # INTP settled kill pair [unsettled, settled-empty]
    handler_pc: int | None = None
    lr: Ring | None = None
    link_out_taps: list = field(default_factory=list)  # [[r0 relay, r1 relay] per bit] of the output port word
    link_in_word: int | None = None

    @property
    def fetch(self) -> Latch:
        return self.fsm.lines[0]

    @property
    def commit(self) -> Latch:
        return self.fsm.lines[1]

    @property
    def next(self) -> Latch:
        return self.fsm.lines[2]


def build_machine(params: Params, n: int = 4, n_prog: int = 8, n_data: int = 8, drive: Drive | None = None, *,
                  ir_hops: int = 15, p_hops: int = 31, rd_hops: int = 35, act_hops: int = 17, next_hops: int = 12,
                  watchdog_hops: int | None = None, commit_wd_hops: int = 150, handler_pc: int | None = None, port_out_word: int | None = None,
                  port_in_word: int | None = None, timer_hops: int | None = None, status_word: int | None = None,
                  x_word: int | None = None, mul: bool = False) -> Machine:
    """`handler_pc`: enables safe-point interrupts with the handler at that program word.
    `port_out_word`: a data word whose rails' rises are exported as FlyLink events (one relay
    per rail, taps in Machine.link_out_taps). `port_in_word`: a data word whose completion
    raises the interrupt (a message arrived); the transport ignites its rails.
    `x_word`: the index register: a data word whose value addresses LOADI/STOREI/ADDI... (a second
    read port and write port over the bank, with that word's rails as the address taps).
    `timer_hops` + `status_word`: a send timer started by the output word's completion and
    cancelled when the handler consumes the input word (its CLR reaches READY); on expiry it
    writes the constant 1 into the status word (all bits) and raises the interrupt, so a
    handler can tell a timeout (status != 0) from an arrival and retransmit (STORE the output
    word again)."""
    drive = drive or Drive.from_params(params)
    a = max(1, (max(n_prog, n_data) - 1).bit_length())
    if watchdog_hops is None:
        watchdog_hops = 260 if mul else 170  # the producer watchdog must outlast a MUL (~1.1 s)
    acc = build_accumulator(params, n, drive=drive, watchdog_hops=watchdog_hops, act_hops=act_hops, ordered_grant=True, mul=mul)
    net = acc.net
    P, S, M, R = acc.producer, acc.reg.stage, acc.reg.master, acc.reg
    nb = word_bits(n, a)
    # program memory: latch-only words
    imem = [[[add_latch(net, drive, f"IM.w{w}.b{i}r{r}") for r in (0, 1)] for i in range(nb)] for w in range(n_prog)]
    pc = add_onehot_ring(net, drive, "PC", n_prog)
    fsm = add_onehot_ring(net, drive, "FSM", 3)
    FETCH, COMMIT, NEXT = fsm.lines
    # instruction register: control fields
    n_ctrl = len(FLAGS) + a
    ir = [add_kill_pair(net, drive, f"IR.b{k}") for k in range(n_ctrl)]
    ir_load, ir_store, ir_jz, ir_jnz, ir_iret, ir_clr, ir_loadi, ir_storei = ir[:8]
    ir_addr = ir[8:]
    addr_taps = [[l.u for l in pair] for pair in ir_addr]
    jt = add_latch(net, drive, "JT")
    sp = add_kill_pair(net, drive, "SP")  # [r0 = no store pending, r1 = pending]: a pair, so both states are rails
    # FETCH is lit by any PC line's rise
    for k, line in enumerate(pc.lines):
        rl = add_edge_relay(net, drive, f"PC.{k}.fetch", line.u)
        net.synapse(rl, FETCH.u, drive.ignite)
    # FETCH's rise clears JT (the IR bits clear themselves: kill pairs)
    add_kill_train(net, drive, "JTkill", FETCH.u, [jt])
    f_ir = add_delay_chain(net, drive, "FETCH.ird", FETCH.u, ir_hops)
    f_p = add_delay_chain(net, drive, "FETCH.pd", FETCH.u, p_hops)
    f_rd = add_delay_chain(net, drive, "FETCH.rdd", FETCH.u, rd_hops)
    # fetch relays: one per (word, bit) into the rail the word holds, vetoed by PC != w and by
    # the word's other rail (so an unloaded word ignites nothing)
    for w in range(n_prog):
        notw = add_veto_neuron(net, drive, f"IM.w{w}.notw", [pc.lines[j].u for j in range(n_prog) if j != w])
        for i in range(nb):
            for r in (0, 1):
                if i < n + N_UNITS + 1:  # B, U, SUB -> P
                    extra = [ir_load[1].u, ir_loadi[1].u] if i < n else []  # a LOAD/LOADI takes B from DMEM
                    add_veto_relay(net, drive, f"IM.w{w}.f{i}r{r}", f_p, [imem[w][i][1 - r].u] + extra, P.rails[i][r],
                                   veto_neurons=[notw])
                else:  # control fields -> IR
                    k = i - (n + N_UNITS + 1)
                    add_veto_relay(net, drive, f"IM.w{w}.f{i}r{r}", f_ir, [imem[w][i][1 - r].u], ir[k][r], veto_neurons=[notw])
    # data memory: read port into P's B rails for LOAD; write port from the master for STORE
    dmem = add_memory(net, drive, "DM", n_data, n)
    add_read_port(net, drive, "LD", dmem, f_rd, addr_taps, [P.rails[i] for i in range(n)], extra_vetoes=[ir_load[0].u])
    if x_word is not None:  # indexed ports: the address taps are the index word's rails
        x_taps = [[dmem.words[x_word].rails[j][0].u, dmem.words[x_word].rails[j][1].u] for j in range(a)]
        add_read_port(net, drive, "LDI", dmem, f_rd, x_taps, [P.rails[i] for i in range(n)], extra_vetoes=[ir_loadi[0].u])
    # FETCH -> COMMIT on the stage's completion; COMMIT lights the register's COMMIT token
    rl = add_edge_relay(net, drive, "FSM.ws", S.completion.u)
    net.synapse(rl, COMMIT.u, drive.ignite)
    rl = add_edge_relay(net, drive, "FSM.commit_token", COMMIT.u, fast_inhibitor=True)
    net.synapse(rl, R.commit_in, drive.ignite)  # one COMMIT token pulse
    # store pending: r1 set during FETCH if STORE; r0 re-asserted at NEXT's rise
    add_veto_relay(net, drive, "SP.set", f_rd, [ir_store[0].u], sp[1])
    add_veto_relay(net, drive, "SP.clear", NEXT.u, [], sp[0])
    # commit done (W_M's pulse): NEXT unless a store is pending; the store write starts on it.
    # Both NEXT-lighting paths are vetoed by FETCH and NEXT: a "written" pulse from the data
    # image at power-up, or any done pulse outside COMMIT, must not advance the PC (measured:
    # the image's completions lit NEXT at 70 ms and the PC moved under the first fetch). A
    # word written by hardware (the send timer's status word) completes without a STORE, so
    # its "written" pulse is also vetoed by "no store pending".
    add_veto_relay(net, drive, "FSM.done_next", R.done_relay, [sp[1].u, FETCH.u, NEXT.u], NEXT)
    wp = add_write_port(net, drive, "ST", dmem, R.done_relay, addr_taps, [[M.rails[i][0].u, M.rails[i][1].u] for i in range(n)],
                        extra_vetoes=[ir_store[0].u], copy_vetoes=[S.fault_latch.u, ir_clr[1].u])
    # The written / cleared pulses count only for the addressed word: a word completed by
    # hardware (a link arrival, the timer's status) while a STORE to another word is pending
    # would otherwise light NEXT ~300 ms early and lose the store (review finding, 2026-09-14).
    for w, d in enumerate(wp.done):  # the word is written: NEXT
        add_veto_relay(net, drive, f"FSM.wdone{w}", d, [FETCH.u, NEXT.u, sp[0].u] + address_vetoes(addr_taps, w), NEXT)
    for w, word in enumerate(dmem.words):  # a CLR is done when the word's READY fires (it stays empty)
        add_veto_relay(net, drive, f"FSM.wclr{w}", word.ready, [FETCH.u, NEXT.u, sp[0].u, ir_clr[0].u] + address_vetoes(addr_taps, w), NEXT)
    if x_word is not None:  # indexed store: the same trigger, the index word's rails as the address
        wpi = add_write_port(net, drive, "STI", dmem, R.done_relay, x_taps, [[M.rails[i][0].u, M.rails[i][1].u] for i in range(n)],
                             extra_vetoes=[ir_storei[0].u], copy_vetoes=[S.fault_latch.u])
        for w, d in enumerate(wpi.done):
            add_veto_relay(net, drive, f"FSM.widone{w}", d, [FETCH.u, NEXT.u, sp[0].u] + address_vetoes(x_taps, w), NEXT)
        for l in wpi.domain_latches:
            for x in l.members:
                net.synapse(S.reset_inh, x, -int(round(0.75 * drive.loop)))
        add_veto_relay(net, drive, "SP.seti", f_rd, [ir_storei[0].u], sp[1])  # a STOREI is a pending store too
    for l in wp.domain_latches:  # COPY_w: cleared with the stage
        for x in l.members:
            net.synapse(S.reset_inh, x, -int(round(0.75 * drive.loop)))
    # Commit watchdog: a stage fault after the grant, or a late rail, can leave the FSM in COMMIT
    # with the master empty and nothing else running (review finding: no timeout covered it).
    # COMMIT's rise starts a chain; if COMMIT still holds when it ends, a TIMEOUT latch lights
    # (fail-stop; the runner and the campaign classifier count it), and it clears the stage.
    cwd = add_delay_chain(net, drive, "FSM.cwd", COMMIT.u, commit_wd_hops)
    cto = add_latch(net, drive, "FSM.cwd.timeout")
    add_veto_relay(net, drive, "FSM.cwd.fire", cwd, [FETCH.u, NEXT.u], cto)
    connect_trigger(net, drive, cto.u, S.reset_trigger, S.reset_edge)
    for x in cto.members:  # the timeout clears with the stage it reset (review: it was permanent)
        net.synapse(S.reset_inh, x, -int(round(0.75 * drive.loop)))
    # NEXT: jump taken? (JZ and Z) or (JNZ and not Z); Z is the master's bit n+1
    z0, z1 = M.rails[n + 1][0].u, M.rails[n + 1][1].u
    add_veto_relay(net, drive, "JT.z", NEXT.u, [ir_jz[0].u, z0], jt)
    add_veto_relay(net, drive, "JT.nz", NEXT.u, [ir_jnz[0].u, z1], jt)
    nx = add_delay_chain(net, drive, "NEXT.d", NEXT.u, next_hops)
    # interrupts (safe point = NEXT, after the commit and any store)
    intp = intq = ints = mask = lr = None
    it = irt = None
    extra_pc_vetoes = []
    if handler_pc is not None:
        intp = add_kill_pair(net, drive, "INTP")  # r1 = pending (host), r0 = none
        # A request that arrives while INTP is already pending must not merge with it: taking
        # the first request asserts INTP.r0, which would otherwise erase both. INTQ stores one
        # additional request (the required depth for one message in flight plus its timeout).
        # Sources below always set INTP.r1 and also try INTQ.r1. INTS distinguishes a genuinely
        # settled-empty INTP from the refractory interval after INTP.clear: r1 vetoes the queue
        # only after that interval, while every actual INTP.r1 rise promptly asserts r0. Thus a
        # first request is taken directly, but a pending or temporarily paralysed INTP queues it.
        # Promotion waits 20 hops (~106 ms) after INTP.clear, beyond INTP's ~80 ms paralysis.
        intq = add_kill_pair(net, drive, "INTQ")  # r1 = one queued request, r0 = empty
        ints = add_kill_pair(net, drive, "INTS")  # r1 = INTP settled-empty, r0 = pending/unsettled
        add_veto_relay(net, drive, "INTS.unsettle", intp[1].u, [], ints[0])
        # Normally the prompt relay above flips a long-settled INTS before another request can
        # arrive. If INTP.r1 rises just after INTS.r1, however, that rail's kill train has made
        # r0 temporarily refractory; each concrete request source also retries after 12 hops
        # (~64 ms), when both INTS members have recovered.
        mask = add_kill_pair(net, drive, "MASK")  # r1 = in the handler
        lr = add_onehot_ring(net, drive, "LR", n_prog)
        it = add_latch(net, drive, "IT")  # interrupt taken this NEXT
        irt = add_latch(net, drive, "IRT")  # IRET taken this NEXT
        add_kill_train(net, drive, "ITkill", FETCH.u, [it, irt])
        add_veto_relay(net, drive, "IT.take", NEXT.u, [intp[0].u, mask[1].u, ir_iret[1].u], it)
        add_veto_relay(net, drive, "IRT.take", NEXT.u, [ir_iret[0].u], irt)
        itd = add_delay_chain(net, drive, "IT.d", it.u, next_hops)
        for w in range(n_prog):  # save the lit PC line into LR, then jump to the handler
            others = [pc.lines[j].u for j in range(n_prog) if j != w]
            add_veto_relay(net, drive, f"LR.save{w}", itd, others, lr.lines[w])
        itd2 = add_delay_chain(net, drive, "IT.d2", itd, 3)
        add_veto_relay(net, drive, "PC.handler", itd2, [], pc.lines[handler_pc])
        add_veto_relay(net, drive, "INTP.clear", itd, [], intp[0])  # taken: no longer pending
        add_veto_relay(net, drive, "MASK.set", itd, [], mask[1])
        settled = add_delay_chain(net, drive, "INTS.settle_d", itd, 8)
        add_veto_relay(net, drive, "INTS.settle", settled, [], ints[1])
        qpromote = add_delay_chain(net, drive, "INTQ.promote_d", itd, 20)
        promote = add_veto_relay(net, drive, "INTQ.promote", qpromote, [intq[0].u], intp[1])
        net.synapse(promote, ints[0].u, drive.ignite)
        qclear = add_delay_chain(net, drive, "INTQ.clear_d", qpromote, 3)
        add_veto_relay(net, drive, "INTQ.clear", qclear, [], intq[0])
        irtd = add_delay_chain(net, drive, "IRT.d", irt.u, next_hops)
        for w in range(n_prog):  # return: the lit LR line -> PC line
            others = [lr.lines[j].u for j in range(n_prog) if j != w]
            add_veto_relay(net, drive, f"PC.ret{w}", irtd, others, pc.lines[w])
        add_veto_relay(net, drive, "MASK.clear", irtd, [], mask[0])
        extra_pc_vetoes = [it.u, irt.u]
    for w in range(n_prog):  # increment: line w lit and no jump -> line w+1
        others = [pc.lines[j].u for j in range(n_prog) if j != w]
        add_veto_relay(net, drive, f"PC.inc{w}", nx, others + [jt.u] + extra_pc_vetoes, pc.lines[(w + 1) % n_prog])
    jd = add_delay_chain(net, drive, "JT.d", jt.u, next_hops)
    for t in range(n_prog):  # jump: target t
        add_veto_relay(net, drive, f"PC.jmp{t}", jd, address_vetoes(addr_taps, t) + extra_pc_vetoes, pc.lines[t])
    link_out = []
    if port_out_word is not None:  # FlyLink send: one event per rail rise of the output port word
        word = dmem.words[port_out_word]
        for i in range(n):
            link_out.append([add_edge_relay(net, drive, f"LINK.out.b{i}r{r}", word.rails[i][r].u, fast_inhibitor=True) for r in (0, 1)])
    arr = None
    if port_in_word is not None:  # FlyLink receive: the input port word's completion is an interrupt
        assert intp is not None, "port_in_word needs handler_pc (the arrival is an interrupt)"
        arr = add_edge_relay(net, drive, "LINK.in.arrive", dmem.words[port_in_word].completion.u, fast_inhibitor=True)
        net.synapse(arr, intp[1].u, drive.ignite)
        add_veto_relay(net, drive, "INTQ.arrive", arr, [ints[1].u], intq[1])
        arr_unsettle_retry = add_delay_chain(net, drive, "INTS.arrive_retry_d", arr, 12)
        net.synapse(arr_unsettle_retry, ints[0].u, drive.ignite)
    if timer_hops is not None:  # send timer: out-word completion starts it, consuming in-word cancels it
        assert port_out_word is not None and status_word is not None and intp is not None
        start = add_edge_relay(net, drive, "TIMER.start", dmem.words[port_out_word].completion.u, fast_inhibitor=True)
        chain, prev = [], start
        for k in range(timer_hops):
            h = net.neuron(f"TIMER.h{k}")
            net.synapse(prev, h, drive.pulse)
            chain.append(h)
            prev = h
        cancel = net.neuron("TIMER.cancel_inh")
        if port_in_word is not None:
            # Any evidence of an ACK cancels the timer, twice over. (1) The arrival's completion
            # train: the ACK is here, the send was not lost (without this the clean link
            # retried once — the handler consumes the word seconds after the timer expires).
            # (2) The word's READY after the handler's CLR: a second ACK (to a retransmission)
            # that lands while the first is still held never completes, so consuming the first
            # must cancel the retransmission's timer too. READY becomes four pulses so the
            # active timer hop cannot slip between one cancellation pulse and the next.
            net.synapse(dmem.words[port_in_word].completion.u, cancel, drive.pulse)
            cancel_src = dmem.words[port_in_word].ready
            net.synapse(cancel_src, cancel, drive.pulse)
            for k in range(1, 4):
                h = net.neuron(f"TIMER.cancel_h{k}")
                net.synapse(cancel_src, h, drive.pulse)
                net.synapse(h, cancel, drive.pulse)
                cancel_src = h
        for h in chain:
            net.synapse(cancel, h, -int(round(1.5 * drive.loop)))
        # status <- 1 through a proper write (reset, READY, copy of a constant-1 level): the word
        # holds 0 between timeouts, so a handler can LOAD it on either path (an empty word would
        # double-rail a LOAD and halt the machine; measured on the exact-channel test)
        one = [[add_latch(net, drive, f"TIMER.one.b{i}r{r}") for r in (0, 1)] for i in range(n)]
        wps = add_write_port(net, drive, "TIMER.wr", Memory([dmem.words[status_word]], n), prev, [],
                             [[one[i][0].u, one[i][1].u] for i in range(n)])
        for l in wps.domain_latches:  # the COPY latch clears with the stage
            for x in l.members:
                net.synapse(S.reset_inh, x, -int(round(0.75 * drive.loop)))
        # Two ports share the status word, and every port's COPY latch lights at the word's
        # READY: without a guard the timer's reset also copies the master (the STORE port's data)
        # into the word, double-railing it (measured: the handler's first LOAD of the status
        # halted the machine). Each port marks "my write is in flight" from its select until the
        # word completes, and that level vetoes the other port's copies (>= 40 ms set-up: READY
        # comes after the word's hold recovery). A program must not STORE to the status word
        # while the timer runs (both marks lit would leave the word empty).
        tsel, msel = add_latch(net, drive, "TIMER.sel"), add_latch(net, drive, "TIMER.msel")
        net.synapse(prev, tsel.u, drive.ignite)
        others = [wp] + ([wpi] if x_word is not None else [])
        for port in others:
            net.synapse(port.selects[status_word], msel.u, drive.ignite)
            for pair in port.copy_relays[status_word]:
                for relay in pair:
                    net.synapse(tsel.u, relay, -int(round(0.5 * drive.loop)))
        for pair in wps.copy_relays[0]:
            for relay in pair:
                net.synapse(msel.u, relay, -int(round(0.5 * drive.loop)))
        add_kill_train(net, drive, "TIMER.sel.kill", wp.done[status_word], [tsel, msel])
        net.synapse(prev, intp[1].u, drive.ignite)
        add_veto_relay(net, drive, "INTQ.timeout", prev, [ints[1].u], intq[1])
        timeout_unsettle_retry = add_delay_chain(net, drive, "INTS.timeout_retry_d", prev, 12)
        net.synapse(timeout_unsettle_retry, ints[0].u, drive.ignite)
        m_status_one = one
    mach = Machine(net, drive, n, a, n_prog, acc, imem, pc, fsm, ir, dmem, jt, sp[1], wp.done,
                   intp, intq, ints, handler_pc, lr, link_out, port_in_word)
    mach.status_word = status_word
    mach.status_one = locals().get("m_status_one")
    mach.commit_timeout = cto
    return mach


# ------------------------------------------------------------------------------ runner
@dataclass
class MachineRun:
    commits: list  # (step, decoded master word)
    pcs: list  # (step, line) rises
    writes: list  # (step, word index, decoded value)
    halted: bool
    faults: int
    timeouts: int


def load_image(sim, m: Machine, program: list[tuple], dmem: dict | None, node: int = 0, step: int = 1) -> None:
    for w, instr in enumerate(program):
        word = encode(instr, m.n, m.a)
        for i, r in rails_for(word, word_bits(m.n, m.a)):
            sim.add_events(node, [step], [m.imem[w][i][r].u], [m.drive.ignite])
    for addr, v in (dmem or {}).items():
        for i, r in rails_for(v, m.n):
            sim.add_events(node, [step], [m.dmem.words[addr].rails[i][r].u], [m.drive.ignite])
    sim.add_events(node, [step], [m.net.roles.index("SPr0.u")], [m.drive.ignite])  # "no store pending" is a rail
    if getattr(m, "status_word", None) is not None:  # the timer's status word starts at 0; its constant-1 source is a level
        for i, r in rails_for(0, m.n):
            sim.add_events(node, [step], [m.dmem.words[m.status_word].rails[i][r].u], [m.drive.ignite])
        for i, r in rails_for(1, m.n):
            sim.add_events(node, [step], [m.status_one[i][r].u], [m.drive.ignite])
    if m.intp is not None:  # "no interrupt pending" and "not masked" are asserted rails, not silence
        sim.add_events(node, [step], [m.intp[0].u], [m.drive.ignite])
        sim.add_events(node, [step], [m.intq[0].u], [m.drive.ignite])
        sim.add_events(node, [step], [m.ints[1].u], [m.drive.ignite])
        mask_r0 = m.net.roles.index("MASKr0.u")
        sim.add_events(node, [step], [mask_r0], [m.drive.ignite])
    sim.add_events(node, [step + 20], [m.pc.lines[0].u], [m.drive.ignite])  # FETCH is lit before the image completes


def run_machine(m: Machine, params: Params, program: list[tuple], dmem: dict | None = None, *, max_ms: float = 20000,
                idle_ms: float = 1500, sim=None, interrupts_at_ms: list[float] = ()) -> tuple[MachineRun, RefSim]:
    """Loads the image, lights PC line 0, and runs until no commit and no PC rise for `idle_ms`
    (halt) or `max_ms`. Decodes the master at every W_M re-ignition and each written word.
    `interrupts_at_ms`: times at which the host ignites the interrupt-pending rail."""
    net, drive = m.net, m.drive
    sim = sim or RefSim(net.topology(), params)
    load_image(sim, m, program, dmem)
    for t in interrupts_at_ms:
        sim.add_events(0, [int(t / params.dt)], [m.intp[1].u], [drive.ignite])
    M = m.acc.reg.master
    wm = M.completion.u
    window = 2 * drive.loop_period_steps
    fault_set = set(m.acc.reg.stage.fault) | {m.acc.reg.stage.fault_latch.u}
    timeout_n = m.acc.producer.watchdog.timeout.u if m.acc.producer.watchdog is not None else -1
    cto_n = m.commit_timeout.u if getattr(m, "commit_timeout", None) is not None else -1
    pc_taps = {l.u: k for k, l in enumerate(m.pc.lines)}
    w_taps = {w.completion.u: k for k, w in enumerate(m.dmem.words)}
    last = {}
    commits, pcs, writes = [], [], []
    faults = timeouts = 0
    last_event = 0
    idle = int(idle_ms / params.dt)
    while sim.step_index < int(max_ms / params.dt):
        sim.step()
        s_ = sim.step_index - 1
        if not sim._spk_step or sim._spk_step[-1][0] != s_:
            if s_ - last_event > idle and s_ > 3000:
                break
            continue
        for n_ in sim._spk_neuron[-1].tolist():
            prev = last.get(n_)
            rise = prev is None or s_ - prev > 3 * drive.loop_period_steps
            last[n_] = s_
            if not rise:
                continue
            if n_ == wm:
                commits.append((s_, decode_recent(sim, M.rail_taps, s_, window)[0]))
                last_event = s_
            elif n_ in pc_taps:
                pcs.append((s_, pc_taps[n_]))
                last_event = s_
            elif n_ in w_taps:
                k = w_taps[n_]
                writes.append((s_, k, decode_recent(sim, m.dmem.words[k].rail_taps, s_, window)[0]))
            elif n_ in fault_set:
                faults += 1
            elif n_ == timeout_n or n_ == cto_n:
                timeouts += 1
        if s_ - last_event > idle and s_ > 3000:
            break
    halted = sim.step_index < int(max_ms / params.dt)
    return MachineRun(commits, pcs, writes, halted, faults, timeouts), sim


# ------------------------------------------------------------------------------ programs and campaigns
def random_program(rng, n: int = 4, a: int = 3, n_prog: int = 8, max_exec: int = 20, mul: bool = False) -> tuple[list[tuple], dict]:
    """A random n_prog-word program that the reference halts within `max_exec` instructions,
    with a data image so that every LOAD reads a written word. The last word is HALT.
    `mul`: include MUL (only for machines built with a multiplier)."""
    ops = ["MOV", "ADD", "SUB", "AND", "OR", "XOR", "STORE", "LOAD", "JZ", "JNZ"] + (["MUL"] if mul else [])
    while True:
        dmem = {int(rng.integers(0, 1 << a)): int(rng.integers(0, 1 << n)) for _ in range(3)}
        prog = []
        for k in range(n_prog - 1):
            op = ops[int(rng.integers(0, len(ops)))] if k else "MOV"  # start with MOV (the master is not preloaded)
            if op in ("STORE", "LOAD"):
                arg = int(rng.integers(0, 1 << a))
            elif op in ("JZ", "JNZ"):
                arg = int(rng.integers(0, n_prog))
            else:
                arg = int(rng.integers(0, 1 << n))
            prog.append((op, arg))
        prog.append(("HALT", n_prog - 1))
        ref = reference_run(prog, n, a, dmem, max_steps=max_exec)
        if ref["halted"] and ref["fault"] is None and len(ref["trace"]) >= 3:
            return prog, dmem


def _rises(steps, neurons, taps: dict, period: int):
    """(step, key) at every rise (first spike after >= 3 periods of silence) of the tapped neurons."""
    out, last = [], {}
    for s_, n_ in zip(steps, neurons):
        k = taps.get(int(n_))
        if k is None:
            continue
        prev = last.get(int(n_))
        if prev is None or s_ - prev > 3 * period:
            out.append((int(s_), k))
        last[int(n_)] = int(s_)
    return out


def classify_machine_run(m: Machine, ev_steps, ev_neurons, program, dmem, params: Params) -> dict:
    """Compare one node's spike record with the reference execution. Classes:
    ok / wrong_value (a committed word differs: silent) / wrong_memory / short (fewer commits:
    hang or refusal) / no_halt (more commits than the reference) ."""
    ref = reference_run(program, m.n, m.a, dmem)
    M = m.acc.reg.master
    period = m.drive.loop_period_steps
    window = 2 * period
    taps = {M.completion.u: ("wm", 0)}
    for w, word in enumerate(m.dmem.words):
        taps[word.completion.u] = ("w", w)
    taps[m.acc.reg.stage.fault_latch.u] = ("fault", 0)
    if m.acc.producer.watchdog is not None:
        taps[m.acc.producer.watchdog.timeout.u] = ("timeout", 0)
    if getattr(m, "commit_timeout", None) is not None:  # a hung COMMIT is a counted timeout too
        taps[m.commit_timeout.u] = ("timeout", 0)
    rises = _rises(ev_steps, ev_neurons, taps, period)

    def decode(rails, step):
        act = set(ev_neurons[(ev_steps > step - window) & (ev_steps <= step)].tolist())
        v = 0
        for i, (r0, r1) in enumerate(rails):
            a0, a1 = r0 in act, r1 in act
            if a0 and a1 or not (a0 or a1):
                return None
            v |= a1 << i
        return v

    commits = [decode(M.rail_taps, s_) for s_, k in rises if k[0] == "wm"]
    dmem_final = dict(dmem or {})
    for s_, k in rises:
        if k[0] == "w":
            dmem_final[k[1]] = decode(m.dmem.words[k[1]].rail_taps, s_)
    n_fault = sum(1 for _, k in rises if k[0] in ("fault", "timeout"))
    exp = [t[1] for t in ref["trace"]]
    if any(c is None for c in commits):
        cls = "master_fault"  # an undecodable committed word (both rails or a missing bit) is a fault, never skipped
    elif any(c is not None and i < len(exp) and c != exp[i] for i, c in enumerate(commits)):
        cls = "wrong_value"
    elif len(commits) < len(exp):
        cls = "short"
    elif len(commits) > len(exp):
        cls = "no_halt"
    elif dmem_final != ref["dmem"]:
        cls = "wrong_memory"
    else:
        cls = "ok"
    return {"class": cls, "commits": commits, "expected": exp, "dmem": dmem_final, "expected_dmem": ref["dmem"],
            "faults": n_fault, "instructions": len(exp)}


def run_machine_batch(m: Machine, params: Params, programs: list, dmems: list, pert, rng, max_ms: float = 25000,
                      device: str = "cpu") -> list[dict]:
    """B perturbed copies of the machine, one program each, on the batched simulator."""
    from .campaign import make_perturbed_sim

    B = len(programs)
    n_steps = int(max_ms / params.dt)
    topo = m.net.topology()
    sim = make_perturbed_sim(topo, params, B, pert, rng, n_steps, device=device)
    for b in range(B):
        load_image(sim, m, programs[b], dmems[b], node=b)
    sim.run(n_steps)
    ev = sim.trace.events
    order = np.lexsort((ev["step"], ev["node"]))
    ev = ev[order]
    bounds = np.searchsorted(ev["node"], np.arange(B + 1))
    out = []
    for b in range(B):
        sl = slice(bounds[b], bounds[b + 1])
        out.append(classify_machine_run(m, ev["step"][sl], ev["neuron"][sl], programs[b], dmems[b], params))
    return out
