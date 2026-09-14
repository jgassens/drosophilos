"""Data memory on veto relays (plan §A2: RAM bank).

A bank is W words of n dual-rail bits; every word is a master register (rails, bit-valid
ORs, completion tree W_w, its own reset controller and READY chain), exactly the staged
register's master. Decoding is exact and needs no threshold margin: a word-select is a veto
relay whose vetoes are the address rails that disagree with w, so any one mismatching rail
blocks it. (Hierarchical predecoding, the plan's remedy for threshold decoders, would only
reduce each relay's fan-in here; the neuron count is 3 per word per port either way, and
only allocated words are instantiated.)

Write port  trigger --WS_w (vetoed by addr != w)--> word w reset --> word READY --> COPY_w
            latch --> per-rail veto relays (driver COPY_w, veto = the data source's other
            rail) --> word rails --> W_w --> edge relay = "written" pulse.
Read port   driver --per-rail veto relays (vetoes: addr != w, the word's other rail)--> the
            destination rails (a producer register, a stage, the ALU's operand rails).

Ordering assumptions (veto relays): the address and data rails are valid >= 15 ms before
the trigger; the words' rails are levels. A read of a word that was never written has no
rail to veto either relay, so both rails of every destination bit ignite: the destination's
fault gates refuse the word (measured: refused and recovered, the stage cleared by the
read consumer's fault latch).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..protocol.celement import add_delay_chain, add_veto_neuron, add_veto_relay
from ..protocol.handshake import Register, add_liveness, add_register, wire_fault_path
from ..protocol.latch import Latch, add_edge_relay, add_latch, connect_trigger
from ..protocol.token import decode_at, rails_for
from ..sim.model import Params
from ..sim.ref64 import RefSim
from .netlist import Drive, Netlist


def address_vetoes(addr_rails, w: int) -> list[int]:
    """The taps that are live when the address is NOT w: for each bit j, the rail opposite
    to w's bit j."""
    return [addr_rails[j][1 - ((w >> j) & 1)] for j in range(len(addr_rails))]


@dataclass
class Memory:
    words: list  # Register per word (masters)
    width: int
    not_word: list = field(default_factory=list)  # shared "address != w" veto neurons, if a port made them

    @property
    def n_words(self) -> int:
        return len(self.words)


def add_memory(net: Netlist, drive: Drive, name: str, n_words: int, width: int) -> Memory:
    return Memory([add_register(net, drive, f"{name}.w{w}", width, with_completion=True) for w in range(n_words)], width)


@dataclass
class WritePort:
    selects: list  # WS_w relay per word
    copies: list  # COPY_w latch per word
    done: list  # "written" pulse relay per word
    domain_latches: list  # latches the caller must put in its reset domain (the COPY_w)


def add_write_port(net: Netlist, drive: Drive, name: str, mem: Memory, trigger: int, addr_taps, data_taps,
                   extra_vetoes: list[int] = (), veto_neurons: list[int] = ()) -> WritePort:
    """`trigger`: a train or a single pulse that starts the write; `addr_taps[j][r]`,
    `data_taps[i][r]`: taps of the address and data rails (levels valid before the trigger).
    `extra_vetoes`: taps that must be silent for the write to happen (e.g. "op is not WRITE")."""
    selects, copies, done, dom = [], [], [], []
    for w, word in enumerate(mem.words):
        vn = add_veto_neuron(net, drive, f"{name}.w{w}.notw", address_vetoes(addr_taps, w) + list(extra_vetoes))
        # word-select: one pulse into the word's edge-detected reset trigger
        ws = add_edge_relay(net, drive, f"{name}.w{w}.sel", trigger, fast_inhibitor=True)
        net.synapse(vn, ws, -int(round(0.5 * drive.loop)))
        for v in veto_neurons:
            net.synapse(v, ws, -int(round(0.5 * drive.loop)))
        net.synapse(ws, word.reset_trigger, drive.relay_in)
        copy = add_latch(net, drive, f"{name}.w{w}.copy")
        net.synapse(word.ready, copy.u, drive.ignite)  # the word is empty and recovered: copy
        for i in range(mem.width):
            for r in (0, 1):
                add_veto_relay(net, drive, f"{name}.w{w}.cp{i}r{r}", copy.u, [data_taps[i][1 - r]], word.rails[i][r])
        d = add_edge_relay(net, drive, f"{name}.w{w}.done", word.completion.u, fast_inhibitor=True)
        selects.append(ws); copies.append(copy); done.append(d); dom.append(copy)
    return WritePort(selects, copies, done, dom)


def add_read_port(net: Netlist, drive: Drive, name: str, mem: Memory, driver: int, addr_taps, target_rails,
                  extra_vetoes: list[int] = (), veto_neurons: list[int] = ()) -> list[int]:
    """`driver`: the train whose rise performs the read; `target_rails[i][r]`: destination
    latches (ignited once each). Returns the relays."""
    relays = []
    for w, word in enumerate(mem.words):
        vn = add_veto_neuron(net, drive, f"{name}.w{w}.notw", address_vetoes(addr_taps, w) + list(extra_vetoes))
        for i in range(mem.width):
            for r in (0, 1):
                relays.append(add_veto_relay(net, drive, f"{name}.w{w}.rd{i}r{r}", driver, [word.rails[i][1 - r].u],
                                             target_rails[i][r], veto_neurons=[vn] + list(veto_neurons)))
    return relays


# ------------------------------------------------------------------ standalone bank as a block
OPS = ("WRITE", "READ")


def ram_word(addr: int, data: int, op: str, a_bits: int, width: int) -> int:
    """Producer word: ADDR[a] | DATA[n] | OP one-hot (WRITE, READ)."""
    return (addr & ((1 << a_bits) - 1)) | ((data & ((1 << width) - 1)) << a_bits) | (1 << (a_bits + width + OPS.index(op)))


@dataclass
class RamBlock:
    net: Netlist
    drive: Drive
    width: int  # producer word width
    producer: Register
    consumer: Register  # the input stage S
    mem: Memory
    out: Register  # read-port output register
    q: Register  # its consumer (host-observable)
    wport: WritePort
    a_bits: int
    data_bits: int


def build_ram_block(params: Params, n_words: int, width: int, drive: Drive | None = None, liveness: bool = True,
                    watchdog_hops: int = 90) -> RamBlock:
    """P(addr, data, op) -> stage S (completion) -> write port into the bank, or read port into
    OUT -> Q. Written words stay; the stage is cleared by the word's "written" pulse or by Q's
    completion; READY is S.ready (writes) then Q.ready (reads)."""
    drive = drive or Drive.from_params(params)
    net = Netlist(params)
    a_bits = max(1, (n_words - 1).bit_length())
    pw = a_bits + width + len(OPS)
    P = add_register(net, drive, "P", pw, with_completion=False)
    S = add_register(net, drive, "Q", pw, with_completion=True)
    for i in range(pw):
        for r in (0, 1):
            relay = add_edge_relay(net, drive, f"data.b{i}r{r}", P.rails[i][r].u)
            net.synapse(relay, S.rails[i][r].u, drive.ignite)
    connect_trigger(net, drive, S.completion.u, P.reset_trigger, P.reset_edge)  # ACCEPT
    wire_fault_path(net, drive, P, S)
    if liveness:
        add_liveness(net, drive, P, S, watchdog_hops)
    mem = add_memory(net, drive, "M", n_words, width)
    addr = [[S.rails[j][0].u, S.rails[j][1].u] for j in range(a_bits)]
    data = [[S.rails[a_bits + i][0].u, S.rails[a_bits + i][1].u] for i in range(width)]
    op_w, op_r = S.rails[a_bits + width], S.rails[a_bits + width + 1]
    trig = add_delay_chain(net, drive, "ports.trig", S.completion.u, 3)  # the stage's completion, 16 ms later
    wp = add_write_port(net, drive, "wr", mem, trig, addr, data, extra_vetoes=[op_w[0].u])
    OUT = add_register(net, drive, "OUT", width, with_completion=False)
    Qo = add_register(net, drive, "R", width, with_completion=True)
    add_read_port(net, drive, "rd", mem, trig, addr, OUT.rails, extra_vetoes=[op_r[0].u])
    for i in range(width):
        for r in (0, 1):
            relay = add_edge_relay(net, drive, f"out.b{i}r{r}", OUT.rails[i][r].u)
            net.synapse(relay, Qo.rails[i][r].u, drive.ignite)
    connect_trigger(net, drive, Qo.completion.u, OUT.reset_trigger, OUT.reset_edge)  # read consumed -> OUT clears
    connect_trigger(net, drive, OUT.ready, Qo.reset_trigger, Qo.reset_edge)  # CLEARED -> Q clears
    wire_fault_path(net, drive, OUT, Qo)
    # the stage clears when the word is written, or when the read has been consumed
    q = -int(round(0.75 * drive.loop))
    for d in wp.done:
        net.synapse(d, S.reset_trigger, drive.relay_in)
    rdone = add_edge_relay(net, drive, "rd.done", Qo.completion.u, fast_inhibitor=True)
    net.synapse(rdone, S.reset_trigger, drive.relay_in)
    # a read of an unwritten word ignites both rails of every OUT bit (no rail vetoes the
    # relays); Q's fault gates refuse it, and that refusal must clear the stage too
    connect_trigger(net, drive, Qo.fault_latch.u, S.reset_trigger, S.reset_edge)
    for l in wp.domain_latches:  # COPY_w clear with the stage
        for x in l.members:
            net.synapse(S.reset_inh, x, q)
    if P.watchdog is not None:
        connect_trigger(net, drive, P.watchdog.timeout.u, S.reset_trigger, S.reset_edge)
    connect_trigger(net, drive, S.fault_latch.u, S.reset_trigger, S.reset_edge)
    return RamBlock(net, drive, pw, P, S, mem, OUT, Qo, wp, a_bits, width)


@dataclass
class RamRecord:
    op: str
    addr: int
    data: int
    expected: int | None
    load_step: int
    accept_step: int | None = None
    done_step: int | None = None
    decoded: int | None = None
    ready_step: int | None = None
    status: str = "incomplete"
    fault_spikes: int = 0
    timeout_spikes: int = 0


def run_ram(rb: RamBlock, params: Params, ops: list[tuple], *, max_steps_per_tx: int = 20000, sim=None,
            init: dict | None = None) -> tuple[list[RamRecord], RefSim, dict]:
    """`ops`: (op, addr, data) tuples; for READ, data is the expected value (None: unwritten).
    `init`: {addr: value} loaded straight into the words at step 1 (the initial image)."""
    net, drive = rb.net, rb.drive
    sim = sim or RefSim(net.topology(), params)
    P, S, Qo, mem = rb.producer, rb.consumer, rb.q, rb.mem
    window = 2 * drive.loop_period_steps
    fault_set = set(S.fault) | set(Qo.fault)
    timeout_n = P.watchdog.timeout.u if P.watchdog is not None else -1
    records, step = [], 0
    if init:  # the image's completions fire "written" pulses at power-up: let them pass
        for a, v in init.items():
            for i, r in rails_for(v, rb.data_bits):
                sim.add_events(0, [1], [mem.words[a].rails[i][r].u], [drive.ignite])
        sim.run(3000)
        step = sim.step_index
    for op, addr, data in ops:
        load_step = max(step, sim.step_index) + 10
        w = ram_word(addr, data if op == "WRITE" else 0, op, rb.a_bits, rb.data_bits)
        for i, r in rails_for(w, rb.width):
            sim.add_events(0, [load_step], [P.rails[i][r].u], [drive.ignite])
        rec = RamRecord(op, addr, data, data if op == "WRITE" else data, load_step)
        word = mem.words[addr]
        wm_prev = sim.trace.events["step"][sim.trace.events["neuron"] == word.completion.u]
        last_wm = int(wm_prev[-1]) if len(wm_prev) else None
        wm_silent = last_wm is None
        reset_seen = False
        deadline = load_step + max_steps_per_tx
        while sim.step_index < deadline:
            sim.step()
            s_ = sim.step_index - 1
            fired = set(sim._spk_neuron[-1].tolist()) if sim._spk_step and sim._spk_step[-1][0] == s_ else set()
            if not fired:
                if reset_seen and not wm_silent and last_wm is not None and s_ - last_wm > 3 * drive.loop_period_steps:
                    wm_silent = True
                continue
            if word.reset_trigger in fired:
                reset_seen = True
            if word.completion.u in fired:
                if op == "WRITE" and reset_seen and rec.done_step is None and wm_silent:
                    rec.done_step = s_
                    rec.decoded, _ = decode_at(sim.trace, word.rail_taps, s_, window)
                last_wm = s_
            if rec.accept_step is None and S.completion.u in fired:
                rec.accept_step = s_
            if op == "READ" and rec.done_step is None and Qo.completion.u in fired:
                rec.done_step = s_
                rec.decoded, _ = decode_at(sim.trace, Qo.rail_taps, s_, window)
            if fired & fault_set or (timeout_n >= 0 and timeout_n in fired):
                rec.fault_spikes += 1
            ready_n = S.ready if op == "WRITE" else Qo.ready
            if (rec.done_step is not None or rec.fault_spikes) and ready_n in fired and s_ > load_step + 200:
                rec.ready_step = s_
                break
        ev = sim.trace.events
        m = (ev["step"] >= load_step) & (ev["step"] <= (rec.ready_step or sim.step_index))
        if timeout_n >= 0:
            rec.timeout_spikes = int((ev["neuron"][m] == timeout_n).sum())
        if rec.done_step is not None and rec.ready_step is not None:
            rec.status = "ok" if rec.decoded == rec.expected else "wrong"
        elif rec.ready_step is not None:
            rec.status = "timeout" if rec.timeout_spikes else "refused"
        records.append(rec)
        step = rec.ready_step if rec.ready_step is not None else sim.step_index
        if rec.ready_step is None:
            break
    stats = {"transactions": len(records), "ok": sum(r.status == "ok" for r in records),
             "latency_ms": [((r.done_step - r.load_step) * params.dt) if r.done_step else None for r in records],
             "cycle_ms": [((r.ready_step - r.load_step) * params.dt) if r.ready_step else None for r in records],
             "neurons": net.n, "synapses": net.nnz, "total_spikes": len(sim.trace)}
    return records, sim, stats


# ------------------------------------------------------------------------------ batched campaign
def random_ram_ops(rng, n_words: int, width: int, k: int) -> tuple[list, dict]:
    """k operations: the first n_words are writes to every word (a known image), then random
    reads and writes. Returns the ops (READ with its expected value) and nothing else."""
    mem = {}
    ops = []
    for w in rng.permutation(n_words).tolist():
        v = int(rng.integers(0, 1 << width))
        ops.append(("WRITE", w, v))
        mem[w] = v
    while len(ops) < k:
        w = int(rng.integers(0, n_words))
        if rng.random() < 0.5:
            v = int(rng.integers(0, 1 << width))
            ops.append(("WRITE", w, v))
            mem[w] = v
        else:
            ops.append(("READ", w, mem[w]))
    return ops[:k], mem


def classify_ram_node(rb: RamBlock, steps, neurons, ops, loads, period_steps: int, drive) -> list[str]:
    """Per transaction: ok / wrong_value / refused (fault or timeout, harness-visible) / hang."""
    S, Qo, mem = rb.consumer, rb.q, rb.mem
    window = 2 * drive.loop_period_steps
    faults = set(S.fault) | set(Qo.fault)
    tn = rb.producer.watchdog.timeout.u if rb.producer.watchdog is not None else -1
    out = []
    held = {}  # what the bank actually holds after the transactions the node completed (no cascades)
    for k, (op, addr, val) in enumerate(ops):
        lo, hi = loads[k], loads[k] + period_steps
        mask = (steps >= lo) & (steps < hi)
        st, nu = steps[mask], neurons[mask]
        word = mem.words[addr]
        if op == "WRITE":
            wm = st[nu == word.completion.u]
            # the word's completion rises after its reset: the first spike >= 3 periods after a gap
            rise = None
            prev = None
            for s_ in wm:
                if prev is not None and s_ - prev > 3 * drive.loop_period_steps:
                    rise = int(s_)
                    break
                prev = int(s_)
            if rise is None and len(wm) and (nu == word.reset_trigger).any() and wm[0] > st[nu == word.reset_trigger][0]:
                rise = int(wm[0])
            done_step, rails = rise, word.rail_taps
        else:
            q = st[nu == Qo.completion.u]
            done_step, rails = (int(q[0]) if len(q) else None), Qo.rail_taps
        if done_step is None:
            n_f = int(np.isin(nu, list(faults)).sum()) + (int((nu == tn).sum()) if tn >= 0 else 0)
            out.append("refused" if n_f else "hang")
            continue
        act = set(nu[(st > done_step - window) & (st <= done_step)].tolist())
        v, ok = 0, True
        for i, (r0, r1) in enumerate(rails):
            a0, a1 = r0 in act, r1 in act
            if (a0 and a1) or not (a0 or a1):
                ok = False
            v |= a1 << i
        if op == "WRITE":
            out.append("ok" if ok and v == val else "wrong_value")
            if ok:
                held[addr] = v
        else:
            exp = held.get(addr, val)  # a read is judged against what was really written
            out.append("ok" if ok and v == exp else ("cascade" if not ok else "wrong_value"))
    return out


def run_ram_batch(rb: RamBlock, params: Params, B: int, k: int, pert, rng, period_ms: float = 700, device="cpu") -> list[list[str]]:
    from .campaign import make_perturbed_sim

    period = int(period_ms / params.dt)
    n_steps = k * period + 500
    topo = rb.net.topology()
    sim = make_perturbed_sim(topo, params, B, pert, rng, n_steps, device=device)
    loads = [50 + j * period for j in range(k)]
    all_ops = []
    for b in range(B):
        ops, _ = random_ram_ops(rng, rb.mem.n_words, rb.data_bits, k)
        all_ops.append(ops)
        for j, (op, addr, val) in enumerate(ops):
            w = ram_word(addr, val if op == "WRITE" else 0, op, rb.a_bits, rb.data_bits)
            for i, r in rails_for(w, rb.width):
                off = int(rng.integers(0, pert.arrival_jitter_steps + 1)) if pert.arrival_jitter_steps else 0
                sim.add_events(b, [loads[j] + off], [rb.producer.rails[i][r].u], [rb.drive.ignite])
    sim.run(n_steps)
    ev = sim.trace.events
    order = np.lexsort((ev["step"], ev["node"]))
    ev = ev[order]
    bounds = np.searchsorted(ev["node"], np.arange(B + 1))
    return [classify_ram_node(rb, ev["step"][bounds[b]:bounds[b + 1]], ev["neuron"][bounds[b]:bounds[b + 1]], all_ops[b], loads,
                              period, rb.drive) for b in range(B)]
