"""Word register with staged commit (plan §A2), and an accumulator that closes the loop
ALU -> staged register -> ALU operand.

    stage S    a consumer register with completion (W_S) and a fault latch (F): receives
               DATA like any channel consumer and holds it until COMMIT or discard
    master M   the architectural value: rails + completion tree; W_M = "M holds a valid
               word". Readers take M's rail taps as tokens while W_M holds
    COMMIT     a control token (one ignition pulse from the FSM; in tests, the host),
               latched in S's reset domain. grant C = AND(COMMIT, W_S)
    sequence   C -> M.reset train -> M.ready -> COPY latch -> per-rail AND(COPY, S rail) ->
               M rails -> W_M -> edge relay -> S.reset (clears S, COMMIT, C, COPY) -> S.ready
               = READY to the upstream
    discard    F (both rails on a staged bit) or the producer's TIMEOUT clears S without
               touching M; F also holds W_S and C down, so a faulty word is never granted

Protocol assumptions (stated, tested where marked):
  * COMMIT is issued at most once per transaction, any time after DATA is sent. A COMMIT
    that arrives before completion waits in its latch (tested: early commit). A second
    COMMIT within the same transaction is absorbed (tested: duplicate). A COMMIT issued
    while nothing is in flight is stored and applies to the next word that completes.
  * Readers of M sample it only while W_M holds; W_M drops for the rewrite window
    (~150 ms: reset train + READY chain + copy + completion).
  * Loading M directly at power-up (the host's "load the initial image") raises W_M once
    and therefore issues one spurious stage reset and READY, which the runner absorbs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..protocol.celement import add_and_latched, add_delay_chain, add_veto_relay
from ..protocol.handshake import Register, add_liveness, add_register, wire_fault_path
from ..protocol.latch import Latch, add_edge_relay, add_latch, connect_trigger
from ..protocol.token import decode_at, rails_for
from ..sim.model import Params
from ..sim.ref64 import RefSim
from .adder import extend_reset
from .alu import N_UNITS, add_alu_logic, wire_alu
from .gates import Gates, Rail2
from .netlist import Drive, Netlist


@dataclass
class StagedRegister:
    stage: Register
    master: Register
    commit: Latch
    granted: Latch
    copy: Latch
    grant_gate: int
    copy_gates: list
    done_relay: int
    commit_in: int = -1  # fire this neuron once to issue COMMIT (set after construction)


def add_staged_commit(net: Netlist, drive: Drive, name: str, S: Register, P: Register,
                      M: Register | None = None, ordered_grant: bool = False, guard_hops: int = 12) -> StagedRegister:
    """Wrap consumer register S (with completion and fault latch, i.e. after wire_fault_path)
    with a master M and a commit controller. The caller must NOT wire the producer's CLEARED
    to S's reset: S is cleared by commit-done, by F, or by the producer's TIMEOUT."""
    n = len(S.rails)
    if M is None:
        M = add_register(net, drive, f"{name}.M", n, with_completion=True)
    assert len(M.rails) == n and S.completion is not None and S.fault_latch is not None
    W, F = S.completion, S.fault_latch
    COMMIT = add_latch(net, drive, f"{name}.commit")
    # Guard window (review finding 2026-09-14): a late opposite rail that the fault gates
    # catch after the grant would find M already reset, so "discarded, M untouched" only
    # held before the grant. The grant is now delayed `guard_hops` (~64 ms) after COMMIT so
    # a rail arriving up to ~60 ms after completion is refused before M is touched; a fault
    # after that leaves M empty or partial, which M's own completion and fault gates make a
    # fail-stop, never a silently wrong master.
    # The guard carries the COMMIT token's own pulse (`commit_in`): a chain fed by the COMMIT
    # latch's train would carry the train and re-ignite the guarded latch continuously, and a
    # one-shot relay on the train could not re-arm within the ~85 ms between transactions.
    commit_in = net.neuron(f"{name}.commit_in")  # the COMMIT token's entry: one pulse from the FSM or the host
    net.synapse(commit_in, COMMIT.u, drive.ignite)
    guard = add_delay_chain(net, drive, f"{name}.guardd", commit_in, guard_hops)
    if ordered_grant:
        # a control machine issues COMMIT only after W_S: the grant is then a veto relay
        # (driver: the guarded pulse, vetoed by the fault latch), with no rate-mode exposure
        C = add_latch(net, drive, f"{name}.grant.L")
        g = add_veto_relay(net, drive, f"{name}.grant", guard, [F.u], C)
    else:
        COMMIT2 = add_latch(net, drive, f"{name}.commit2")
        net.synapse(guard, COMMIT2.u, drive.ignite)  # the guarded COMMIT as a standard-rate train
        g, C = add_and_latched(net, drive, f"{name}.grant", [COMMIT2.u, W.u])
    connect_trigger(net, drive, C.u, M.reset_trigger, M.reset_edge)  # granted -> clear M
    COPY = add_latch(net, drive, f"{name}.copy")
    net.synapse(M.ready, COPY.u, drive.ignite)  # M empty and recovered -> copy enable
    copy_gates = []  # veto relays: COPY's rise ignites M rail r unless the stage holds rail 1-r
    for i in range(n):  # (a rate-mode AND(COPY, S rail) sat on COPY alone for ~100 ms per commit and
        for r in (0, 1):  # leaked in nodes whose COPY latch ran fast: master faults, 2 per 5,000)
            copy_gates.append(add_veto_relay(net, drive, f"{name}.cp{i}r{r}", COPY.u, [S.rails[i][1 - r].u, F.u], M.rails[i][r]))
    # commit done = W_M's first spike, through an edge relay as a single pulse: W_M then holds
    # for as long as M is valid, and a train into the stage's edge-detected reset trigger would
    # hold that trigger down and block the F / TIMEOUT discards
    done = add_edge_relay(net, drive, f"{name}.done", M.completion.u, fast_inhibitor=True)
    net.synapse(done, S.reset_trigger, drive.relay_in)
    # discard: a faulty staged word clears the stage (P is cleared by FAULT-ACCEPT already);
    # F also holds the grant down so the word can never be committed
    connect_trigger(net, drive, F.u, S.reset_trigger, S.reset_edge)
    for x in list(C.members) + ([] if ordered_grant else [g]):
        net.synapse(F.u, x, drive.reset)
    if P.watchdog is not None:
        connect_trigger(net, drive, P.watchdog.timeout.u, S.reset_trigger, S.reset_edge)
    q = -int(round(0.75 * drive.loop))
    for l in (COMMIT, C, COPY) + (() if ordered_grant else (COMMIT2,)):
        for x in l.members:
            net.synapse(S.reset_inh, x, q)
    if not ordered_grant:
        net.synapse(S.reset_inh, g, q)
    net.group(f"{name}.master_taps", [t for pair in M.rail_taps for t in pair])
    sr = StagedRegister(S, M, COMMIT, C, COPY, g, copy_gates, done)
    sr.commit_in = commit_in
    return sr


@dataclass
class StagedChannel:
    net: Netlist
    drive: Drive
    width: int  # producer word width
    producer: Register
    reg: StagedRegister
    alu_width: int = 0  # >0: accumulator (A operand = master[0:alu_width])

    @property
    def consumer(self) -> Register:  # the stage; lets the campaign harness treat this as a channel
        return self.reg.stage

    @property
    def master(self) -> Register:
        return self.reg.master

    @property
    def accept(self) -> int:
        return self.reg.stage.completion.u

    @property
    def ready(self) -> int:
        return self.reg.stage.ready

    @property
    def commit_done(self) -> int:
        return self.reg.master.completion.u


def build_staged_register(params: Params, width: int, drive: Drive | None = None, liveness: bool = True,
                          watchdog_hops: int = 55) -> StagedChannel:
    """P -> stage -> master: the register on its own, loaded by a plain transport channel."""
    drive = drive or Drive.from_params(params)
    net = Netlist(params)
    P = add_register(net, drive, "P", width, with_completion=False)
    S = add_register(net, drive, "Q", width, with_completion=True)
    for i in range(width):
        for r in (0, 1):
            relay = add_edge_relay(net, drive, f"data.b{i}r{r}", P.rails[i][r].u)
            net.synapse(relay, S.rails[i][r].u, drive.ignite)
    connect_trigger(net, drive, S.completion.u, P.reset_trigger, P.reset_edge)  # ACCEPT
    wire_fault_path(net, drive, P, S)
    if liveness:
        add_liveness(net, drive, P, S, watchdog_hops)
    sr = add_staged_commit(net, drive, "R", S, P)
    return StagedChannel(net, drive, width, P, sr)


def build_accumulator(params: Params, width: int, drive: Drive | None = None, liveness: bool = True,
                      watchdog_hops: int = 150, act_hops: int = 11, ordered_grant: bool = False) -> StagedChannel:
    """P(B, U, SUB) + master[0:n] as A -> ALU -> stage(R, C, Z, V) -> commit -> master."""
    drive = drive or Drive.from_params(params)
    net = Netlist(params)
    pw = width + N_UNITS + 1
    P = add_register(net, drive, "P", pw, with_completion=False)
    M = add_register(net, drive, "R.M", width + 3, with_completion=True)
    S = add_register(net, drive, "Q", width + 3, with_completion=True)
    A = [Rail2(*M.rails[i]) for i in range(width)]  # the master is a level: the ALU's operand gate tokenises it
    B = [Rail2(*P.rails[i]) for i in range(width)]
    U = [Rail2(*P.rails[width + k]) for k in range(N_UNITS)]
    SUB = Rail2(*P.rails[width + N_UNITS])
    G = Gates(net, drive)
    R, C, V = add_alu_logic(G, "alu", A, B, U, SUB, P.reset_inh, act_hops=act_hops)
    wire_alu(net, drive, R, C, V, S)
    extend_reset(net, drive, S, G.latches, G.gates)
    connect_trigger(net, drive, S.completion.u, P.reset_trigger, P.reset_edge)  # ACCEPT
    wire_fault_path(net, drive, P, S)
    if liveness:
        add_liveness(net, drive, P, S, watchdog_hops)
    sr = add_staged_commit(net, drive, "R", S, P, M, ordered_grant=ordered_grant)
    net.group("alu_latches", [x for l in G.latches for x in l.members])
    return StagedChannel(net, drive, pw, P, sr, alu_width=width)


@dataclass
class CommitRecord:
    word: int
    expected: int
    load_step: int
    commit_inject_step: int | None
    accept_step: int | None = None
    grant_step: int | None = None
    m_reset_step: int | None = None
    done_step: int | None = None
    ready_step: int | None = None
    decoded_stage: int | None = None
    decoded_master: int | None = None
    master_at_ready: int | None = None
    status: str = "incomplete"  # committed | discarded | timeout | incomplete
    fault_spikes: int = 0
    m_fault_spikes: int = 0
    timeout_spikes: int = 0
    grant_spikes: int = 0
    injected: list = field(default_factory=list)


def run_commits(sc: StagedChannel, params: Params, words: list[int], expected: list[int], *,
                commit_delay_steps=0, faults: dict[int, list[tuple]] | None = None, init_master: int | None = None,
                max_steps_per_tx: int = 20000, gap_steps: int = 0, sim=None) -> tuple[list[CommitRecord], RefSim, dict]:
    """Drive the register through `words`. `commit_delay_steps` (int or per-word list; None =
    no COMMIT for that word) is measured from the load. Fault specs per word:
        ("corrupt", bit)                 both rails of stage bit `bit` (10 ms after load)
        ("duplicate_commit", delay)      a second COMMIT pulse `delay` steps after the first
    `expected[k]` is the master's value after word k commits (or the unchanged value if the
    word is expected to be discarded)."""
    net, drive = sc.net, sc.drive
    sim = sim or RefSim(net.topology(), params)
    P, S, M, R = sc.producer, sc.reg.stage, sc.reg.master, sc.reg
    window = 2 * drive.loop_period_steps
    faults = faults or {}
    fault_set = set(S.fault)
    timeout_n = P.watchdog.timeout.u if P.watchdog is not None else -1
    if timeout_n >= 0:
        fault_set.add(timeout_n)
    delays = commit_delay_steps if isinstance(commit_delay_steps, list) else [commit_delay_steps] * len(words)
    records: list[CommitRecord] = []
    step = 0
    if init_master is not None:
        for i, r in rails_for(init_master, len(M.rails)):
            sim.add_events(0, [1], [M.rails[i][r].u], [drive.ignite])
        # power-up: W_M's first rise clears the (empty) stage once and issues one READY
        while sim.step_index < 4000:
            sim.step()
            s = sim.step_index - 1
            if sim._spk_step and sim._spk_step[-1][0] == s and S.ready in set(sim._spk_neuron[-1].tolist()):
                break
        step = sim.step_index + 1
    for k, w in enumerate(words):
        load_step = max(step, sim.step_index)
        for i, r in rails_for(w, sc.width):
            sim.add_events(0, [load_step], [P.rails[i][r].u], [drive.ignite])
        rec = CommitRecord(w, expected[k], load_step, None)
        if delays[k] is not None:
            rec.commit_inject_step = load_step + int(delays[k])
            sim.add_events(0, [rec.commit_inject_step], [R.commit_in], [drive.ignite])
        for spec in faults.get(k, []):
            if spec[0] == "corrupt":
                i = spec[1]
                staged = w if sc.alu_width == 0 else expected[k]  # the word the stage will receive
                r = 1 - ((staged >> i) & 1)
                t = load_step + 100
                sim.add_events(0, [t], [S.rails[i][r].u], [drive.ignite])
                rec.injected.append(("corrupt", i, r, t))
            elif spec[0] == "duplicate_commit":
                t = rec.commit_inject_step + int(spec[1])
                sim.add_events(0, [t], [R.commit_in], [drive.ignite])
                rec.injected.append(("duplicate_commit", t))
        first_fault = None
        ev0 = sim.trace.events
        wm_prev = ev0["step"][ev0["neuron"] == M.completion.u]
        last_wm = int(wm_prev[-1]) if len(wm_prev) else None
        wm_silent = False  # W_M holds from the previous commit until the M reset kills it
        deadline = load_step + max_steps_per_tx
        while sim.step_index < deadline:
            sim.step()
            s = sim.step_index - 1
            fired = set(sim._spk_neuron[-1].tolist()) if sim._spk_step and sim._spk_step[-1][0] == s else set()
            if rec.m_reset_step is not None and not wm_silent and last_wm is not None and s - last_wm > 3 * drive.loop_period_steps:
                wm_silent = True
            if M.completion.u in fired:
                if rec.m_reset_step is not None and rec.done_step is None and (wm_silent or last_wm is None):
                    rec.done_step = s  # W_M's re-ignition after the rewrite = commit done
                    rec.decoded_master, _ = decode_at(sim.trace, M.rail_taps, s, window)
                last_wm = s
            if not fired:
                continue
            if first_fault is None and fired & fault_set:
                first_fault = s
            if rec.accept_step is None and S.completion.u in fired:
                rec.accept_step = s
                rec.decoded_stage, _ = decode_at(sim.trace, S.rail_taps, s, window)
            if rec.grant_step is None and R.granted.u in fired:
                rec.grant_step = s
            if rec.m_reset_step is None and M.reset_trigger in fired:
                rec.m_reset_step = s
            if (rec.done_step is not None or first_fault is not None) and S.ready in fired:
                rec.ready_step = s
                break
        tr = sim.trace
        ev = tr.events
        end = rec.ready_step if rec.ready_step is not None else sim.step_index
        if rec.ready_step is not None:
            rec.master_at_ready, _ = decode_at(tr, M.rail_taps, rec.ready_step, window)
        m = (ev["step"] >= load_step) & (ev["step"] <= end)
        rec.fault_spikes = int(np.isin(ev["neuron"][m], list(S.fault)).sum())
        rec.m_fault_spikes = int(np.isin(ev["neuron"][m], list(M.fault)).sum())
        rec.grant_spikes = int((ev["neuron"][m] == R.granted.u).sum())
        if timeout_n >= 0:
            rec.timeout_spikes = int((ev["neuron"][m] == timeout_n).sum())
        if rec.done_step is not None and rec.ready_step is not None:
            rec.status = "committed"
        elif rec.ready_step is not None and rec.timeout_spikes:
            rec.status = "timeout"
        elif rec.ready_step is not None:
            rec.status = "discarded"
        records.append(rec)
        step = (rec.ready_step if rec.ready_step is not None else sim.step_index) + gap_steps
        if rec.ready_step is None:
            break
    stats = {
        "transactions": len(records),
        "committed": sum(1 for r in records if r.status == "committed"),
        "correct": sum(1 for r in records if r.status == "committed" and r.decoded_master == r.expected),
        "accept_ms": [((r.accept_step - r.load_step) * params.dt) if r.accept_step is not None else None for r in records],
        "commit_ms": [((r.done_step - r.grant_step) * params.dt) if r.done_step is not None and r.grant_step is not None else None for r in records],
        "cycle_ms": [((r.ready_step - r.load_step) * params.dt) if r.ready_step is not None else None for r in records],
        "neurons": net.n,
        "synapses": net.nnz,
        "total_spikes": len(sim.trace),
    }
    return records, sim, stats
