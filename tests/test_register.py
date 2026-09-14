"""Word register with staged commit: commit exactly once per staged word, early and late
COMMIT, duplicate COMMIT absorbed, faulty and timed-out words discarded with the master
untouched, retention, and the master's valid flag dropping only for the rewrite."""

import numpy as np

from drosophilos.lib.staged import build_staged_register, run_commits
from drosophilos.sim.model import Params

PARAMS = Params()


def _spikes(sim, neuron, lo, hi):
    ev = sim.trace.events
    return ev["step"][(ev["neuron"] == neuron) & (ev["step"] >= lo) & (ev["step"] <= hi)]


def test_commits_early_and_late():
    sc = build_staged_register(PARAMS, 4)
    words = [0b1010, 0b0101, 0b1111, 0b0000, 0b0110]
    recs, sim, st = run_commits(sc, PARAMS, words, words, commit_delay_steps=[0, 6000, 0, 3000, 0])
    assert st["committed"] == 5 and st["correct"] == 5, [(r.word, r.decoded_master, r.status) for r in recs]
    assert all(r.master_at_ready == r.word and r.fault_spikes == 0 and r.m_fault_spikes == 0 for r in recs)
    M = sc.master
    for r in recs[1:]:  # W_M drops for the rewrite only: silent from the M reset until commit-done
        wm = _spikes(sim, M.completion.u, r.m_reset_step + 100, r.done_step - 1)
        assert len(wm) == 0, (r.word, len(wm))
        assert 200 < (r.done_step - r.grant_step) * PARAMS.dt < 350, (r.done_step - r.grant_step) * PARAMS.dt
    early, late = recs[0], recs[1]
    assert early.grant_step > early.accept_step  # an early COMMIT waits for completion
    assert late.grant_step - late.load_step > 6000  # a late COMMIT is honoured when it arrives
    print("register:", sc.net.n, "neurons; commit ms", [round(c) for c in st["commit_ms"]], "cycle ms", [round(c) for c in st["cycle_ms"]])


def test_duplicate_commit_applies_once():
    sc = build_staged_register(PARAMS, 4)
    words = [0b0011, 0b1100, 0b1001]
    recs, sim, st = run_commits(sc, PARAMS, words, words, commit_delay_steps=[0, 0, 0],
                                faults={1: [("duplicate_commit", 300)], 2: [("duplicate_commit", 2500)]})
    assert st["committed"] == 3 and st["correct"] == 3, [(r.word, r.decoded_master, r.status) for r in recs]
    M = sc.master
    for r in recs:  # exactly one M reset (one commit) per staged word
        assert len(_spikes(sim, M.reset_trigger, r.load_step, r.ready_step)) == 1, r


def test_faulty_word_is_discarded_and_master_unchanged():
    sc = build_staged_register(PARAMS, 4)
    words = [0b1010, 0b0101, 0b1111]
    expected = [0b1010, 0b1010, 0b1111]  # word 1 is corrupted: the master keeps word 0
    recs, sim, st = run_commits(sc, PARAMS, words, expected, commit_delay_steps=0, faults={1: [("corrupt", 2)]})
    r0, r1, r2 = recs
    assert r0.status == "committed" and r0.decoded_master == 0b1010
    assert r1.status == "discarded" and r1.fault_spikes > 0 and r1.grant_spikes == 0 and r1.done_step is None, r1
    assert r1.master_at_ready == 0b1010 and r1.m_fault_spikes == 0
    assert r2.status == "committed" and r2.decoded_master == 0b1111 and r2.fault_spikes == 0, r2


def test_timeout_discards_stage_and_master_unchanged():
    sc = build_staged_register(PARAMS, 4)
    net = sc.net
    relay = net.roles.index("data.b2r1.edge")
    for k in range(net.nnz):  # bit 2, rail 1 can never reach the stage
        if net.src[k] == relay:
            net.quanta[k] = 0
    words = [0b0010, 0b0100, 0b1001]
    expected = [0b0010, 0b0010, 0b1001]
    recs, sim, st = run_commits(sc, PARAMS, words, expected, commit_delay_steps=0)
    r0, r1, r2 = recs
    assert r0.status == "committed" and r0.decoded_master == 0b0010
    assert r1.status == "timeout" and r1.timeout_spikes > 0 and r1.accept_step is None and r1.done_step is None, r1
    assert r1.master_at_ready == 0b0010 and r1.ready_step is not None
    assert r2.status == "committed" and r2.decoded_master == 0b1001 and r2.timeout_spikes == 0, r2


def test_commit_issued_before_data_applies_to_next_word():
    sc = build_staged_register(PARAMS, 4)
    words = [0b0111, 0b1000]
    recs, sim, st = run_commits(sc, PARAMS, words, words, commit_delay_steps=[0, -1500], gap_steps=2000)
    assert st["committed"] == 2 and st["correct"] == 2, [(r.word, r.decoded_master, r.status) for r in recs]
    assert recs[1].commit_inject_step < recs[1].load_step


def test_retention_two_seconds():
    sc = build_staged_register(PARAMS, 4)
    words = [0b1101, 0b0010]
    recs, sim, st = run_commits(sc, PARAMS, words, words, commit_delay_steps=[20000, 0], max_steps_per_tx=30000)
    assert st["committed"] == 2 and st["correct"] == 2, [(r.word, r.decoded_master, r.status) for r in recs]
    r = recs[0]
    assert r.grant_step - r.load_step >= 20000 and r.decoded_stage == 0b1101
    # while it waits, nothing touches the master and nothing is granted
    M = sc.master
    assert len(_spikes(sim, M.reset_trigger, r.load_step, r.commit_inject_step)) == 0
    assert len(_spikes(sim, sc.reg.granted.u, r.load_step, r.commit_inject_step)) == 0
