"""spec.md §3 rules on the neural channel: asynchronous arrival, duplicates, corruption,
late opposite rail, stale tokens, lost ACCEPT."""

import numpy as np
import pytest

from drosophilos.protocol.handshake import build_channel
from drosophilos.protocol.run import run_transactions
from drosophilos.sim.model import Params

PARAMS = Params()


def test_asynchronous_operand_arrival_4bit():
    rng = np.random.default_rng(3)
    words = [0b1010, 0b0111, 0b1100, 0b0001, 0b1111]
    offsets = [list(rng.integers(0, 400, size=4)) for _ in words]  # bits arrive up to 40 ms apart
    ch = build_channel(PARAMS, 4)
    recs, sim, st = run_transactions(ch, PARAMS, words, bit_offsets=offsets)
    assert st["completed"] == len(words) and st["correct"] == len(words), [(r.word, r.decoded, r.status) for r in recs]
    # completion must wait for the last bit: accept after the latest offset
    for r, off in zip(recs, offsets):
        assert r.accept_step - r.load_step > max(off)


def test_same_rail_duplicate_is_absorbed():
    ch = build_channel(PARAMS, 4)
    recs, sim, st = run_transactions(ch, PARAMS, [0b0110, 0b1001], faults={0: [("duplicate", 1, 150), ("duplicate", 2, 600)]})
    assert st["completed"] == 2 and st["correct"] == 2
    assert all(r.fault_spikes == 0 for r in recs)


def test_corrupted_rail_is_a_fault_never_consumed_and_channel_recovers():
    ch = build_channel(PARAMS, 4)
    recs, sim, st = run_transactions(ch, PARAMS, [0b0110, 0b1001, 0b0011], faults={0: [("corrupt", 2)]})
    r0 = recs[0]
    assert r0.fault_spikes > 0
    assert r0.accept_step is None and r0.status == "fault", r0  # completion latch blocked
    assert r0.cleared_step is not None and r0.ready_step is not None  # FAULT-ACCEPT ran the four phases
    assert st["completed"] == 3
    assert [r.decoded for r in recs[1:]] == [0b1001, 0b0011]  # the channel is clean afterwards


def test_late_opposite_rail_after_accept_is_flagged_only():
    ch = build_channel(PARAMS, 4)
    recs, sim, st = run_transactions(ch, PARAMS, [0b0110, 0b1001], faults={0: [("late_opposite", 0, 50)]})
    r0 = recs[0]
    assert r0.status == "valid" and r0.decoded == 0b0110  # consumed before the late spike
    assert r0.fault_spikes > 0  # flagged
    assert st["completed"] == 2 and recs[1].decoded == 0b1001


def test_stale_spike_is_absorbed_or_detected_never_consumed_wrong():
    """A stale rail spike (full ignition strength) d ms after the consumer's reset trigger.
    Measured: it rides through the half-strength reset train (d < ~20 ms), is absorbed only
    while the members' after-hyperpolarisation blocks ignition (d ~ 20-30 ms), and ignites
    a stale latch afterwards. In every case the next word is either decoded correctly or
    flagged as a fault and never consumed, and the channel recovers on the word after.
    By construction no stale source exists in this window (the producer is cleared >= 48 ms
    before the consumer resets); the fault path is the safety net."""
    outcomes = {}
    for d_ms in (5, 15, 25, 35, 45):
        ch = build_channel(PARAMS, 4)
        # word 0 sets bit 0 = 0 (rail 0); the stale spike re-asserts rail 0; word 1 sets bit 0 = 1
        recs, sim, st = run_transactions(ch, PARAMS, [0b0000, 0b0001, 0b0011], faults={0: [("stale", 0, 0, int(d_ms * 10))]})
        r1, r2 = recs[1], recs[2]
        outcomes[d_ms] = (r1.status, r1.decoded, r1.fault_spikes, r2.status, r2.decoded)
        assert r1.status in ("valid", "fault"), r1
        if r1.status == "valid":
            assert r1.decoded == 0b0001
        else:
            assert r1.accept_step is None and r1.fault_spikes > 0  # detected, never consumed
        assert r2.status == "valid" and r2.decoded == 0b0011, r2  # channel recovered
    print("stale outcomes (status, decoded, faults, next status, next decoded):", outcomes)
    assert any(o[0] == "valid" for o in outcomes.values()) and any(o[0] == "fault" for o in outcomes.values())


def test_lost_accept_deadlocks_the_channel():
    ch = build_channel(PARAMS, 4)
    net = ch.net
    # sever ACCEPT: zero every synapse from the completion latch tap into the producer's trigger/edge
    for k in range(net.nnz):
        if net.src[k] == ch.consumer.completion.u and net.dst[k] in (ch.producer.reset_trigger, ch.producer.reset_edge):
            net.quanta[k] = 0
    recs, sim, st = run_transactions(ch, PARAMS, [0b0110, 0b1001], max_steps_per_tx=4000)
    r0 = recs[0]
    assert r0.accept_step is not None and r0.decoded == 0b0110  # the consumer completed
    assert r0.cleared_step is None and r0.ready_step is None  # ...but the channel stalls
    assert st["completed"] == 0
