"""spec.md §5 properties, checked by exhaustive state enumeration of the abstract machine."""

import pytest

from drosophilos.protocol.explore import explore

WORDS = ((1, 0), (0, 1), (1, 1))


def test_no_deadlock_no_double_consume_no_illegal_transition():
    r = explore(WORDS, 2)
    assert not r.deadlocks and r.double_consumes == 0 and not r.illegal
    assert r.terminal_states >= 1
    # every terminal state consumed exactly the sent words, in order
    assert r.consumed_sets == {tuple((i, w) for i, w in enumerate(WORDS))}


def test_duplicate_tokens_are_idempotent():
    r = explore(WORDS, 2, inject_duplicates=True)
    assert not r.deadlocks and r.double_consumes == 0 and not r.illegal
    assert r.consumed_sets == {tuple((i, w) for i, w in enumerate(WORDS))}


def test_late_opposite_rail_is_a_fault_not_a_value():
    r = explore(WORDS, 2, inject_late_opposite=True)
    assert not r.deadlocks and r.double_consumes == 0 and not r.illegal
    assert r.faults_max >= 1
    # a faulted word is never consumed with a wrong value: every consumed value is a sent word
    for cs in r.consumed_sets:
        for seq, val in cs:
            assert val == WORDS[seq]


def test_stale_data_is_accepted_without_protection():
    r = explore(WORDS, 2, timing_bound=False, phase_rails=False, inject_stale=True)
    assert r.stale_accepted_max >= 1, "the checker must demonstrate the failure mode"


@pytest.mark.parametrize("kw", [dict(timing_bound=True, phase_rails=False), dict(timing_bound=False, phase_rails=True)])
def test_stale_data_rejected_with_timing_bound_or_phase_rails(kw):
    r = explore(WORDS, 2, inject_stale=True, **kw)
    assert r.stale_accepted_max == 0 and not r.deadlocks and r.double_consumes == 0 and not r.illegal


@pytest.mark.parametrize("kind", ["ACCEPT", "CLEARED", "READY", "DATA"])
def test_lost_token_deadlocks_without_retransmission(kind):
    r = explore(WORDS, 2, lose={kind})
    assert r.deadlocks, f"losing {kind} must stall the channel (motivates timeouts)"
    assert r.double_consumes == 0 and not r.illegal
