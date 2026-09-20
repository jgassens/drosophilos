"""A refused input word is re-sent, not skipped (docs/tick_stalls.md, seed-108 copy 77).

Mechanism found in the copy-77 capture: one rail of a host-loaded word failed to latch in the
input stage, the stage never completed, the producer's watchdog timed out (~460 ms) and reset
producer and stage, READY rose again, and the host — which counts READY rises — loaded the
*next* token. The word was lost without a fault and every later output shifted by one; the
campaign scored it as five wrong values. The refusal is the kernel doing its job (fail-stop);
the host has to resend. These tests construct the refusal directly by dropping one rail's
ignition from one load."""

import pytest

from drosophilos.lib.kernel import build_pipeline, run_pipeline, run_pipeline_batched
from drosophilos.sim.model import Params

P = Params()
TOKENS = [1, 2, 3, 5]


def _mov_kernel():
    spec = [{"name": "out", "op": "MOV", "a": ("const", "zero"), "b": "input"}]
    return build_pipeline(P, 4, spec, consts={"zero": 0})


def _dark_bit0(times: int):
    """Drop bit 0's rail ignition from the first `times` loads of token index 1 (value 2): the
    stage cannot complete that word, so the producer's watchdog refuses it."""
    seen = {"n": 0}

    def rail_filter(node, index, stream, value, rails):
        if index == 1 and seen["n"] < times:
            seen["n"] += 1
            return [(i, r) for i, r in rails if i != 0]
        return rails
    return rail_filter


def _run_batched(times=1, max_ms=20000, **kw):
    pl = _mov_kernel()
    outs, _, st = run_pipeline_batched(pl, P, [TOKENS], max_ms=max_ms, device="cpu", progress=0,
                                       rail_filter=_dark_bit0(times), **kw)
    return [v for _, v in outs[0]["out"]], st


def test_refused_word_is_resent_and_the_schedule_is_delivered_whole():
    got, st = _run_batched()
    assert got == TOKENS, (got, st["refused"])
    assert st["faults"] == 0
    assert st["timeouts"] == 1 and st["refusals"] == 1 and st["retries"] == 1, st["refused"]
    ev = st["refused"][0][0]
    assert ev["schedule_index"] == 1 and ev["value"] == 2 and ev["retried"]
    loads = st["load_events"][0]
    assert [e["value"] for e in loads] == [1, 2, 2, 3, 5]
    assert [e["retry"] for e in loads] == [False, False, True, False, False]


def test_without_retry_the_refused_word_is_dropped_and_later_outputs_shift():
    # the pre-fix behaviour, kept as the description of what the seed-108 campaign scored
    got, st = _run_batched(retry_refused=False)
    assert got == [1, 3, 5], got
    assert st["refusals"] == 1 and st["retries"] == 0 and st["faults"] == 0


def test_a_word_refused_past_max_retries_blocks_the_node_instead_of_skipping_it():
    got, st = _run_batched(times=99, max_ms=8000, max_retries=2)
    assert got == [1], got  # nothing after the refused word: unfinished, not shifted
    assert st["refusals"] == 3 and st["retries"] == 2 and st["blocked_nodes"] == [0]
    assert [e["attempt"] for e in st["refused"][0]] == [1, 2, 3]
    assert st["host_stalls"]


def test_single_runner_resends_too():
    pl = _mov_kernel()
    out, _, st = run_pipeline(pl, P, TOKENS, max_ms=20000, rail_filter=_dark_bit0(1))
    assert [v for _, v in out] == TOKENS
    assert st["refusals"] == 1 and st["retries"] == 1
