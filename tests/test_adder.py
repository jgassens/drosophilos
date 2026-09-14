"""Dual-rail full adder and 4-bit ripple adder as four-phase channels."""

import numpy as np
import pytest

from drosophilos.lib.adder import build_adder_channel, operand_word
from drosophilos.protocol.run import run_transactions
from drosophilos.sim.model import Params

PARAMS = Params()


def test_full_adder_all_eight_inputs():
    ch = build_adder_channel(PARAMS, 1)
    cases = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]
    words = [operand_word(a, b, c, 1) for a, b, c in cases]
    expected = [a + b + c for a, b, c in cases]
    recs, sim, st = run_transactions(ch, PARAMS, words, expected=expected, max_steps_per_tx=12000)
    assert st["completed"] == 8, st
    assert [r.decoded for r in recs] == expected, [(r.word, r.decoded, r.status) for r in recs]
    assert all(r.fault_spikes == 0 for r in recs)
    print("1-bit FA:", {k: v for k, v in st.items() if k != "accept_latency_ms"}, "accept ms", [round(a) for a in st["accept_latency_ms"]])


def test_ripple_adder_4bit_random_operands():
    rng = np.random.default_rng(11)
    ch = build_adder_channel(PARAMS, 4)
    cases = [(int(rng.integers(0, 16)), int(rng.integers(0, 16)), int(rng.integers(0, 2))) for _ in range(6)]
    cases[0] = (15, 15, 1)  # longest carry chain
    words = [operand_word(a, b, c, 4) for a, b, c in cases]
    expected = [a + b + c for a, b, c in cases]
    recs, sim, st = run_transactions(ch, PARAMS, words, expected=expected, max_steps_per_tx=20000)
    assert st["completed"] == len(cases), [(r.word, r.decoded, r.status) for r in recs]
    assert [r.decoded for r in recs] == expected, [(r.word, r.decoded, r.status) for r in recs]
    assert all(r.fault_spikes == 0 for r in recs)
    print("4-bit adder:", {k: v for k, v in st.items() if k != "accept_latency_ms"}, "accept ms", [round(a) for a in st["accept_latency_ms"]])


def test_ordered_adder_1bit_exhaustive_and_4bit_corners():
    """The veto-relay adder (operand gate, delayed B, delayed carries): no rate-mode gate."""
    ch = build_adder_channel(PARAMS, 1, ordered=True)
    cases = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]
    recs, sim, st = run_transactions(ch, PARAMS, [operand_word(a, b, c, 1) for a, b, c in cases],
                                     expected=[a + b + c for a, b, c in cases], max_steps_per_tx=12000)
    assert [r.decoded for r in recs] == [a + b + c for a, b, c in cases], [(r.word, r.decoded, r.status) for r in recs]
    assert all(r.fault_spikes == 0 for r in recs)
    ch = build_adder_channel(PARAMS, 4, ordered=True)
    cases = [(15, 15, 1), (0, 0, 0), (7, 8, 1), (9, 6, 0), (15, 0, 1), (5, 11, 0), (1, 15, 0)]
    recs, sim, st = run_transactions(ch, PARAMS, [operand_word(a, b, c, 4) for a, b, c in cases],
                                     expected=[a + b + c for a, b, c in cases], max_steps_per_tx=12000)
    assert [r.decoded for r in recs] == [a + b + c for a, b, c in cases], [(r.word, r.decoded, r.status) for r in recs]
    assert all(r.fault_spikes == 0 and r.timeout_spikes == 0 for r in recs)
    print("ordered 4-bit adder:", ch.net.n, "neurons; accept ms", [round(a) for a in st["accept_latency_ms"]])
