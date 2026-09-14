"""Dual-rail ALU as a four-phase channel: reference vs ISA semantics, exhaustive 1-bit,
4-bit operations at the corners, and fault refusal."""

import numpy as np

from drosophilos.isa import semantics as sem
from drosophilos.lib.alu import OPS, alu_reference, alu_word, build_alu_channel, decode_alu_word
from drosophilos.protocol.run import run_transactions
from drosophilos.sim.model import Params

PARAMS = Params()


def _signed(x: int, w: int) -> int:
    return x - (1 << w) if x >> (w - 1) else x


def test_reference_matches_isa_semantics_4bit():
    w = 4
    for a in range(16):
        for b in range(16):
            sa, sb = _signed(a, w), _signed(b, w)
            add = sem.add_wrap(sa, sb, w)
            sub = sem.sub_wrap(sa, sb, w)
            ra, rs = alu_reference(a, b, "ADD", w), alu_reference(a, b, "SUB", w)
            assert ra["r"] == add.value & 15 and ra["v"] == add.flags.ovf, (a, b, ra, add)
            assert rs["r"] == sub.value & 15 and rs["v"] == sub.flags.ovf, (a, b, rs, sub)
            assert ra["c"] == int(a + b > 15) and rs["c"] == int(a >= b)  # unsigned carry / no-borrow
            assert alu_reference(a, b, "AND", w)["r"] == sem.and_(a, b, w).value & 15
            assert alu_reference(a, b, "OR", w)["r"] == sem.or_(a, b, w).value & 15
            assert alu_reference(a, b, "XOR", w)["r"] == sem.xor(a, b, w).value & 15
            assert alu_reference(a, b, "MOV", w)["r"] == b
            assert alu_reference(a, b, "MUL", w)["r"] == sem.mul_wrap(sa, sb, w).value & 15
            for op in OPS:
                assert decode_alu_word(alu_word(a, b, op, w), w) == (a, b, op)


def test_alu_1bit_exhaustive():
    ch = build_alu_channel(PARAMS, 1, mul=True)
    cases = [(a, b, op) for op in OPS for a in (0, 1) for b in (0, 1)]
    words = [alu_word(a, b, op, 1) for a, b, op in cases]
    expected = [alu_reference(a, b, op, 1)["word"] for a, b, op in cases]
    recs, sim, st = run_transactions(ch, PARAMS, words, expected=expected, max_steps_per_tx=15000)
    assert st["completed"] == len(cases), st
    assert [r.decoded for r in recs] == expected, [(c, r.decoded, r.status) for c, r in zip(cases, recs)]
    assert all(r.fault_spikes == 0 and r.timeout_spikes == 0 for r in recs)


def test_alu_4bit_corners():
    ch = build_alu_channel(PARAMS, 4)
    cases = [(15, 15, "ADD"), (0, 0, "ADD"), (7, 1, "ADD"), (8, 8, "ADD"), (3, 9, "SUB"), (9, 3, "SUB"), (8, 8, "SUB"),
             (0, 1, "SUB"), (6, 5, "AND"), (15, 0, "AND"), (6, 5, "OR"), (10, 5, "XOR"), (15, 15, "XOR"), (0, 11, "MOV"),
             (9, 0, "MOV")]
    words = [alu_word(a, b, op, 4) for a, b, op in cases]
    expected = [alu_reference(a, b, op, 4)["word"] for a, b, op in cases]
    recs, sim, st = run_transactions(ch, PARAMS, words, expected=expected, max_steps_per_tx=20000)
    assert st["completed"] == len(cases), st
    assert [r.decoded for r in recs] == expected, [(c, r.decoded, r.status) for c, r in zip(cases, recs)]
    assert all(r.fault_spikes == 0 and r.timeout_spikes == 0 for r in recs)
    lat = [a for a in st["accept_latency_ms"]]
    assert max(lat) < 700, lat
    print("4-bit ALU:", ch.net.n, "neurons; accept ms", [round(a) for a in lat])


def test_alu_fault_refused_and_recovers():
    ch = build_alu_channel(PARAMS, 4)
    cases = [(5, 3, "ADD"), (5, 3, "ADD"), (6, 5, "XOR")]
    words = [alu_word(a, b, op, 4) for a, b, op in cases]
    expected = [alu_reference(a, b, op, 4)["word"] for a, b, op in cases]
    recs, sim, st = run_transactions(ch, PARAMS, words, expected=expected, max_steps_per_tx=20000,
                                     faults={1: [("corrupt", 2)]})
    assert recs[0].decoded == expected[0] and recs[0].status == "valid"
    assert recs[1].status == "fault" and recs[1].fault_spikes > 0 and recs[1].ready_step is not None, recs[1]
    assert recs[2].decoded == expected[2] and recs[2].status == "valid" and recs[2].fault_spikes == 0, recs[2]


def test_alu_4bit_multiplier():
    ch = build_alu_channel(PARAMS, 4, mul=True, watchdog_hops=300)
    cases = [(3, 5, "MUL"), (7, 7, "MUL"), (0, 9, "MUL"), (15, 15, "MUL"), (2, 4, "MUL"), (13, 11, "MUL"), (6, 5, "ADD")]
    words = [alu_word(a, b, op, 4) for a, b, op in cases]
    expected = [alu_reference(a, b, op, 4)["word"] for a, b, op in cases]
    recs, sim, st = run_transactions(ch, PARAMS, words, expected=expected, max_steps_per_tx=30000)
    assert [r.decoded for r in recs] == expected, [(c, r.decoded, r.status) for c, r in zip(cases, recs)]
    assert all(r.fault_spikes == 0 and r.timeout_spikes == 0 for r in recs)
    print("4-bit ALU with multiplier:", ch.net.n, "neurons; MUL accept ms", [round(a) for a in st["accept_latency_ms"][:6]])
