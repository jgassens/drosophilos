"""The first machine: ALU -> staged register -> ALU operand. The host loads the initial
accumulator, then for every instruction injects (B, op) and one COMMIT token and reads
the committed accumulator; everything else, including the value that feeds back as
operand A, is neural."""

import numpy as np

from drosophilos.lib.alu import OPS, alu_operand_word, alu_reference, decode_alu_word
from drosophilos.lib.staged import build_accumulator, run_commits
from drosophilos.sim.model import Params

PARAMS = Params()
W = 4


def _program(instrs, acc0):
    acc, words, expected = acc0, [], []
    for op, b in instrs:
        ref = alu_reference(acc, b, op, W)
        acc = ref["r"]
        words.append(alu_operand_word(b, op, W))
        expected.append(ref["word"])
    return words, expected


def test_seven_instruction_program():
    sc = build_accumulator(PARAMS, W)
    instrs = [("MOV", 5), ("ADD", 3), ("SUB", 9), ("AND", 6), ("OR", 1), ("XOR", 15), ("ADD", 15)]
    words, expected = _program(instrs, 0)
    recs, sim, st = run_commits(sc, PARAMS, words, expected, commit_delay_steps=0, init_master=0)
    assert st["committed"] == 7 and st["correct"] == 7, [(i, r.decoded_master, r.status) for i, r in zip(instrs, recs)]
    assert all(r.fault_spikes == 0 and r.m_fault_spikes == 0 and r.timeout_spikes == 0 for r in recs)
    flags = [(r.decoded_master >> W) & 7 for r in recs]  # C, Z, V
    assert flags[1] == 0b100 and flags[2] == 0b000 and flags[6] == 0b101, flags  # 5+3: V; 8-9: borrow (C=0); 8+15: C and V (-8 + -1)
    print("accumulator:", sc.net.n, "neurons; accept ms", [round(a) for a in st["accept_ms"]],
          "commit ms", [round(c) for c in st["commit_ms"]], "cycle ms", [round(c) for c in st["cycle_ms"]])


def test_random_program_matches_reference():
    rng = np.random.default_rng(7)
    ops = list(OPS)
    instrs = [(ops[int(rng.integers(0, len(ops)))], int(rng.integers(0, 16))) for _ in range(10)]
    acc0 = int(rng.integers(0, 16))
    words, expected = _program(instrs, acc0)
    for w, (op, b) in zip(words, instrs):
        assert decode_alu_word(w, W, with_a=False) == (None, b, op)
    sc = build_accumulator(PARAMS, W, mul=True)
    recs, sim, st = run_commits(sc, PARAMS, words, expected, commit_delay_steps=[int(rng.integers(0, 4000)) for _ in instrs],
                                init_master=acc0)
    assert st["committed"] == len(instrs) and st["correct"] == len(instrs), [(i, r.decoded_master, r.expected, r.status) for i, r in zip(instrs, recs)]
    assert all(r.fault_spikes == 0 and r.m_fault_spikes == 0 for r in recs)
