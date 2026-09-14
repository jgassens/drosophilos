"""The control machine: programs in neural memory executed by the one-hot sequencer, with
the accumulator, data RAM, loads, stores, conditional jumps and halt, checked against the
Python reference execution of the same ISA."""

import numpy as np

from drosophilos.lib.control import build_machine, random_program, reference_run, run_machine
from drosophilos.sim.model import Params

PARAMS = Params()


def _check(m, prog, dmem=None, max_ms=20000):
    ref = reference_run(prog, m.n, m.a, dmem)
    run, sim = run_machine(m, PARAMS, prog, dmem, max_ms=max_ms)
    got = [v for _, v in run.commits]
    exp = [v for _, v in ref["trace"]]
    assert got == exp, (got, exp, run.pcs, run.writes, run.faults, run.timeouts)
    assert [k for _, k in run.pcs] == [0] + [pc for pc, _ in ref["trace"][1:]], (run.pcs, ref["trace"])
    final = dict(dmem or {})
    for _, k, v in run.writes:
        final[k] = v
    assert final == ref["dmem"], (final, ref["dmem"])
    assert run.halted and run.faults == 0 and run.timeouts == 0
    return run


def test_arithmetic_store_load_jump_halt():
    m = build_machine(PARAMS)
    prog = [("MOV", 5), ("ADD", 3), ("STORE", 2), ("LOAD", 2), ("SUB", 8), ("JZ", 7), ("MOV", 1), ("HALT", 7)]
    run = _check(m, prog)
    t = [s for s, _ in run.commits]
    print("machine:", m.net.n, "neurons; instruction ms", [round((b - a) * PARAMS.dt) for a, b in zip(t, t[1:])])


def test_countdown_loop_with_jnz_and_data_image():
    m = build_machine(PARAMS)
    prog = [("LOAD", 3), ("SUB", 1), ("STORE", 3), ("JNZ", 1), ("LOAD", 5), ("XOR", 15), ("STORE", 6), ("HALT", 7)]
    _check(m, prog, dmem={3: 3, 5: 9}, max_ms=20000)


def test_random_programs_match_reference():
    rng = np.random.default_rng(5)
    m = build_machine(PARAMS)
    for _ in range(2):
        prog, dmem = random_program(rng)
        _check(m, prog, dmem, max_ms=26000)
