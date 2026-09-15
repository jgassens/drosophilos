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


def test_interrupt_lands_at_safe_point_and_resumes():
    """Two host interrupts during a countdown loop: the handler (at word 10) saves the
    accumulator, increments a counter, restores the accumulator and returns; the main
    program's result is unchanged and the counter reads 2."""
    main = [("MOV", 3), ("SUB", 1), ("JNZ", 1), ("MOV", 9), ("STORE", 1), ("LOAD", 1), ("ADD", 2), ("STORE", 3), ("HALT", 8)]
    handler = [("STORE", 5), ("LOAD", 6), ("ADD", 1), ("STORE", 6), ("LOAD", 5), ("IRET", 0)]
    program = main + [("HALT", 9)] + handler
    m = build_machine(PARAMS, n=4, n_prog=16, n_data=8, handler_pc=10)
    run, sim = run_machine(m, PARAMS, program, {6: 0}, max_ms=30000, interrupts_at_ms=[2500, 6000])
    final = {6: 0}
    for _, k, v in run.writes:
        final[k] = v
    ref = reference_run(program, 4, 4, {6: 0}, max_steps=80)  # the main program alone
    assert final[1] == ref["dmem"][1] and final[3] == ref["dmem"][3], (final, ref["dmem"])
    assert final[6] == 2, final
    pcs = [k for _, k in run.pcs]
    assert pcs.count(10) == 2 and run.halted and run.faults == 0 and run.timeouts == 0, (pcs, run.faults)
    # each handler entry returns to the interrupted word
    for i, k in enumerate(pcs):
        if k == 10:
            assert pcs[i + 6] == pcs[i - 1], pcs


def test_clr_empties_a_word_and_the_machine_continues():
    program = [("MOV", 3), ("STORE", 2), ("CLR", 2), ("MOV", 1), ("STORE", 3), ("HALT", 5)]
    m = build_machine(PARAMS, n=4, n_prog=8, n_data=8)
    run, sim = run_machine(m, PARAMS, program, {}, max_ms=12000)
    final = {}
    for _, k, v in run.writes:
        final[k] = v
    assert [k for _, k in run.pcs] == [0, 1, 2, 3, 4, 5], run.pcs
    assert final == {2: 3, 3: 1} and run.halted and run.faults == 0, (final, run.faults)
    # word 2 is empty at the end: no rail of it is active in the last 50 ms
    ev = sim.trace.events
    w2 = m.dmem.words[2]
    late = ev["step"] > ev["step"].max() - 500
    assert not any(np.isin(ev["neuron"][late], [t for pair in w2.rail_taps for t in pair])), "word 2 not emptied"


def test_send_timer_expires_and_handler_retransmits():
    """No link: the send times out, the handler counts the timeout, clears the status word and
    re-sends once; after the second timeout it stops. Word 3 = counter, 4 = saved acc,
    5 = status (written by the timer), 7 = output port."""
    main = [("MOV", 9), ("STORE", 7), ("JMP", 3), ("JMP", 2)] + [("HALT", k) for k in range(4, 8)]
    handler = [("STORE", 4), ("LOAD", 5), ("JZ", 21), ("LOAD", 3), ("ADD", 1), ("STORE", 3), ("MOV", 0), ("STORE", 5), ("LOAD", 3), ("SUB", 2), ("JZ", 21),
               ("LOAD", 4), ("STORE", 7), ("LOAD", 4), ("IRET", 0)]  # 8..22; status <- 0 by a STORE (a CLR would empty it)
    program = main + handler + [("HALT", k) for k in range(23, 32)]
    assert len(program) == 32 and program[21] == ("LOAD", 4)
    m = build_machine(PARAMS, n=4, n_prog=32, n_data=8, handler_pc=8, port_out_word=7, timer_hops=300, status_word=5)
    run, sim = run_machine(m, PARAMS, program, {3: 0}, max_ms=36000, idle_ms=6000)
    final = {3: 0}
    for _, k, v in run.writes:
        final[k] = v
    pcs = [k for _, k in run.pcs]
    assert pcs.count(8) == 2, pcs  # the handler ran twice (two timeouts)
    assert final[3] == 2, final  # the counter counted them
    assert [k for _, k, _ in run.writes].count(7) == 2, run.writes  # the original send and one retransmit
    assert run.faults == 0 and run.timeouts == 0


def test_indexed_addressing_through_the_index_word():
    """X = word 7. Write 9 to DMEM[2] through X, then read DMEM[3] (image) and add DMEM[3] again
    through X; store the sum at word 4."""
    prog = [("MOV", 2), ("STORE", 7), ("MOV", 9), ("STOREI", 0), ("MOV", 3), ("STORE", 7), ("LOADI", 0), ("ADDI", 0), ("STORE", 4), ("HALT", 9)]
    m = build_machine(PARAMS, n=4, n_prog=16, n_data=8, x_word=7)
    ref = reference_run(prog, 4, 4, {3: 5}, max_steps=20, x_word=7)
    run, sim = run_machine(m, PARAMS, prog, {3: 5}, max_ms=16000)
    got = [v for _, v in run.commits]
    assert got == [v for _, v in ref["trace"]], (got, ref["trace"], run.faults, run.timeouts)
    final = {3: 5}
    for _, k, v in run.writes:
        final[k] = v
    assert final == ref["dmem"] == {3: 5, 7: 3, 2: 9, 4: 10}, (final, ref["dmem"])
    assert run.halted and run.faults == 0
