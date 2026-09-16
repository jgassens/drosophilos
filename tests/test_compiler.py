"""DrosoC -> IR -> (interpreter, machine reference, neural machine) agree with the C golden
reference on canonical state and emitted pixels, for fresh inputs."""

import numpy as np
import pytest

from drosophilos.compiler.frontend_c import compile_c
from drosophilos.compiler.golden import run_golden
from drosophilos.compiler.lower import lower
from drosophilos.isa.ir import interpret
from drosophilos.lib.control import build_machine, reference_run, run_machine
from drosophilos.sim.model import Params

PARAMS = Params()

SRC = """
static u8 x, acc, n, mask;
static void accumulate(void) { acc = acc + x; x = x - 1; }
int main(void) {
    x = in_read();
    acc = 0;
    n = 3;
    while (x != 0) { accumulate(); }
    mask = acc & 0x0F;
    if (mask == 0) { acc = acc ^ 0x21; } else { acc = acc + n; }
    out_pixel(acc);
    return 0;
}
"""


def _expected_state(prog, inputs):
    ref = run_golden(SRC, inputs, [v for v in prog.variables if not v.startswith("__")], prog.width)
    return ref


def test_frontend_interpreter_and_lowering_match_golden():
    prog = compile_c(SRC)
    assert prog.width == 8
    for inputs in ([4], [0], [7], [200]):
        gold = _expected_state(prog, inputs)
        ir = interpret(prog, inputs)
        assert ir["halted"] and ir["outs"] == gold["outs"], (inputs, ir["outs"], gold["outs"])
        state = [(v, val) for v, val in ir["state"] if not v.startswith("__")]
        assert state == gold["state"], (inputs, state, gold["state"])
        # the machine-level reference of the lowered program
        words = lower(prog)
        a_bits = max(1, (max(len(words), len(prog.variables)) - 1).bit_length())
        dmem = {prog.ports["in"]: inputs[0]}
        mref = reference_run(words, prog.width, a_bits, dmem, max_steps=2000)
        assert mref["halted"] and mref["fault"] is None, mref
        got_state = [(v, mref["dmem"].get(a, 0)) for v, a in sorted(prog.variables.items(), key=lambda kv: kv[1]) if not v.startswith("__")]
        assert got_state == gold["state"], (inputs, got_state, gold["state"])
        assert mref["dmem"].get(prog.ports["out"]) == gold["outs"][-1]
    print("lowered program:", len(words), "words;", len(prog.variables), "data words")


@pytest.mark.slow
def test_neural_machine_runs_compiled_program():
    prog = compile_c(SRC)
    words = lower(prog)
    n_prog = 1 << (len(words) - 1).bit_length()
    n_data = 1 << (len(prog.variables) - 1).bit_length()
    m = build_machine(PARAMS, n=prog.width, n_prog=n_prog, n_data=n_data)
    inputs = [2]
    gold = _expected_state(prog, inputs)
    dmem = {prog.ports["in"]: inputs[0]}
    ref = reference_run(words, m.n, m.a, dmem, max_steps=2000)
    run, sim = run_machine(m, PARAMS, words, dmem, max_ms=1400 * (len(ref["trace"]) + 4))  # the cycle is ~1.17 s since the 09-14 review fixes (commit guard, fetch at +166 ms); 1,000 ms per instruction cut the run two instructions short
    got = [v for _, v in run.commits]
    exp = [v for _, v in ref["trace"]]
    assert got == exp, (got[:20], exp[:20], run.faults, run.timeouts)
    final = dict(dmem)
    for _, k, v in run.writes:
        final[k] = v
    state = [(v, final.get(a, 0)) for v, a in sorted(prog.variables.items(), key=lambda kv: kv[1]) if not v.startswith("__")]
    assert state == gold["state"], (state, gold["state"])
    assert final.get(prog.ports["out"]) == gold["outs"][-1]
    print("neural machine:", m.net.n, "neurons;", len(exp), "instructions executed")


ARRAY_SRC = """
static u8 a[4] = {3, 1, 4, 1};
static u8 i, s;
int main(void) {
    i = 0; s = 0;
    while (i != 4) { s = s + a[i]; i = i + 1; }
    a[2] = s;
    out_pixel(s);
    return 0;
}
"""


MUL_SRC = """
static u8 x, y;
int main(void) { x = in_read(); y = x * 7; y = y + 1; out_pixel(y); return 0; }
"""


def test_multiply_matches_golden():
    prog = compile_c(MUL_SRC)
    for inputs in ([3], [40], [255]):
        gold = run_golden(MUL_SRC, inputs, ["x", "y"], 8)
        ir = interpret(prog, inputs)
        assert ir["outs"] == gold["outs"] and [(v, val) for v, val in ir["state"] if not v.startswith("__")] == gold["state"]
        words = lower(prog)
        a_bits = max(1, (max(len(words), len(prog.variables)) - 1).bit_length())
        ref = reference_run(words, 8, a_bits, {prog.ports["in"]: inputs[0]}, max_steps=500)
        assert ref["halted"] and ref["dmem"][prog.ports["out"]] == gold["outs"][-1], (ref, gold)


def test_arrays_through_the_index_word_match_golden():
    prog = compile_c(ARRAY_SRC)
    words = lower(prog)
    names = [v for v in prog.variables if not v.startswith("__")]
    gold = run_golden(ARRAY_SRC, [], names, prog.width)
    ir = interpret(prog, [])
    assert [(v, val) for v, val in ir["state"] if not v.startswith("__")] == gold["state"] and ir["outs"] == gold["outs"]
    a_bits = max(1, (max(len(words), len(prog.variables)) - 1).bit_length())
    ref = reference_run(words, prog.width, a_bits, {}, max_steps=3000, x_word=prog.ports["x"])
    state = [(v, ref["dmem"].get(a, 0)) for v, a in sorted(prog.variables.items(), key=lambda kv: kv[1]) if not v.startswith("__")]
    assert ref["halted"] and state == gold["state"] and ref["dmem"].get(prog.ports["out"]) == gold["outs"][-1]
    assert gold["state"][1] == ("s", 9) and gold["state"][4] == ("a[2]", 9)


@pytest.mark.slow
def test_neural_machine_runs_array_program():
    prog = compile_c(ARRAY_SRC)
    words = lower(prog)
    n_prog = 1 << (len(words) - 1).bit_length()
    n_data = 1 << (len(prog.variables) - 1).bit_length()
    m = build_machine(PARAMS, n=prog.width, n_prog=n_prog, n_data=n_data, x_word=prog.ports["x"])
    dmem = {prog.variables[f"a[{k}]"]: v for k, v in enumerate((3, 1, 4, 1))}
    ref = reference_run(words, m.n, m.a, dmem, max_steps=3000, x_word=prog.ports["x"])
    run, sim = run_machine(m, PARAMS, words, dmem, max_ms=1500 * (len(ref["trace"]) + 4))  # see above: the indexed cycle is longer still
    assert [v for _, v in run.commits] == [v for _, v in ref["trace"]], (run.faults, run.timeouts)
    final = dict(dmem)
    for _, k, v in run.writes:
        final[k] = v
    assert final.get(prog.variables["s"]) == 9 and final.get(prog.variables["a[2]"]) == 9 and final.get(prog.ports["out"]) == 9
