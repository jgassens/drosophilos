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
    run, sim = run_machine(m, PARAMS, words, dmem, max_ms=1000 * (len(ref["trace"]) + 4))
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
