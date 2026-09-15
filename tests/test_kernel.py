"""Resident kernels: the renderer's column loop compiled to a pipeline (compiler/kernel.py) and
run as spatial dataflow (lib/kernel.py)."""

import pytest

from drosophilos.compiler.frontend_c import compile_c
from drosophilos.compiler.kernel import NotAKernel, compile_kernel, kernel_reference, loop_body
from drosophilos.isa.ir import interpret
from drosophilos.lib.kernel import build_pipeline, run_pipeline
from drosophilos.sim.model import Params

PARAMS = Params()
RENDER = open("examples/render.c").read()


def test_column_loop_compiles_to_four_cells_and_matches_the_interpreter():
    prog = compile_c(RENDER)
    body = loop_body(prog, "main", "loop3")
    ks = compile_kernel(prog, body, "col", params={"heading": 3})
    assert [c["op"] for c in ks.cells] == ["ADD", "AND", "LOAD", "LOAD"]
    assert ks.cells[0]["a"] == "input" and ks.cells[2]["mem"] == "map" and ks.cells[3]["mem"] == "htab"
    assert ks.consts == {"k3": 3, "k7": 7}  # the array bases and the induction step folded away
    assert ks.mems["map"] == (8, {0: 6, 1: 6, 2: 5, 3: 4, 4: 3, 5: 4, 6: 5, 7: 6})
    for heading in (0, 3, 5):
        ks = compile_kernel(prog, body, "col", params={"heading": heading})
        assert kernel_reference(ks, list(range(8))) == interpret(prog, [heading])["outs"][:8]


def test_bodies_with_branches_or_missing_parameters_are_rejected():
    prog = compile_c(RENDER)
    with pytest.raises(NotAKernel):
        compile_kernel(prog, loop_body(prog, "main", "loop1"), "frame", params={"heading": 1})  # holds the inner loop
    with pytest.raises(NotAKernel):
        compile_kernel(prog, loop_body(prog, "main", "loop3"), "col")  # heading is not given


@pytest.mark.slow
def test_neural_pipeline_renders_eight_columns():
    prog = compile_c(RENDER)
    ks = compile_kernel(prog, loop_body(prog, "main", "loop3"), "col", params={"heading": 3})
    pl = build_pipeline(PARAMS, prog.width, ks.cells, consts=ks.consts, mems=ks.mems)
    outs, sim, st = run_pipeline(pl, PARAMS, list(range(8)), max_ms=40000)
    assert [v for _, v in outs] == kernel_reference(ks, list(range(8))), (outs, st)
    print("pipeline", st)


@pytest.mark.slow
def test_neural_pipeline_with_a_fan_out_value():
    """One master read by two cells: the producer commits only when both have sampled it."""
    spec = [{"name": "c1", "op": "ADD", "a": "input", "b": ("const", "k1")},
            {"name": "c2", "op": "AND", "a": "c1", "b": ("const", "k7")},
            {"name": "c3", "op": "XOR", "a": "c1", "b": ("const", "k3")},
            {"name": "c4", "op": "ADD", "a": "c2", "b": "c3"}]
    pl = build_pipeline(PARAMS, 8, spec, consts={"k1": 1, "k7": 7, "k3": 3})
    tokens = [0, 5, 9, 14, 14, 200]
    outs, sim, st = run_pipeline(pl, PARAMS, tokens, max_ms=30000)
    expect = [(((t + 1) & 7) + ((t + 1) ^ 3)) & 255 for t in tokens]
    assert [v for _, v in outs] == expect, (outs, st)
    print("fan-out pipeline", st)


TICK = open("examples/tick.c").read()


def test_tick_loop_compiles_to_a_state_kernel_and_matches_the_interpreter():
    """The toy world update: loop-carried state (px, mx, health), two ifs per call inlined into
    select cells, two outputs per tick."""
    from drosophilos.compiler.kernel import kernel_outputs
    prog = compile_c(TICK)
    ks = compile_kernel(prog, loop_body(prog, "main", "loop9"), "i", params={"vel": 5})
    assert [c["op"] for c in ks.cells].count("SEL") == 4  # px twice, mx, health
    assert ks.state_cells.keys() == {"px", "mx", "health"} and len(ks.outputs) == 2
    assert [c.get("init") for c in ks.cells if c.get("init") is not None] == [20, 90, 100]
    for vel in (5, 250, 0):
        ks = compile_kernel(prog, loop_body(prog, "main", "loop9"), "i", params={"vel": vel})
        ir = interpret(prog, [vel])["outs"]
        assert [v for pair in kernel_outputs(ks, [3, 2, 1]) for v in pair] == ir[:6]


@pytest.mark.slow
def test_neural_state_kernel_runs_three_ticks():
    from drosophilos.compiler.kernel import kernel_outputs
    prog = compile_c(TICK)
    ks = compile_kernel(prog, loop_body(prog, "main", "loop9"), "i", params={"vel": 5})
    pl = build_pipeline(PARAMS, prog.width, ks.cells, consts=ks.consts, mems=ks.mems, outputs=ks.outputs)
    outs, sim, st = run_pipeline(pl, PARAMS, [3, 2, 1], max_ms=45000)
    got = [[v for _, v in st["outputs_by_cell"][o]] for o in ks.outputs]
    assert got == [list(col) for col in zip(*kernel_outputs(ks, [3, 2, 1]))], (got, st)
    print("state kernel", {k: v for k, v in st.items() if k != "outputs_by_cell"})
