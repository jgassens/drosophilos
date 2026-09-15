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
