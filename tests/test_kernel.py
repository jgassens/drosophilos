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


TICK2 = open("examples/tick2.c").read()
VELS = [5, 5, 250, 3, 0, 40, 40, 40]  # includes a west-wall wrap and an east-wall clamp


def test_per_tick_input_makes_the_velocity_the_token():
    from drosophilos.compiler.kernel import kernel_outputs
    prog = compile_c(TICK2)
    ks = compile_kernel(prog, loop_body(prog), "i")
    assert "input" in (ks.cells[0]["a"], ks.cells[0].get("b")) and not ks.cells[0].get("trigger")  # px + vel reads the token itself
    assert [v for pair in kernel_outputs(ks, VELS) for v in pair] == interpret(prog, VELS)["outs"][:16]


@pytest.mark.slow
def test_neural_state_kernel_with_fresh_input_every_tick():
    from drosophilos.compiler.kernel import kernel_outputs
    prog = compile_c(TICK2)
    ks = compile_kernel(prog, loop_body(prog), "i")
    pl = build_pipeline(PARAMS, prog.width, ks.cells, consts=ks.consts, mems=ks.mems, outputs=ks.outputs)
    outs, sim, st = run_pipeline(pl, PARAMS, VELS, max_ms=90000)
    got = [[v for _, v in st["outputs_by_cell"][o]] for o in ks.outputs]
    assert got == [list(col) for col in zip(*kernel_outputs(ks, VELS))], (got, st)
    assert st["faults"] == 0 and st["timeouts"] == 0
    print("state kernel, fresh input", {k: v for k, v in st.items() if k != "outputs_by_cell"})


SHIFTS = """static u8 x, y, i;
int main(void) { i = 4; while (i != 0) { x = in_read(); y = (x >> 3) + (x << 2); out_pixel(y); i = i - 1; } return 0; }"""


def test_constant_shifts_agree_across_the_references():
    from drosophilos.compiler.golden import run_golden
    from drosophilos.compiler.kernel import kernel_outputs
    prog = compile_c(SHIFTS)
    ins = [7, 200, 255, 1]
    ks = compile_kernel(prog, loop_body(prog), "i")
    assert [c["op"] for c in ks.cells] == ["SHR", "SHL", "ADD"]
    assert interpret(prog, ins)["outs"] == run_golden(SHIFTS, ins, ["x", "y", "i"], 8)["outs"] == [v for p in kernel_outputs(ks, ins) for v in p]


@pytest.mark.slow
def test_neural_shift_cells_are_wiring():
    from drosophilos.compiler.kernel import kernel_outputs
    prog = compile_c(SHIFTS)
    ins = [7, 200, 255, 1]
    ks = compile_kernel(prog, loop_body(prog), "i")
    pl = build_pipeline(PARAMS, prog.width, ks.cells, consts=ks.consts, mems=ks.mems, outputs=ks.outputs)
    outs, sim, st = run_pipeline(pl, PARAMS, ins, max_ms=30000)
    assert [v for _, v in outs] == [p[0] for p in kernel_outputs(ks, ins)], (outs, st)
    assert st["faults"] == 0 and st["timeouts"] == 0
    print("shift kernel", {k: v for k, v in st.items() if k != "outputs_by_cell"})


RENDER2 = open("examples/render2.c").read()


def test_perspective_column_loop_matches_the_references():
    from drosophilos.compiler.golden import run_golden
    from drosophilos.compiler.kernel import kernel_outputs
    prog = compile_c(RENDER2)
    ks = compile_kernel(prog, loop_body(prog), "col", params={"heading": 3})
    assert [c["op"] for c in ks.cells] == ["ADD", "AND", "LOAD", "LOAD", "MUL", "SHR"]
    ko = [v for p in kernel_outputs(ks, list(range(8))) for v in p]
    assert ko == interpret(prog, [3])["outs"] == run_golden(RENDER2, [3], ["col", "d", "r", "h", "heading"], 16)["outs"]


@pytest.mark.slow
def test_neural_perspective_kernel_renders_eight_columns():
    """16-bit cells with a multiplier: the heights of eight columns (~6.6 s per column, the
    array multiplier's latency; ~6 minutes of wall time)."""
    from drosophilos.compiler.kernel import kernel_outputs
    prog = compile_c(RENDER2)
    ks = compile_kernel(prog, loop_body(prog), "col", params={"heading": 3})
    pl = build_pipeline(PARAMS, prog.width, ks.cells, consts=ks.consts, mems=ks.mems, outputs=ks.outputs)
    outs, sim, st = run_pipeline(pl, PARAMS, list(range(8)), max_ms=120000)
    assert [v for _, v in outs] == [p[0] for p in kernel_outputs(ks, list(range(8)))], (outs, st)
    assert st["faults"] == 0 and st["timeouts"] == 0
    print("perspective kernel", {k: v for k, v in st.items() if k != "outputs_by_cell"})


@pytest.mark.slow
def test_two_streams_with_a_parameter_edge_host_paced():
    """A world-update cell (state: heading += 1 per tick token) and a column kernel reading the
    heading as a parameter; the host streams a frame's columns, then a tick, then the next frame."""
    spec = [{"name": "hd", "op": "ADD", "a": "hd", "b": ("const", "k1"), "init": 3, "trigger": ["input:tick"]},
            {"name": "c1", "op": "ADD", "a": "input", "b": ("param", "hd")},
            {"name": "c2", "op": "AND", "a": "c1", "b": ("const", "k7")}]
    pl = build_pipeline(PARAMS, 8, spec, consts={"k1": 1, "k7": 7}, outputs=["c2", "hd"], streams=["input", "tick"])
    # the host paces both ways: a tick waits for the frame's columns (their pixels are out), and
    # the next frame's columns wait for the tick's state to have landed (its output is out);
    # the third number is the count of outputs, over all cells, that must be out first
    sched = [0, 1, 2, ("tick", 0, 3), ("input", 0, 4), ("input", 1, 4), ("input", 2, 4), ("tick", 1, 7), ("input", 5, 8)]
    outs, sim, st = run_pipeline(pl, PARAMS, sched, max_ms=40000)
    expect = [(0 + 3) & 7, (1 + 3) & 7, (2 + 3) & 7, (0 + 4) & 7, (1 + 4) & 7, (2 + 4) & 7, (5 + 5) & 7]
    assert [v for _, v in outs] == expect, (outs, st)
    assert st["faults"] == 0 and st["timeouts"] == 0
    print("two streams", {k: v for k, v in st.items() if k != "outputs_by_cell"})


def frame_schedule(spec, frames: int, columns: int) -> list:
    """The host schedule for a two-loop renderer: a frame's columns, then the tick; each token
    waits for every output the earlier tokens owe (columns behind the tick's committed state,
    the tick behind the frame's pixels)."""
    n_tick_outs = sum(1 for c in spec.cells if c["name"] in spec.outputs and c.get("stream") != "input")
    sched, owed = [], 0
    for f in range(frames):
        for c in range(columns):  # the frame's columns wait only for the previous tick's outputs, not for each other
            sched.append(("input", c, owed))
        owed += columns
        sched.append((spec.streams[1], f, owed))
        owed += n_tick_outs
    return sched


def test_two_loop_renderer_compiles_to_two_kernels_and_matches_the_interpreter():
    from drosophilos.compiler.kernel import compile_program, kernel_outputs
    prog = compile_c(RENDER)
    ks = compile_program(prog, params={"heading": 3})
    assert ks.streams == ["input", "frame"] and ks.state_cells == {"heading": "f1_add"}
    assert [c["op"] for c in ks.cells if c["stream"] == "input"] == ["ADD", "AND", "LOAD", "LOAD"]
    assert ks.cells[2]["b"] == ("param", "f1_add")  # the columns read the heading as a parameter
    ko = kernel_outputs(ks, [(s, v) for s, v, _ in frame_schedule(ks, 2, 8)])
    pixels_and_records = [v for outs in ko for v in outs[:1]]  # a column's pixel, a tick's frame record
    assert pixels_and_records == interpret(prog, [3])["outs"]


@pytest.mark.slow
def test_neural_two_loop_renderer_two_frames():
    """Two kernels, two token streams, the host pacing frames: sixteen pixels and two frame
    records equal the interpreter's, with the heading advanced by the tick between frames."""
    from drosophilos.compiler.kernel import compile_program
    prog = compile_c(RENDER)
    ks = compile_program(prog, params={"heading": 3})
    pl = build_pipeline(PARAMS, prog.width, ks.cells, consts=ks.consts, mems=ks.mems, outputs=ks.outputs, streams=ks.streams)
    sched = frame_schedule(ks, 2, 8)
    n_tick_outs = sum(1 for c in ks.cells if c["name"] in ks.outputs and c["stream"] != "input")
    outs, sim, st = run_pipeline(pl, PARAMS, sched, max_ms=120000, expect_outputs=16 + 2 * n_tick_outs)
    by = st["outputs_by_cell"]
    pixels = [v for _, v in by["c3_load"]]
    records = [v for _, v in by["f0_mov"]]
    ir = interpret(prog, [3])["outs"]
    assert pixels == [x for k, x in enumerate(ir) if k % 9 != 8] and records == [0, 1], (by, st)
    assert st["faults"] == 0 and st["timeouts"] == 0
    print("two-loop renderer", {k: v for k, v in st.items() if k != "outputs_by_cell"})
