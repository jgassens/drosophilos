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


def test_chained_guard_rechecks_sources_when_passed_pair_is_reignited():
    """A stale intermediate `passed` rail must not replay a multi-source cell.

    The two nodes run the same unperturbed three-input guard. After its legitimate start,
    both get IDLE again, but node 1 also gets a strong synthetic stray on the cached passed
    rail. This deterministically models the mix-B guard-doublet failure: before the final
    source recheck, node 1 emitted a second start with both source REQs false.
    """
    import torch

    from drosophilos.lib.control import add_kill_pair, add_kill_train
    from drosophilos.lib.kernel import _chain_true
    from drosophilos.lib.netlist import Drive, Netlist
    from drosophilos.sim.lif_torch import TorchSim

    drive = Drive.from_params(PARAMS)
    net = Netlist(PARAMS)
    image = []
    a = add_kill_pair(net, drive, "a")
    b = add_kill_pair(net, drive, "b")
    idle = add_kill_pair(net, drive, "idle")
    image += [a[0], b[0], idle[1]]
    start = net.neuron("start")
    _chain_true(net, drive, "go", [a, b, idle], start, image, start)
    for latch in (a[0], b[0], idle[0]):
        net.synapse(start, latch.u, drive.ignite)
    add_kill_train(net, drive, "start.kill", start, [a[1], b[1], idle[1]])
    passed = net.roles.index("go.p0r1.u")

    class InjectPassed(TorchSim):
        def __init__(self):
            super().__init__(net.topology(), PARAMS, n_nodes=2, device="cpu", dtype=torch.float32)
            self.injected_at = None

        def step(self):
            before = len(self._spk_step)
            super().step()
            for k in range(before, len(self._spk_step)):
                for node, neuron in zip(self._spk_node[k].tolist(), self._spk_neuron[k].tolist()):
                    if node == 1 and neuron == start and self.injected_at is None:
                        now = self.step_index
                        self.injected_at = now + 1000  # after the start reset has recovered
                        self.add_events(1, [self.injected_at, now + 2000], [passed, idle[1].u],
                                        [drive.ignite, drive.ignite])
                        self.add_events(0, [now + 2000], [idle[1].u], [drive.ignite])

    sim = InjectPassed()
    for node in range(2):
        for latch in image:
            sim.add_events(node, [1], [latch.u], [drive.ignite])
        sim.add_events(node, [1000, 1000], [a[1].u, b[1].u], [drive.ignite, drive.ignite])
    sim.run(7000)

    starts = [[] for _ in range(2)]
    passed_after_injection = False
    for steps, nodes, neurons in zip(sim._spk_step, sim._spk_node, sim._spk_neuron):
        step = int(steps[0])
        for node, neuron in zip(nodes.tolist(), neurons.tolist()):
            if neuron == start and (not starts[node] or step - starts[node][-1] > 100):
                starts[node].append(step)
            if node == 1 and neuron == passed and sim.injected_at is not None and step >= sim.injected_at:
                passed_after_injection = True
    assert sim.injected_at is not None and passed_after_injection
    assert [len(node_starts) for node_starts in starts] == [1, 1], starts


def _run_request_ambiguity_reproduction(pl, cell_name, source, token, max_ms):
    """Two CPU-reference copies receive the same forced both-live REQ transition.

    Node 0 is the fixed topology.  On node 1 only, the new REQ-self-false input to the
    start guard is zeroed, exactly reconstructing the old guard.  Repeated ignition of the
    losing rail holds the deterministic state which a mix-B hit during an unlucky kill-pair
    transition can leave behind; it is not random fault search.
    """
    import numpy as np

    from drosophilos.lib.kernel import run_pipeline_batched
    from drosophilos.sim.ref64 import RefSim

    cell = next(c for c in pl.cells if c.name == cell_name)
    req = cell.reqs[source]
    topo = pl.net.topology()
    veto = pl.net.roles.index(f"{cell.name}.go.g0.pa.veto")
    self_veto = np.flatnonzero((topo.src == req[0].u) & (topo.dst == veto))
    assert len(self_veto) == 1
    quanta = np.broadcast_to(topo.quanta, (2, topo.nnz)).copy()
    quanta[1, self_veto] = 0  # node 1 is the pre-fix guard

    class InjectAmbiguousRequest(RefSim):
        def __init__(self):
            super().__init__(topo, PARAMS, n_nodes=2, quanta=quanta)
            self.saw_done = [False, False]
            self.injected = [False, False]
            self.starts = [[], []]

        def step(self):
            before = len(self._spk_step)
            super().step()
            for k in range(before, len(self._spk_step)):
                step = int(self._spk_step[k][0])
                for node, neuron in zip(self._spk_node[k].tolist(), self._spk_neuron[k].tolist()):
                    if neuron == cell.reg.done_relay:
                        self.saw_done[node] = True
                    if neuron == cell.idle[1].u and self.saw_done[node] and not self.injected[node]:
                        self.injected[node] = True
                        now = self.step_index
                        steps = [now + 5] + [now + d for d in (400, 500, 600, 700, 800)]
                        neurons = [req[1].u] + [req[0].u] * 5
                        self.add_events(node, steps, neurons, [pl.drive.ignite] * len(steps))
                    if neuron == cell.start and (not self.starts[node] or step - self.starts[node][-1] > 100):
                        self.starts[node].append(step)

    sim = InjectAmbiguousRequest()
    outs, sim, stats = run_pipeline_batched(pl, PARAMS, [[token], [token]], max_ms=max_ms,
                                             expect_outputs=[2, 2], sim=sim, progress=0)
    return [[v for _, v in node[pl.output.name]] for node in outs], sim.starts, stats


@pytest.mark.xfail(strict=False, reason="§10.4's remedy (one-hot guards, four-pulse REQ arbitration, 14/22-hop margins) stalled copies under mix B and was reverted; this reproduction records the failure it was written against")
def test_mulp_row_request_ambiguity_reproduces_an_extra_output_and_is_vetoed():
    """Shape A: one stale row request adds a second product without another input token."""
    pl = build_pipeline(PARAMS, 4, [{"name": "m", "op": "MULP", "a": "input", "b": ("const", "k")}],
                        consts={"k": 3})
    assert pl.net.n < 30000
    got, starts, stats = _run_request_ambiguity_reproduction(pl, "m.r1", "m.r0", 3, 8000)
    assert got == [[9], [9, 9]], (got, starts, stats)
    assert [len(x) for x in starts] == [1, 2]
    # The fixed per-source pair has the fourth kill pulse on both transitions.
    assert "m.r1.req.m.r0.k0.h3" in pl.net.roles and "m.r1.req.m.r0.k1.h3" in pl.net.roles


@pytest.mark.xfail(strict=False, reason="§10.4's remedy (one-hot guards, four-pulse REQ arbitration, 14/22-hop margins) stalled copies under mix B and was reverted; this reproduction records the failure it was written against")
def test_load_request_ambiguity_reproduces_an_old_address_and_is_vetoed():
    """Shape B: a final ROM reader re-runs on the preceding address while no token arrives."""
    mem1 = {i: (i + 1) & 15 for i in range(16)}
    mem2 = {i: (i * 3) & 15 for i in range(16)}
    spec = [{"name": "a", "op": "LOAD", "a": "input", "mem": "m1"},
            {"name": "out", "op": "LOAD", "a": "a", "mem": "m2"}]
    pl = build_pipeline(PARAMS, 4, spec, mems={"m1": (16, mem1), "m2": (16, mem2)})
    assert pl.net.n < 30000
    got, starts, stats = _run_request_ambiguity_reproduction(pl, "out", "a", 2, 5000)
    assert got == [[9], [9, 9]], (got, starts, stats)
    assert [len(x) for x in starts] == [1, 2]


@pytest.mark.xfail(strict=False, reason="§10.5's remedy (ACT^d re-lights the request and IDLE rails) removed the perspective duplicates but stalled copies in every block under mix B and was reverted; this reproduction records the failure it was written against")
def test_dark_request_rail_replays_the_next_word_and_actd_relights_it():
    """§10.5: a request's false rail re-lit by the start pulse ~55 ms after its own kill train
    can fail to catch; the dark pair lets the row's next IDLE start it with no request, and the
    real request then replays the word. Both copies carry the same 3 sigma corner on one rail
    (`m.r2.req.m.r1r0`: loop -12 %, kill +12 %, V_th +0.4 mV, bias -0.4 mV); copy 1 also has the
    ACT^d re-light synapse zeroed, which is the pre-fix circuit. Three tokens 1.4 s apart, the
    perspective pipeline's regime (token interval longer than a row's cycle)."""
    import numpy as np

    from drosophilos.lib.kernel import run_pipeline_batched
    from drosophilos.sim.ref64 import RefSim

    pl = build_pipeline(PARAMS, 4, [{"name": "m", "op": "MULP", "a": "input", "b": ("const", "k")}],
                        consts={"k": 3})
    assert pl.net.n < 30000
    cell = next(c for c in pl.cells if c.name == "m.r2")
    req = cell.reqs["m.r1"]
    roles = pl.net.roles
    topo = pl.net.topology()
    u, v = req[0].u, req[0].v
    k1_inh = roles.index("m.r2.req.m.r1.k1.inh")
    act_d = roles.index("m.r2.actd.d10")
    trigger = roles.index("m.r2.trigger.m.r1")
    loop = ((topo.src == u) & (topo.dst == v)) | ((topo.src == v) & (topo.dst == u))
    kill = (topo.src == k1_inh) & ((topo.dst == u) | (topo.dst == v))
    relight = (topo.src == act_d) & (topo.dst == u)
    assert loop.sum() == 2 and kill.sum() == 2 and relight.sum() == 1
    quanta = np.broadcast_to(topo.quanta, (2, topo.nnz)).astype(np.float64).copy()
    quanta[:, loop] *= 0.88
    quanta[:, kill] *= 1.12
    quanta[1, relight] = 0  # copy 1: the start pulse's ignition is the rail's only chance
    quanta = np.rint(quanta).astype(np.int32)
    vth = np.full((2, topo.n), PARAMS.V_th)
    vth[:, [u, v]] += 0.4
    bias = np.zeros((2, topo.n))
    bias[:, [u, v]] -= 0.4

    class Record(RefSim):
        def __init__(self):
            super().__init__(topo, PARAMS, n_nodes=2, quanta=quanta, V_th=vth, bias=bias)
            self.rises = {(b, w): [] for b in range(2) for w in ("start", "trigger", "req_false")}
            self.watch = {cell.start: "start", trigger: "trigger", u: "req_false"}

        def step(self):
            before = len(self._spk_step)
            super().step()
            for k in range(before, len(self._spk_step)):
                step = int(self._spk_step[k][0])
                for node, neuron in zip(self._spk_node[k].tolist(), self._spk_neuron[k].tolist()):
                    what = self.watch.get(neuron)
                    if what is not None:
                        seen = self.rises[(node, what)]
                        if not seen or step - seen[-1] > 150:
                            seen.append(step)

    sim = Record()
    outs, sim, stats = run_pipeline_batched(pl, PARAMS, [[3, 5, 2], [3, 5, 2]], max_ms=9000, expect_outputs=[3, 3],
                                             sim=sim, progress=0, gap_ms=1400)
    got = [[value for _, value in outs[b]["m"]] for b in range(2)]
    assert got == [[9, 15, 6], [9, 15, 15]], (got, stats)
    starts = [sim.rises[(b, "start")] for b in range(2)]
    triggers = [sim.rises[(b, "trigger")] for b in range(2)]
    assert [len(x) for x in starts] == [3, 4], starts  # the pre-fix copy: token 2 early, token 2 again, then token 3
    # fixed copy: every start follows its trigger by the guard's 12 hops
    assert all(0 < s - t < 1000 for s, t in zip(starts[0], triggers[0])), (starts[0], triggers[0])
    # pre-fix copy: the second start precedes the second trigger (a start with no request)
    assert starts[1][1] < triggers[1][1], (starts[1], triggers[1])
    # the rail: one spike from the start's ignition on both copies, a train only where ACT^d re-lit it
    first_start = starts[0][0]
    rail = [[r for r in sim.rises[(b, "req_false")] if first_start < r < first_start + 1500] for b in range(2)]
    assert len(rail[1]) == 1 and len(rail[0]) >= 2 and rail[0][1] > first_start + 500, rail


def test_batched_runner_captures_selected_spikes_before_trace_trimming():
    import numpy as np

    from drosophilos.lib.kernel import run_pipeline_batched
    from drosophilos.sim.ref64 import RefSim

    pl = build_pipeline(PARAMS, 1, [{"name": "out", "op": "MOV", "a": "input", "b": ("const", "zero")}],
                        consts={"zero": 0})
    wanted = {pl.cells[0].start, pl.cells[0].reg.done_relay}
    sim = RefSim(pl.net.topology(), PARAMS, n_nodes=2)
    outs, sim, stats = run_pipeline_batched(pl, PARAMS, [[1], [0]], max_ms=3000, sim=sim, progress=0,
                                             capture_spikes=(1, wanted))
    steps, neurons = stats["captured_spikes"]
    assert len(steps) == len(neurons) > 0
    assert set(neurons.tolist()) == wanted
    assert np.all(steps[1:] >= steps[:-1])


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


@pytest.mark.slow
def test_batched_nodes_run_the_renderer_on_different_columns():
    """Two copies of the column kernel on the batched simulator, each with half the columns:
    the cluster's shape (one kernel per brain, the host dealing tokens)."""
    from drosophilos.compiler.kernel import kernel_outputs
    from drosophilos.lib.kernel import run_pipeline_batched
    prog = compile_c(RENDER)
    ks = compile_kernel(prog, loop_body(prog, "main", "loop3"), "col", params={"heading": 3})
    pl = build_pipeline(PARAMS, prog.width, ks.cells, consts=ks.consts, mems=ks.mems)
    outs, sim, st = run_pipeline_batched(pl, PARAMS, [[0, 1, 2, 3], [4, 5, 6, 7]], max_ms=30000)
    ref = kernel_reference(ks, list(range(8)))
    assert [v for _, v in outs[0]["c3_load"]] == ref[:4] and [v for _, v in outs[1]["c3_load"]] == ref[4:], (outs, st)
    print("batched", st)


def test_game_program_compiles_to_a_tick_kernel_and_a_pixel_kernel():
    from drosophilos.compiler.kernel import compile_program, kernel_outputs
    prog = compile_c(open("examples/game.c").read())
    ks = compile_program(prog, params={"heading": 0})
    assert ks.streams == ["input", "f"] and set(ks.state_cells) == {"heading"}
    ticks = [c for c in ks.cells if c["stream"] != "input"]
    assert [c["op"] for c in ticks] == ["ADD", "AND"]  # heading = (heading + turn) & 63
    toks = [(50 << 8) | x for x in (0, 40, 80, 120)]
    sched = [("input", t) for t in toks] + [("f", 10)] + [("input", t) for t in toks] + [("f", 10)]
    pixels = [o[0] for (st, _), o in zip(sched, kernel_outputs(ks, sched)) if st == "input"]  # a tick's output is its state
    assert pixels == interpret(prog, [0] + toks + [10] + toks + [10] + toks + [10])["outs"][:8]


@pytest.mark.slow
def test_neural_32bit_kernel_three_cells():
    """32-bit cells: ADD, XOR, SUB on four tokens (the input register's watchdog scales with the width)."""
    spec = [{"name": "c1", "op": "ADD", "a": "input", "b": ("const", "k1")},
            {"name": "c2", "op": "XOR", "a": "c1", "b": ("const", "k2")},
            {"name": "c3", "op": "SUB", "a": "c2", "b": "c1"}]
    consts = {"k1": 0x12345678, "k2": 0x0F0F0F0F}
    pl = build_pipeline(PARAMS, 32, spec, consts=consts)
    tokens = [0, 1, 0xFFFFFFFF, 0x7FFFFFFF]
    mask = 0xFFFFFFFF
    exp = [((((t + consts["k1"]) & mask) ^ consts["k2"]) - ((t + consts["k1"]) & mask)) & mask for t in tokens]
    outs, sim, st = run_pipeline(pl, PARAMS, tokens, max_ms=30000)
    assert [v for _, v in outs] == exp and st["faults"] == 0 and st["timeouts"] == 0, (outs, st)
    print("32-bit kernel", {k: v for k, v in st.items() if k != "outputs_by_cell"})


@pytest.mark.slow
def test_neural_pipelined_multiplier_eight_bits():
    """MULP: n row cells; one product per cell latency instead of the array's n^2 ripple."""
    pl = build_pipeline(PARAMS, 8, [{"name": "m", "op": "MULP", "a": "input", "b": ("const", "k")}], consts={"k": 3})
    toks = [3, 7, 0, 255, 16, 100]
    outs, sim, st = run_pipeline(pl, PARAMS, toks, max_ms=40000)
    assert [v for _, v in outs] == [(x * 3) & 255 for x in toks] and st["faults"] == 0 and st["bad_outputs"] == 0, (outs, st)
    print("MULP 8-bit", {k: v for k, v in st.items() if k not in ("outputs_by_cell", "load_steps")})


@pytest.mark.slow
def test_neural_perspective_kernel_with_the_pipelined_multiplier():
    from drosophilos.compiler.kernel import kernel_outputs
    prog = compile_c(RENDER2)
    ks = compile_kernel(prog, loop_body(prog), "col", params={"heading": 3}, mul="pipelined")
    assert "MULP" in [c["op"] for c in ks.cells]
    pl = build_pipeline(PARAMS, prog.width, ks.cells, consts=ks.consts, mems=ks.mems, outputs=ks.outputs)
    outs, sim, st = run_pipeline(pl, PARAMS, list(range(8)), max_ms=120000)
    assert [v for _, v in outs] == [p[0] for p in kernel_outputs(ks, list(range(8)))], (outs, st)
    print("perspective kernel, MULP", {k: v for k, v in st.items() if k not in ("outputs_by_cell", "load_steps")})


@pytest.mark.slow
def test_kernel_ram_written_by_one_pass_and_read_by_the_next():
    """A column pass stores heights into a RAM buffer; a pixel pass reads them back by column
    (the host paces the passes: the reads start once every write has landed)."""
    spec = [{"name": "h", "op": "LOAD", "a": "input", "mem": "htab"},
            {"name": "st", "op": "STORE", "a": "input", "b": "h", "mem": "hbuf"},
            {"name": "col", "op": "AND", "a": "input:pix", "b": ("const", "k255")},
            {"name": "hb", "op": "LOAD", "a": "col", "mem": "hbuf"},
            {"name": "out", "op": "ADD", "a": "hb", "b": ("const", "k1")}]
    htab = {0: 40, 1: 33, 2: 25, 3: 19}
    pl = build_pipeline(PARAMS, 8, spec, consts={"k255": 255, "k1": 1}, mems={"htab": (4, htab), "hbuf": (4, {}, "ram")},
                        outputs=["st", "out"], streams=["input", "pix"])
    sched = [0, 1, 2, 3] + [("pix", (1 << 8) | c, 4) for c in (2, 0, 3, 1)]
    outs, sim, st = run_pipeline(pl, PARAMS, sched, max_ms=60000, expect_outputs=8)
    by = st["outputs_by_cell"]
    assert [v for _, v in by["st"]] == [40, 33, 25, 19], (by, st)
    assert [v for _, v in by["out"]] == [26, 41, 20, 34], (by, st)
    assert st["faults"] == 0 and st["timeouts"] == 0 and st["bad_outputs"] == 0
    print("kernel RAM", {k: v for k, v in st.items() if k not in ("outputs_by_cell", "load_steps")})


@pytest.mark.slow
def test_kernel_ram_written_by_two_store_cells_in_one_pass():
    """Two STORE cells write disjoint words of one 8-word RAM on every token (word t and word
    t + 4), each through its own write port; the ports' marks keep either port's COPY off the
    other's write (lib/kernel.py, `_share_write_ports`). A LOAD pass then reads all eight words
    back in a shuffled order (the host paces the passes: the reads start once every write is out)."""
    spec = [{"name": "h", "op": "LOAD", "a": "input", "mem": "htab"},
            {"name": "d", "op": "XOR", "a": "h", "b": ("const", "k255")},
            {"name": "hi", "op": "OR", "a": "input", "b": ("const", "k4")},
            {"name": "st_lo", "op": "STORE", "a": "input", "b": "h", "mem": "hbuf"},
            {"name": "st_hi", "op": "STORE", "a": "hi", "b": "d", "mem": "hbuf"},
            {"name": "col", "op": "AND", "a": "input:pix", "b": ("const", "k7")},
            {"name": "hb", "op": "LOAD", "a": "col", "mem": "hbuf"},
            {"name": "out", "op": "ADD", "a": "hb", "b": ("const", "k1")}]
    htab = {0: 40, 1: 33, 2: 25, 3: 19}
    pl = build_pipeline(PARAMS, 8, spec, consts={"k255": 255, "k4": 4, "k7": 7, "k1": 1},
                        mems={"htab": (4, htab), "hbuf": (8, {}, "ram")}, outputs=["st_lo", "st_hi", "out"], streams=["input", "pix"])
    assert pl.net.n < 30000, pl.net.n
    reads = [5, 2, 7, 0, 3, 6, 1, 4]
    sched = [0, 1, 2, 3] + [("pix", (1 << 8) | c, 8) for c in reads]
    outs, sim, st = run_pipeline(pl, PARAMS, sched, max_ms=80000, expect_outputs=16)
    by = st["outputs_by_cell"]
    hbuf = {t: htab[t] for t in range(4)} | {t + 4: htab[t] ^ 255 for t in range(4)}
    assert [v for _, v in by["st_lo"]] == [40, 33, 25, 19], (by, st)
    assert [v for _, v in by["st_hi"]] == [215, 222, 230, 236], (by, st)
    assert [v for _, v in by["out"]] == [(hbuf[c] + 1) & 255 for c in reads], (by, st)
    assert st["faults"] == 0 and st["timeouts"] == 0 and st["bad_outputs"] == 0
    print("kernel RAM, two store cells", {k: v for k, v in st.items() if k not in ("outputs_by_cell", "load_steps")})


def test_compiler_accepts_two_stores_into_one_array():
    """Two STORE cells on one array compile (their write ports carry marks in the substrate);
    the reference writes both words per token; a store followed by a load of the same array in
    one body stays rejected (no ordering edge)."""
    from drosophilos.compiler.kernel import kernel_outputs
    src = "static u8 a[8]; static u8 i, x; int main(void) { i = 4; while (i != 0) { x = in_read(); a[x] = x + 1; a[x + 4] = x + 2; out_pixel(x); i = i - 1; } return 0; }"
    prog = compile_c(src)
    ks = compile_kernel(prog, loop_body(prog), "i")
    stores = [c for c in ks.cells if c["op"] == "STORE"]
    assert len(stores) == 2 and {c["mem"] for c in stores} == {"a"} and ks.mems["a"][2] == "ram"
    assert kernel_outputs(ks, [0, 1, 2, 3]) == [[1, 2, 0], [2, 3, 1], [3, 4, 2], [4, 5, 3]]
    prog = compile_c("static u8 a[4]; static u8 i, x; int main(void) { i = 3; while (i != 0) { a[i] = i; x = a[i]; out_pixel(x); i = i - 1; } return 0; }")
    with pytest.raises(NotAKernel):
        compile_kernel(prog, loop_body(prog), "i")


TWO_PASS = """static u8 htab[4] = {40, 33, 25, 19};
static u8 hbuf[4];
static u8 col, h, t, c, r, f, p, heading;
int main(void) {
    heading = 1;
    f = 2;
    while (f != 0) {
        col = 0;
        while (col != 4) { h = htab[(col + heading) & 3]; hbuf[col] = h; col = col + 1; }
        p = 8;
        while (p != 0) { t = in_read(); c = t & 3; r = t >> 2; h = hbuf[c]; out_pixel(h + r); p = p - 1; }
        heading = heading + 1;
        f = f - 1;
    }
    return 0;
}"""


def two_pass_schedule(ks, frames, cols, pix):
    """Per frame: the column pass (stores), the pixel pass once the stores are out, the tick once the pixels are out."""
    n_tick = sum(1 for o in ks.outputs if next(c for c in ks.cells if c["name"] == o)["stream"] == "input:f")
    sched, owed = [], 0
    for f in range(frames):
        sched += [("input", c, owed) for c in cols]
        owed += len(cols)
        sched += [("p", t, owed) for t in pix]
        owed += len(pix)
        sched.append(("f", f, owed))
        owed += n_tick
    return sched, owed


def test_two_pass_renderer_compiles_to_three_kernels():
    from drosophilos.compiler.kernel import compile_program, kernel_outputs
    prog = compile_c(TWO_PASS)
    ks = compile_program(prog, params={"heading": 1})
    assert ks.streams == ["input", "f", "p"] and ks.rams == {"hbuf"}
    assert [c["op"] for c in ks.cells if c["stream"] == "input"][-1] == "STORE"
    pix = [(r << 2) | c for r in range(2) for c in range(4)]
    sched, _ = two_pass_schedule(ks, 2, range(4), pix)
    pixels = [o[0] for (st, _, _), o in zip(sched, kernel_outputs(ks, [(s, v) for s, v, _ in sched])) if st == "p"]
    assert pixels == interpret(prog, pix + pix)["outs"]


@pytest.mark.slow
def test_neural_two_pass_renderer_with_a_ram_buffer():
    """Three kernels: the tick, a column pass writing heights into a RAM buffer, a pixel pass
    reading them; the host paces the three phases of each frame."""
    from drosophilos.compiler.kernel import compile_program, kernel_outputs
    prog = compile_c(TWO_PASS)
    ks = compile_program(prog, params={"heading": 1})
    pl = build_pipeline(PARAMS, prog.width, ks.cells, consts=ks.consts, mems=ks.mems, outputs=ks.outputs, streams=ks.streams)
    pix = [(r << 2) | c for r in range(2) for c in range(4)]
    sched, owed = two_pass_schedule(ks, 2, range(4), pix)
    outs, sim, st = run_pipeline(pl, PARAMS, sched, max_ms=150000, expect_outputs=owed)
    pixel_cell = [o for o in ks.outputs if next(c for c in ks.cells if c["name"] == o)["stream"] == "input:p"][0]
    got = [v for _, v in st["outputs_by_cell"][pixel_cell]]
    assert got == interpret(prog, pix + pix)["outs"], (st["outputs_by_cell"], st)
    assert st["faults"] == 0 and st["timeouts"] == 0 and st["bad_outputs"] == 0
    print("two-pass renderer", {k: v for k, v in st.items() if k not in ("outputs_by_cell", "load_steps")})


def test_compiler_rejects_what_the_kernels_cannot_carry():
    """Review round three: programs that would compile to something other than the interpreter's
    meaning are refused with a reason, and two that were wrongly refused compile."""
    from drosophilos.compiler.kernel import compile_program
    rejected = {
        "counter read beside in_read": "static u8 i, x; int main(void) { i = 3; while (i != 0) { x = in_read(); out_pixel(x + i); i = i - 1; } return 0; }",
        "store then load of one array": "static u8 a[4]; static u8 i, x; int main(void) { i = 3; while (i != 0) { a[i] = i; x = a[i]; out_pixel(x); i = i - 1; } return 0; }",
    }
    for name, src in rejected.items():
        prog = compile_c(src)
        with pytest.raises(NotAKernel):
            compile_kernel(prog, loop_body(prog), "i")
    with pytest.raises(NotAKernel):  # an inner loop writing a variable the outer loop carries
        compile_program(compile_c("static u8 f, i, h; int main(void) { h = 1; f = 2; while (f != 0) { i = 2; while (i != 0) { h = h + 1; out_pixel(h); i = i - 1; } h = h + 1; f = f - 1; } return 0; }"), params={"h": 1})
    prog = compile_c("static u8 i, x, k; int main(void) { i = 2; while (i != 0) { x = in_read(); if (x != 0) { x = x + k; } out_pixel(x); i = i - 1; } return 0; }")
    ks = compile_kernel(prog, loop_body(prog), "i", params={"k": 5})  # a parameter first read inside one arm
    assert [c["op"] for c in ks.cells] == ["ADD", "MOV", "SEL"]  # the token as a condition gets a MOV cell for its Z flag
    prog = compile_c("static u8 c, x, f, p; int main(void) { f = 1; while (f != 0) { p = 2; while (p != 0) { x = in_read(); if (x != 0) { c = 1; } else { c = 2; } out_pixel(c); p = p - 1; } f = f - 1; } return 0; }")
    ks = compile_program(prog)  # an if on a second stream's token
    assert "SEL" in [c["op"] for c in ks.cells]


def test_neural_pacing_compiles_a_ring_per_pass():
    """The phase counter is a RING pseudo-cell (K lines, advanced by the pass's last output and
    every STORE), not the three ALU cells of the first step: no cell of the pass gets a reader,
    so commit gating holds nothing back for the count. The reference skips the rings."""
    from drosophilos.compiler.kernel import compile_program, kernel_outputs
    prog = compile_c(TWO_PASS)
    host = compile_program(prog, params={"heading": 1})
    ks = compile_program(prog, params={"heading": 1}, pacing="neural", counts={"input": 4, "p": 8})
    rings = [c for c in ks.cells if c["op"] == "RING"]
    assert [(c["name"], c["k"], c["stream"], c["trigger"]) for c in rings] == [("ph0_ring", 4, "input", ["c3_store"]), ("ph1_ring", 8, "input:p", ["c1_3_add"])]
    assert ks.phases == [("input", "ph0_ring", "wrap"), ("p", "ph1_ring", "wrap"), ("f", "f0_add", "each")]
    assert [c for c in ks.cells if c["op"] != "RING"] == host.cells and ks.consts == host.consts  # the pass itself is untouched
    assert not [c for c in ks.cells if c["name"].startswith("ph") and c["op"] in ("XOR", "ADD", "SEL")]
    for c in ks.cells:  # a ring is nobody's source: no cell reads it, so no commit waits on it
        assert all(c.get(k) not in ("ph0_ring", "ph1_ring") for k in ("a", "b", "c"))
    pix = [(r << 2) | c for r in range(2) for c in range(4)]
    sched, _ = two_pass_schedule(ks, 2, range(4), pix)
    assert kernel_outputs(ks, [(s, v) for s, v, _ in sched]) == kernel_outputs(host, [(s, v) for s, v, _ in sched])
    with pytest.raises(NotAKernel):
        compile_program(prog, params={"heading": 1}, pacing="neural", counts={"input": 4})  # every inner stream needs its count


def test_pacing_ring_wraps_every_k_tokens():
    """A two-cell pass (K = 3) and a one-cell tick, built by hand: the ring advances at every
    done of its trigger and wraps at the third, the wrap closes the pass's phase and opens the
    tick's, the tick's done reopens the pass's; the register's commits run at the pipeline's own
    pace (the ring holds no commit). One trigger for the pass; the tick's is 'each'."""
    from drosophilos.sim.ref64 import RefSim
    K, n, frames = 3, 4, 3
    spec = [{"name": "c1", "op": "ADD", "a": "input", "b": ("const", "k1"), "stream": "input"},
            {"name": "c2", "op": "XOR", "a": "c1", "b": ("const", "k3"), "stream": "input"},
            {"name": "f1", "op": "MOV", "a": "input:f", "b": "input:f", "stream": "input:f"},
            {"name": "ph0_ring", "op": "RING", "k": K, "stream": "input", "trigger": ["c2"]}]
    pl = build_pipeline(PARAMS, n, spec, consts={"k1": 1, "k3": 3}, outputs=["c2", "f1"], streams=["input", "f"],
                        phases=[("input", "ph0_ring", "wrap"), ("f", "f1", "each")])
    assert pl.net.n < 6000, pl.net.n
    lines = pl.rings["input"][0]
    watch = {"wrap": pl.phase_ends["input"], "done": pl.cells[1].reg.done_relay, "commit": pl.net.roles.index("input.commit_pulse"),
             "ok_in": pl.phase_ok["input"][1].u, "ok_f": pl.phase_ok["f"][1].u, **{f"l{k}": lines[k][1].u for k in range(K)}}

    class Watched(RefSim):  # the runner trims old spikes; keep these neurons' rises
        def __init__(self):
            super().__init__(pl.net.topology(), PARAMS)
            self.rec = {u: [] for u in watch.values()}

        def step(self):
            b = len(self._spk_step)
            super().step()
            for k in range(b, len(self._spk_step)):
                for u in self._spk_neuron[k].tolist():
                    if u in self.rec:
                        self.rec[u].append(int(self._spk_step[k][0]))

    # the host deals each frame behind the previous frame's tick output (a two-phase schedule
    # has no held token to stop it dealing the next frame's tokens into the open, draining pass)
    sched = []
    for f in range(frames):
        sched += [("input", K * f + i, f * (K + 1) if i == 0 else 0) for i in range(K)] + [("f", f, 0)]
    outs, sim, st = run_pipeline(pl, PARAMS, sched, max_ms=40000, expect_outputs=frames * (K + 1), sim=Watched())
    assert st["faults"] == 0 and st["timeouts"] == 0 and st["bad_outputs"] == 0, st
    assert [v for _, v in st["outputs_by_cell"]["c2"]] == [((K * f + i + 1) ^ 3) & 15 for f in range(frames) for i in range(K)]
    order = [c for _, c in sorted((s, c) for c, lst in st["outputs_by_cell"].items() for s, _ in lst)]
    assert order == (["c2"] * K + ["f1"]) * frames, order  # the tick waits for the wrap, the next frame for the tick

    def rises(u, gap_ms=300):
        out, last = [], -10**9
        for s in sim.rec[u]:
            if s - last > gap_ms / PARAMS.dt:
                out.append(s * PARAMS.dt)
            last = s
        return out

    done, wrap = rises(watch["done"]), rises(watch["wrap"])
    assert len(done) == K * frames and len(wrap) == frames, (done, wrap)
    for f, w in enumerate(wrap):  # one wrap per K dones, within 50 ms of the K-th (the old counter: ~5 s)
        assert 0 < w - done[K * f + K - 1] < 50, (f, w, done)
    for k in range(K):  # line k lights at the k-th done of every frame (line 0: by the image, then at each wrap)
        lit = rises(watch[f"l{k}"])
        if k == 0:
            assert lit[0] < 10 and lit[1:] == wrap, (lit, wrap)
        else:
            assert len(lit) == frames and all(0 <= lit[f] - done[K * f + k - 1] < 50 for f in range(frames)), (k, lit, done)
    ok_in, ok_f = rises(watch["ok_in"]), rises(watch["ok_f"])
    assert len(ok_f) == frames and all(0 <= o - w < 20 for o, w in zip(ok_f, wrap)), (ok_f, wrap)  # the wrap opens the tick's phase
    assert ok_in[0] < 10 and len(ok_in) == frames and all(o > w for o, w in zip(ok_in[1:], wrap)), (ok_in, wrap)  # and the tick reopens the pass's (the run ends at the last tick's output, before it does)
    starts = rises(watch["commit"])  # the register's commit pulses: one per token
    assert len(starts) == K * frames
    within = [b - a for a, b in zip(starts, starts[1:]) if b - a < 3000]  # the gaps inside a frame (between frames: the host's barrier)
    assert len(within) == frames * (K - 1) and max(within) < 1500, within  # the pass runs at its own pace, no commit waits for the count


@pytest.mark.slow
def test_neural_pacing_replaces_the_host_barriers():
    """Stage F2, first step: the phase order (columns, pixels, tick) is enforced by phase gates in
    the substrate; the host deals the tokens in program order with no barrier at all. The phase
    counters are one-hot rings (24,721 neurons; the three-cell ALU counters made it 36,880)."""
    from drosophilos.compiler.kernel import compile_program, kernel_outputs
    prog = compile_c(TWO_PASS)
    ks = compile_program(prog, params={"heading": 1}, pacing="neural", counts={"input": 4, "p": 8})
    assert ks.phases == [("input", "ph0_ring", "wrap"), ("p", "ph1_ring", "wrap"), ("f", "f0_add", "each")]
    assert len(ks.cells) == 11  # nine cells and two rings
    pl = build_pipeline(PARAMS, prog.width, ks.cells, consts=ks.consts, mems=ks.mems, outputs=ks.outputs, streams=ks.streams, phases=ks.phases)
    pix = [(r << 2) | c for r in range(2) for c in range(4)]
    sched = []
    for f in range(2):
        sched += [("input", c, 0) for c in range(4)] + [("p", t, 0) for t in pix] + [("f", f, 0)]  # no barriers
    outs, sim, st = run_pipeline(pl, PARAMS, sched, max_ms=200000, expect_outputs=8 + 16 + 2)
    pixel_cell = [o for o in ks.outputs if next(c for c in ks.cells if c["name"] == o)["stream"] == "input:p"][0]
    got = [v for _, v in st["outputs_by_cell"][pixel_cell]]
    assert got == interpret(prog, pix + pix)["outs"], (st["outputs_by_cell"], st)
    assert st["faults"] == 0 and st["timeouts"] == 0 and st["bad_outputs"] == 0
    print("neural pacing", {k: v for k, v in st.items() if k not in ("outputs_by_cell", "load_steps")})
