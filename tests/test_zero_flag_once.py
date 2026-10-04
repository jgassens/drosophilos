"""RefSim regressions for the optional, once-per-word Z0 ignition circuit."""

from dataclasses import replace

import numpy as np
import pytest

from drosophilos.bench import kernel_campaign
from drosophilos.lib.alu import add_zero_flag
from drosophilos.lib.kernel import build_pipeline, run_pipeline_batched
from drosophilos.lib.netlist import Drive, Netlist
from drosophilos.protocol.handshake import add_register
from drosophilos.sim.model import Params
from drosophilos.sim.ref64 import RefSim


PARAMS = Params()
WIDTH = 8
Z_BIT = WIDTH + 1
STOP = 4500
# All simultaneous, the four near-simultaneous copy-18 drivers, late ripple bits,
# one active input (the OR's slowest rise), and zero (the unchanged Z1 path).
CASES = [
    (0xFF, [100] * WIDTH),
    (0xD1, [101, 100, 100, 100, 113, 100, 100, 104]),
    (0xFE, [100 + 300 * i for i in range(WIDTH)]),
    (0x01, [100] * WIDTH),
    (0x00, [100] * WIDTH),
]
RELOAD_WORDS = (0xFF, 0x01, 0x00)


def _circuit(zero_once, rate_robust):
    drive = replace(Drive.from_params(PARAMS), rate_robust=rate_robust)
    net = Netlist(PARAMS)
    q = add_register(net, drive, "Q", WIDTH + 3, with_completion=True)
    add_zero_flag(net, drive, q, WIDTH, zero_once=zero_once)
    drivers = [i for i, role in enumerate(net.roles)
               if role.startswith("alu.z0.") and role.endswith(".edge")]
    return net, drive, q, drivers


def _load(sim, drive, q, node, word, times):
    sim.add_events(node, times + [times[0]] * 2,
                   [q.rails[i][(word >> i) & 1].u for i in range(WIDTH)]
                   + [q.rails[WIDTH][0].u, q.rails[WIDTH + 2][0].u],
                   [drive.ignite] * (WIDTH + 2))


def _ignitions(trace, drivers, node, start=0, stop=STOP):
    return sorted(int(s) for i in drivers for s in trace.neuron_steps(i, node)
                  if start <= s < stop)


def _period(trace, neuron, node):
    steps = trace.neuron_steps(neuron, node)
    steady = steps[(steps >= 3000) & (steps < STOP)]
    assert len(steady) > 20
    return float(np.median(np.diff(steady)))


def _clear_and_reload(sim, drive, q, words):
    for node in range(len(words)):
        sim.add_events(node, [STOP], [q.reset_trigger], [drive.ignite])
    ready = {}
    while len(ready) < len(words) and sim.step_index < STOP + 1500:
        before = len(sim._spk_step)
        sim.step()
        for k in range(before, len(sim._spk_step)):
            for node, neuron in zip(sim._spk_node[k], sim._spk_neuron[k]):
                if neuron == q.ready and int(node) not in ready:
                    ready[int(node)] = int(sim._spk_step[k][0])
                    _load(sim, drive, q, int(node), words[int(node)], [sim.step_index] * WIDTH)
    assert len(ready) == len(words), ready
    sim.run(4500)
    return dict(reloaded=sim.trace, ready=ready, reload_words=words)


@pytest.fixture(scope="module", params=[False, True], ids=["base", "rate_robust"])
def captures(request):
    result = {}
    for once in (False, True):
        net, drive, q, drivers = _circuit(once, request.param)
        words = [word for word in RELOAD_WORDS for _ in CASES]
        sim = RefSim(net.topology(), PARAMS, n_nodes=len(words))
        for node, (word, times) in enumerate(CASES * len(RELOAD_WORDS)):
            _load(sim, drive, q, node, word, times)
        sim.run(STOP)
        result[once] = dict(net=net, drive=drive, q=q, drivers=drivers, trace=sim.trace)
        if not once:
            continue
        # Exercise the real four-tap stage controller and reload at READY, not
        # after an arbitrary long recovery interval or a simulator state reset.
        result[once].update(_clear_and_reload(sim, drive, q, words))
    return result


def test_many_bits_and_late_ripple_ignite_z0_exactly_once(captures):
    fixed, legacy = captures[True], captures[False]
    assert len(fixed["drivers"]) == 1
    assert len(legacy["drivers"]) == WIDTH
    for node, (word, _) in enumerate(CASES[:-1]):
        assert len(_ignitions(fixed["trace"], fixed["drivers"], node)) == 1
        assert len(_ignitions(legacy["trace"], legacy["drivers"], node)) == word.bit_count()
        # Check both members of the circulating latch, not only its output tap.
        for member in fixed["q"].rails[Z_BIT][0].members:
            period = _period(fixed["trace"], member, node)
            assert abs(period - fixed["drive"].loop_period_steps) <= 1
        old_first = legacy["trace"].neuron_steps(legacy["q"].rails[Z_BIT][0].u, node)[0]
        new_first = fixed["trace"].neuron_steps(fixed["q"].rails[Z_BIT][0].u, node)[0]
        print({"rate_robust": fixed["drive"].rate_robust, "word": word,
               "Z0_period_steps": period,
               "legacy_period_steps": _period(legacy["trace"], legacy["q"].rails[Z_BIT][0].u, node),
               "Z0_latency_delta_ms": round((new_first - old_first) * PARAMS.dt, 1)})
        assert not len(fixed["trace"].neuron_steps(fixed["q"].rails[Z_BIT][1].u, node))


def test_zero_still_uses_only_z1(captures):
    fixed, legacy = captures[True], captures[False]
    node = len(CASES) - 1
    assert _ignitions(fixed["trace"], fixed["drivers"], node) == []
    for member in fixed["q"].rails[Z_BIT][0].members:
        assert not len(fixed["trace"].neuron_steps(member, node))
    for member in fixed["q"].rails[Z_BIT][1].members:
        spikes = fixed["trace"].neuron_steps(member, node)
        assert len(spikes) > 20
        np.testing.assert_array_equal(spikes, legacy["trace"].neuron_steps(member, node))


def test_stage_clear_leaves_both_z_members_dark_and_rearms_at_ready(captures):
    fixed = captures[True]
    q, trace = fixed["q"], fixed["reloaded"]
    for node, ready in fixed["ready"].items():
        for rail in q.rails[Z_BIT]:
            for member in rail.members:
                steps = trace.neuron_steps(member, node)
                assert not np.any((steps >= ready - 100) & (steps <= ready)), steps
        word = fixed["reload_words"][node]
        assert len(_ignitions(trace, fixed["drivers"], node, ready, 12000)) == int(word != 0)
        expected_z = int(word == 0)
        for member in q.rails[Z_BIT][expected_z].members:
            assert len(trace.neuron_steps(member, node)[trace.neuron_steps(member, node) > ready]) > 20
        for member in q.rails[Z_BIT][1 - expected_z].members:
            assert not np.any(trace.neuron_steps(member, node) > ready)


def test_z_valid_and_completion_rise_for_both_words_without_faults(captures):
    fixed = captures[True]
    q, trace = fixed["q"], fixed["reloaded"]
    for node, ready in fixed["ready"].items():
        for latch in (q.valid[Z_BIT], q.completion):
            for member in latch.members:
                steps = trace.neuron_steps(member, node)
                assert np.any(steps < STOP)
                assert np.any(steps > ready)
        assert all(not len(trace.neuron_steps(fault, node)) for fault in q.fault)


@pytest.mark.parametrize("rate_robust", [False, True])
def test_mix_b_style_reload_at_ready(rate_robust):
    """Small seeded smoke: independent static draws and strays on every neuron.

    This uses mix B's laws, not copy 18's draws or the CUDA stray stream, and
    checks 24 two-word transfers per build rather than campaign reliability.
    """
    net, drive, q, drivers = _circuit(True, rate_robust)
    topo = net.topology()
    words = list(RELOAD_WORDS) * 8
    rng = np.random.default_rng(108)
    quanta = np.rint(topo.quanta[None, :] * np.exp(rng.normal(0, 0.04, (len(words), topo.nnz)))).astype(np.int32)
    vth = PARAMS.V_th + rng.normal(0, 0.2, (len(words), topo.n))
    bias = topo.sim_bias() + rng.normal(0, 0.2, (len(words), topo.n))
    sim = RefSim(topo, PARAMS, n_nodes=len(words), quanta=quanta, V_th=vth, bias=bias)
    # Geometric gaps implement the same per-step Bernoulli 5-Hz law; schedule
    # background input before the probe so it continues across clear and reload.
    for node in range(len(words)):
        steps, neurons = [], []
        for neuron in range(topo.n):
            times = np.cumsum(rng.geometric(5 * PARAMS.dt / 1000, 32)) - 1
            assert times[-1] >= 12000  # this seed's samples cover the whole run
            times = times[times < 12000]
            steps.extend(times); neurons.extend([neuron] * len(times))
        sim.add_events(node, steps, neurons, [150] * len(steps))
        _load(sim, drive, q, node, 0xD1, CASES[1][1])
    sim.run(STOP)
    initial = sim.trace
    result = _clear_and_reload(sim, drive, q, words)
    trace = result["reloaded"]
    for node, ready in result["ready"].items():
        assert len(_ignitions(initial, drivers, node)) == 1
        assert len(_ignitions(trace, drivers, node, ready, sim.step_index)) == int(words[node] != 0)
        for rail, latch in enumerate(q.rails[Z_BIT]):
            for member in latch.members:
                count = np.count_nonzero(trace.neuron_steps(member, node) > ready + 2000)
                assert (count > 20) == (rail == int(words[node] == 0)), (node, words[node], rail, count)
        assert np.any(trace.neuron_steps(q.valid[Z_BIT].u, node) > ready)
        assert np.any(trace.neuron_steps(q.completion.u, node) > ready)
        assert all(not len(trace.neuron_steps(fault, node)) for fault in q.fault)


@pytest.mark.parametrize("rate_robust", [False, True])
def test_shared_or_and_relay_are_in_the_stage_reset_domain(rate_robust):
    net, drive, q, _ = _circuit(True, rate_robust)
    gate = net.roles.index("alu.z0.or")
    hold = net.roles.index("alu.z0.ign.hold_inh")
    assert any(net.src[k] == hold and net.quanta[k] < 0 for k in net.incoming[gate])
    for i, role in enumerate(net.roles):
        if role == "alu.z0.or" or role.startswith("alu.z0.ign."):
            assert any(net.src[k] == q.reset_inh and net.quanta[k] < 0 for k in net.incoming[i]), role
    if rate_robust:
        gate = net.roles.index("alu.z0.or")
        assert gate in net.rate_gates
        inputs = [net.src[k] for k in net.incoming[gate] if net.quanta[k] > 0]
        assert inputs == [net.rate_readouts[q.rails[i][1].u] for i in range(WIDTH)]
        for source, tap in net.rate_readouts.items():
            expected = sorted((net.src[k], 16 * net.quanta[k], net.delay[k])
                              for k in net.incoming[source] if net.quanta[k] < 0)
            actual = sorted((net.src[k], net.quanta[k], net.delay[k])
                            for k in net.incoming[tap] if net.quanta[k] < 0)
            assert actual == expected


@pytest.mark.parametrize("rate_robust", [False, True])
def test_tick_kernel_build_cost(rate_robust):
    _, legacy, _, _ = kernel_campaign.block("tick", PARAMS, rate_robust=rate_robust)
    _, fixed, _, _ = kernel_campaign.block("tick", PARAMS, rate_robust=rate_robust, zero_once=True)
    generators = fixed.net.roles.count("alu.z0.or")
    delta = (fixed.net.n - legacy.net.n, fixed.net.nnz - legacy.net.nnz)
    # Four neurons / nineteen edges replace sixteen / thirty-two. Relative to
    # the analysis's fifteen-edge estimate, three edges reset the relay and its
    # inhibitors and one quiets the OR during hold to permit re-arming at READY.
    assert generators > 0
    assert delta == (-12 * generators, -13 * generators)
    print({"rate_robust": rate_robust, "Z_generators": generators,
           "legacy_neurons_synapses": (legacy.net.n, legacy.net.nnz),
           "zero_once_neurons_synapses": (fixed.net.n, fixed.net.nnz), "delta": delta})


@pytest.mark.parametrize("datapath", ["generic", "specialized"])
@pytest.mark.parametrize("rate_robust", [False, True])
def test_xor_kernel_computes_correct_values_and_flags(datapath, rate_robust):
    pl = build_pipeline(PARAMS, WIDTH,
                        [{"name": "out", "op": "XOR", "a": "input", "b": ("const", "k")}],
                        consts={"k": 85}, zero_once=True,
                        datapath=datapath, rate_robust=rate_robust)
    sim = RefSim(pl.net.topology(), PARAMS)
    outputs, sim, stats = run_pipeline_batched(pl, PARAMS, [[170, 85, 255]],
                                               max_ms=10000, sim=sim, progress=0, full_trace=True)
    assert [v for _, v in outputs[0]["out"]] == [255, 0, 170], stats
    assert stats["faults"] == stats["timeouts"] == stats["bad_outputs"] == 0
    stage = pl.cells[0].stage
    trace = sim.trace
    drivers = [i for i, role in enumerate(pl.net.roles) if role == "alu.z0.ign.edge"]
    assert len(_ignitions(trace, drivers, 0, 0, sim.step_index)) == 2
    # The committed value alone omits Z; inspect the stage's flag at each commit.
    for (commit, value), expected_z in zip(outputs[0]["out"], [0, 1, 0]):
        assert (value == 0) == bool(expected_z)
        for r in (0, 1):
            steps = trace.neuron_steps(stage.rails[Z_BIT][r].u)
            live = np.any((steps > commit - 200) & (steps < commit))
            assert live == (r == expected_z), (commit, value, r, steps)
