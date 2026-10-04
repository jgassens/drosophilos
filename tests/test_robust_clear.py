"""Real-primitive candidate evaluation for the opt-in request clear.

The copy-28 draws and ignition protocol come from the reviewed entrainment
regression. Candidate changes are applied to actual controller edges, never to
prescribed inhibitor arrivals. Run the slow tests with -s to reproduce the
candidate/reload tables and the 40,000-clear strayed qualification per corner.
"""

import numpy as np
import pytest

from drosophilos.bench.a2_campaigns import MIXES
from drosophilos.lib.campaign import make_perturbed_sim
from drosophilos.lib.kernel import build_pipeline, run_pipeline, run_pipeline_batched
from drosophilos.lib.control import add_kill_train
from drosophilos.protocol.celement import add_delay_chain
from drosophilos.sim.ref64 import RefSim
from test_request_clear_entrainment import PARAMS, DRIVE, _circuit


CORNERS = {
    "copy28": "copy28",
    "mirror28": "copy28",
    "nominal": (1.0, 0.0),
    "slow8": (0.92, 0.4),
    "slow12": (0.88, 0.6),
    "fast20": (1.2, -0.8),
    "fast30": (1.3, -1.2),
}
CANDIDATES = ("baseline", "phase", "verify", "stagger", "stronger", "robust")
# The reviewed budget is relative to the first actual inhibitor arrival, not
# to the external source event (which precedes the source neuron's own spike).
RELOAD_BUDGET = 862


def _candidate_circuit(candidate="robust", corner="copy28", *, copy_controller=True):
    net, latch, source, inh, vth, bias = _circuit(
        4, copy_controller=copy_controller, corner=CORNERS[corner], robust=candidate == "robust")
    if corner == "mirror28":
        vth[list(latch.members)] = vth[list(reversed(latch.members))]
        bias[list(latch.members)] = bias[list(reversed(latch.members))]
        loop = [e for e, (s, d) in enumerate(zip(net.src, net.dst))
                if s in latch.members and d in latch.members]
        net.quanta[loop[0]], net.quanta[loop[1]] = net.quanta[loop[1]], net.quanta[loop[0]]
    for e, (s, d) in enumerate(zip(net.src, net.dst)):
        role = net.roles[d]
        if candidate == "phase" and role in ("kill.h1", "kill.h2", "kill.h3"):
            net.delay[e] = {"kill.h1": 0, "kill.h2": 36, "kill.h3": 18}[role]
        if candidate == "stagger" and s == inh and d == latch.v:
            net.delay[e] = 40  # one refractory period later than u
        if candidate == "stronger" and s == inh and d in latch.members:
            net.quanta[e] = round(net.quanta[e] / 0.75)
    if candidate == "verify":
        late = add_delay_chain(net, DRIVE, "verify.delay", source, 12)
        gate = net.neuron("verify.gate")
        net.synapse(late, gate, round(0.5 * DRIVE.single_need))
        for member in latch.members:
            net.synapse(member, gate, round(0.35 * DRIVE.rate_need))
        add_kill_train(net, DRIVE, "verify.kill", gate, [latch], pulses=1, strength=0.75)
        vth = np.pad(vth, (0, net.n - len(vth)), constant_values=PARAMS.V_th)
        bias = np.pad(bias, (0, net.n - len(bias)))
    return net, latch, source, inh, vth, bias


def _live_nodes(trace, latch, after):
    events = trace.events
    lit = np.isin(events["neuron"], latch.members) & (events["step"] > after[events["node"]])
    return np.unique(events["node"][lit])


def _phase_probe(candidate="robust", corner="copy28", *, copy_controller=True):
    net, latch, source, inh, vth, bias = _candidate_circuit(
        candidate, corner, copy_controller=copy_controller)
    clears = np.arange(5000, 6500)
    sim = RefSim(net.topology(), PARAMS, n_nodes=len(clears), V_th=vth, bias=bias)
    kick = latch.u if corner == "mirror28" else latch.v
    for node, clear in enumerate(clears):
        sim.add_events(node, [10, 5020, clear], [latch.u, kick, source],
                       [4522, 300, DRIVE.ignite])
    sim.run(int(clears[-1]) + 1800)
    trace = sim.trace
    return len(_live_nodes(trace, latch, clears + 1500)), trace.neuron_steps(inh, 0)


def _reload_probe(candidate="robust", corner="slow12", offsets=range(0, 1001, 50), *, copy_controller=False):
    net, latch, source, inh, vth, bias = _candidate_circuit(
        candidate, corner, copy_controller=copy_controller)
    phases = list(range(0, 44, 4))
    sim = RefSim(net.topology(), PARAMS, n_nodes=len(phases), V_th=vth, bias=bias)
    for node, phase in enumerate(phases):
        sim.add_events(node, [10, 2000 + phase], [latch.u, source], [4522, DRIVE.ignite])
    sim.run(3040)
    trace = sim.trace
    train = trace.neuron_steps(inh)
    delays = [net.delay[e] for e, (s, d) in enumerate(zip(net.src, net.dst))
              if s == inh and d in latch.members]
    first, last = int(train[0]) + min(delays), int(train[-1]) + max(delays)
    timing = {"arrivals": (train + 18 - first).tolist(), "last": last - first}
    for node in range(len(phases)):
        for member in latch.members:
            pre = trace.neuron_steps(member, node)
            assert np.any((pre > 1500) & (pre < 2000))
    if len(_live_nodes(trace, latch, last + np.asarray(phases) + 200)):
        # A surviving orbit is not a successful reload. Rejected candidates at
        # fast corners have no meaningful recovery threshold to report.
        return {**timing, "cleared": False, "after_last": None,
                "after_first": None, "margin": None}, np.empty((0, 11), dtype=bool)
    if candidate == "verify":
        assert not np.any(trace.events["neuron"] == net.roles.index("verify.kill.inh"))
    offsets = np.asarray(list(offsets))
    cases = [(offset, phase) for offset in offsets for phase in range(0, 44, 4)]
    sim = RefSim(net.topology(), PARAMS, n_nodes=len(cases), V_th=vth, bias=bias)
    reloads = np.array([last + offset + phase for offset, phase in cases])
    for node, ((offset, phase), reload) in enumerate(zip(cases, reloads)):
        sim.add_events(node, [10, 2000 + phase, reload], [latch.u, source, latch.u],
                       [4522, DRIVE.ignite, 4655])
    sim.run(int(reloads.max()) + 1800)
    success = np.zeros(len(cases), dtype=bool)
    success[_live_nodes(sim.trace, latch, reloads + 1500)] = True
    success = success.reshape(-1, 11)
    reliable = offsets[np.logical_and.accumulate(np.all(success, axis=1)[::-1])[::-1]]
    minimum = int(reliable[0]) if len(reliable) else None
    return {**timing, "cleared": True,
            "after_last": minimum, "after_first": last - first + minimum if minimum is not None else None,
            "margin": RELOAD_BUDGET - (last - first + minimum) if minimum is not None else None}, success


def _strayed_probe(candidate="robust", corner="copy28", copies=40000, *, copy_controller=True):
    rng = np.random.default_rng(108)
    survived = 0
    for start in range(0, copies, 1000):
        batch = min(1000, copies - start)
        net, latch, source, _, vth, bias = _candidate_circuit(
            candidate, corner, copy_controller=copy_controller)
        sim = RefSim(net.topology(), PARAMS, n_nodes=batch, V_th=vth, bias=bias)
        clears = rng.integers(3000, 5000, batch)
        for node, clear in enumerate(clears):
            steps, neurons, quanta = [10, int(clear)], [latch.u, source], [4522, DRIVE.ignite]
            for member in latch.members:
                times = np.flatnonzero(rng.random(6800) < 5 * PARAMS.dt / 1000)
                steps.extend(times)
                neurons.extend([member] * len(times))
                quanta.extend([150] * len(times))
            sim.add_events(node, steps, neurons, quanta)
        sim.run(6800)
        survived += len(_live_nodes(sim.trace, latch, clears + 1500))
    return survived


@pytest.mark.parametrize("corner", CORNERS)
@pytest.mark.parametrize("copy_controller", [True, False], ids=["copy28_controller", "nominal_controller"])
def test_robust_phase_survival(corner, copy_controller):
    survived, train = _phase_probe(corner=corner, copy_controller=copy_controller)
    assert survived == 0, (corner, survived)
    assert (train - train[0]).tolist() == ([0, 47, 88, 151] if copy_controller else [0, 43, 81, 138])


@pytest.mark.parametrize("corner", CORNERS)
def test_robust_reload_margin(corner):
    result, _ = _reload_probe(corner=corner)
    assert result["arrivals"] == [0, 43, 81, 138]  # four actual spikes, including the tail
    assert result["margin"] > 0, (corner, result)
    if corner == "slow12":
        assert result["after_last"] == 700
        assert result["after_first"] == 838
        assert result["margin"] == 24


def test_slow_corner_reloads_at_the_earliest_start():
    result, success = _reload_probe(corner="slow12", offsets=(RELOAD_BUDGET - 138,))
    assert result["after_first"] == RELOAD_BUDGET
    assert np.all(success)


def test_copy28_controller_also_has_positive_slow_corner_reload_margin():
    result, success = _reload_probe(corner="slow12", offsets=(700, 711), copy_controller=True)
    assert result["arrivals"] == [0, 47, 88, 151]
    assert result["margin"] == 11
    assert np.all(success)


@pytest.mark.parametrize("rate_robust", [False, True])
def test_small_kernel_computes_and_reloads_with_robust_requests(rate_robust):
    pl = build_pipeline(PARAMS, 2,
                        [{"name": "out", "op": "ADD", "a": "input", "b": ("const", "one")}],
                        consts={"one": 1}, robust_request_clear=True, rate_robust=rate_robust)
    outputs, sim, stats = run_pipeline(pl, PARAMS, [0, 1, 3], max_ms=12000)
    assert isinstance(sim, RefSim)
    assert [value for _, value in outputs] == [1, 2, 0]
    assert stats["faults"] == stats["timeouts"] == stats["bad_outputs"] == 0
    assert not stats["refusals"]


@pytest.mark.parametrize("options, requirement", [
    ({"start_relight_hops": 0}, "start_relight_hops=5"),
    ({"start_relight_hops": 4}, "start_relight_hops=5"),
    ({"start_relight_hops": 6}, "start_relight_hops=5"),
    ({"relight_repair_delay": False}, "relight_repair_delay=True"),
    ({"true_guards": False}, "true_guards=True"),
])
def test_robust_clear_rejects_unqualified_relight_settings(options, requirement):
    with pytest.raises(ValueError, match=f"robust_request_clear requires.*{requirement}"):
        build_pipeline(PARAMS, 1,
                       [{"name": "out", "op": "MOV", "a": "input", "b": ("const", "zero")}],
                       consts={"zero": 0}, robust_request_clear=True, **options)


@pytest.mark.parametrize("zero_once", [False, True])
@pytest.mark.parametrize("noisy", [False, True], ids=["nominal", "mix_B"])
def test_fanout_join_kernel_computes_and_reloads_with_robust_requests(zero_once, noisy):
    """Four cells exercise inter-cell requests, fanout, join and zero/nonzero reloads.

    The noisy case uses three independent mix-B copies (weight, threshold, bias
    and seeded 5 Hz strays), rather than qualifying a full Stage D campaign.
    """
    spec = [{"name": "c1", "op": "ADD", "a": "input", "b": ("const", "one")},
            {"name": "c2", "op": "AND", "a": "c1", "b": ("const", "mask")},
            {"name": "c3", "op": "XOR", "a": "c1", "b": ("const", "flip")},
            {"name": "c4", "op": "ADD", "a": "c2", "b": "c3"}]
    pl = build_pipeline(PARAMS, 4, spec, consts={"one": 1, "mask": 5, "flip": 3},
                        outputs=[cell["name"] for cell in spec], rate_robust=True,
                        robust_request_clear=True, zero_once=zero_once)
    tokens = [0, 2, 15]  # c1 wraps to zero; c3 also transitions through zero
    expected = {"c1": [1, 3, 0], "c2": [1, 1, 0], "c3": [2, 0, 3], "c4": [3, 1, 3]}
    copies, max_ms = (3 if noisy else 1), 12000
    if noisy:
        sim = make_perturbed_sim(pl.net.topology(), PARAMS, copies, MIXES["B"],
                                 np.random.default_rng(108), int(max_ms / PARAMS.dt),
                                 device="cpu", backend="torch-fast")
    else:
        sim = RefSim(pl.net.topology(), PARAMS)
    outputs, _, stats = run_pipeline_batched(
        pl, PARAMS, [tokens.copy() for _ in range(copies)], sim=sim, max_ms=max_ms,
        expect_outputs=[len(tokens) * len(spec)] * copies, progress=0)
    for node in outputs:
        assert {name: [value for _, value in events] for name, events in node.items()} == expected, stats
    assert stats["faults"] == stats["timeouts"] == stats["bad_outputs"] == stats["refusals"] == 0
    assert not stats["blocked_nodes"] and not stats["host_stalls"] and not stats["truncated"]


@pytest.mark.slow
@pytest.mark.parametrize("corner", CORNERS)
def test_40000_strayed_clears_per_corner(corner):
    survived = _strayed_probe(corner=corner)
    print({"corner": corner, "survivors": survived, "clears": 40000}, flush=True)
    assert survived == 0


@pytest.mark.slow
@pytest.mark.parametrize("corner", ["copy28", "fast30"])
def test_40000_strayed_clears_nominal_controller(corner):
    survived = _strayed_probe(corner=corner, copy_controller=False)
    print({"controller": "nominal", "corner": corner, "survivors": survived, "clears": 40000}, flush=True)
    assert survived == 0


@pytest.mark.slow
def test_candidate_evaluation_table():
    for candidate in CANDIDATES:
        for corner in CORNERS:
            survived, train = _phase_probe(candidate, corner)
            reload, _ = _reload_probe(candidate, corner)
            print({"candidate": candidate, "corner": corner, "survived": survived,
                   "clears": 1500, "copy_spikes": len(train), **reload}, flush=True)
