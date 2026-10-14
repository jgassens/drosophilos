"""Experimental operand one-shot: bounded local probes and separate cluster cases.

No campaign qualification is claimed by these laptop tests. Prescribed trains
are synaptic-boundary inputs; receivers, spent memory and resets are real LIFs.
"""

import os
from time import perf_counter, process_time

import numpy as np
import pytest

from drosophilos.lib.gates import Gates, Rail2, Tap
from drosophilos.lib.netlist import Drive, Netlist
from drosophilos.protocol.latch import add_experimental_autapses, add_reset
from drosophilos.sim.model import Topology
from drosophilos.sim.ref64 import RefSim
from test_ordered_xor import P, SurveySim, arithmetic_survey, prescribe


SCENARIOS = ("second-pulse", "order-15ms", "order-100ms", "mix-B", "copy85",
             "addsub4-mix-B", "addsub8-mix-B", "addsub16-nominal")
FAILURES = ("double", "missing", "wrong", "repeated", "ignition_repeat", "survivors")


def operand_fixture(enabled=True):
    net, drive = Netlist(P), Drive.from_params(P)
    early = Rail2(Tap(net.neuron("early0")), Tap(net.neuron("early1")))
    late = Rail2(Tap(net.neuron("late0")), Tap(net.neuron("late1")))
    cin = Rail2(Tap(net.neuron("cin0")), Tap(net.neuron("cin1")))
    g = Gates(net, drive, ordered_xor_oneshot=True, operand_xor_oneshot=enabled)
    _, _, xs, _ = g.ripple_adder_ordered("add", [late], [early], cin)
    reset = add_reset(net, drive, "reset", g.latches, g.gates)
    add_experimental_autapses(net, drive)
    return net, drive, early, late, xs[0], reset, g


def copy85_receiver(*, nominal=False, first_only=False):
    """Frozen reconstructed receiving draws and captured arrivals, origin 198000.

    The original sorted topology has 123323 neurons / 249940 edges, mix B,
    seed 110, B=100, row 85. Values are preserved from the diagnostic report,
    independent of changed topology IDs or newly allocated random draws.
    """
    sim = RefSim(Topology.empty(1), P, V_th=[-45.17450109365213],
                 bias=[.022141032716653797])
    sim.V[:] = P.E_L + sim.bias
    excitation, veto, source_inhibition = ((4655, -1810, -7966) if nominal else
                                           (4919, -1792, -7663))
    times = [566] if first_only else [566, 607, 645]
    sim.add_events(0, times, [0] * len(times), [excitation] * len(times))
    sim.add_events(0, [480, 548, 614], [0] * 3, [veto] * 3)
    sim.add_events(0, [624, 671], [0] * 2, [source_inhibition] * 2)
    sim.run(700)
    return sim.trace.events["step"].tolist()


def operand_survey(scenario, trials, *, enabled=True):
    net, drive, early, late, out, reset, g = operand_fixture(enabled)
    topo = net.topology()
    rng = np.random.default_rng(8585)
    result = dict(copies=trials, trials=trials * (1 if scenario == "copy85" else 4),
                  double=0, missing=0, wrong=0, repeated=0,
                  ignition_repeat=0, survivors=0, min_latency=10000, max_latency=0)
    for start in range(0, trials, 256):
        size = min(256, trials - start)
        gaussian = scenario == "mix-B"
        w = (rng.normal(0, .04, (size, topo.nnz)) if gaussian else
             rng.choice([-.12, .12], (size, topo.nnz)))
        th = (rng.normal(0, .2, (size, topo.n)) if gaussian else
              rng.choice([-.6, .6], (size, topo.n)))
        bias = (rng.normal(0, .2, (size, topo.n)) if gaussian else
                rng.choice([-.6, .6], (size, topo.n)))
        sim = SurveySim(topo, P, n_nodes=size, quanta=np.rint(topo.quanta * np.exp(w)),
                        V_th=P.V_th + th, bias=np.asarray(net.bias) + bias)
        # Each physical copy visits every truth-table row, changes polarities,
        # holds consumption over long gaps and reloads after real reset.
        for tx in range(4 if scenario != "copy85" else 1):
            origin = sim.step_index
            ev, lv = tx // 2, tx % 2
            first = 1000 if scenario == "order-100ms" else 200 if scenario == "second-pulse" else 150
            spacing = 41 if scenario == "second-pulse" else 30
            lt = np.array([first, first + spacing, first + 2 * spacing, first + 3 * spacing,
                           first + 1000, first + 1042, first + 2300])
            if scenario == "copy85":
                ev = lv = 0
                # Captured delayed acc6 emissions. Three real captured pulses
                # suffice for the escape; added late pulses probe retention.
                jitter = rng.integers(-5, 6, (size, 1))
                jitter[0] = 0
                lt = np.array([548, 589, 627, 1600, 1642, 2700])[None, :] + jitter
                wrong = net.roles.index("add.fa0.x.y1.a.edge")
                sim.V_th[:, wrong] = -45.17450109365213
                sim.bias[:, wrong] = .022141032716653797
                weights = {".edge": 4919, ".edge_inh": -7663, ".veto": -1792}
                for e, (src, dst) in enumerate(zip(topo.src, topo.dst)):
                    if dst == wrong:
                        role = net.roles[src]
                        sim.quanta[:, e] = (weights[".veto"] if role.endswith(".veto") else
                                            weights[".edge_inh"] if role.endswith(".edge_inh") else
                                            weights[".edge"])
                veto = net.roles.index("add.fa0.x.y1.a.veto")
                sim.silenced[:, veto] = True
                prescribe(sim, veto, [462, 530, 596] + list(range(662, 3200, 66)))
                if not enabled:
                    inh = net.roles.index("add.fa0.x.y1.a.edge_inh")
                    sim.silenced[:, inh] = True
                    prescribe(sim, inh, [606, 653] + list(range(700, 3200, 47)))
                prescribe(sim, early.taps[ev], np.arange(404, 3200, 60))
            else:
                # Minimum 15-ms EARLY lead, with independent pulse jitter, and
                # a much earlier settled EARLY. Reversed EARLY/LATE arrival is
                # outside the ordered primitive's contract, not qualified here.
                lt = lt[None, :] + rng.integers(0, 6, (size, len(lt)))
                prescribe(sim, early.taps[ev], origin + np.arange(0, 4000, 60))
            if tx == 0:
                sim.V[:] = P.E_L + sim.bias
            before = sim.counts.copy()
            sim.first.fill(-1)
            prescribe(sim, late.taps[lv], origin + lt)
            sim.run(4200)
            counts = sim.counts - before
            seen = counts[:, out.taps] > 0
            result["double"] += int(seen.all(axis=1).sum())
            result["missing"] += int((~seen.any(axis=1)).sum())
            result["wrong"] += int(seen[:, 1 - (ev ^ lv)].sum())
            edges = [i for i, r in enumerate(net.roles) if r.startswith("add.fa0.x.y") and r.endswith(".edge")]
            result["ignition_repeat"] += int((counts[:, edges].sum(axis=1) > 1).sum())
            if enabled:
                once = [net.roles.index(f"add.fa0.x.once{r}.edge") for r in (0, 1)]
                result["repeated"] += int((counts[:, once].sum(axis=1) != 1).sum())
            lat = sim.first[:, out.taps[ev ^ lv]] - origin - lt[:, 0]
            lat = lat[sim.first[:, out.taps[ev ^ lv]] >= 0]
            if len(lat):
                result["min_latency"] = min(result["min_latency"], int(lat.min()))
                result["max_latency"] = max(result["max_latency"], int(lat.max()))
            for node in range(size):
                sim.add_events(node, [origin + 4200], [reset[0]], [drive.ignite])
            sim.run(2000)
            members = [m for latch in g.latches for m in latch.members]
            result["survivors"] += int((sim.last[:, members] >= origin + 5900).sum())
    return result


def test_copy85_reconstruction_and_suppression():
    assert copy85_receiver() == [621]
    assert copy85_receiver(nominal=True) == []
    assert copy85_receiver(first_only=True) == []
    legacy = operand_survey("copy85", 1, enabled=False)
    assert legacy["double"] == legacy["wrong"] == 1, legacy
    fixed = operand_survey("copy85", 1)
    assert all(fixed[key] == 0 for key in FAILURES), fixed


def test_operand_spent_latches_are_in_stage_reset_domain():
    net, _, _, _, _, reset, g = operand_fixture()
    spent = {m for latch in g.latches if ".x.once" in net.roles[latch.u] for m in latch.members}
    targets = {d for s, d, q in zip(net.src, net.dst, net.quanta) if s == reset[1] and q < 0}
    assert len(spent) == 4
    assert spent <= targets
    assert not any(".bx" in r and ".once" in r for r in net.roles)


def test_nominal_operand_latency():
    latencies = []
    for enabled in (False, True):
        net, _, early, late, out, _, _ = operand_fixture(enabled)
        sim = SurveySim(net.topology(), P, n_nodes=4)
        for node in range(4):
            ev, lv = node // 2, node % 2
            prescribe(sim, early.taps[ev], np.arange(0, 800, 60), [node])
            prescribe(sim, late.taps[lv], [200, 241, 282, 700], [node])
        sim.run(1000)
        assert not (sim.counts[:, out.taps] > 0).all(axis=1).any()
        first = [sim.first[node, out.taps[(node // 2) ^ (node % 2)]] for node in range(4)]
        assert min(first) > 200
        latencies.append(np.asarray(first) - 200)
    print("nominal operand XOR latency steps off/on", [lat.tolist() for lat in latencies])
    assert (latencies[1] - latencies[0] == 42).all()


@pytest.mark.parametrize("datapath", ["generic", "specialized"])
def test_pipeline_rearms_with_master_ignition_and_zero_once(datapath):
    from test_ordered_xor import test_pipeline_combinations_rearm
    test_pipeline_combinations_rearm(True, datapath, True, operand=True, ignition=True)


@pytest.mark.skipif(os.environ.get("DROSO_OPXOR_SMOKE") != "1",
                    reason="operand-XOR laptop frontier requires DROSO_OPXOR_SMOKE=1")
def test_noisy16_carry_escape_frontier():
    """Preserve a rejected wider envelope; this is not a passing qualification.

    The operand XOR consumes once and stays exclusive. An ordinary carry
    propagate relay escapes its veto and feeds both sum one-shots downstream.
    Extending protection to carry generation/propagation is a separate change.
    """
    result = arithmetic_survey(16, 12, True, operand_oneshot=True)
    assert result["double"] == result["wrong"] == 1, result
    assert result["missing"] == result["repeated"] == result["survivors"] == 0, result
    failure, = result["failure_details"]
    assert {key: failure[key] for key in ("copy", "tx", "bit", "a", "b", "sub")} == dict(
        copy=9, tx=1, bit=12, a=0xaaaa, b=0x5555, sub=1)
    probes = failure["probes"]
    assert probes["add.fa11.x.once1.edge"][0] == 1
    assert probes["add.fa11.x.y1.u"][0] == 0
    assert probes["add.fa11.x.y0.u"][0] > 0
    assert probes["add.fa11.p0.edge"] == (1, 11527)
    assert probes["add.fa11.c0.u"][0] > 0 and probes["add.fa11.c1.u"][0] > 0
    print("UNQUALIFIED noisy16 carry frontier: one double/wrong sum in 24 transactions;",
          "fa11.p0.edge at 11527, operand XOR exclusive", flush=True)


def run_scenario(scenario, *, smoke=False):
    wall, cpu = perf_counter(), process_time()
    if scenario.startswith("addsub"):
        width = int(scenario.split("-")[0].removeprefix("addsub"))
        result = arithmetic_survey(width, 12 if smoke else 512 if width == 4 else 2048,
                                   scenario.endswith("mix-B"),
                                   operand_oneshot=True)
        keys = ("wrong", "double", "missing", "repeated", "survivors")
    else:
        result = operand_survey(scenario, 8 if smoke else 40_000)
        keys = FAILURES
    print("operand XOR", scenario, "smoke" if smoke else "cluster", result,
          f"wall_s={perf_counter() - wall:.3f}", f"cpu_s={process_time() - cpu:.3f}", flush=True)
    assert all(result[key] == 0 for key in keys), result


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("DROSO_OPXOR_CLUSTER") != "1",
                    reason="operand-XOR cluster surveys require DROSO_OPXOR_CLUSTER=1")
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_operand_xor_cluster(scenario):
    run_scenario(scenario)


@pytest.mark.skipif(os.environ.get("DROSO_OPXOR_SMOKE") != "1",
                    reason="operand-XOR laptop smoke requires DROSO_OPXOR_SMOKE=1")
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_operand_xor_smoke(scenario):
    run_scenario(scenario, smoke=True)
