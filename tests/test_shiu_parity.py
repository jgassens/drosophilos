"""Compare to Shiu's released equations, including refractory-gated synaptic writes."""

from textwrap import dedent

import numpy as np
import pytest

from drosophilos.sim import ModelRules, Params, RefSim
from drosophilos.sim import Topology
from drosophilos.bench.model_compare import small_fixtures

b2 = pytest.importorskip("brian2")

SHIU = ModelRules(True, True, "v", True)


@pytest.mark.parametrize("arrival", ["synaptic", "external"])
def test_freeze_only_discards_arrival_on_postsynaptic_spike_step(arrival):
    """M2 must gate writes after threshold, independently of reset clearing g."""
    b2.prefs.codegen.target = "numpy"
    params = Params()
    dt = params.dt * b2.ms
    group = b2.NeuronGroup(
        2, dedent('''
            dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
            dg/dt = -g / tau              : volt (unless refractory)
        '''),
        threshold="v > v_th", reset="v = v_rst",
        refractory=params.t_ref * b2.ms, method="exact", dt=dt,
        namespace=dict(v_0=params.E_L * b2.mV, v_th=params.V_th * b2.mV,
                       v_rst=params.V_reset * b2.mV, t_mbr=params.tau_m * b2.ms,
                       tau=params.tau_s * b2.ms),
    )
    group.v = -40 * b2.mV
    group.g = 20 * b2.mV
    objects = [group]
    if arrival == "synaptic":
        source = group
        topo = Topology.from_edges(2, [0], [1], [4000], [0])
    else:
        source = b2.SpikeGeneratorGroup(1, [0], [0] * b2.ms, dt=dt)
        objects.append(source)
        topo = Topology.empty(2)
    syn = b2.Synapses(source, group, on_pre="g += 68.75 * mV", dt=dt)
    syn.connect(i=0, j=1)
    syn.delay = 0 * b2.ms
    spikes = b2.SpikeMonitor(group)
    b2.Network(*objects, syn, spikes).run(dt)

    rules = ModelRules(freeze_g_in_refractory=True, clear_g_on_spike=False)
    ref = RefSim(topo, params, rules=rules)
    ref.V[:] = -40
    ref.g[:] = 20
    if arrival == "external":
        ref.add_events(0, [0], [1], [4000])
    ref.step()

    np.testing.assert_array_equal(spikes.i, [0, 1])
    np.testing.assert_array_equal(ref.trace.neuron_steps(1), [0])
    expected = np.asarray(group.g / b2.mV)
    np.testing.assert_allclose(expected, 20 * ref.c, rtol=1e-12)
    np.testing.assert_allclose(ref.g[0], expected, rtol=1e-12)


@pytest.mark.parametrize("pair", [False, True], ids=["single", "recurrent_pair"])
def test_shiu_equations_match_ref64(pair):
    b2.prefs.codegen.target = "numpy"
    params = Params()
    dt = params.dt * b2.ms
    # Same pair weight as the bounded experiment; no refractory exemption on neuron 1.
    topo = small_fixtures(params)[int(pair)].topo
    eqs = dedent('''
        dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
        dg/dt = -g / tau               : volt (unless refractory)
        rfc                            : second
        ''')
    # The release's reset also assigns w=0; w is unused in these equations.
    group = b2.NeuronGroup(
        topo.n, eqs + "\nw : 1", threshold="v > v_th",
        reset="v = v_rst; w = 0; g = 0 * mV", refractory="rfc",
        method="exact", dt=dt,
        namespace=dict(v_0=params.E_L * b2.mV, v_th=params.V_th * b2.mV,
                       v_rst=params.V_reset * b2.mV, t_mbr=params.tau_m * b2.ms,
                       tau=params.tau_s * b2.ms),
    )
    group.v = params.E_L * b2.mV
    group.rfc = params.t_ref * b2.ms
    group.rfc[0] = 0 * b2.ms
    objects = [group]
    if pair:
        syn = b2.Synapses(group, group, "q : volt", on_pre="g += q", dt=dt)
        syn.connect(i=topo.src, j=topo.dst)
        syn.q = topo.quanta * params.w_unit * b2.mV
        syn.delay = topo.delay * dt
        objects.append(syn)
    # Replay fixed PoissonInput-like arrivals, in the PoissonInput slot, to avoid
    # comparing unrelated RNG streams. Includes adjacent events and a 50-ms gap.
    steps = np.unique(np.r_[np.arange(50, 100, 3),
                            [50, 51, 52, 150, 350, 700, 701, 1200, 1600, 2000, 2400]])
    drive = np.zeros((2500, topo.n))
    drive[steps, 0] = 68.75
    inputs = b2.TimedArray(drive * b2.mV, dt=dt)
    # run_regularly resolves the TimedArray through the group's namespace.
    group.namespace["inputs"] = inputs
    runner = group.run_regularly("v += inputs(t, i)", when="synapses", order=0)
    mon = b2.SpikeMonitor(group)
    objects.extend([runner, mon])
    b2.Network(*objects).run(250 * b2.ms)

    ref = RefSim(topo, params, rules=SHIU)
    ref.add_events(0, steps, np.zeros(len(steps), int), np.full(len(steps), 4000))
    ref.run(2500)
    for neuron in range(topo.n):
        expected = np.rint(np.asarray(mon.t[mon.i == neuron] / dt)).astype(int)
        actual = ref.trace.neuron_steps(neuron)
        assert len(actual) == len(expected), (neuron, actual, expected)
        assert len(actual) > 0
        assert np.max(np.abs(actual - expected)) <= 1, (neuron, actual, expected)
        print(f"{'pair' if pair else 'single'} neuron {neuron}: {len(actual)} spikes, "
              f"maximum difference {np.max(np.abs(actual - expected))} dt")
