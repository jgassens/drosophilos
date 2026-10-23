"""Compare to Shiu's released equations, including refractory-gated synaptic writes."""

from textwrap import dedent

import numpy as np
import pytest

from drosophilos.sim import ModelRules, Params, RefSim
from drosophilos.bench.model_compare import small_fixtures

b2 = pytest.importorskip("brian2")

SHIU = ModelRules(True, True, "v", True)


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
