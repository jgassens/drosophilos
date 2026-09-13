"""Cross-simulator check: the same circuit in Brian2 (method='exact', default slot
order) must produce the same spike-event trace as ref64 (schedule.md §5)."""

import numpy as np
import pytest

from drosophilos.sim import Params, RefSim, SpikeTrace

brian2 = pytest.importorskip("brian2")


def _brian2_trace(topo, params: Params, n_steps: int, events) -> SpikeTrace:
    from brian2 import (
        Network,
        NeuronGroup,
        SpikeGeneratorGroup,
        SpikeMonitor,
        Synapses,
        ms,
        mV,
        prefs,
    )

    prefs.codegen.target = "numpy"
    dt = params.dt * ms
    n = topo.n
    ns = dict(
        E_L=params.E_L * mV,
        V_th=params.V_th * mV,
        V_reset=params.V_reset * mV,
        tau_m=params.tau_m * ms,
        tau_s=params.tau_s * ms,
    )
    eqs = """
    dv/dt = (E_L - v + g)/tau_m : volt (unless refractory)
    dg/dt = -g/tau_s : volt
    """
    G = NeuronGroup(
        n,
        eqs,
        threshold="v > V_th",
        reset="v = V_reset",
        refractory=params.t_ref * ms,
        method="exact",
        dt=dt,
        namespace=ns,
    )
    G.v = params.E_L * mV
    G.g = 0 * mV
    S = Synapses(G, G, "w : volt", on_pre="g += w", dt=dt)
    S.connect(i=topo.src.astype(int), j=topo.dst.astype(int))
    S.w = topo.quanta * params.w_unit * mV
    S.delay = topo.delay * dt

    steps, neurons, quanta = events
    q_by_neuron = {}
    for ne, q in zip(neurons, quanta):
        q_by_neuron.setdefault(int(ne), int(q))
        assert q_by_neuron[int(ne)] == int(q), "Brian2 harness needs one weight per input neuron"
    inputs = sorted(q_by_neuron)
    idx_map = {ne: k for k, ne in enumerate(inputs)}
    P = SpikeGeneratorGroup(len(inputs), [idx_map[int(ne)] for ne in neurons], steps * dt, dt=dt)
    Sin = Synapses(P, G, "w : volt", on_pre="g += w", dt=dt)
    Sin.connect(i=list(range(len(inputs))), j=inputs)
    Sin.w = np.array([q_by_neuron[ne] for ne in inputs]) * params.w_unit * mV
    mon = SpikeMonitor(G)
    net = Network(G, S, P, Sin, mon)
    net.run(n_steps * dt)
    st = np.round(np.asarray(mon.t / dt)).astype(np.int64)
    return SpikeTrace.from_arrays(st, np.zeros(len(st), np.int64), np.asarray(mon.i, dtype=np.int64))


def test_ref64_matches_brian2_spike_for_spike(small_circuit):
    topo, params, n_steps, events = small_circuit
    ev = events[0]
    ref = RefSim(topo, params)
    ref.add_events(0, *ev)
    ref.run(n_steps)
    b2 = _brian2_trace(topo, params, n_steps, ev)
    only_ref, only_b2 = ref.trace.diff(b2)
    assert len(ref.trace) > 200
    assert len(only_ref) == 0 and len(only_b2) == 0, (
        f"{len(only_ref)} spikes only in ref64, {len(only_b2)} only in Brian2; "
        f"first: {only_ref[:5].tolist()} / {only_b2[:5].tolist()}"
    )
