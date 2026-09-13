"""Stage 0 simulator certification: reference vs production, batch independence,
snapshots, refractory semantics (schedule.md)."""

import numpy as np
import pytest
import torch

from drosophilos.sim import Params, RefSim, SpikeTrace, Topology, TorchSim


def _run_ref(topo, params, n_steps, events, n_nodes=1, **kw):
    sim = RefSim(topo, params, n_nodes=n_nodes, **kw)
    for b, (st, ne, q) in enumerate(events[:n_nodes]):
        sim.add_events(b, st, ne, q)
    sim.run(n_steps)
    return sim


def _run_torch(topo, params, n_steps, events, n_nodes=1, **kw):
    sim = TorchSim(topo, params, n_nodes=n_nodes, **kw)
    for b, (st, ne, q) in enumerate(events[:n_nodes]):
        sim.add_events(b, st, ne, q)
    sim.run(n_steps)
    return sim


def test_reference_produces_activity(small_circuit):
    topo, params, n_steps, events = small_circuit
    sim = _run_ref(topo, params, n_steps, events)
    assert len(sim.trace) > 200, "circuit should be active enough to test anything"
    # recurrent activity: neurons that receive no external drive must spike too
    assert (sim.trace.events["neuron"] >= 12).sum() > 50


def test_torch_cpu_float64_matches_reference_spike_for_spike(small_circuit):
    topo, params, n_steps, events = small_circuit
    ref = _run_ref(topo, params, n_steps, events, n_nodes=2, record=[(0, 3), (1, 7)])
    prod = _run_torch(topo, params, n_steps, events, n_nodes=2, record=[(0, 3), (1, 7)])
    only_ref, only_prod = ref.trace.diff(prod.trace)
    assert len(only_ref) == 0 and len(only_prod) == 0, (only_ref[:10], only_prod[:10])
    assert ref.trace == prod.trace
    Vr, gr = ref.recorded()
    Vp, gp = prod.recorded()
    assert np.array_equal(Vr, Vp) and np.array_equal(gr, gp)


def test_batch_independence(small_circuit):
    topo, params, n_steps, events = small_circuit
    batched = _run_ref(topo, params, n_steps, events, n_nodes=2)
    alone0 = _run_ref(topo, params, n_steps, events[:1], n_nodes=1)
    alone1 = _run_ref(topo, params, n_steps, [events[1]], n_nodes=1)
    assert batched.trace.node(0) == alone0.trace
    assert batched.trace.node(1) == alone1.trace
    tb = _run_torch(topo, params, n_steps, events, n_nodes=2)
    assert tb.trace.node(1) == alone1.trace


@pytest.mark.parametrize("cls", [RefSim, TorchSim])
def test_snapshot_restore_reproduces_trace(small_circuit, cls):
    topo, params, n_steps, events = small_circuit
    full = cls(topo, params, n_nodes=1)
    full.add_events(0, *events[0])
    full.run(n_steps)

    part = cls(topo, params, n_nodes=1)
    part.add_events(0, *events[0])
    part.run(1000)
    snap = part.snapshot()
    first = part.trace
    part.run(500)  # diverge from the snapshot point, then go back
    part.restore(snap)
    part.run(n_steps - 1000)
    assert first.concat(part.trace) == full.trace


def test_refractory_resumes_exactly_n_ref_steps_after_spike():
    params = Params()
    topo = Topology.empty(1)
    sim = RefSim(topo, params)
    steps = np.arange(0, 400)
    sim.add_events(0, steps, np.zeros_like(steps), np.full_like(steps, 200 * 16))  # huge drive every step
    sim.run(400)
    st = sim.trace.neuron_steps(0)
    assert len(st) > 5
    isi = np.diff(st)
    # the drive keeps growing, so once it is large the neuron fires the moment integration
    # resumes: exactly n_ref steps after the previous spike, never sooner
    assert isi.min() == params.n_ref == 22
    assert (isi[-3:] == 22).all()


def test_single_step_response_needs_about_five_thousand_synapses():
    """Physics check recorded in schedule.md: V moves by k*g per step, k ~ 0.0049."""
    params = Params()
    a, c, k = params.constants()
    # smallest quanta that crosses threshold on the first felt step
    need = (params.V_th - params.E_L) / (k * params.w_unit)
    assert 80_000 < need < 85_000


def test_delivery_is_felt_one_step_after_delivery_step():
    params = Params()
    topo = Topology.empty(2)
    sim = RefSim(topo, params, record=[(0, 1)])
    sim.add_events(0, [10], [1], [16])  # one synapse's worth at step 10
    sim.run(13)
    V, g = sim.recorded()
    # g jumps at the end of step 10 (observed after delivery); V moves at step 11
    assert g[9, 0] == 0.0 and g[10, 0] > 0.0
    assert V[10, 0] == params.E_L and V[11, 0] > params.E_L


def test_synaptic_delay_in_steps():
    params = Params()
    # neuron 0 -> neuron 1 with a 7-step delay and a weight big enough to fire in one step
    big = 100_000
    topo = Topology.from_edges(2, [0], [1], [big], [7])
    sim = RefSim(topo, params, record=[(0, 1)])
    sim.add_events(0, [5], [0], [big])
    sim.run(30)
    s0 = sim.trace.neuron_steps(0)
    s1 = sim.trace.neuron_steps(1)
    # event delivered at 5, felt at 6, spike at 6 (later spikes come from the lingering drive)
    assert s0[0] == 6
    # spike at 6 delivered at 6+7=13, felt at 14 -> spike at 14
    assert s1[0] == 14


def test_silenced_neuron_never_spikes_and_never_delivers():
    params = Params()
    topo = Topology.from_edges(2, [0], [1], [100_000], [0])
    sim = RefSim(topo, params, silenced=np.array([True, False]))
    sim.add_events(0, [5], [0], [100_000])
    sim.run(30)
    assert len(sim.trace) == 0


def test_zero_delay_self_consistency_torch_vs_ref():
    rng = np.random.default_rng(7)
    n = 40
    src = np.repeat(np.arange(n), 6)
    dst = rng.integers(0, n, len(src))
    keep = dst != src
    # unphysiological weights on purpose: many coincident zero-delay deliveries per step
    topo = Topology.from_edges(n, src[keep], dst[keep], rng.integers(1, 6, keep.sum()) * 6000, np.zeros(keep.sum(), int))
    ev = (np.arange(0, 800, 3), np.arange(0, 800, 3) % 10, np.full(267, 90_000))
    ref = RefSim(topo, Params())
    ref.add_events(0, *ev)
    ref.run(800)
    prod = TorchSim(topo, Params())
    prod.add_events(0, *ev)
    prod.run(800)
    assert len(ref.trace) > 50
    assert ref.trace == prod.trace


def test_topology_sorting_and_validation():
    t = Topology.from_edges(3, [2, 0, 0], [1, 2, 1], [16, 32, -16], [3, 1, 0])
    assert list(t.src) == [0, 0, 2]
    assert list(t.dst) == [1, 2, 1]
    assert list(t.quanta) == [-16, 32, 16]
    assert list(t.indptr) == [0, 2, 2, 3]
    with pytest.raises(ValueError):
        Topology.from_edges(3, [0], [1], [16], [101])
