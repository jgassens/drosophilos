"""Track A (docs/perf_campaign.md §4): the static-shape FastSim and the Observer against
RefSim/TorchSim. The circuit is identical; only the simulator changes."""

import numpy as np
import pytest
import torch

from drosophilos.lib.kernel import build_pipeline, run_pipeline, run_pipeline_batched
from drosophilos.sim import FastSim, Observer, Params, RefSim, Topology, TorchSim
from drosophilos.sim.model import D_MAX

from conftest import random_events, random_topology


def _run(cls, topo, params, n_steps, events, n_nodes=1, **kw):
    sim = cls(topo, params, n_nodes=n_nodes, **kw)
    for b, (st, ne, q) in enumerate(events[:n_nodes]):
        sim.add_events(b, st, ne, q)
    sim.run(n_steps)
    return sim


# ---- (1) spike-for-spike parity ------------------------------------------------------------
def test_fast_cpu_float64_matches_reference_and_torch_spike_for_spike(small_circuit):
    topo, params, n_steps, events = small_circuit
    rec = [(0, 3), (1, 7)]
    ref = _run(RefSim, topo, params, n_steps, events, n_nodes=2, record=rec)
    prod = _run(TorchSim, topo, params, n_steps, events, n_nodes=2, record=rec)
    fast = _run(FastSim, topo, params, n_steps, events, n_nodes=2, record=rec)
    assert len(ref.trace) > 200
    only_ref, only_fast = ref.trace.diff(fast.trace)
    assert len(only_ref) == 0 and len(only_fast) == 0, (only_ref[:10], only_fast[:10])
    assert ref.trace == fast.trace == prod.trace
    Vr, gr = ref.recorded()
    Vf, gf = fast.recorded()
    assert np.array_equal(Vr, Vf) and np.array_equal(gr, gf)
    assert fast.delivery_path.startswith("edge-wise")  # the CPU choice; CUDA takes the CSR product
    sparse = _run(FastSim, topo, params, n_steps, events, n_nodes=2, record=rec, delivery="sparse")
    assert sparse.delivery_path.startswith("sparse CSR") and sparse.trace == ref.trace
    assert np.array_equal(Vr, sparse.recorded()[0]) and np.array_equal(gr, sparse.recorded()[1])


def test_fast_float32_matches_torch_float32_bit_for_bit(small_circuit):
    """Same ops in the same order: float32 rounds identically in both backends."""
    topo, params, n_steps, events = small_circuit
    rec = [(0, 3), (1, 7)]
    prod = _run(TorchSim, topo, params, n_steps, events, n_nodes=2, record=rec, dtype=torch.float32)
    fast = _run(FastSim, topo, params, n_steps, events, n_nodes=2, record=rec, dtype=torch.float32)
    assert prod.trace == fast.trace
    assert all(np.array_equal(a, b) for a, b in zip(prod.recorded(), fast.recorded()))


def test_scatter_delivery_and_per_node_quanta_match_reference(small_circuit):
    topo, params, n_steps, events = small_circuit
    ref = _run(RefSim, topo, params, n_steps, events, n_nodes=2)
    fast = _run(FastSim, topo, params, n_steps, events, n_nodes=2, delivery="scatter")
    assert ref.trace == fast.trace and fast.delivery_path.startswith("edge-wise")
    quanta = np.broadcast_to(topo.quanta, (2, topo.nnz)).copy()
    quanta[1, ::3] = 0  # node 1 loses a third of its synapses
    ref_q = _run(RefSim, topo, params, n_steps, events, n_nodes=2, quanta=quanta)
    fast_q = _run(FastSim, topo, params, n_steps, events, n_nodes=2, quanta=quanta)
    assert ref_q.trace == fast_q.trace
    assert ref_q.trace.node(1) != ref.trace.node(1)
    with pytest.raises(ValueError, match="per-node quanta"):
        FastSim(topo, params, n_nodes=2, quanta=quanta, delivery="sparse")


def test_k_step_block_matches_single_steps(small_circuit):
    """graph_steps=K on CPU runs the same block function the CUDA graph captures, eagerly."""
    topo, params, n_steps, events = small_circuit
    ref = _run(RefSim, topo, params, n_steps, events, n_nodes=2, record=[(0, 3)])
    for K in (1, 37, D_MAX + 1):
        fast = _run(FastSim, topo, params, n_steps, events, n_nodes=2, record=[(0, 3)], graph_steps=K)
        assert ref.trace == fast.trace, K
        assert np.array_equal(ref.recorded()[0], fast.recorded()[0])
        assert not fast.graph_active and "CUDA" in fast.graph_fallback_reason
    with pytest.raises(ValueError, match="graph_steps"):
        FastSim(topo, params, graph_steps=D_MAX + 2)


def test_snapshot_restore_reproduces_trace(small_circuit):
    topo, params, n_steps, events = small_circuit
    full = FastSim(topo, params, n_nodes=1)
    full.add_events(0, *events[0])
    full.run(n_steps)
    part = FastSim(topo, params, n_nodes=1, graph_steps=30)
    part.add_events(0, *events[0])
    part.run(1000)
    snap = part.snapshot()
    first = part.trace
    part.run(500)
    part.restore(snap)
    part.run(n_steps - 1000)
    assert first.concat(part.trace) == full.trace


# ---- (3) stray input ----------------------------------------------------------------------
@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_stray_input_stream_is_identical(small_circuit, seed):
    """One torch.rand((B, n)) per step from the same seeded generator: the same stream."""
    topo, params, n_steps, events = small_circuit
    kw = dict(stray_rate_hz=50.0, stray_quanta=3000, stray_seed=seed)
    prod = _run(TorchSim, topo, params, 1500, events, n_nodes=2, **kw)
    fast = _run(FastSim, topo, params, 1500, events, n_nodes=2, **kw)
    block = _run(FastSim, topo, params, 1500, events, n_nodes=2, graph_steps=30, **kw)
    assert prod.trace == fast.trace == block.trace
    assert len(prod.trace) > len(_run(TorchSim, topo, params, 1500, events, n_nodes=2).trace)


# ---- (4) events inside a block ----------------------------------------------------------------
def test_events_scheduled_inside_a_block_land_on_their_step():
    params = Params()
    topo = Topology.empty(2)
    sim = FastSim(topo, params, record=[(0, 1)], graph_steps=40)
    sim.run(40)  # one whole block
    sim.add_events(0, [50, 79], [1, 1], [16, 16])  # inside the next block, first and last step
    sim.run(40)
    V, g = sim.recorded()
    # delivered at 50 (g jumps after delivery), felt at 51; again at 79
    assert g[49, 0] == 0.0 and g[50, 0] > 0.0 and V[50, 0] == params.E_L and V[51, 0] > params.E_L
    assert g[79, 0] > g[78, 0]
    ref = RefSim(topo, params, record=[(0, 1)])
    ref.run(40)
    ref.add_events(0, [50, 79], [1, 1], [16, 16])
    ref.run(40)
    assert np.array_equal(ref.recorded()[1], g)


def test_events_added_between_blocks_match_reference(small_circuit):
    topo, params, n_steps, events = small_circuit
    ref, fast = RefSim(topo, params, n_nodes=2), FastSim(topo, params, n_nodes=2, graph_steps=64)
    for s in (ref, fast):
        s.add_events(0, *events[0])
        s.add_events(1, *events[1])
        s.run(1000)
        s.add_events(0, [1000, 1017, 1063, 1064], [0, 1, 2, 3], [5000, 6000, 7000, 8000])
        s.add_events(1, [1005, 1100], [3, 4], [8000, 9000])
        s.run(1000)
    assert ref.trace == fast.trace
    with pytest.raises(ValueError, match="past"):
        fast.add_events(0, [10], [0], [16])


# ---- (5) several distinct delays ----------------------------------------------------------------
def test_two_distinct_delays_take_the_grouped_path_with_identical_traces():
    rng = np.random.default_rng(3)
    n = 50
    src = np.repeat(np.arange(n), 6)
    dst = rng.integers(0, n, len(src))
    keep = dst != src
    delay = np.where(rng.random(keep.sum()) < 0.5, 7, 18)
    topo = Topology.from_edges(n, src[keep], dst[keep], rng.integers(30, 80, keep.sum()) * 16, delay)
    ev = random_events(n, 2000, rng)
    ref = _run(RefSim, topo, Params(), 2000, [ev])
    fast = _run(FastSim, topo, Params(), 2000, [ev])
    assert fast.delays == (7, 18)
    assert len(ref.trace) > 50 and ref.trace == fast.trace


# ---- observer contract --------------------------------------------------------------------------
def test_observer_never_drops_spikes_silently(small_circuit):
    topo, params, n_steps, events = small_circuit
    obs = Observer([1, 2, 3], 1, n_neurons=topo.n, observe_every=4, retain_steps=8)
    with pytest.raises(RuntimeError, match="filtered"):
        obs.trace
    with pytest.raises(ValueError, match="subset"):
        Observer([1, 2], 1, n_neurons=topo.n, capture=(0, [5]))
    with pytest.raises(OverflowError):
        obs.commit_block(0, 5)
    sim = FastSim(topo, params, observer=obs)
    sim.add_events(0, *events[0])
    sim.run(6)
    with pytest.raises(RuntimeError, match="not observed"):
        obs.fired_at(6)
    sim.run(20)
    with pytest.raises(RuntimeError, match="trimmed"):
        obs.fired_at(3)
    # the watched subset is exactly the reference's spikes of those neurons
    ref = _run(RefSim, topo, params, 26, events)
    ev = ref.trace.events
    want = ev[np.isin(ev["neuron"], [1, 2, 3]) & (ev["step"] >= obs.retained_from)]
    got = obs.watched_trace().events
    assert np.array_equal(want, got)


def test_float32_sparse_refused_when_integer_sums_could_round():
    big = 2**23
    topo = Topology.from_edges(3, [0, 1], [2, 2], [big, big], [1, 1])
    with pytest.raises(ValueError, match="not exact"):
        FastSim(topo, Params(), dtype=torch.float32, delivery="sparse")
    assert FastSim(topo, Params(), dtype=torch.float32).delivery_path.startswith("edge-wise")
    assert FastSim(topo, Params(), dtype=torch.float64, delivery="sparse").delivery_path.startswith("sparse")


# ---- (1)/(2) a compiled kernel through the runners -------------------------------------------------
PARAMS = Params()


def _mov_kernel():
    spec = [{"name": "out", "op": "MOV", "a": "input", "b": ("const", "zero")}]
    return build_pipeline(PARAMS, 4, spec, consts={"zero": 0})


def _values(outs):
    return [[v for _, v in node["out"]] for node in outs]


def test_batched_runner_backends_agree_on_a_kernel():
    pl = _mov_kernel()
    sched = [[1, 5], [0, 3]]
    kw = dict(max_ms=5000, progress=0)
    outs_t, sim_t, st_t = run_pipeline_batched(pl, PARAMS, sched, full_trace=True, **kw)
    outs_f, sim_f, st_f = run_pipeline_batched(pl, PARAMS, sched, backend="torch-fast", full_trace=True,
                                               observe_every=1, **kw)
    assert st_t["simulator"] == "TorchSim" and st_f["simulator"] == "FastSim"
    assert outs_t == outs_f  # same (step, value) pairs
    assert st_t["load_steps"] == st_f["load_steps"]
    assert len(sim_t.trace) > 1000 and sim_t.trace == sim_f.trace
    assert st_t["faults"] == st_f["faults"] == 0 and st_t["bad_outputs"] == st_f["bad_outputs"] == 0
    # (2) the watched-set observer with a K-step transfer: the same decoded outputs, and only
    # the runner's neurons leave the device
    window = 2 * pl.drive.loop_period_steps
    outs_w, sim_w, st_w = run_pipeline_batched(pl, PARAMS, sched, backend="torch-fast", **kw)
    assert st_w["observe_every"] == window and not st_w["full_trace"]
    assert len(sim_w.observer.watch_ids) < pl.net.n // 10
    assert _values(outs_w) == _values(outs_t)
    with pytest.raises(RuntimeError, match="filtered"):
        sim_w.trace
    # (4) K-step blocks: the host schedule runs once per block, observation per block
    outs_g, sim_g, st_g = run_pipeline_batched(pl, PARAMS, sched, backend="torch-fast", graph_steps=window, **kw)
    assert st_g["graph_steps"] == window and st_g["observe_every"] == window and not st_g["graph_active"]
    assert _values(outs_g) == _values(outs_t)
    assert "graph_replay" not in st_g["profile"]["regions"]


def test_single_node_runner_reads_a_fastsim_through_the_observer():
    pl = _mov_kernel()
    outs_r, _, st_r = run_pipeline(pl, PARAMS, [1, 5], max_ms=5000)
    outs_f, sim_f, st_f = run_pipeline(pl, PARAMS, [1, 5], max_ms=5000, sim=FastSim(pl.net.topology(), PARAMS))
    assert st_r["simulator"] == "RefSim" and st_f["simulator"] == "FastSim"
    assert outs_r == outs_f


def test_capture_spikes_agree_between_backends():
    pl = _mov_kernel()
    wanted = {pl.cells[0].start, pl.cells[0].reg.done_relay}
    _, _, st_t = run_pipeline_batched(pl, PARAMS, [[1], [0]], max_ms=3000, progress=0, capture_spikes=(1, wanted))
    _, _, st_f = run_pipeline_batched(pl, PARAMS, [[1], [0]], max_ms=3000, progress=0, capture_spikes=(1, wanted),
                                      backend="torch-fast", observe_every=1)
    for a, b in zip(st_t["captured_spikes"], st_f["captured_spikes"]):
        assert np.array_equal(a, b)
    assert set(st_f["captured_spikes"][1].tolist()) == wanted


def test_backend_rejections_are_explicit():
    pl = _mov_kernel()
    with pytest.raises(ValueError, match="backend"):
        run_pipeline_batched(pl, PARAMS, [[1]], max_ms=100, progress=0, backend="numpy")
    with pytest.raises(ValueError, match="FastSim"):
        run_pipeline_batched(pl, PARAMS, [[1]], max_ms=100, progress=0, backend="torch-fast",
                             sim=TorchSim(pl.net.topology(), PARAMS))


# ---- (6) MPS -------------------------------------------------------------------------------------
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS device")
def test_mps_float32_matches_cpu_float32(small_circuit):
    topo, params, n_steps, events = small_circuit
    cpu = _run(TorchSim, topo, params, n_steps, events, n_nodes=2, dtype=torch.float32)
    mps = _run(FastSim, topo, params, n_steps, events, n_nodes=2, dtype=torch.float32, device="mps")
    block = _run(FastSim, topo, params, n_steps, events, n_nodes=2, dtype=torch.float32, device="mps", graph_steps=64)
    assert mps.delivery_path.startswith("edge-wise")
    # measured bit-identical on this laptop (torch 2.14); a future MPS kernel change would show
    # here, which is the point: any difference must be documented, not assumed away
    assert cpu.trace == mps.trace == block.trace


# ---- bench wiring ------------------------------------------------------------------------------
def test_render_doom_accepts_torch_fast_and_rejects_its_flags_elsewhere():
    from drosophilos.bench.render_doom import RenderConfig, _effective_backend

    backend, device, dtype_name, torch_dtype, simulator, _gpu = _effective_backend(
        RenderConfig(backend="torch-fast", graph_steps=94, observe_every=94, delivery="scatter"))
    assert (backend, device, dtype_name, simulator) == ("torch-fast", "cpu", "float64", "FastSim")
    with pytest.raises(SystemExit, match="torch-fast only"):
        _effective_backend(RenderConfig(backend="torch", graph_steps=94))
    with pytest.raises(SystemExit, match="graph-steps"):
        _effective_backend(RenderConfig(backend="torch-fast", graph_steps=D_MAX + 2))
    with pytest.raises(SystemExit, match="float32"):
        _effective_backend(RenderConfig(backend="torch-fast", device="mps", dtype="float64"))


def test_perf_campaign_backend_variant_is_a_simulator_only_comparison(tmp_path):
    import argparse
    from drosophilos.bench.perf_campaign import run_campaign

    args = argparse.Namespace(levels="primitive", backend="torch", device="cpu", dtype="float64", mul="array",
                              copies="1", pacing="host", repeats=1, tokens=16, primitive_blocks="cells-mov",
                              primitive_max_s=150, profile_steps=0, historical=False, allow_cpu_renders=False,
                              dry_run=True, out=str(tmp_path / "c"), graph_steps=0, observe_every=None,
                              delivery="auto", variant_backend="torch-fast")
    report = run_campaign(args)
    names = [c["name"] for c in report["configurations"]]
    assert names == ["baseline", "backend-torch-fast"]
    fast = report["configurations"][1]
    assert fast["backend"] == "torch-fast" and fast["graph_steps"] == 0 and "graph_steps" not in report["configurations"][0]
    assert all(e["comparison"] == "simulator-only" for e in report["cost_estimate"])
    args.variant_backend, args.graph_steps = None, 94
    with pytest.raises(ValueError, match="torch-fast"):
        run_campaign(args)
