"""Compact register-domain qualification using the unchanged float64 RefSim.

The reduced controllers and static draws are extracted from the committed kernel,
not prescribed inhibitory arrivals. Slow tests retain the 40,000-reset surveys.
"""

import numpy as np
import pytest

from drosophilos.lib.kernel import build_pipeline, run_pipeline
from drosophilos.protocol.latch import add_latch, add_ready, add_reset, compact_reset_domains
from drosophilos.protocol.token import decode_at, rails_for
from drosophilos.lib.netlist import Netlist
from drosophilos.sim.ref64 import RefSim
from test_completion_stall import COMMIT, DRIVE, LOAD, PARAMS, WIDTH, _cell, _map, _observed_run, _outcome, _run


CAPTURES = {
    "master74": (74, False, "c5_sub.M", "b3r0"),
    "master17": (17, False, "c4_sel.M", "b8r0"),
    "master24": (24, True, "c2_sel.M", "b6r0"),
    "stage18": (18, False, "c3_xor.Q", "b9r0"),
}
# Independent loop / clear / ignition corners. The asymmetric cases reverse
# both loop directions and thresholds; weak-clear fast cases are the kill stress.
CORNERS = {
    "nominal": ((1., 1.), (0., 0.), (1., 1.), 1.),
    "slow8": ((.92, .92), (.4, .4), (1.08, 1.08), .92),
    "slow12": ((.88, .88), (.6, .6), (1.12, 1.12), .88),
    "fast20": ((1.2, 1.2), (-.8, -.8), (.88, .88), 1.),
    "fast30": ((1.3, 1.3), (-1.2, -1.2), (.88, .88), 1.),
    "asymmetric": ((1.3, .88), (.6, -1.2), (.88, 1.12), .88),
    "mirror": ((.88, 1.3), (-1.2, .6), (1.12, .88), .88),
}


def _primitive(case, *, compact=True, ready=False, strength=1.75, controller="nominal"):
    copy, rate, name, rail = CAPTURES.get(case, (None, False, "R", "L"))
    net = Netlist(PARAMS)
    latch = add_latch(net, DRIVE, f"{name}.{rail}")
    trigger, inh, _ = add_reset(net, DRIVE, name, [latch])
    ready_id = add_ready(net, DRIVE, name, trigger, hops=15) if ready else None
    before = net.topology()
    if copy is None:
        q = before.quanta.astype(np.int64)
        vth, bias = np.full(net.n, PARAMS.V_th), np.zeros(net.n)
    else:
        q, vth, bias, _ = _map(copy, net, rate_robust=rate)
    if compact:
        compact_reset_domains(net, DRIVE, [(trigger, inh)], strength=strength)
        q = np.rint(q * net.topology().quanta / before.quanta).astype(np.int64)
    ignition = DRIVE.ignite
    if copy is None:
        loop, threshold, clear, ignite = CORNERS[case]
        for e, (s, d) in enumerate(zip(before.src, before.dst)):
            if s in latch.members and d in latch.members:
                q[e] = round(DRIVE.loop * loop[latch.members.index(s)])
            if s == inh and d in latch.members:
                q[e] = round(q[e] * clear[latch.members.index(d)])
        vth[list(latch.members)] += threshold
        ignition = round(ignition * ignite)
    if controller != "nominal":
        scale, threshold = {"fast": (1.12, -.6), "slow": (.88, .6),
                            "fast_ready": (1.12, -.6)}[controller]
        control = [i for i, role in enumerate(net.roles)
                   if i not in latch.members and (controller != "fast_ready" or ".ready" in role)]
        vth[control] = PARAMS.V_th + threshold
        for e, d in enumerate(before.dst):
            if d in control and q[e] > 0:
                q[e] = round(q[e] * scale)
    return net.topology(), latch, trigger, inh, ready_id, q, vth, bias, ignition


def _live(trace, latch, after):
    ev = trace.events
    mask = np.isin(ev["neuron"], latch.members) & (ev["step"] > after[ev["node"]])
    return np.unique(ev["node"][mask])


def _survival(case, n=256, *, compact=True, noisy=True, seed=108, strength=1.75, controller="nominal"):
    topo, latch, trigger, inh, _, q, vth, bias, _ = _primitive(
        case, compact=compact, strength=strength, controller=controller)
    rng = np.random.default_rng(seed)
    survivors = 0
    for lo in range(0, n, 1000):
        count = min(1000, n - lo)
        resets = rng.integers(3000, 4500, count) if noisy else 3000 + np.arange(count)
        sim = RefSim(topo, PARAMS, n_nodes=count, V_th=vth, bias=bias,
                     quanta=np.broadcast_to(q, (count, topo.nnz)))
        for b, reset in enumerate(resets):
            # Captured c3's four ignition offsets; the later small kick also
            # challenges already-circulating master orbits and both asymmetries.
            steps = [10, 11, 14, 23, 2520, int(reset)]
            neurons = [latch.u] * 4 + [latch.v if b % 2 else latch.u, trigger]
            quanta = [DRIVE.ignite] * 4 + [300, DRIVE.relay_in]
            if noisy:
                st, ne = np.nonzero(rng.random((6200, topo.n)) < 5 * PARAMS.dt / 1000)
                steps += st.tolist(); neurons += ne.tolist(); quanta += [150] * len(st)
            sim.add_events(b, steps, neurons, quanta)
        trace = _observed_run(sim, 6200, latch.members)
        survivors += len(_live(trace, latch, resets + 1500))
    return survivors


def _reload(case, offsets=range(400, 1251, 25), *, strength=1.75, controller="nominal"):
    topo, latch, trigger, inh, ready, q, vth, bias, ignition = _primitive(
        case, ready=True, strength=strength, controller=controller)
    probe = RefSim(topo, PARAMS, V_th=vth, bias=bias, quanta=q[None, :])
    probe.add_events(0, [10, 2000], [latch.u, trigger], [DRIVE.ignite, DRIVE.relay_in])
    trace = _observed_run(probe, 4000, [*latch.members, inh, ready])
    arrivals = trace.neuron_steps(inh) + PARAMS.default_delay_steps
    first, last = int(arrivals[0]), int(arrivals[-1])
    ready_at = int(trace.neuron_steps(ready)[0])
    assert not len(_live(trace, latch, np.array([last + 200])))
    assert np.any(trace.neuron_steps(latch.u) > 1500)
    offsets = sorted(set(offsets) | {ready_at - first})  # also test the exact READY time
    cases = [(int(offset), phase) for offset in offsets for phase in range(0, 44, 4)]
    sim = RefSim(topo, PARAMS, n_nodes=len(cases), V_th=vth, bias=bias,
                 quanta=np.broadcast_to(q, (len(cases), topo.nnz)))
    reloads = np.array([first + offset + phase for offset, phase in cases])
    for b, ((_, phase), reload) in enumerate(zip(cases, reloads)):
        sim.add_events(b, [10, 2000 + phase, reload], [latch.u, trigger, latch.u],
                       [DRIVE.ignite, DRIVE.relay_in, ignition])
    trace = _observed_run(sim, int(reloads.max()) + 1600, latch.members)
    success = np.zeros(len(cases), bool)
    success[_live(trace, latch, reloads + 1500)] = True
    success = success.reshape(-1, 11).all(axis=1)
    reliable = np.logical_and.accumulate(success[::-1])[::-1]
    minimum = int(np.asarray(list(offsets))[reliable][0]) if np.any(reliable) else None
    return dict(arrivals=(arrivals - first).tolist(), ready=ready_at - first,
                recovery=minimum, margin=None if minimum is None else ready_at - first - minimum,
                ready_reloads=bool(success[offsets.index(ready_at - first)]))


@pytest.mark.parametrize("case", [*CAPTURES, *CORNERS])
def test_entrained_register_latches_clear(case):
    assert _survival(case, noisy=False) == 0


@pytest.mark.parametrize("case", [*CAPTURES, *CORNERS])
def test_reset_recovery_before_ready(case):
    result = _reload(case)
    print(case, result, flush=True)
    expected = {"master74": [0, 45, 84, 144], "master17": [0, 42, 82, 148],
                "master24": [0, 48, 89, 149], "stage18": [0, 41, 78, 141]}
    assert result["arrivals"] == expected.get(case, [0, 43, 81, 138])
    assert result["margin"] > 0 and result["ready_reloads"], (case, result)


@pytest.mark.parametrize("controller", ["fast", "slow", "fast_ready"])
def test_independent_controller_corners(controller):
    assert _survival("fast30", noisy=False, controller=controller) == 0
    result = _reload("slow12", controller=controller)
    print(controller, result, flush=True)
    assert result["arrivals"] == {"fast": [0, 34, 66, 111], "slow": [0, 60, 115],
                                  "fast_ready": [0, 43, 81, 138]}[controller]
    assert result["margin"] > 0 and result["ready_reloads"], result


def test_regression_phases_fail_with_the_original_reset():
    assert _survival("master74", compact=False, noisy=False) == 52
    assert _survival("master24", compact=False, noisy=False) == 2


@pytest.mark.parametrize("zero_once,request_clear", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("rate", [False, True])
def test_multi_cell_kernel_reloads(zero_once, request_clear, rate):
    pl = build_pipeline(PARAMS, 2,
                        [{"name": "sum", "op": "ADD", "a": "input", "b": ("const", "one")},
                         {"name": "out", "op": "XOR", "a": "sum", "b": ("const", "one")}],
                        consts={"one": 1}, robust_register_reset=True,
                        zero_once=zero_once, robust_request_clear=request_clear, rate_robust=rate,
                        datapath="specialized" if request_clear else "generic")
    outputs, sim, stats = run_pipeline(pl, PARAMS, [0, 1, 3], max_ms=15000)
    assert isinstance(sim, RefSim)
    assert [v for _, v in outputs] == [0, 3, 1]
    assert stats["faults"] == stats["timeouts"] == stats["bad_outputs"] == 0
    assert not stats["refusals"]


@pytest.mark.parametrize("copy", [74, 24], ids=["base", "rate_robust"])
def test_next_copy_and_stage_word_at_ready_with_slow_storage(copy):
    c = _cell(copy, nominal=True, compact=True)
    topo, Q, M = c["net"].topology(), c["Q"], c["M"]
    q = c["q"].copy()
    storage = {x for reg in (Q, M) for pair in reg.rails for l in pair for x in l.members}
    ready_nodes = {x for reg in (Q, M) for x in [*reg.ready_chain, reg.ready]}
    c["vth"][list(storage)] += .6
    c["vth"][list(ready_nodes)] -= .6
    for e, (s, d) in enumerate(zip(topo.src, topo.dst)):
        if d in storage:
            q[e] = round(q[e] * (.88 if q[e] > 0 else 1.12))
        elif d in ready_nodes and q[e] > 0:
            q[e] = round(q[e] * 1.12)

    def simulate(next_load=None):
        sim = RefSim(topo, PARAMS, V_th=c["vth"], bias=c["bias"], quanta=q[None, :])
        events = [(10, M.rails[i][r].u, DRIVE.ignite) for i, r in rails_for(c["old"], WIDTH)]
        events += [(LOAD, Q.rails[i][r].u, DRIVE.ignite) for i, r in rails_for(c["new"], WIDTH)]
        events.append((COMMIT, c["reg"].commit_in, DRIVE.ignite))
        if next_load is not None:
            events += [(next_load, Q.rails[i][r].u, round(.88 * DRIVE.ignite))
                       for i, r in rails_for(c["old"], WIDTH)]
            events.append((next_load + 4300, c["reg"].commit_in, DRIVE.ignite))
        sim.add_events(0, *zip(*events))
        watched = [Q.ready, M.ready, c["reg"].done_relay, *M.fault, *Q.fault]
        watched += [x for reg in (Q, M) for pair in reg.rails for l in pair for x in l.members]
        return _observed_run(sim, 21000 if next_load is None else next_load + 11500, watched)

    first = simulate()
    ready = first.neuron_steps(Q.ready)
    next_load = int(ready[ready > COMMIT][0])  # deliberately no grace interval
    trace = simulate(next_load)
    done = trace.neuron_steps(c["reg"].done_relay)
    done = done[done > COMMIT]
    assert len(done) == 2
    for step, word in zip(done, (c["new"], c["old"])):
        assert decode_at(trace, M.rail_taps, int(step), window=120) == (word, "valid")
    for f in Q.fault + M.fault:
        assert not len(trace.neuron_steps(f))


def test_small_noisy_multi_cell_kernel():
    pl = build_pipeline(PARAMS, 2,
                        [{"name": "sum", "op": "ADD", "a": "input", "b": ("const", "one")},
                         {"name": "out", "op": "XOR", "a": "sum", "b": ("const", "one")}],
                        consts={"one": 1}, robust_register_reset=True, rate_robust=True,
                        zero_once=True, robust_request_clear=True)
    topo = pl.net.topology()
    rng = np.random.default_rng(108)
    q = np.rint(topo.quanta * np.exp(rng.normal(0, .04, topo.nnz))).astype(np.int64)
    sim = RefSim(topo, PARAMS, quanta=q[None, :], V_th=PARAMS.V_th + rng.normal(0, .2, topo.n),
                 bias=np.asarray(pl.net.bias) + rng.normal(0, .2, topo.n))
    # Poisson-law strays at mix-B rate/charge; generate sparse times per neuron
    # instead of allocating a steps x whole-kernel random matrix.
    steps, neurons = [], []
    for neuron in range(topo.n):
        times = np.cumsum(rng.geometric(5 * PARAMS.dt / 1000, 100)) - 1
        while times[-1] < 100000:
            times = np.r_[times, times[-1] + np.cumsum(rng.geometric(5 * PARAMS.dt / 1000, 100))]
        times = times[times < 100000]
        steps.extend(times); neurons.extend([neuron] * len(times))
    # RefSim.add_events masks its input once per distinct step. Bound that work
    # without changing any event or RNG draw: one huge noisy-kernel call is quadratic.
    order = np.argsort(steps)
    steps, neurons = np.asarray(steps)[order], np.asarray(neurons)[order]
    for start in range(0, len(steps), 1000):
        stop = min(start + 1000, len(steps))
        sim.add_events(0, steps[start:stop], neurons[start:stop], np.full(stop - start, 150))
    outputs, _, stats = run_pipeline(pl, PARAMS, [0, 3], sim=sim, max_ms=10000)
    assert [v for _, v in outputs] == [0, 1]
    assert stats["faults"] == stats["timeouts"] == stats["bad_outputs"] == 0
    assert not stats["refusals"]


@pytest.mark.parametrize("rate", [False, True])
def test_extended_stage_domain_is_dark_at_ready_and_rearms(rate):
    pl = build_pipeline(PARAMS, 2,
                        [{"name": "out", "op": "ADD", "a": "input", "b": ("const", "one")}],
                        consts={"one": 1}, robust_register_reset=True, zero_once=True, rate_robust=rate)
    outputs, sim, stats = run_pipeline(pl, PARAMS, [0, 3], max_ms=10000, full_trace=True)
    assert [v for _, v in outputs] == [1, 0]
    assert stats["faults"] == stats["timeouts"] == stats["bad_outputs"] == 0
    sim.run(1800)  # the runner stops at DONE; observe the final stage reset/READY too
    trace = sim.trace
    stage = pl.cells[0].stage
    ready = int(trace.neuron_steps(stage.ready)[-1])
    targets = {d for s, d, q in zip(pl.net.src, pl.net.dst, pl.net.quanta)
               if s == stage.reset_inh and q < 0}
    ev = trace.events
    assert not np.any(np.isin(ev["neuron"], list(targets)) & (ev["step"] > ready - 100))


def test_observation_optimization_preserves_refsim_spikes():
    topo, latch, trigger, inh, _, q, vth, bias, _ = _primitive("master74")
    sims = [RefSim(topo, PARAMS, n_nodes=2, V_th=vth, bias=bias,
                   quanta=np.broadcast_to(q, (2, topo.nnz))) for _ in range(2)]
    for sim in sims:
        for node in (0, 1):
            sim.add_events(node, [10, 501, 730], [latch.u, latch.v, trigger],
                           [DRIVE.ignite, 300, DRIVE.relay_in])
    observed = _observed_run(sims[0], 2100, [*latch.members, inh])
    sims[1].run(2100)
    ev = sims[1].trace.events
    assert np.array_equal(observed.events, ev[np.isin(ev["neuron"], [*latch.members, inh])])


@pytest.mark.parametrize("copy", [74, 24])
def test_fault_discard_clears_fault_and_accepts_next_word(copy):
    c = _cell(copy, nominal=True, compact=True)
    Q, M = c["Q"], c["M"]

    def run(next_load=None):
        sim = RefSim(c["net"].topology(), PARAMS)
        events = [(10, M.rails[i][r].u, DRIVE.ignite) for i, r in rails_for(c["old"], WIDTH)]
        events += [(LOAD, Q.rails[i][r].u, DRIVE.ignite) for i, r in rails_for(c["new"], WIDTH)]
        events.append((LOAD, Q.rails[0][1 - (c["new"] & 1)].u, DRIVE.ignite))
        if next_load is not None:
            events += [(next_load, Q.rails[i][r].u, DRIVE.ignite) for i, r in rails_for(c["new"], WIDTH)]
            events.append((next_load + 4300, c["reg"].commit_in, DRIVE.ignite))
        sim.add_events(0, *zip(*events))
        watched = [Q.ready, Q.fault_latch.u, c["reg"].done_relay]
        watched += [l.u for pair in M.rails for l in pair]
        return _observed_run(sim, 12000 if next_load is None else next_load + 10000, watched)

    first = run()
    ready = first.neuron_steps(Q.ready)
    next_load = int(ready[ready > LOAD][0])
    trace = run(next_load)
    faults = trace.neuron_steps(Q.fault_latch.u)
    assert np.any(faults > LOAD) and not np.any(faults > next_load - 100)
    done = trace.neuron_steps(c["reg"].done_relay)
    done = done[done > LOAD]
    assert len(done) == 1 and done[0] > next_load + 4300
    assert decode_at(trace, M.rail_taps, int(done[0]), window=120) == (c["new"], "valid")


@pytest.mark.parametrize("copy", [74, 17, 24])
def test_small_noisy_completion(copy):
    c = _cell(copy, compact=True)
    trace, commits = _run(c, [23, 108, 329, 731])
    for b, tc in enumerate(commits):
        out = _outcome(c, trace, b, tc)
        assert out["done"] and not out["survivors"] and not out["faults"]


@pytest.mark.slow
@pytest.mark.parametrize("case,controller", [(c, "nominal") for c in [*CAPTURES, *CORNERS]] +
                         [("fast30", c) for c in ("fast", "slow")])
def test_40000_noisy_resets(case, controller):
    survivors = _survival(case, 40000, controller=controller)
    print(dict(case=case, controller=controller, resets=40000, survivors=survivors), flush=True)
    assert survivors == 0


@pytest.mark.slow
def test_candidate_evaluation_table():
    for strength, nominal, slow in ((1.1, 76, 256), (1.25, 0, 256), (1.35, 0, 208), (1.5, 0, 0), (1.75, 0, 0)):
        a = _survival("fast30", noisy=False, strength=strength)
        b = _survival("fast30", noisy=False, strength=strength, controller="slow")
        reload = _reload("slow12", strength=strength, controller="fast_ready")
        print(dict(strength=strength, phases=256, nominal_controller=a, slow_controller=b, **reload), flush=True)
        assert (a, b) == (nominal, slow)
