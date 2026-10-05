"""Compact register-domain qualification using the unchanged float64 RefSim.

The reduced controllers and static draws are extracted from the committed kernel,
not prescribed inhibitory arrivals. Slow tests retain the 40,000-reset surveys.
"""

from dataclasses import replace

import numpy as np
import pytest

from drosophilos.lib.kernel import _NoProducer, _fault_latch, build_pipeline, run_pipeline
from drosophilos.lib.staged import add_staged_commit
from drosophilos.protocol.handshake import add_register
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
    "slow3sigma": ((np.exp(-.12),) * 2, (.6,) * 2, (np.exp(.12),) * 2, np.exp(-.12)),
    "fast3sigma": ((np.exp(.12),) * 2, (-.6,) * 2, (np.exp(-.12),) * 2, 1.),
    "fast30_bias": ((1.3,) * 2, (-1.2,) * 2, (.88,) * 2, 1.),
}
CORNER_BIAS = {"slow3sigma": -.6, "fast3sigma": .6, "fast30_bias": .6}


def _candidate_pipeline(*args, strength=1.75, tail=8, **kwargs):
    """Exercise the public experiment; direct compilation compares rejected policies."""
    kwargs.pop("robust_register_reset", None)
    if strength == 1.75 and tail == 8:
        return build_pipeline(*args, **kwargs, experimental_register_reset=True)
    pl = build_pipeline(*args, **kwargs)
    regs = [r for sr, _ in pl.inputs.values() for r in (sr.stage, sr.master)]
    regs += [r for c in pl.cells for r in (c.stage, c.master)]
    compact_reset_domains(pl.net, pl.drive, [(r.reset_trigger, r.reset_inh) for r in regs],
                          strength=strength, ready_tail_links=tail)
    return pl


def _sparse_strays(sim, rng, steps):
    """Independent Bernoulli(5 Hz * dt) arrivals, using geometric waiting times."""
    for node in range(sim.B):
        times, neurons = [], []
        for neuron in range(sim.n):
            t = -1
            while True:
                t += int(rng.geometric(5 * PARAMS.dt / 1000))
                if t >= steps:
                    break
                times.append(t)
                neurons.append(neuron)
        sim.add_events(node, times, neurons, [150] * len(times))


def _path_trials(n, *, rate=False, strength=1.75, tail=8, advance=0, seed=108,
                 corner=True, noisy=True, progress=False, verified=False):
    """Two actual staged commits, loading the SAME Q rail at its observed READY.

    A one-bit word isolates the reset/reload contract from ALU rate-gate failure.
    No inhibitory arrivals or COPY ignitions are prescribed: real Q/M reset,
    READY, COMMIT, grant, selected-rail arms and completion circuits execute.
    Advancing both READY chains tests a positive margin on Q load AND M COPY.
    It does not qualify a whole ALU or substitute for the extended-domain tests.
    """
    drive = replace(DRIVE, rate_robust=rate)
    net = Netlist(PARAMS)
    Q = add_register(net, drive, "cell.Q", 1, with_completion=True)
    _fault_latch(net, drive, "cell", Q)
    reg = add_staged_commit(net, drive, "cell", Q, _NoProducer(), ordered_grant=True,
                            copy_requires_rail=True)
    M = reg.master
    if verified:
        from drosophilos.protocol.handshake import verify_register_resets
        verify_register_resets(net, drive, [Q, M])
    else:
        compact_reset_domains(net, drive, [(r.reset_trigger, r.reset_inh) for r in (Q, M)],
                              strength=strength, ready_tail_links=tail)
    topo = net.topology()
    # Move the actual Q READY and M READY->COPY path earlier, not an external
    # proxy ignition. 100 steps is one extended synaptic delay.
    for r in (Q, M):
        edge = np.flatnonzero((topo.dst == r.ready) & (topo.quanta > 0))
        assert len(edge) == (4 if verified else 1) and np.all(advance <= topo.delay[edge])
        topo.delay[edge] -= advance
    rails = {x for r in (Q, M) for pair in r.rails for latch in pair for x in latch.members}
    ready = {x for r in (Q, M) for x in [*r.ready_chain, r.ready]}
    if verified:
        ready.update(i for i, role in enumerate(net.roles)
                     if ".verify." in role and (".quiet" in role or ".recovery" in role))
    rng = np.random.default_rng(seed)
    # A verified path can spend three retry windows at each reset. The timed
    # candidate's 1.8-second horizon would misclassify a late second DONE.
    trial_steps = 32000 if verified else 18000
    failures = ready_missing = faults = survivors = 0
    min_copy_after_ready = 10**9
    for lo in range(0, n, 250):
        count = min(250, n - lo)
        if corner:
            q = np.broadcast_to(topo.quanta, (count, topo.nnz)).copy()
            vth = np.full((count, topo.n), PARAMS.V_th)
            bias = np.broadcast_to(net.bias, (count, topo.n)).copy()
            vth[:, list(rails)] += .6
            bias[:, list(rails)] -= .6
            vth[:, list(ready)] -= .6
            bias[:, list(ready)] += .6
            for e, d in enumerate(topo.dst):
                scale = (np.exp(-.12) if q[0, e] > 0 else np.exp(.12)) if d in rails else (
                    np.exp(.12) if d in ready and q[0, e] > 0 else 1.)
                q[:, e] = np.rint(q[:, e] * scale).astype(np.int64)
        else:
            q = np.rint(topo.quanta * np.exp(rng.normal(0, .04, (count, topo.nnz)))).astype(np.int64)
            vth = PARAMS.V_th + rng.normal(0, .2, (count, topo.n))
            bias = np.asarray(net.bias) + rng.normal(0, .2, (count, topo.n))
        sim = RefSim(topo, PARAMS, n_nodes=count, quanta=q, V_th=vth, bias=bias)
        if noisy:
            _sparse_strays(sim, rng, trial_steps)
        for b in range(count):
            sim.add_events(b, [10, 3000, 6000], [M.rails[0][0].u, Q.rails[0][0].u, reg.commit_in],
                           [DRIVE.ignite] * 3)
        loaded = np.zeros(count, bool)
        dones = np.zeros(count, int)
        mready = np.full(count, -1)
        last_m_spike = np.full(count, -1000)
        cp_gate = reg.copy_gates[0]
        copy_delay = int(topo.delay[(topo.src == cp_gate) & (topo.dst == M.rails[0][0].u)][0])
        # Inspect every step: scheduling READY+1 must not slip by a host polling interval.
        for step in range(trial_steps):
            sim.step()
            if not sim._spk_step:
                continue
            nodes, neurons = sim._spk_node[-1], sim._spk_neuron[-1]
            if step > 6000:
                dones[nodes[neurons == reg.done_relay]] += 1
                faults += int(np.isin(neurons, Q.fault + M.fault).sum())
                for b in nodes[neurons == M.ready]:
                    survivors += int(last_m_spike[b] >= step - 100)
                    mready[b] = step
                for b in nodes[neurons == cp_gate]:
                    if mready[b] >= 0:
                        min_copy_after_ready = min(min_copy_after_ready, step + copy_delay - mready[b])
                for b in nodes[neurons == Q.ready]:
                    if not loaded[b]:
                        sim.add_events(int(b), [step + 1, step + 1000], [Q.rails[0][0].u, reg.commit_in],
                                       [round(np.exp(-.12) * DRIVE.ignite) if corner else DRIVE.ignite,
                                        DRIVE.ignite])
                        loaded[b] = True
            last_m_spike[nodes[neurons == M.rails[0][0].u]] = step
            sim._spk_step.clear(); sim._spk_node.clear(); sim._spk_neuron.clear()
        failures += int(np.count_nonzero(dones != 2))
        ready_missing += int(np.count_nonzero(~loaded))
        if progress:
            print(dict(rate=rate, corner=corner, strength=strength, tail=tail, advance=advance,
                       trials=lo + count, failures=failures, ready_missing=ready_missing,
                       faults=faults, survivors=survivors), flush=True)
    return dict(trials=n, failures=failures, ready_missing=ready_missing, faults=faults,
                survivors=survivors, copy_arrival_after_ready=int(min_copy_after_ready),
                advance=advance, tail=None if verified else tail,
                strength=.75 if verified else strength, verified=verified)


@pytest.mark.parametrize("rate", [False, True])
def test_actual_q_ready_and_master_copy_have_100_step_margin(rate):
    result = _path_trials(10, rate=rate, advance=100)
    assert result["failures"] == result["ready_missing"] == result["faults"] == result["survivors"] == 0
    assert result["advance"] == 100 and 0 < result["copy_arrival_after_ready"] < 10**9


@pytest.mark.slow
@pytest.mark.parametrize("rate", [False, True])
@pytest.mark.parametrize("corner", [False, True], ids=["mix_b", "adverse_3sigma"])
def test_10000_actual_noisy_reload_paths(rate, corner):
    result = _path_trials(10000, rate=rate, corner=corner, advance=100, progress=True)
    assert result["failures"] == result["ready_missing"] == result["faults"] == result["survivors"] == 0, result
    assert result["advance"] == 100 and 0 < result["copy_arrival_after_ready"] < 10**9, result


def _primitive(case, *, compact=True, ready=False, strength=1.75, controller="nominal",
               ready_tail_links=8):
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
        compact_reset_domains(net, DRIVE, [(trigger, inh)], strength=strength,
                              ready_tail_links=ready_tail_links)
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
        bias[list(latch.members)] = CORNER_BIAS.get(case, 0.)
        ignition = round(ignition * ignite)
    if controller != "nominal":
        kind = controller.removesuffix("_bias")
        scale, threshold = {"fast": (1.12, -.6), "slow": (.88, .6),
                            "fast_ready": (np.exp(.12), -.6)}[kind]
        control = [i for i, role in enumerate(net.roles)
                   if i not in latch.members and (kind != "fast_ready" or ".ready" in role)]
        vth[control] = PARAMS.V_th + threshold
        if controller.endswith("_bias"):
            bias[control] = -threshold
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


def _reload(case, offsets=range(400, 1801, 25), *, strength=1.75, controller="nominal",
            ready_tail_links=8):
    topo, latch, trigger, inh, ready, q, vth, bias, ignition = _primitive(
        case, ready=True, strength=strength, controller=controller, ready_tail_links=ready_tail_links)
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
    assert result["margin"] >= 300 and result["ready_reloads"], (case, result)


@pytest.mark.parametrize("controller", ["fast", "slow", "fast_ready", "fast_ready_bias"])
def test_independent_controller_corners(controller):
    assert _survival("fast30", noisy=False, controller=controller) == 0
    result = _reload("slow3sigma" if controller.endswith("bias") else "slow12", controller=controller)
    print(controller, result, flush=True)
    assert result["arrivals"] == {"fast": [0, 34, 66, 111], "slow": [0, 60, 115],
                                  "fast_ready": [0, 43, 81, 138],
                                  "fast_ready_bias": [0, 43, 81, 138]}[controller]
    assert result["margin"] >= 100 and result["ready_reloads"], result


def test_regression_phases_fail_with_the_original_reset():
    assert _survival("master74", compact=False, noisy=False) == 52
    assert _survival("master24", compact=False, noisy=False) == 2


def test_compaction_is_idempotent_including_overlapping_domains():
    c = _cell(24, nominal=True)
    net = c["net"]
    controllers = [(r.reset_trigger, r.reset_inh) for r in (c["Q"], c["M"])]
    compact_reset_domains(net, DRIVE, controllers)
    first = (net.quanta.copy(), net.delay.copy())
    compact_reset_domains(net, DRIVE, controllers[::-1] + controllers[:1])
    assert (net.quanta, net.delay) == first
    compact_reset_domains(net, DRIVE, controllers[:1])
    assert (net.quanta, net.delay) == first


@pytest.mark.parametrize("hops,pulses", [(3, 4), (16, 4), (15, 3), (15, 5)])
def test_compaction_rejects_short_or_long_chains_before_mutation(hops, pulses):
    net = Netlist(PARAMS)
    latch = add_latch(net, DRIVE, "R.L")
    trigger, inh, _ = add_reset(net, DRIVE, "R", [latch], pulses=pulses)
    add_ready(net, DRIVE, "R", trigger, hops=hops)
    before = (net.quanta.copy(), net.delay.copy())
    with pytest.raises(ValueError, match="compact register reset requires"):
        compact_reset_domains(net, DRIVE, [(trigger, inh)])
    assert (net.quanta, net.delay) == before


@pytest.mark.parametrize("zero_once,request_clear", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("rate", [False, True])
@pytest.mark.parametrize("datapath", ["generic", "specialized"])
def test_multi_cell_kernel_reloads(zero_once, request_clear, rate, datapath):
    pl = _candidate_pipeline(PARAMS, 2,
                        [{"name": "sum", "op": "ADD", "a": "input", "b": ("const", "one")},
                         {"name": "out", "op": "XOR", "a": "sum", "b": ("const", "one")}],
                        consts={"one": 1}, robust_register_reset=True,
                        zero_once=zero_once, robust_request_clear=request_clear, rate_robust=rate,
                        datapath=datapath)
    outputs, sim, stats = run_pipeline(pl, PARAMS, [0, 1, 3], max_ms=15000)
    assert isinstance(sim, RefSim)
    assert [v for _, v in outputs] == [0, 3, 1]
    assert stats["faults"] == stats["timeouts"] == stats["bad_outputs"] == 0
    assert not stats["refusals"]


@pytest.mark.parametrize("copy", [74, 24], ids=["base", "rate_robust"])
@pytest.mark.parametrize("storage_bias", [0., -.2, -.6])
def test_next_copy_and_stage_word_at_ready_with_slow_storage(copy, storage_bias):
    c = _cell(copy, nominal=True, compact=True)
    topo, Q, M = c["net"].topology(), c["Q"], c["M"]
    q = c["q"].copy()
    storage = {x for reg in (Q, M) for pair in reg.rails for l in pair for x in l.members}
    ready_nodes = {x for reg in (Q, M) for x in [*reg.ready_chain, reg.ready]}
    c["vth"][list(storage)] += .6
    c["bias"][list(storage)] += storage_bias
    c["vth"][list(ready_nodes)] -= .6
    c["bias"][list(ready_nodes)] += .6
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
    # Qualify a positive 100-step margin on the actual Q-word reload, not only
    # recovery measured from the first inhibitory pulse of a reduced latch.
    next_load = int(ready[ready > COMMIT][0]) - 100
    trace = simulate(next_load)
    done = trace.neuron_steps(c["reg"].done_relay)
    done = done[done > COMMIT]
    assert len(done) == 2
    for step, word in zip(done, (c["new"], c["old"])):
        assert decode_at(trace, M.rail_taps, int(step), window=120) == (word, "valid")
    for f in Q.fault + M.fault:
        assert not len(trace.neuron_steps(f))


def test_small_noisy_multi_cell_kernel():
    pl = _candidate_pipeline(PARAMS, 2,
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
    pl = _candidate_pipeline(PARAMS, 2,
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


def _extended_domain_trial(rate, *, strength=1.75, tail=8, scope="all", noise_seed=None,
                           verified=False):
    """Real ADD transactions at combined three-sigma draws on the reset domain.

    All incoming weights (including ignition, loop, reset and mirrored edges),
    thresholds and tonic biases are perturbed together; READY is independently
    fast. This deliberately includes the rate gates that isolated rails omit.
    """
    args = (PARAMS, 2, [{"name": "out", "op": "ADD", "a": "input", "b": ("const", "one")}])
    opts = dict(consts={"one": 1}, zero_once=True, rate_robust=rate)
    pl = (build_pipeline(*args, **opts) if strength is None or verified else
          _candidate_pipeline(*args, **opts, strength=strength, tail=tail))
    if verified:
        from drosophilos.protocol.handshake import verify_register_resets
        regs = [r for sr, _ in pl.inputs.values() for r in (sr.stage, sr.master)]
        regs += [r for c in pl.cells for r in (c.stage, c.master)]
        verify_register_resets(pl.net, pl.drive, regs)
    topo, cell = pl.net.topology(), pl.cells[0]
    targets = {d for s, d, q in zip(topo.src, topo.dst, topo.quanta)
               if s in {cell.stage.reset_inh, cell.master.reset_inh} and q < 0}
    if scope != "all":
        def selected(i):
            role = pl.net.roles[i]
            if scope == "alu":
                return role.endswith((".u", ".v")) and not any(
                    role.startswith("out." + prefix) for prefix in
                    ("Q.", "M.", "act.", "commit.", "grant.", "copy.", "faultL."))
            if scope == "control":
                return any("." + prefix + "." in role for prefix in ("act", "commit", "grant", "copy"))
            return i in pl.net.rate_readouts.values()
        targets = {i for i in targets if selected(i)}
    ready = {x for r in (cell.stage, cell.master) for x in [*r.ready_chain, r.ready]}
    if verified:
        ready.update(i for i, role in enumerate(pl.net.roles)
                     if ".verify." in role and (".quiet" in role or ".recovery" in role))
    q = np.rint(topo.quanta * np.where(
        np.isin(topo.dst, list(targets)), np.where(topo.quanta > 0, np.exp(-.12), np.exp(.12)),
        np.where(np.isin(topo.dst, list(ready)), np.exp(.12), 1.))).astype(np.int64)
    vth, bias = np.full(topo.n, PARAMS.V_th), np.array(pl.net.bias)
    vth[list(targets)] += .6; bias[list(targets)] -= .6
    vth[list(ready)] -= .6; bias[list(ready)] += .6
    sim = RefSim(topo, PARAMS, quanta=q[None, :], V_th=vth, bias=bias)
    if noise_seed is not None:
        _sparse_strays(sim, np.random.default_rng(noise_seed), 50000)
    outputs, _, stats = run_pipeline(pl, PARAMS, [0, 0], sim=sim, max_ms=5000, full_trace=True)
    trace = sim.trace
    result = dict(rate=rate, strength=strength, tail=tail, scope=scope, outputs=[v for _, v in outputs],
                  stage_resets=len(trace.neuron_steps(cell.stage.reset_inh)),
                  master_resets=len(trace.neuron_steps(cell.master.reset_inh)),
                  stage_done=len(trace.neuron_steps(cell.stage.completion.u)),
                  faults=stats["faults"])
    return result


@pytest.mark.parametrize("rate", [False, True])
@pytest.mark.parametrize("scope", ["alu", "control", "mirrors"])
def test_extended_groups_rearm_at_combined_adverse_draws(rate, scope):
    result = _extended_domain_trial(rate, scope=scope)
    assert result["outputs"] == [1, 1] and result["faults"] == 0, result


@pytest.mark.parametrize("rate", [False, True])
@pytest.mark.parametrize("strength", [None, 1.75], ids=["off", "candidate"])
def test_full_domain_corner_prevents_qualification(rate, strength):
    """Do not relabel a rail-only pass as full mix-B qualification.

    Base stalls before either cell reset; RR gets through Q but never commits M.
    The same failures with the option off distinguish an existing analogue limit
    from the rejected four-link READY recovery race.
    """
    result = _extended_domain_trial(rate, strength=strength)
    assert result["outputs"] == [] and result["stage_resets"] == 0, result
    if not rate:
        assert result["master_resets"] == result["stage_done"] == 0, result
    else:
        assert result["master_resets"] > 0 and result["stage_done"] > 0, result


@pytest.mark.slow
@pytest.mark.parametrize("rate", [False, True])
def test_extended_domain_frontier(rate):
    for strength, tail in ((1.5, 4), (1.75, 4), (1.75, 8), (2., 8), (1.75, 16)):
        result = _extended_domain_trial(rate, strength=strength, tail=tail)
        print(result, flush=True)
        assert result["outputs"] == [] and result["stage_resets"] == 0, result


@pytest.mark.slow
@pytest.mark.parametrize("rate", [False, True])
def test_full_domain_corner_also_fails_with_mix_b_strays(rate):
    results = [_extended_domain_trial(rate, noise_seed=seed) for seed in range(4)]
    print(results, flush=True)
    assert all(r["outputs"] != [1, 1] for r in results), results


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
@pytest.mark.parametrize("adverse", [False, True])
def test_fault_discard_clears_fault_and_accepts_next_word(copy, adverse):
    c = _cell(copy, nominal=True, compact=True)
    Q, M = c["Q"], c["M"]
    topo = c["net"].topology()
    if adverse:
        targets = set(Q.fault_latch.members)
        ready = {x for r in (Q, M) for x in [*r.ready_chain, r.ready]}
        c["vth"][list(targets)] += .6; c["bias"][list(targets)] -= .6
        c["vth"][list(ready)] -= .6; c["bias"][list(ready)] += .6
        for e, d in enumerate(topo.dst):
            scale = (np.exp(-.12) if c["q"][e] > 0 else np.exp(.12)) if d in targets else (
                np.exp(.12) if d in ready and c["q"][e] > 0 else 1.)
            c["q"][e] = round(c["q"][e] * scale)

    def run(next_fault=None, next_load=None):
        sim = RefSim(topo, PARAMS, quanta=c["q"][None, :], V_th=c["vth"], bias=c["bias"])
        events = [(10, M.rails[i][r].u, DRIVE.ignite) for i, r in rails_for(c["old"], WIDTH)]
        events += [(LOAD, Q.rails[i][r].u, DRIVE.ignite) for i, r in rails_for(c["new"], WIDTH)]
        events.append((LOAD, Q.rails[0][1 - (c["new"] & 1)].u, DRIVE.ignite))
        if next_fault is not None:
            events += [(next_fault, Q.rails[i][r].u, DRIVE.ignite) for i, r in rails_for(c["new"], WIDTH)]
            events.append((next_fault, Q.rails[0][1 - (c["new"] & 1)].u, DRIVE.ignite))
        if next_load is not None:
            events += [(next_load, Q.rails[i][r].u, DRIVE.ignite) for i, r in rails_for(c["new"], WIDTH)]
            events.append((next_load + 4300, c["reg"].commit_in, DRIVE.ignite))
        sim.add_events(0, *zip(*events))
        watched = [Q.ready, Q.fault_latch.u, c["reg"].done_relay]
        watched += [l.u for pair in M.rails for l in pair]
        return _observed_run(sim, (next_load or next_fault or LOAD) + 10000, watched)

    first = run()
    ready = first.neuron_steps(Q.ready)
    next_fault = int(ready[ready > LOAD][0]) - 100
    second = run(next_fault=next_fault)
    ready = second.neuron_steps(Q.ready)
    next_load = int(ready[ready > next_fault + 100][0]) - 100
    trace = run(next_fault=next_fault, next_load=next_load)
    faults = trace.neuron_steps(Q.fault_latch.u)
    assert np.any((faults > LOAD) & (faults < next_fault - 100))
    assert np.any((faults > next_fault) & (faults < next_load - 100))
    assert not np.any(faults > next_load - 100)
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
                         [("fast30", c) for c in ("fast", "slow")] +
                         [(c, "slow_bias") for c in ("fast3sigma", "fast30_bias")])
def test_40000_noisy_resets(case, controller):
    survivors = _survival(case, 40000, controller=controller)
    print(dict(case=case, controller=controller, resets=40000, survivors=survivors), flush=True)
    assert survivors == 0


@pytest.mark.slow
def test_candidate_evaluation_table():
    for strength, nominal, slow in ((1.1, 76, 256), (1.25, 0, 256), (1.35, 0, 208), (1.5, 0, 0), (1.75, 0, 0)):
        a = _survival("fast30", noisy=False, strength=strength)
        b = _survival("fast30", noisy=False, strength=strength, controller="slow")
        reload = _reload("slow12", strength=strength, controller="fast_ready", ready_tail_links=4)
        print(dict(strength=strength, phases=256, nominal_controller=a, slow_controller=b, **reload), flush=True)
        assert (a, b) == (nominal, slow)
        assert reload["margin"] == {1.1: 120, 1.25: 95, 1.35: 70, 1.5: 45, 1.75: 20}[strength]


@pytest.mark.parametrize("tail,margin", [(4, -178), (6, -14), (8, 150)])
def test_biased_recovery_frontier(tail, margin):
    result = _reload("slow3sigma", controller="fast_ready_bias", ready_tail_links=tail)
    assert result["margin"] == margin
    assert result["ready_reloads"] == (margin > 0)


@pytest.mark.slow
@pytest.mark.parametrize("rate", [False, True])
@pytest.mark.parametrize("tail", [4, 6])
def test_rejected_ready_delays_on_actual_noisy_paths(rate, tail):
    result = _path_trials(1000, rate=rate, tail=tail)
    print(dict(rate=rate, **result), flush=True)
    assert result["failures"] == {(False, 4): 1000, (False, 6): 238,
                                   (True, 4): 998, (True, 6): 227}[rate, tail], result


@pytest.mark.slow
def test_rejected_1_5_strength_has_59_survivors():
    assert _survival("fast30", 40000, strength=1.5, controller="slow") == 59
