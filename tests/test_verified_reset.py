"""Closed-loop reset qualification, including the counterexamples blocking release.

The prototype is an actual neural netlist running on unmodified RefSim. A host
observer only records its decisions; it never supplies READY, retry or FAULT.
The public verified option is rejected. The timed fallback is explicitly
experimental and is exercised through its public flag in test_robust_reset.py.
"""

from dataclasses import replace

import numpy as np
import pytest

from drosophilos.bench import kernel_campaign, stage_d
from drosophilos.bench.stall_diag import build_tick_pipeline
from drosophilos.lib.kernel import _NoProducer, _fault_latch, build_pipeline, run_pipeline
from drosophilos.lib.netlist import Netlist
from drosophilos.lib.staged import add_staged_commit
from drosophilos.protocol.handshake import add_register, verify_register_resets
from drosophilos.protocol.latch import (VERIFY_MAX_RETRIES, add_latch, add_ready, add_reset,
                                       add_reset_verification)
from drosophilos.sim.ref64 import RefSim
from test_robust_reset import (CAPTURES, CORNERS, DRIVE, PARAMS, _extended_domain_trial,
                               _observed_run, _primitive, _sparse_strays)


def _verified_primitive(case="nominal", controller="nominal"):
    """Keep the captured old nodes/edges; new detector nodes have explicit draws."""
    exact = controller == "fast_exact_bias"
    before, latch, trigger, inh, ready, q0, v0, b0, _ = _primitive(
        case, compact=False, ready=True, controller="nominal" if exact else controller)
    _, _, name, rail = CAPTURES.get(case, (None, False, "R", "L"))
    net = Netlist(PARAMS)
    add_latch(net, DRIVE, f"{name}.{rail}")
    add_reset(net, DRIVE, name, [latch])
    add_ready(net, DRIVE, name, trigger, hops=15)
    chain = [i for i, role in enumerate(net.roles) if ".ready_delay" in role]
    verification = add_reset_verification(net, DRIVE, name, trigger, inh, ready, chain)
    topo = net.topology()
    old = {(int(s), int(d)): q for s, d, q in zip(before.src, before.dst, q0)}
    q = topo.quanta.astype(np.int64)
    new_edges = np.ones(topo.nnz, bool)
    for e, (s, d) in enumerate(zip(topo.src, topo.dst)):
        if (s, d) in old:
            q[e], new_edges[e] = old[s, d], False
        elif s == chain[-1] and net.roles[d].endswith(".a0.quiet0"):
            q[e], new_edges[e] = old[s, ready], False
    vth = np.r_[v0, np.full(net.n - before.n, PARAMS.V_th)]
    bias = np.r_[b0, np.zeros(net.n - before.n)]
    if exact:
        # Exact three-sigma log-normal factors, alongside the original survey's
        # rounded +/-12% controller. Keep the storage corner's existing draws.
        controls = np.ones(topo.n, bool)
        controls[list(latch.members)] = False
        vth[controls] -= .6
        bias[controls] += .6
        edges = controls[topo.dst]
        q[edges] = np.rint(q[edges] * np.where(q[edges] > 0, np.exp(.12), np.exp(-.12)))
    elif controller != "nominal":
        fast = controller.startswith("fast")
        vth[before.n:] += -.6 if fast else .6
        if controller.endswith("bias"):
            bias[before.n:] += .6 if fast else -.6
        q[new_edges & (q > 0)] = np.rint(q[new_edges & (q > 0)] * (1.12 if fast else .88))
        # A fast certificate with weak veto is the detector's adverse corner.
        q[new_edges & (q < 0)] = np.rint(q[new_edges & (q < 0)] * (.88 if fast else 1.12))
    return net, latch, trigger, inh, ready, verification, q, vth, bias, before.n, new_edges


def _verification_survey(case, n=32, *, controller="nominal", noisy=False, seed=108,
                         batch_size=1000, progress=False):
    net, latch, trigger, inh, ready, v, q, vth, bias, old_n, new_edges = _verified_primitive(case, controller)
    topo = net.topology()
    rng = np.random.default_rng(seed)
    result = dict(case=case, controller=controller, trials=n, ready=0, false_ready=0,
                  exhausted=0, undecided=0, terminal_survivors=0, max_attempt_spikes=0,
                  duplicate_ready=0, live_at_ready=0, trigger_doublets=0, retry_doublets=0,
                  first_failure=None,
                  attempts=[0] * 4, common_ms=[], retry_ms=[], exhausted_ms=[],
                  max_latch_period_ms=0., min_window_ms=None)
    recovery_ends = [i for i, role in enumerate(net.roles)
                     if role.endswith((".ready_delay14", ".recovery14"))]
    for lo in range(0, n, batch_size):
        count = min(batch_size, n - lo)
        resets = rng.integers(3000, 4500, count) if noisy else 3000 + np.arange(lo, lo + count)
        quanta = np.broadcast_to(q, (count, topo.nnz)).copy()
        thresholds = np.broadcast_to(vth, (count, topo.n)).copy()
        biases = np.broadcast_to(bias, (count, topo.n)).copy()
        if noisy and controller == "nominal":
            # Original captured/corner parameters remain fixed. Only the newly
            # introduced verification circuit receives independent static draws.
            quanta[:, new_edges] = np.rint(quanta[:, new_edges] * np.exp(
                rng.normal(0, .04, (count, int(new_edges.sum())))))
            thresholds[:, old_n:] += rng.normal(0, .2, (count, topo.n - old_n))
            biases[:, old_n:] += rng.normal(0, .2, (count, topo.n - old_n))
        sim = RefSim(topo, PARAMS, n_nodes=count, quanta=quanta, V_th=thresholds, bias=biases)
        steps = int(resets.max()) + 10500
        if noisy:
            _sparse_strays(sim, rng, steps)
        for b, reset in enumerate(resets):
            sim.add_events(b, [10, 11, 14, 23, 2520, int(reset)],
                           [latch.u] * 4 + [latch.v if b % 2 else latch.u, trigger],
                           [DRIVE.ignite] * 4 + [300, DRIVE.relay_in])
        trace = _observed_run(sim, steps, [*latch.members, ready, *v.attempts, *v.checks,
                                         *recovery_ends, v.exhausted.u])
        for b, reset in enumerate(resets):
            t = trace.node(b)
            decisions = t.neuron_steps(ready)
            failures = t.neuron_steps(v.exhausted.u)
            counts = [len(t.neuron_steps(a)) for a in v.attempts]
            result["max_attempt_spikes"] = max(result["max_attempt_spikes"], max(counts))
            result["trigger_doublets"] += int(counts[0] > 1)
            result["retry_doublets"] += int(any(c > 1 for c in counts[1:]))
            for i, spikes in enumerate(counts):
                result["attempts"][i] += int(spikes > 0)
            result["ready"] += int(bool(len(decisions)))
            result["exhausted"] += int(bool(len(failures)))
            result["undecided"] += int(not len(decisions) and not len(failures))
            pre = t.neuron_steps(latch.u)
            pre = pre[(pre > 1000) & (pre < 2500)]
            if len(pre) > 1:
                result["max_latch_period_ms"] = max(result["max_latch_period_ms"],
                                                     float(np.diff(pre).max()) * PARAMS.dt)
            live = np.r_[t.neuron_steps(latch.u), t.neuron_steps(latch.v)]
            result["terminal_survivors"] += int(not len(decisions) and np.any(live > reset + 9000))
            if len(failures):
                result["exhausted_ms"].append((int(failures[0]) - int(t.neuron_steps(trigger)[0])) * PARAMS.dt)
            if len(decisions):
                duplicate = len(decisions) != 1
                unsafe = bool(np.any(live >= decisions[0] - 100))
                result["duplicate_ready"] += int(duplicate)
                result["live_at_ready"] += int(unsafe)
                result["false_ready"] += int(duplicate or unsafe)
                if (duplicate or unsafe or max(counts) > 1) and result["first_failure"] is None:
                    result["first_failure"] = dict(trial=lo + b, reset=int(reset),
                                                   ready=decisions.tolist(),
                                                   attempts=[t.neuron_steps(a).tolist() for a in v.attempts],
                                                   last_latch_spike=int(live.max()))
                start = t.neuron_steps(trigger)[0]
                result["common_ms" if sum(c > 0 for c in counts) == 1 else "retry_ms"].append(
                    (int(decisions[0]) - int(start)) * PARAMS.dt)
                for end, check in zip(recovery_ends, v.checks):
                    end_times, check_times = t.neuron_steps(end), t.neuron_steps(check)
                    if len(end_times) and len(check_times):
                        window = float(check_times[0] - end_times[0]) * PARAMS.dt
                        old_window = result["min_window_ms"]
                        result["min_window_ms"] = window if old_window is None else min(window, old_window)
        if progress:
            print({key: value for key, value in result.items()
                   if key not in ("common_ms", "retry_ms", "exhausted_ms")} | {"completed": lo + count},
                  flush=True)
    for key in ("common_ms", "retry_ms", "exhausted_ms"):
        values = result[key]
        result[key] = (dict(count=len(values), quantiles=np.quantile(values, [0, .5, .95, 1]).tolist())
                       if values else dict(count=0, quantiles=[]))
    return result


def _coalesced_strays(sim, rng, steps):
    """Exactly _sparse_strays' draws, without millions of tiny event arrays.

    Keep one array per column/node, then group by time. Integer delivery is
    unchanged; only the external schedule's storage differs. This bounds the
    large reload surveys' setup memory as well as their observation memory.
    """
    columns = [[], [], []]
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
        for column, values in zip(columns, (times, np.full(len(times), node), neurons)):
            column.append(np.asarray(values, dtype=np.int64))
    times, nodes, neurons = [np.concatenate(c) for c in columns]
    order = np.argsort(times, kind="stable")
    times, nodes, neurons = times[order], nodes[order], neurons[order]
    unique, starts = np.unique(times, return_index=True)
    ends = np.r_[starts[1:], len(times)]
    quanta = np.full(len(times), 150, dtype=np.int64)
    for step, lo, hi in zip(unique, starts, ends):
        sim._events[int(step)].append((nodes[lo:hi], neurons[lo:hi], quanta[lo:hi]))


def test_coalesced_strays_preserve_every_external_event():
    net, *_ = _verified_primitive()
    old, new = [RefSim(net.topology(), PARAMS, n_nodes=3) for _ in range(2)]
    _sparse_strays(old, np.random.default_rng(108), 10000)
    _coalesced_strays(new, np.random.default_rng(108), 10000)
    assert old._events.keys() == new._events.keys()
    for step in old._events:
        for left, right in zip(zip(*old._events[step]), zip(*new._events[step])):
            np.testing.assert_array_equal(np.concatenate(left), np.concatenate(right))


def _verified_reload_trials(n, *, rate=False, seed=108, progress=False, batch_size=250):
    """Actual Q READY+1 reload and M READY->COPY paths at the adverse rail corner.

    Same two-commit experiment as test_robust_reset._path_trials(verified=True).
    Coalesce simultaneous external events as _observed_run does; RefSim's
    integer delivery, integration, strays and 32,000-step horizon are unchanged.
    """
    drive = replace(DRIVE, rate_robust=rate)
    net = Netlist(PARAMS)
    Q = add_register(net, drive, "cell.Q", 1, with_completion=True)
    _fault_latch(net, drive, "cell", Q)
    reg = add_staged_commit(net, drive, "cell", Q, _NoProducer(), ordered_grant=True,
                            copy_requires_rail=True)
    M = reg.master
    verify_register_resets(net, drive, [Q, M])
    topo = net.topology()
    rails = {x for r in (Q, M) for pair in r.rails for latch in pair for x in latch.members}
    ready = {x for r in (Q, M) for x in [*r.ready_chain, r.ready]}
    ready.update(i for i, role in enumerate(net.roles)
                 if ".verify." in role and (".quiet" in role or ".recovery" in role))
    q = np.rint(topo.quanta * np.where(
        np.isin(topo.dst, list(rails)), np.where(topo.quanta > 0, np.exp(-.12), np.exp(.12)),
        np.where(np.isin(topo.dst, list(ready)) & (topo.quanta > 0), np.exp(.12), 1.)))
    vth, bias = np.full(net.n, PARAMS.V_th), np.array(net.bias)
    vth[list(rails)] += .6
    bias[list(rails)] -= .6
    vth[list(ready)] -= .6
    bias[list(ready)] += .6
    rng = np.random.default_rng(seed)
    result = dict(trials=n, rate=rate, failures=0, ready_missing=0, faults=0, survivors=0,
                  copy_arrival_after_ready=10**9)
    cp = reg.copy_gates[0]
    copy_delay = int(topo.delay[(topo.src == cp) & (topo.dst == M.rails[0][0].u)][0])
    for lo in range(0, n, batch_size):
        count = min(batch_size, n - lo)
        sim = RefSim(topo, PARAMS, n_nodes=count,
                     quanta=np.broadcast_to(q, (count, topo.nnz)), V_th=vth, bias=bias)
        _coalesced_strays(sim, rng, 32000)
        for b in range(count):
            sim.add_events(b, [10, 3000, 6000], [M.rails[0][0].u, Q.rails[0][0].u, reg.commit_in],
                           [DRIVE.ignite] * 3)
        for step, groups in sim._events.items():
            if len(groups) > 1:
                sim._events[step] = [tuple(np.concatenate(c) for c in zip(*groups))]
        loaded, dones = np.zeros(count, bool), np.zeros(count, int)
        mready, last_m = np.full(count, -1), np.full(count, -1000)
        for step in range(32000):
            sim.step()
            if not sim._spk_step:
                continue
            nodes, neurons = sim._spk_node[-1], sim._spk_neuron[-1]
            if step > 6000:
                dones[nodes[neurons == reg.done_relay]] += 1
                result["faults"] += int(np.isin(neurons, Q.fault + M.fault).sum())
                for b in nodes[neurons == M.ready]:
                    result["survivors"] += int(last_m[b] >= step - 100)
                    mready[b] = step
                for b in nodes[neurons == cp]:
                    if mready[b] >= 0:
                        result["copy_arrival_after_ready"] = min(
                            result["copy_arrival_after_ready"], int(step + copy_delay - mready[b]))
                for b in nodes[neurons == Q.ready]:
                    if not loaded[b]:
                        sim.add_events(int(b), [step + 1, step + 1000], [Q.rails[0][0].u, reg.commit_in],
                                       [round(np.exp(-.12) * DRIVE.ignite), DRIVE.ignite])
                        loaded[b] = True
            last_m[nodes[neurons == M.rails[0][0].u]] = step
            sim._spk_step.clear()
            sim._spk_node.clear()
            sim._spk_neuron.clear()
        result["failures"] += int(np.count_nonzero(dones != 2))
        result["ready_missing"] += int(np.count_nonzero(~loaded))
        if progress:
            print(result | {"completed": lo + count}, flush=True)
    return result


def test_silence_certificate_and_bounded_terminal_fault():
    net = Netlist(PARAMS)
    reg = add_register(net, DRIVE, "R", 1, False)
    fault = add_latch(net, DRIVE, "fault")
    reg.fault_latch = fault
    verify_register_resets(net, DRIVE, [reg])
    v = reg.verification
    assert len(v.attempts) == VERIFY_MAX_RETRIES + 1 == 4
    assert v.added_neurons == 104 and v.added_synapses == len(v.domain) + 153
    topo = net.topology()
    q = np.broadcast_to(topo.quanta, (2, topo.nnz)).copy()
    q[1, topo.src == reg.reset_inh] = 0  # a guaranteed survivor must NEVER get READY
    sim = RefSim(topo, PARAMS, n_nodes=2, quanta=q)
    for b in (0, 1):
        sim.add_events(b, [10, 3000], [reg.rails[0][0].u, reg.reset_trigger],
                       [DRIVE.ignite, DRIVE.relay_in])
    sim.add_events(1, [12000], [reg.reset_trigger], [DRIVE.relay_in])
    trace = _observed_run(sim, 14500, [reg.ready, *v.attempts, v.exhausted.u, fault.u])
    assert trace.neuron_steps(reg.ready, 0).tolist() == [4451]
    assert not len(trace.neuron_steps(reg.ready, 1))
    for i, entry in enumerate(v.attempts):
        assert len(trace.neuron_steps(entry, 0)) == int(i == 0)
        assert len(trace.neuron_steps(entry, 1)) == 1
    assert len(trace.neuron_steps(v.exhausted.u, 1)) and len(trace.neuron_steps(fault.u, 1))
    assert not len(trace.neuron_steps(v.exhausted.u, 0))


@pytest.mark.parametrize("case", [*CAPTURES, *CORNERS])
def test_silence_verification_at_captured_draws_and_corners(case):
    result = _verification_survey(case)
    print(result, flush=True)
    assert result["false_ready"] == result["undecided"] == 0, result
    assert result["max_attempt_spikes"] == 1, result
    assert result["ready"] + result["exhausted"] == result["trials"], result
    if result["ready"]:
        assert result["min_window_ms"] > 5 * result["max_latch_period_ms"], result


@pytest.mark.slow
@pytest.mark.parametrize("case,controller", [(c, "nominal") for c in [*CAPTURES, *CORNERS]] +
                         [("fast3sigma", "fast_bias"), ("fast30_bias", "slow_bias")])
def test_40000_verified_reset_survey(case, controller):
    result = _verification_survey(case, 40000, controller=controller, noisy=True)
    print(result, flush=True)
    # This is a safety screen, not a claim of 40,000 successful resets: exhausted
    # trials and survivors at exhaustion are reported separately and fail-stop.
    assert result["false_ready"] == 0, result
    assert result["max_attempt_spikes"] <= 1, result
    if case in CAPTURES:
        assert result["ready"] == result["trials"], result
        assert result["exhausted"] == result["undecided"] == result["terminal_survivors"] == 0, result
        assert result["min_window_ms"] > 5 * result["max_latch_period_ms"], result


@pytest.mark.parametrize("case,controller,survivors", [
    ("fast20", "nominal", 31), ("fast30_bias", "slow_bias", 32)])
def test_entrained_survivors_still_prevent_release(case, controller, survivors):
    result = _verification_survey(case, 32, controller=controller, noisy=True)
    assert result["terminal_survivors"] == result["exhausted"] == survivors, result
    assert result["false_ready"] == result["undecided"] == 0, result
    assert result["attempts"] == [32] * 4, result


@pytest.mark.parametrize("controller,first,last", [("fast_bias", 4597, 5497),
                                                  ("fast_exact_bias", 4596, 5465)])
def test_three_sigma_controller_can_emit_a_ready_train(controller, first, last):
    result = _verification_survey("fast3sigma", 32, controller=controller, noisy=True, seed=112)
    assert result["ready"] == 32 and result["false_ready"] == result["duplicate_ready"] == 1, result
    assert result["live_at_ready"] == result["exhausted"] == result["undecided"] == 0, result
    assert result["max_attempt_spikes"] == 2 and result["trigger_doublets"] == 1, result
    failure = result["first_failure"]
    assert failure["trial"] == 15 and failure["attempts"][0] == [3410, 3535], result
    assert len(failure["ready"]) == 24, result
    assert (failure["ready"][0], failure["ready"][-1], failure["last_latch_spike"]) == (first, last, 3528)


@pytest.mark.parametrize("rate", [False, True])
def test_full_domain_corner_prevents_verified_release(rate):
    result = _extended_domain_trial(rate, verified=True)
    assert result["outputs"] == [] and result["stage_resets"] == 0, result
    if not rate:
        # A reset-only circuit cannot repair failure before its first input.
        assert result["master_resets"] == result["stage_done"] == 0, result
    else:
        assert result["master_resets"] > 0 and result["stage_done"] > 0, result


@pytest.mark.parametrize("rate", [False, True])
def test_verified_actual_reload_is_observed_through_retry_windows(rate):
    result = _verified_reload_trials(10, rate=rate)
    assert result["failures"] == result["ready_missing"] == result["faults"] == result["survivors"] == 0, result
    # Check the optimized observation against the original actual-path harness.
    from test_robust_reset import _path_trials
    original = _path_trials(10, rate=rate, verified=True)
    for key in ("failures", "ready_missing", "faults", "survivors", "copy_arrival_after_ready"):
        assert result[key] == original[key]


@pytest.mark.slow
@pytest.mark.parametrize("rate", [False, True])
def test_10000_verified_actual_noisy_reload_paths(rate):
    result = _verified_reload_trials(10000, rate=rate, progress=True)
    assert result["failures"] == result["ready_missing"] == result["faults"] == result["survivors"] == 0, result


@pytest.mark.parametrize("rate", [False, True])
@pytest.mark.parametrize("other_options", [False, True])
def test_prototype_needs_a_stage_ready_interlock(rate, other_options):
    pl = build_pipeline(PARAMS, 2,
                        [{"name": "sum", "op": "ADD", "a": "input", "b": ("const", "one")},
                         {"name": "out", "op": "XOR", "a": "sum", "b": ("const", "one")}],
                        consts={"one": 1}, rate_robust=rate, zero_once=other_options,
                        robust_request_clear=other_options)
    regs = [r for sr, _ in pl.inputs.values() for r in (sr.stage, sr.master)]
    regs += [r for cell in pl.cells for r in (cell.stage, cell.master)]
    original_edges = list(zip(pl.net.src, pl.net.dst, pl.net.quanta, pl.net.delay))
    original_targets = {r.reset_inh: {d for s, d, q, _ in original_edges
                                    if s == r.reset_inh and q < 0} for r in regs}
    verify_register_resets(pl.net, pl.drive, regs)
    for r in regs:
        assert set(r.verification.domain) == original_targets[r.reset_inh]
        assert r.verification.added_neurons == 104
        assert r.verification.added_synapses == len(r.verification.domain) + 152 + int(r.fault_latch is not None)
        # Actual full fan-out remains the ordinary 0.75 train, including mirrors.
        assert [(s, d, q, delay) for s, d, q, delay in zip(
            pl.net.src, pl.net.dst, pl.net.quanta, pl.net.delay) if s == r.reset_inh] == [
                edge for edge in original_edges if edge[0] == r.reset_inh]
    outputs, sim, _ = run_pipeline(pl, PARAMS, [0, 1, 3], max_ms=8000, full_trace=True)
    assert [value for _, value in outputs] == [0]  # expected complete result is [0, 3, 1]
    cell = pl.cells[0]
    starts = sim.trace.neuron_steps(cell.start)
    ready = sim.trace.neuron_steps(cell.stage.ready)
    assert len(starts) >= 2 and starts[1] < ready[0]
    assert any(np.any(sim.trace.neuron_steps(entry) > starts[1])
               for entry in cell.stage.verification.attempts[1:])


@pytest.mark.parametrize("entry", ["kernel", "stage_d", "campaign", "recheck", "diagnostic"])
def test_verified_option_is_never_silently_enabled_or_ignored(entry):
    with pytest.raises(ValueError, match="verified_register_reset is not qualified"):
        if entry == "kernel":
            kernel_campaign.block("tick", PARAMS, verified_register_reset=True)
        elif entry == "stage_d":
            stage_d.main(["--verified-register-reset", "--ticks", "1", "--c-ticks", "0"])
        elif entry == "campaign":
            kernel_campaign.main(["tick", "--verified-register-reset", "--copies", "1"])
        elif entry == "recheck":
            stage_d.recheck_record({"build_options": {"verified_register_reset": True}})
        else:
            _, pl, _, _ = kernel_campaign.block("tick", PARAMS)
            build_tick_pipeline({"block": "tick", **pl.build_options, "verified_register_reset": True})


@pytest.mark.parametrize("option", ["verified_register_reset", "experimental_register_reset"])
@pytest.mark.parametrize("options", [{"idle_hops": 19}, {"copy_requires_rail": False},
                                     {"retry_clear": True}, {"streams": ["other"]},
                                     {"drive": replace(DRIVE, kill_strength=1.)}])
def test_new_reset_options_reject_unsupported_policies(option, options):
    with pytest.raises(ValueError, match=option + " requires"):
        build_pipeline(PARAMS, 2, [{"name": "out", "op": "ADD", "a": "input", "b": ("const", "k")}],
                       consts={"k": 1}, **{option: True}, **options)


@pytest.mark.parametrize("other", ["robust_register_reset", "verified_register_reset"])
def test_reset_modes_are_mutually_exclusive(other):
    with pytest.raises(ValueError, match="mutually exclusive"):
        kernel_campaign.block("tick", PARAMS, experimental_register_reset=True, **{other: True})
