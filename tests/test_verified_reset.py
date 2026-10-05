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
from drosophilos.lib.kernel import build_pipeline, run_pipeline
from drosophilos.lib.netlist import Netlist
from drosophilos.protocol.handshake import add_register, verify_register_resets
from drosophilos.protocol.latch import (VERIFY_MAX_RETRIES, add_latch, add_ready, add_reset,
                                       add_reset_verification)
from drosophilos.sim.ref64 import RefSim
from test_robust_reset import (CAPTURES, CORNERS, DRIVE, PARAMS, _extended_domain_trial,
                               _observed_run, _path_trials, _primitive, _sparse_strays)


def _verified_primitive(case="nominal", controller="nominal"):
    """Keep the captured old nodes/edges; new detector nodes have explicit draws."""
    before, latch, trigger, inh, ready, q0, v0, b0, _ = _primitive(
        case, compact=False, ready=True, controller=controller)
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
    if controller != "nominal":
        fast = controller.startswith("fast")
        vth[before.n:] += -.6 if fast else .6
        if controller.endswith("bias"):
            bias[before.n:] += .6 if fast else -.6
        q[new_edges & (q > 0)] = np.rint(q[new_edges & (q > 0)] * (1.12 if fast else .88))
        # A fast certificate with weak veto is the detector's adverse corner.
        q[new_edges & (q < 0)] = np.rint(q[new_edges & (q < 0)] * (.88 if fast else 1.12))
    return net, latch, trigger, inh, ready, verification, q, vth, bias, before.n, new_edges


def _verification_survey(case, n=32, *, controller="nominal", noisy=False, seed=108):
    net, latch, trigger, inh, ready, v, q, vth, bias, old_n, new_edges = _verified_primitive(case, controller)
    topo = net.topology()
    rng = np.random.default_rng(seed)
    result = dict(case=case, controller=controller, trials=n, ready=0, false_ready=0,
                  exhausted=0, undecided=0, terminal_survivors=0, max_attempt_spikes=0,
                  attempts=[0] * 4, common_ms=[], retry_ms=[], exhausted_ms=[],
                  max_latch_period_ms=0., min_window_ms=None)
    recovery_ends = [i for i, role in enumerate(net.roles)
                     if role.endswith((".ready_delay14", ".recovery14"))]
    for lo in range(0, n, 250):
        count = min(250, n - lo)
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
                result["false_ready"] += int(len(decisions) != 1 or np.any(live >= decisions[0] - 100))
                start = t.neuron_steps(trigger)[0]
                result["common_ms" if sum(c > 0 for c in counts) == 1 else "retry_ms"].append(
                    (int(decisions[0]) - int(start)) * PARAMS.dt)
                for end, check in zip(recovery_ends, v.checks):
                    end_times, check_times = t.neuron_steps(end), t.neuron_steps(check)
                    if len(end_times) and len(check_times):
                        window = float(check_times[0] - end_times[0]) * PARAMS.dt
                        old_window = result["min_window_ms"]
                        result["min_window_ms"] = window if old_window is None else min(window, old_window)
    for key in ("common_ms", "retry_ms", "exhausted_ms"):
        values = result[key]
        result[key] = (dict(count=len(values), quantiles=np.quantile(values, [0, .5, .95, 1]).tolist())
                       if values else dict(count=0, quantiles=[]))
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
    result = _path_trials(10, verified=True, rate=rate)
    assert result["failures"] == result["ready_missing"] == result["faults"] == result["survivors"] == 0, result


@pytest.mark.slow
@pytest.mark.parametrize("rate", [False, True])
def test_10000_verified_actual_noisy_reload_paths(rate):
    result = _path_trials(10000, rate=rate, verified=True, progress=True)
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
