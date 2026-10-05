"""READY/IDLE qualification on the unchanged neural simulator.

The two public options remain rejected if reset qualification fails. Direct
experiments use the same final wiring pass, without a hidden builder bypass.
"""

import numpy as np
import pytest

from drosophilos.bench import stage_d
from drosophilos.lib.kernel import build_pipeline, interlock_cell_ready, run_pipeline
from drosophilos.protocol.handshake import verify_register_resets
from drosophilos.sim.ref64 import RefSim
from test_robust_reset import PARAMS


SPEC = [dict(name="sum", op="ADD", a="input", b=("const", "one")),
        dict(name="out", op="XOR", a="sum", b=("const", "one"))]


def _interlock(pl, *, verified=False):
    interlock_cell_ready(pl.net, pl.cells, idle_hops=pl.build_options["idle_hops"])
    if verified:
        registers = [r for sr, _ in pl.inputs.values() for r in (sr.stage, sr.master)]
        registers += [r for cell in pl.cells for r in (cell.stage, cell.master)]
        verify_register_resets(pl.net, pl.drive, registers)
    # A saved candidate must be rejected by production rebuild/recheck, rather
    # than being silently rebuilt with its experimental wiring omitted.
    pl.build_options.update(ready_interlock=True, verified_register_reset=verified)
    return pl


def _pipeline(*, verified=False, **options):
    return _interlock(build_pipeline(PARAMS, 2, SPEC, consts={"one": 1}, **options),
                      verified=verified)


def _first(trace, neuron, after):
    steps = trace.neuron_steps(neuron)
    following = steps[steps > after]
    assert len(following), (neuron, after)
    return int(following[0])


@pytest.mark.parametrize("rate", [False, True])
@pytest.mark.parametrize("zero_once", [False, True])
@pytest.mark.parametrize("request_clear", [False, True])
@pytest.mark.parametrize("verified", [False, True])
def test_multi_cell_results_and_four_phase_order(rate, zero_once, request_clear, verified):
    pl = _pipeline(verified=verified, rate_robust=rate, zero_once=zero_once,
                   robust_request_clear=request_clear)
    outputs, sim, stats = run_pipeline(pl, PARAMS, [0, 1, 3], max_ms=9000, full_trace=True)
    assert [value for _, value in outputs] == [0, 3, 1]
    assert stats["faults"] == stats["timeouts"] == stats["bad_outputs"] == 0
    # Drain the final clear/verification and go tails too: stopping at the last
    # output alone could hide a stuck final reset or a cached ghost START.
    sim.run(10000)
    trace = sim.trace
    retry_count = 0
    for cell in pl.cells:
        starts = trace.neuron_steps(cell.start)
        dones = trace.neuron_steps(cell.reg.done_relay)
        assert len(starts) == len(dones) == 3
        assert len(trace.neuron_steps(cell.stage.ready)) == len(trace.neuron_steps(cell.master.ready)) == 3
        assert not any(len(trace.neuron_steps(gate)) for reg in (cell.stage, cell.master) for gate in reg.fault)
        for i, start in enumerate(starts):
            # Requests are latched; go rechecks them and IDLE before issuing START.
            for pair in cell.reqs.values():
                assert np.any(trace.neuron_steps(pair[1].u) < start)
            sequence = [int(start)]
            for neuron in (pl.net.roles.index(f"{cell.name}.actd.d10"),
                           cell.stage.completion.u, cell.commit_pulse,
                           cell.reg.commit_in, cell.master.reset_trigger,
                           cell.master.ready, cell.reg.copy.u, cell.master.completion.u,
                           cell.reg.done_relay):
                sequence.append(_first(trace, neuron, sequence[-1]))
            assert sequence[-1] == dones[i]
            for neuron in (cell.stage.reset_trigger, cell.stage.ready, cell.idle[1].u):
                sequence.append(_first(trace, neuron, sequence[-1]))
            # The stage-ready tail has four pulse relays, with no DONE bypass.
            assert sequence[-1] - sequence[-2] >= 4 * 50
            if i + 1 < len(starts):
                assert sequence[-1] < starts[i + 1]
        if verified:
            v = cell.stage.verification
            retry_count += sum(len(trace.neuron_steps(entry)) for entry in v.attempts[1:])
            assert not len(trace.neuron_steps(v.exhausted.u))
    if verified:
        # Natural settling of the full reset domain exercises real neural retries.
        assert retry_count > 0


@pytest.mark.parametrize("verified", [False, True])
def test_no_timeout_bypasses_missing_ready(verified):
    pl = _pipeline(verified=verified)
    cell = pl.cells[0]
    silenced = np.zeros(pl.net.n, bool)
    silenced[cell.stage.ready] = True
    sim = RefSim(pl.net.topology(), PARAMS, silenced=silenced)
    outputs, sim, _ = run_pipeline(pl, PARAMS, [0, 1, 3], sim=sim, max_ms=8000, full_trace=True)
    assert [value for _, value in outputs] == [0]
    assert len(sim.trace.neuron_steps(cell.start)) == 1
    assert len(sim.trace.neuron_steps(cell.reg.done_relay)) == 1
    done = sim.trace.neuron_steps(cell.reg.done_relay)[0]
    assert not np.any(sim.trace.neuron_steps(cell.idle[1].u) > done)


def test_wiring_changes_only_the_ready_dependency_and_rejects_double_install():
    pl = build_pipeline(PARAMS, 2, SPEC, consts={"one": 1})
    before = (list(pl.net.roles), list(pl.net.src), list(pl.net.dst),
              list(pl.net.quanta), list(pl.net.delay), list(pl.net.bias))
    _interlock(pl)
    after = tuple(list(x) for x in (pl.net.roles, pl.net.src, pl.net.dst,
                                   pl.net.quanta, pl.net.delay, pl.net.bias))
    for i in (0, 2, 3, 4, 5):
        assert before[i] == after[i]
    changed = [i for i, (a, b) in enumerate(zip(before[1], after[1])) if a != b]
    assert len(changed) == len(pl.cells)
    for cell, edge in zip(pl.cells, changed):
        assert pl.net.src[edge] == cell.stage.ready
        assert pl.net.roles[pl.net.dst[edge]] == f"{cell.name}.idled.d16"
        assert pl.net.roles[before[1][edge]] == f"{cell.name}.idled.d15"
    with pytest.raises(ValueError, match="unmodified IDLE"):
        _interlock(pl)
    assert after == (pl.net.roles, pl.net.src, pl.net.dst, pl.net.quanta, pl.net.delay, pl.net.bias)


def _full_domain_corner(rate, *, interlock=False, verified=False):
    """Same complete three-sigma domain as the previous reset frontier, including baseline."""
    pl = build_pipeline(PARAMS, 2, SPEC[:1], consts={"one": 1}, zero_once=True, rate_robust=rate)
    if interlock:
        _interlock(pl, verified=verified)
    topo, cell = pl.net.topology(), pl.cells[0]
    targets = {d for s, d, q in zip(topo.src, topo.dst, topo.quanta)
               if s in {cell.stage.reset_inh, cell.master.reset_inh} and q < 0}
    ready = {x for r in (cell.stage, cell.master) for x in [*r.ready_chain, r.ready]}
    ready.update(i for i, role in enumerate(pl.net.roles)
                 if ".verify." in role and (".quiet" in role or ".recovery" in role))
    q = np.rint(topo.quanta * np.where(
        np.isin(topo.dst, list(targets)), np.where(topo.quanta > 0, np.exp(-.12), np.exp(.12)),
        np.where(np.isin(topo.dst, list(ready)), np.exp(.12), 1.))).astype(np.int64)
    vth, bias = np.full(topo.n, PARAMS.V_th), np.array(pl.net.bias)
    vth[list(targets)] += .6
    bias[list(targets)] -= .6
    vth[list(ready)] -= .6
    bias[list(ready)] += .6
    sim = RefSim(topo, PARAMS, quanta=q[None, :], V_th=vth, bias=bias)
    outputs, sim, stats = run_pipeline(pl, PARAMS, [0, 0], sim=sim, max_ms=5000, full_trace=True)
    return dict(rate=rate, interlock=interlock, verified=verified, outputs=[v for _, v in outputs],
                stage_resets=len(sim.trace.neuron_steps(cell.stage.reset_trigger)),
                master_resets=len(sim.trace.neuron_steps(cell.master.reset_trigger)),
                stage_done=len(sim.trace.neuron_steps(cell.stage.completion.u)), faults=stats["faults"])


@pytest.mark.parametrize("rate", [False, True])
def test_full_domain_corner_is_also_a_baseline_failure(rate):
    rows = [_full_domain_corner(rate, interlock=i, verified=v)
            for i, v in ((False, False), (True, False), (True, True))]
    print(rows, flush=True)
    for row in rows:
        assert row["outputs"] == [] and row["stage_resets"] == 0, row
        if not rate:
            assert row["master_resets"] == row["stage_done"] == 0, row
        else:
            assert row["master_resets"] > 0 and row["stage_done"] > 0, row


def _tick_latency(rate=False, *, interlock=False, verified=False, ticks=3):
    k = stage_d.load_kernel(stage_d.PROGRAM)
    pl = stage_d.build(k, PARAMS, rate_robust=rate)
    if interlock:
        _interlock(pl, verified=verified)
    tokens = [5, 5, 250, 3, 0, 40, 40, 40][:ticks]
    _, _, stats = run_pipeline(pl, PARAMS, tokens, max_ms=60000)
    comparison = stage_d.compare_copy(
        k, stage_d.reference_states(k, tokens), tokens, stats["outputs_by_cell"],
        t_load0=stats["load_steps"][0], dt=PARAMS.dt, load_events=stats["load_events"])
    times = comparison["tick_ms"]
    assert comparison["status"] == "matched" and comparison["matched"] == len(tokens), comparison
    assert stats["faults"] == stats["timeouts"] == stats["bad_outputs"] == 0
    return dict(rate=rate, interlock=interlock, verified=verified, ticks=len(times), tick_ms=times,
                first_tick_ms=times[0], per_tick_ms=float(np.mean(np.diff(times))))


@pytest.mark.slow
@pytest.mark.parametrize("rate", [False, True])
@pytest.mark.parametrize("interlock,verified", [(False, False), (True, False), (True, True)])
def test_tick2_latency_and_feedback_liveness(rate, interlock, verified):
    print(_tick_latency(rate, interlock=interlock, verified=verified), flush=True)
