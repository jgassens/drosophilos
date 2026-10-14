"""Master recovery candidate: separate opt-in cluster jobs and laptop smokes.

RefSim and MixProbe retain the existing simulator, real four-tap reset, real
READY chain and required-rail COPY arms. They do not prescribe storage spikes.
"""

from dataclasses import replace
import os
from time import perf_counter, process_time

import numpy as np
import pytest

from drosophilos.lib.netlist import Drive, Netlist
from drosophilos.protocol.celement import add_veto_relay
from drosophilos.protocol.latch import (
    MASTER_IGNITION_PARTNER_DELAY_STEPS, MASTER_IGNITION_VERSION,
    add_experimental_autapses, add_latch, add_ready, add_reset,
)
from drosophilos.protocol.rate import rate_tap
from drosophilos.sim.model import Params, Topology
from drosophilos.sim.ref64 import RefSim
from test_fast_latch import MixProbe


P = Params()
D = replace(Drive.from_params(P), rate_robust=True)
CASES = ("nominal", "weak_loop", "fast3sigma", "slow3sigma", "high_threshold", "negative_bias", "mix_b",
         "independent3sigma", "copy75")
INITIALS = (pytest.param("empty", id="first-load"), pytest.param("reload", id="reload"))
CLEAR_CASES = ("request28", "master74", "master17", "master24", "stage18",
               "fast3sigma", "slow3sigma", "mix_b")
CLUSTER = pytest.mark.skipif(os.environ.get("DROSO_IGNITION_CLUSTER") != "1",
                             reason="ignition cluster evidence requires DROSO_IGNITION_CLUSTER=1")
SMOKE = pytest.mark.skipif(os.environ.get("DROSO_IGNITION_SMOKE") != "1",
                           reason="ignition laptop smoke requires DROSO_IGNITION_SMOKE=1")


@pytest.mark.parametrize("block", ["tick", "render", "perspective"])
def test_configuration_f_cost_and_only_master_ignition_changes(block):
    from collections import Counter
    from drosophilos.bench.kernel_campaign import block as build
    from drosophilos.bench.stall_diag import build_tick_pipeline
    options = dict(rate_robust=True, experimental_autapse=True, ordered_xor_oneshot=True,
                   f_width16_experimental=block == "perspective")
    _, old, _, _ = build(block, P, **options)
    _, new, _, _ = build(block, P, master_ignition=True, **options)
    assert old.net.n == new.net.n
    assert old.net.roles == new.net.roles and old.net.bias == new.net.bias
    registers = [reg for reg, _ in new.inputs.values()] + [c.reg for c in new.cells]
    expected = Counter((arm, latch.v, D.ignite, MASTER_IGNITION_PARTNER_DELAY_STEPS)
                       for reg in registers for arm, latch in
                       zip(reg.copy_gates, [l for pair in reg.master.rails for l in pair]))
    def edges(net):
        return Counter(zip(net.src, net.dst, net.quanta, net.delay))
    assert edges(new.net) == edges(old.net) + expected
    assert new.net.nnz - old.net.nnz == sum(expected.values())
    _, _, rebuilt = build_tick_pipeline(dict(block=block, **new.build_options))
    for field in ("roles", "src", "dst", "quanta", "bias", "delay"):
        assert getattr(new.net, field) == getattr(rebuilt.net, field)
    assert new.build_options["master_ignition_version"] == MASTER_IGNITION_VERSION


def test_copy75_partner_handoff_at_reconstructed_draws_with_ignition():
    # Reuse the diagnostic's frozen reconstruction and prove its adverse control.
    from test_stall_diag import test_copy75_partner_handoff_at_reconstructed_draws
    for reset in (False, True):
        test_copy75_partner_handoff_at_reconstructed_draws(reset)
    sim = RefSim(Topology.empty(1), P, V_th=[-44.42221141037541],
                 bias=[-0.7152384644970224])
    sim.V[:] = P.E_L - .7152384644970224
    sim.add_events(0, [249, 293, 337, 384, 457], [0] * 5, [-2707] * 5)
    sim.add_events(0, [1182 + MASTER_IGNITION_PARTNER_DELAY_STEPS - 18, 1226],
                   [0, 0], [round(D.ignite * np.exp(-.12)), 3281])
    sim.run(3200)
    assert len(sim.trace.events) >= 1


@pytest.mark.parametrize("repair", [False, True])
def test_copy75_complete_loop_at_reconstructed_draws(repair):
    net = Netlist(P)
    latch = add_latch(net, D, "M.b40r1")
    add_experimental_autapses(net, D)
    topo = net.topology()
    weights = {(0, 1): 3281, (1, 0): 3792, (0, 0): -737, (1, 1): -692}
    q = np.array([[weights[int(s), int(d)] for s, d in zip(topo.src, topo.dst)]])
    sim = RefSim(topo, P, quanta=q, V_th=[-45.12705001632755, -44.42221141037541],
                 bias=[.03693153370274249, -.7152384644970224])
    sim.V[:] = P.E_L + sim.bias
    for member, reset in zip(latch.members, (-2618, -2707)):
        sim.add_events(0, [249, 293, 337, 384, 457], [member] * 5, [reset] * 5)
    sim.add_events(0, [1182], [latch.u], [4661])
    if repair:
        sim.add_events(0, [1182 + MASTER_IGNITION_PARTNER_DELAY_STEPS - 18],
                       [latch.v], [round(D.ignite * np.exp(-.12))])
    sim.run(5800)
    ev = sim.trace.events
    if repair:
        for member in latch.members:
            assert np.count_nonzero((ev["neuron"] == member) & (ev["step"] > 5000)) >= 2
    else:
        assert ev["neuron"].tolist() == [latch.u]
        assert ev["step"].tolist() == [1210]


def _fixture(*, repair=True, partner_delay=MASTER_IGNITION_PARTNER_DELAY_STEPS):
    net = Netlist(P)
    master = [add_latch(net, D, f"M.b0r{r}") for r in (0, 1)]
    stage = [add_latch(net, D, f"Q.b0r{r}") for r in (0, 1)]
    fault = add_latch(net, D, "Q.fault")
    copy = add_latch(net, D, "copy")
    trigger, inh, _ = add_reset(net, D, "M", master + [copy])
    ready = add_ready(net, D, "M", trigger, hops=15)
    net.synapse(ready, copy.u, D.ignite)
    arms = [add_veto_relay(net, D, f"cp0r{r}", copy.u,
                          [stage[1-r].u, fault.u], master[r], require=[stage[r].u])
            for r in (0, 1)]
    readers = [rate_tap(net, D, l.u) for l in master]
    add_experimental_autapses(net, D)
    if repair:
        for arm, latch in zip(arms, master):
            net.synapse(arm, latch.v, D.ignite, partner_delay)
    return net, master, stage, fault, copy, trigger, ready, arms, readers


def test_copy_timing_is_preserved_and_partner_ignition_is_staggered():
    times = []
    for repair in (False, True):
        net, master, stage, fault, copy, trigger, ready, arms, readers = _fixture(repair=repair)
        sim = RefSim(net.topology(), P)
        sim.add_events(0, [10, 1000], [stage[1].u, trigger], [D.ignite, D.relay_in])
        sim.run(4000)
        ev = sim.trace.events
        times.append([int(ev["step"][ev["neuron"] == m][0])
                      for m in [ready, arms[1], *master[1].members]])
    assert times[0][:3] == times[1][:3] == [1872, 2009, 2056]
    assert times[1][2] < times[1][3] < times[0][3]
    print("READY, COPY, u, v steps (off/on)", times, flush=True)


def ignition_survey(case, initial, copies, seed=17531, *, stray=True,
                    partner_delay=MASTER_IGNITION_PARTNER_DELAY_STEPS):
    """Empty reset -> first COPY or live old rail -> reset -> opposite COPY.

    A second ordinary reset proves extinction of the newly established orbit.
    It is observed before its READY could permit another write. Random phases,
    independent mix-B strays on every neuron, both members and readouts counted.
    """
    net, master, stage, fault, copy, trigger, ready, arms, readers = _fixture(partner_delay=partner_delay)
    # Clear the source stage too, so a later legitimate READY cannot reload
    # from live input and be miscounted as a reset survivor.
    final_clear, final_inh, _ = add_reset(net, D, "final", master + stage + [copy, fault])
    topo = net.topology()
    members = [m for latch in master for m in latch.members]
    targets = np.isin(topo.dst, members)
    q = topo.quanta.copy()
    th, bias = np.full(net.n, P.V_th), np.asarray(net.bias).copy()
    exc, neg, threshold, tonic = {
        "nominal": (1., 1., 0., 0.),
        "weak_loop": (np.exp(-.12), np.exp(.12), 0., 0.),
        "fast3sigma": (np.exp(.12), np.exp(-.12), -.6, .6),
        "slow3sigma": (np.exp(-.12), np.exp(.12), .6, -.6),
        "high_threshold": (1., 1., .6, 0.),
        "negative_bias": (1., 1., 0., -.6),
        "mix_b": (1., 1., 0., 0.),
        "independent3sigma": (1., 1., 0., 0.),
        "copy75": (1., 1., 0., 0.),
    }[case]
    q[targets] = np.rint(q[targets] * np.where(q[targets] > 0, exc, neg))
    th[members] += threshold
    bias[members] += tonic
    if case == "copy75":
        inh = net.roles.index("M.reset_inh")
        for latch in master:
            weights = {(latch.u, latch.v): 3281, (latch.v, latch.u): 3792,
                       (latch.u, latch.u): -737, (latch.v, latch.v): -692,
                       (inh, latch.u): -2618, (inh, latch.v): -2707}
            weights.update({(final_inh, latch.u): -2618, (final_inh, latch.v): -2707})
            for e, pair in enumerate(zip(topo.src, topo.dst)):
                if pair in weights:
                    q[e] = weights[pair]
                elif pair[0] in arms and pair[1] == latch.u:
                    q[e] = 4661
            th[list(latch.members)] = [-45.12705001632755, -44.42221141037541]
            bias[list(latch.members)] = [.03693153370274249, -.7152384644970224]
    totals = dict(copies=copies, missing=0, double_rails=0, survivors=0,
                  not_live_before_clear=0, old_survivors=0, multiple_copy=0)
    for lo in range(0, copies, 1000):
        count = min(1000, copies - lo)
        rng = np.random.default_rng(seed + lo)
        weights = np.broadcast_to(q, (count, topo.nnz)).copy()
        thresholds = np.broadcast_to(th, (count, net.n)).copy()
        biases = np.broadcast_to(bias, (count, net.n)).copy()
        if case == "copy75":
            added = np.isin(topo.src, arms) & np.isin(topo.dst, [l.v for l in master])
            weights[:, added] = np.rint(weights[:, added] * np.exp(rng.normal(0, .04, (count, int(added.sum())))))
        if case == "independent3sigma":
            # All 4,096 signs of loop, feedback, reset, threshold, bias and
            # both COPY ignition weights. Repeat the complete grid 10 times.
            # Coprime stride covers the complete grid; small smokes also vary
            # upper axes instead of testing only the first adjacent corners.
            signs = 2 * ((((17 * np.arange(lo, lo+count)[:, None]) % 4096) >> np.arange(12)) & 1) - 1
            for latch in master:
                for e, (s, d, weight) in enumerate(zip(topo.src, topo.dst, topo.quanta)):
                    if s in latch.members and d in latch.members:
                        axis = (2 + latch.members.index(int(d))) if s == d else latch.members.index(int(s))
                    elif s in (net.roles.index("M.reset_inh"), final_inh) and d in latch.members:
                        axis = 4 + latch.members.index(int(d))
                    elif s in arms and d in latch.members:
                        axis = 10 + latch.members.index(int(d))
                    else:
                        continue
                    weights[:, e] = np.rint(weight * np.exp(.12 * signs[:, axis]))
                thresholds[:, list(latch.members)] += .6 * signs[:, 6:8]
                biases[:, list(latch.members)] += .6 * signs[:, 8:10]
        sim = MixProbe(net, count, seed + lo, [*members, *readers, ready, *arms],
                       noisy=case == "mix_b", stray=stray, quanta=weights, V_th=thresholds, bias=biases)
        # Empty storage begins at its long-empty equilibrium, including bias.
        sim.V[:] = P.E_L + sim.bias
        resets = rng.integers(1000, 1100, count)
        clears = rng.integers(5200, 5300, count)
        for b, (reset, clear) in enumerate(zip(resets, clears)):
            sim.add_events(b, [10, int(reset), int(clear)], [stage[1].u, trigger, final_clear],
                           [D.ignite, D.relay_in, D.relay_in])
            if initial == "reload":
                sim.add_events(b, [10, 10+partner_delay-18], list(master[0].members), [round(D.ignite * exc)] * 2)
        sim.run(7000)
        first_copy = np.full(count, 7000)
        arm_steps, arm_nodes = sim.spikes(arms[1])
        np.minimum.at(first_copy, arm_nodes, arm_steps)
        live = np.ones(count, bool)
        established = np.ones(count, bool)
        survived = np.zeros(count, bool)
        old_survived = np.zeros(count, bool)
        old_live = np.ones(count, bool)
        for member in master[1].members:
            steps, nodes = sim.spikes(member)
            hit = (steps > resets[nodes] + 1600) & (steps < clears[nodes] - 200)
            established &= np.bincount(nodes[hit], minlength=count) >= 2
            hit = (steps > clears[nodes] - 200) & (steps < clears[nodes])
            live &= np.bincount(nodes[hit], minlength=count) >= 1
        for member in master[0].members:
            steps, nodes = sim.spikes(member)
            old_live &= np.bincount(nodes[(steps > resets[nodes]-200) & (steps < resets[nodes])], minlength=count) >= 1
            old_survived[nodes[(steps >= first_copy[nodes]) & (steps < clears[nodes])]] = True
        for member in [*members, *readers]:
            steps, nodes = sim.spikes(member)
            survived[nodes[steps > clears[nodes]+1500]] = True
        totals["missing"] += int((~established | ~live).sum())
        totals["double_rails"] += int(old_survived.sum())
        totals["old_survivors"] += int(old_survived.sum())
        totals["survivors"] += int(survived.sum())
        totals["not_live_before_clear"] += int((~live).sum()) + (int((~old_live).sum()) if initial == "reload" else 0)
        totals["multiple_copy"] += int((sim.counts(arms[1]) != 1).sum())
    return totals


@SMOKE
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("initial", INITIALS)
def test_master_ignition_corner_smoke(case, initial):
    run_ignition_survey(case, initial, smoke=True)


@pytest.mark.slow
@CLUSTER
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("initial", INITIALS)
def test_master_ignition_cluster_survey(case, initial):
    run_ignition_survey(case, initial)


def run_ignition_survey(case, initial, *, smoke=False):
    wall, cpu = perf_counter(), process_time()
    count = 256 if smoke else (40_960 if case == "independent3sigma" else 40_000)
    result = ignition_survey(case, initial, copies=count)
    print("ignition", MASTER_IGNITION_VERSION, case, initial, "smoke" if smoke else "cluster",
          result, f"wall_s={perf_counter()-wall:.3f}", f"cpu_s={process_time()-cpu:.3f}", flush=True)
    assert result == dict(copies=count, missing=0, double_rails=0, survivors=0,
                         not_live_before_clear=0, old_survivors=0, multiple_copy=0), result


@pytest.mark.slow
@CLUSTER
@pytest.mark.parametrize("initial", INITIALS)
def test_all_independent_corners_establish_without_strays(initial):
    result = ignition_survey("independent3sigma", initial, copies=4096, stray=False)
    print("deterministic independent corners", initial, result, flush=True)
    assert result == dict(copies=4096, missing=0, double_rails=0, survivors=0,
                         not_live_before_clear=0, old_survivors=0, multiple_copy=0)


@pytest.mark.slow
@CLUSTER
@pytest.mark.parametrize("case", CLEAR_CASES)
def test_master_ignition_clear_cluster_survey(case):
    run_clear_survey(case)


@SMOKE
@pytest.mark.parametrize("case", CLEAR_CASES)
def test_master_ignition_clear_smoke(case):
    run_clear_survey(case, smoke=True)


def run_clear_survey(case, *, smoke=False):
    wall, cpu = perf_counter(), process_time()
    count = 256 if smoke else 40_000
    result = delayed_clear_survey(case, copies=count)
    print("ignition clear", MASTER_IGNITION_VERSION, case, "smoke" if smoke else "cluster",
          result, f"wall_s={perf_counter()-wall:.3f}", f"cpu_s={process_time()-cpu:.3f}", flush=True)
    assert result == dict(copies=count, survivors=0, not_live_before_clear=0, reader_survivors=0), result


def delayed_clear_survey(case, copies, seed=108):
    """The eight earlier clear scenarios, with staggered member ignition.

    Keep their real controllers and reconstructed old edges. Sample added
    autapses/readouts independently as before; changing the initial orbit is
    precisely why the old single-member clear results cannot qualify this.
    """
    from test_stable_latch import AUTAPSE, CAPTURES, clear_fixture
    net, latch, trigger, inh, _, q, th, bias, new = clear_fixture(AUTAPSE, case, rate=True)
    readers = list(net.rate_readouts.values())
    totals = dict(copies=copies, survivors=0, not_live_before_clear=0, reader_survivors=0)
    for lo in range(0, copies, 1000):
        count = min(1000, copies-lo)
        rng = np.random.default_rng(seed+lo)
        weights = np.broadcast_to(q, (count, len(q))).copy()
        thresholds = np.broadcast_to(th, (count, net.n)).copy()
        biases = np.broadcast_to(bias, (count, net.n)).copy()
        if case in CAPTURES:
            weights[:, new] = np.rint(weights[:, new] * np.exp(rng.normal(0, .04, (count, int(new.sum())))))
            thresholds[:, readers] += rng.normal(0, .2, (count, len(readers)))
            biases[:, readers] += rng.normal(0, .2, (count, len(readers)))
        sim = MixProbe(net, count, seed+lo, [*latch.members, *readers], noisy=case == "mix_b",
                       quanta=weights, V_th=thresholds, bias=biases)
        clears = rng.integers(3000, 4500, count)
        for b, clear in enumerate(clears):
            for member, offset in zip(latch.members, (0, MASTER_IGNITION_PARTNER_DELAY_STEPS - 18)):
                sim.add_events(b, [s + offset for s in (10, 11, 14, 23)], [member] * 4, [D.ignite] * 4)
            sim.add_events(b, [2520, int(clear)], [latch.v if b % 2 else latch.u, trigger],
                           [300, D.relay_in])
        sim.run(6200)
        live, survived = np.ones(count, bool), np.zeros(count, bool)
        for member in latch.members:
            steps, nodes = sim.spikes(member)
            live &= np.bincount(nodes[(steps > clears[nodes]-200) & (steps < clears[nodes])], minlength=count) > 0
            survived[nodes[steps > clears[nodes]+1500]] = True
        totals["not_live_before_clear"] += int((~live).sum())
        totals["survivors"] += int(survived.sum())
        for reader in readers:
            steps, nodes = sim.spikes(reader)
            totals["reader_survivors"] += len(np.unique(nodes[steps > clears[nodes]+1500]))
    return totals


@SMOKE
def test_simultaneous_ignition_fast_clear_counterexample():
    # Fixed small adverse control: the same draws/strays/clear phases must
    # expose the rejected circuit, and the staggered candidate must clear.
    old = ignition_survey("fast3sigma", "empty", copies=256, partner_delay=18)
    new = ignition_survey("fast3sigma", "empty", copies=256)
    print("simultaneous / staggered fast clear", old, new, flush=True)
    assert old["survivors"] == 2, old
    assert new == dict(copies=256, missing=0, double_rails=0, survivors=0,
                       not_live_before_clear=0, old_survivors=0, multiple_copy=0), new


@pytest.mark.parametrize("stage_value,fault_on", [(None, False), (0, False), (1, False), (1, True), ("both", False)])
def test_copy_authorization_and_opposite_rail_veto(stage_value, fault_on):
    net, master, stage, fault, copy, trigger, ready, arms, readers = _fixture()
    sim = RefSim(net.topology(), P)
    for r in (0, 1):
        if stage_value in (r, "both"):
            sim.add_events(0, [10], [stage[r].u], [D.ignite])
    if fault_on:
        sim.add_events(0, [10], [fault.u], [D.ignite])
    sim.add_events(0, [1000], [trigger], [D.relay_in])
    sim.run(5000)
    ev = sim.trace.events
    for r in (0, 1):
        selected = stage_value == r and not fault_on
        for member in master[r].members:
            count = np.count_nonzero(ev["neuron"] == member)
            assert (count >= 2) if selected else (count == 0)


def test_no_master_write_without_copy():
    net, master, stage, fault, copy, trigger, ready, arms, readers = _fixture()
    sim = RefSim(net.topology(), P)
    sim.add_events(0, [10], [stage[1].u], [D.ignite])
    sim.run(5000)
    assert not np.isin(sim.trace.events["neuron"], [m for l in master for m in l.members]).any()


def test_stage_discard_before_copy_cannot_write():
    net, master, stage, fault, copy, trigger, ready, arms, readers = _fixture()
    discard, _, _ = add_reset(net, D, "discard", stage + [fault])
    sim = RefSim(net.topology(), P)
    sim.add_events(0, [10, 1000, 1500], [stage[1].u, trigger, discard],
                   [D.ignite, D.relay_in, D.relay_in])
    sim.run(5000)
    ev = sim.trace.events
    assert np.count_nonzero(ev["neuron"] == copy.u) > 0
    assert not np.isin(ev["neuron"], [m for l in master for m in l.members]).any()


@pytest.mark.parametrize("candidate", ["legacy", "dual", "delay"])
def test_rejected_delay_candidate_copies_an_uncommitted_next_word(candidate):
    net, master, stage, fault, copy, trigger, ready, arms, readers = _fixture(repair=candidate == "dual")
    if candidate == "delay":
        roles = {role: i for i, role in enumerate(net.roles)}
        chain = [trigger, *[roles[f"M.ready_delay{k}"] for k in range(15)], ready]
        pairs = set(zip(chain, chain[1:]))
        for e, pair in enumerate(zip(net.src, net.dst)):
            if pair in pairs:
                net.delay[e] = 100
    discard, _, _ = add_reset(net, D, "discard", stage + [fault])
    release = add_ready(net, D, "discard", discard, hops=15)
    net.synapse(release, stage[0].u, D.ignite)  # next DATA obeys stage READY; no next COMMIT
    sim = RefSim(net.topology(), P)
    sim.add_events(0, [10, 1000, 1500], [stage[1].u, trigger, discard],
                   [D.ignite, D.relay_in, D.relay_in])
    sim.run(5000)
    written = np.isin(sim.trace.events["neuron"], master[0].members).any()
    assert bool(written) is (candidate == "delay")


@pytest.mark.slow
@CLUSTER
def test_master_ignition_two_cell_kernel_computes_and_reloads():
    from drosophilos.lib.kernel import build_pipeline, run_pipeline
    pl = build_pipeline(P, 2, [dict(name="sum", op="ADD", a="input", b=("const", "one")),
                               dict(name="out", op="XOR", a="sum", b=("const", "one"))],
                        consts={"one": 1}, rate_robust=True, experimental_autapse=True,
                        ordered_xor_oneshot=True, master_ignition=True)
    outputs, sim, stats = run_pipeline(pl, P, [0, 1, 3], max_ms=6000)
    assert isinstance(sim, RefSim)
    assert [v for _, v in outputs] == [0, 3, 1]
    assert stats["faults"] == stats["timeouts"] == stats["bad_outputs"] == 0
