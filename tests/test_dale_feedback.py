"""Bounded RefSim comparisons; a measured failure is not qualification.

Larger same-seed surveys: DROSO_DALE_SURVEY=1 pytest -s -m slow
tests/test_dale_feedback.py. All feedback is installed after consumers.
"""

from dataclasses import replace
import hashlib
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from drosophilos.connectome.embed_image import adder_cases, simulate_channel
from drosophilos.lib.adder import build_adder_channel
from drosophilos.lib.kernel import build_pipeline, run_pipeline_batched
from drosophilos.lib.netlist import Drive, Netlist
from drosophilos.protocol import latch as feedback
from drosophilos.protocol.rate import rate_tap
from drosophilos.sim.model import Params
from test_experimental_autapse import _NoisyRefSim
from test_fast_latch import MixProbe, _periods
import test_stable_latch as stable


P = Params()
D = Drive.from_params(P)
F = dict(rate_robust=True, experimental_autapse=True, ordered_xor_oneshot=True)
SPEC = [dict(name="sum", op="ADD", a="input", b=("const", "one")),
        dict(name="out", op="XOR", a="sum", b=("const", "one"))]
# Captured BEFORE editing any production file (2026-10-15). Ordered JSON arrays
# of src, dst, quanta, delay, roles; full SHA256, not edge-count assertions.
FROZEN = {
    ("kernel", False): "a4a12041ed3f14368d21c2b13ccd753a4b1416e49a4d70cc05ffea22efaf81fe",
    ("adder", False): "eed4a4a2b40375ff802d81eebd041d4151354e9557ac01776cface380fea6c5c",
    ("kernel", True): "6fcd4004647d069f6c19b9a4a90c906bb9d2870646820ef4200b3154f67774f2",
    ("adder", True): "9d9ab9eb0113ce036b28e7bbacd67fc29156dccae07a7b766dd055e398705488",
}


def kernel(**options):
    return build_pipeline(P, 2, SPEC, consts={"one": 1}, **options)


def adder(**options):
    return build_adder_channel(P, 4, ordered=True, watchdog_hops=100,
                               storage="flipflop", **options)


def digest(net):
    data = [getattr(net, k) for k in ("src", "dst", "quanta", "delay", "roles")]
    return hashlib.sha256(json.dumps(data, separators=(",", ":")).encode()).hexdigest()


@pytest.mark.parametrize("kind,enabled", FROZEN)
def test_frozen_defaults_and_historical_f(kind, enabled):
    build = kernel if kind == "kernel" else adder if enabled else lambda **kw: build_adder_channel(P, 4, **kw)
    options = F if enabled else {}
    assert digest(build(**options).net) == FROZEN[kind, enabled]
    assert digest(build(**options, latch_feedback="autapse").net) == FROZEN[kind, enabled]


@pytest.mark.parametrize("build", [kernel, adder])
def test_private_structure_and_record(build):
    old, new = build(**F), build(**F, latch_feedback="dale_private")
    net = new.net
    members = [i for i, role in enumerate(old.net.roles)
               if any(s == i == d and q == -round(.2 * D.loop)
                      for s, d, q in zip(old.net.src, old.net.dst, old.net.quanta))]
    assert net.n - old.net.n == len(members)
    assert net.nnz - old.net.nnz == len(members)
    roles = {r: i for i, r in enumerate(net.roles)}
    for member in members:
        inh = roles[net.roles[member] + ".fb"]
        assert [(net.src[e], net.quanta[e], net.delay[e]) for e in net.incoming[inh]] == [
            (member, round(D.loop * feedback.DALE_FEEDBACK_DRIVE_GAIN), 0)]
        outgoing = [(d, q, dl) for s, d, q, dl in zip(net.src, net.dst, net.quanta, net.delay) if s == inh]
        assert outgoing == [(member, -round(.2 * D.loop), 0)]
        assert not any(net.src[e] == member for e in net.incoming[member])
    # Feedback never enters a reader/qualifier's reset mirror.
    for source, mirrors in net.inhibition_mirrors.items():
        inh = roles.get(net.roles[source] + ".fb")
        for target, _ in mirrors:
            assert not any(net.src[e] == inh for e in net.incoming[target])
    assert new.build_options["latch_feedback"] == "dale_private"
    assert new.build_options["dale_feedback_version"] == feedback.DALE_FEEDBACK_VERSION == 1
    print("counts", build.__name__, (old.net.n, old.net.nnz), (net.n, net.nnz))


@pytest.mark.parametrize("build", [kernel, adder])
def test_options_and_version_validation(build):
    with pytest.raises(ValueError, match="requires experimental_autapse"):
        build(latch_feedback="dale_private")
    with pytest.raises(ValueError, match="unsupported latch_feedback"):
        build(**F, latch_feedback="shared")
    for version in (0, 2, None, "1", True):
        with pytest.raises(ValueError, match="unsupported Dale-feedback circuit version"):
            build(**F, latch_feedback="dale_private", dale_feedback_version=version)
    build(**F, dale_feedback_version=None)  # irrelevant to historical autapse


def test_local_signs_and_real_clear_mirroring():
    net = Netlist(P)
    latch = feedback.add_latch(net, D, "L")
    reader = rate_tap(net, replace(D, rate_robust=True), latch.u)
    clear, _, _ = feedback.add_reset(net, D, "clear", [latch])
    feedback.add_dale_latch_feedback(net, D)
    for member in latch.members:
        assert all(q > 0 for s, q in zip(net.src, net.quanta) if s == member)
    # Later genuine clears still mirror; no new readers may be installed here.
    net.synapse(clear, latch.u, -D.loop)
    assert any(net.src[e] == clear and net.quanta[e] == -16 * D.loop for e in net.incoming[reader])
    before = digest(net)
    with pytest.raises(ValueError, match="already installed"):
        feedback.add_dale_latch_feedback(net, D)
    assert digest(net) == before


def test_stage_d_record_replay_and_version_check():
    from drosophilos.bench import stage_d
    from drosophilos.bench.stall_diag import build_tick_pipeline

    pl = stage_d.build(stage_d.load_kernel(), P, **F, latch_feedback="dale_private")
    record = {"stage": "D", "build_options": dict(pl.build_options)}
    rebuilt = build_tick_pipeline(record)[2]
    assert digest(rebuilt.net) == digest(pl.net)
    stage_d.recheck_record(record)
    record["build_options"]["dale_feedback_version"] = 2
    with pytest.raises(ValueError, match="unsupported Dale-feedback circuit version"):
        build_tick_pipeline(record)
    with pytest.raises(ValueError, match="unsupported Dale-feedback circuit version"):
        stage_d.recheck_record(record)


def test_nominal_period_and_feedback_latency():
    for policy in ("autapse", "dale_private"):
        net = Netlist(P)
        latch = feedback.add_latch(net, D, "L")
        (feedback.add_experimental_autapses if policy == "autapse" else
         feedback.add_dale_latch_feedback)(net, D)
        sim = MixProbe(net, 1, 0, range(net.n), noisy=False, stray=False)
        sim.ignite([latch.u])
        sim.run(6000)
        assert _periods(sim, latch.u)["interval"] == [58.] * 7
        if policy == "dale_private":
            for member, settled_lag in ((latch.u, 2), (latch.v, 1)):
                spikes = sim.spikes(member)[0]
                inhibitory = sim.spikes(member + 2)[0]
                assert inhibitory[0] - spikes[0] == 35
                settled = inhibitory[inhibitory >= 2000]
                previous = spikes[np.searchsorted(spikes, settled, side="right") - 1]
                assert np.unique(settled - previous).tolist() == [settled_lag]


def period_probe(policy, copies=128, steps=6000, seed=109):
    net = Netlist(P)
    latch = feedback.add_latch(net, D, "L")
    rate_tap(net, replace(D, rate_robust=True), latch.u)
    (feedback.add_experimental_autapses if policy == "autapse" else
     feedback.add_dale_latch_feedback)(net, D)
    sim = MixProbe(net, copies, seed, [*latch.members, *range(3, net.n)])
    sim.ignite([latch.u])
    sim.run(steps)
    return {**_periods(sim, latch.u), **stable._holding(sim, latch.u, steps)}


def corner_probe(policy, patterns=256):
    """Sample independent three-sigma signs; this is NOT exhaustive corners."""
    net = Netlist(P)
    latch = feedback.add_latch(net, D, "L")
    rate_tap(net, replace(D, rate_robust=True), latch.u)
    (feedback.add_experimental_autapses if policy == "autapse" else
     feedback.add_dale_latch_feedback)(net, D)
    topo = net.topology()
    rng = np.random.default_rng(109)
    signs = np.repeat(rng.choice([-1, 1], (patterns, topo.nnz + 2 * net.n)), 2, axis=0)
    sim = MixProbe(net, 2 * patterns, 109, latch.members, noisy=False, stray=False,
                   quanta=np.rint(topo.quanta * np.exp(.12 * signs[:, :topo.nnz])),
                   V_th=P.V_th + .6 * signs[:, topo.nnz:topo.nnz + net.n],
                   bias=np.asarray(net.bias) + .6 * signs[:, -net.n:])
    sim.ignite([latch.u], quanta=round(D.ignite * np.exp(-.12)))
    for b in range(sim.B):
        if b % 2:
            sim.add_events(b, [11, 14, 23], [latch.u] * 3, [D.ignite] * 3)
        sim.add_events(b, [2520], [latch.v if b % 2 else latch.u], [300])
    sim.run(6000)
    return {**_periods(sim, latch.u), **stable._holding(sim, latch.u, 6000)}


def test_sampled_three_sigma_corners():
    rows = {policy: corner_probe(policy) for policy in ("autapse", "dale_private")}
    print("sampled corners", json.dumps(rows), flush=True)
    assert rows["autapse"]["interval"][0] == 36
    assert rows["autapse"]["non_ignition"] == rows["autapse"]["loss_of_hold"] == 0
    assert rows["dale_private"]["interval"][0] == 37
    assert rows["dale_private"]["non_ignition"] == 41
    assert rows["dale_private"]["loss_of_hold"] == 43
    assert rows["dale_private"]["not_holding_at_end"] == 84


def noisy_reload_probe(copies):
    net, latch, source, _, _, *_ = stable.clear_fixture(stable.AUTAPSE, "mix_b", rate=True)
    watch = [*latch.members, *net.rate_readouts.values()]
    rows = {}
    for offset in (None, 700, 1000):
        sim = MixProbe(net, copies, 108, watch)
        for b in range(copies):
            clear = 3000 + 4 * (b % 15)
            sim.add_events(b, [10, 11, 14, 23, 2520, clear],
                           [latch.u] * 4 + [latch.v if b % 2 else latch.u, source],
                           [D.ignite] * 4 + [300, D.relay_in])
            if offset is not None:
                sim.add_events(b, [clear + offset], [latch.u], [D.ignite])
        sim.run(6200)
        held = [sim.counts(tap, since=6000) > 0 for tap in watch]
        rows[str(offset)] = int(np.logical_and.reduce(held).sum()) if offset is not None else int(
            np.logical_or.reduce(held).sum())
    return rows


def stable_surveys(monkeypatch, copies=128):
    """Reuse the qualification probe's actual controller, kick and recovery grid."""
    rows = {}
    for policy in ("autapse", "dale_private"):
        row = {"period": period_probe(policy, copies=copies)}
        with monkeypatch.context() as patch:
            if policy == "dale_private":
                patch.setattr(stable, "_modify", lambda net, latch, config:
                              feedback.add_dale_latch_feedback(net, D))
            row["clear"] = {case: stable.clear_survey(stable.AUTAPSE, case, copies=copies,
                                                       seed=108, rate=True)
                            for case in ("mix_b", "master74", "fast3sigma")}
            row["reload"] = stable.reload_survey(stable.AUTAPSE, "nominal", rate=True)
            row["noisy_reload"] = noisy_reload_probe(copies)
        rows[policy] = row
    return rows


def test_bounded_stable_surveys(monkeypatch):
    rows = stable_surveys(monkeypatch)
    print("stable", json.dumps(rows), flush=True)
    for row in rows.values():
        assert row["period"]["non_ignition"] == row["period"]["loss_of_hold"] == 0
        assert all(v["not_live_before_clear"] == v["survivors"] == v["reader_survivors"] == 0
                   for v in row["clear"].values())
        assert row["reload"] == dict(recovery=700, ready=872, margin=172, control_survivors=0)
    # Record the frontier rather than weakening any autapse qualification gate.
    assert rows["autapse"]["period"]["interval"][0] == 50
    assert rows["dale_private"]["period"]["interval"][0] == 52
    assert rows["autapse"]["noisy_reload"] == {"None": 0, "700": 114, "1000": 128}
    assert rows["dale_private"]["noisy_reload"] == {"None": 0, "700": 102, "1000": 128}


def kernel_survey(policy, shape="rom", seed=110, copies=1):
    from test_config_f_kernels import small_shape

    pl, tokens, expected = small_shape(shape, latch_feedback=policy)
    outputs, _, stats = run_pipeline_batched(
        pl, P, [tokens] * copies, max_ms=6000, sim=_NoisyRefSim(pl.net, copies=copies, seed=seed),
        progress=0, expect_outputs=[sum(map(len, expected.values()))] * copies)
    got = [{name: [v for _, v in values] for name, values in row.items()} for row in outputs]
    return dict(correct=sum(row == expected for row in got), copies=copies, outputs=got,
                faults=stats["faults"], timeouts=stats["timeouts"], bad_outputs=stats["bad_outputs"])


def test_bounded_noisy_f_kernel():
    rows = {policy: kernel_survey(policy) for policy in ("autapse", "dale_private")}
    print("kernel", json.dumps(rows), flush=True)
    for row in rows.values():
        assert row["correct"] == 1
        assert row["faults"] == row["timeouts"] == row["bad_outputs"] == 0


def adder_survey(policy, fresh_each, count=8, seed=110):
    ch = adder(**F, latch_feedback=policy)
    _, words, expected = adder_cases(4, count, seed=seed)
    result = simulate_channel(ch, SimpleNamespace(topology=ch.net.topology()), words, expected,
                              fresh_each=fresh_each)
    return {k: v for k, v in result.items() if k not in ("records", "reached", "statuses")}


@pytest.mark.parametrize("fresh_each", [True, False])
def test_bounded_f_adder(fresh_each):
    rows = {policy: adder_survey(policy, fresh_each) for policy in ("autapse", "dale_private")}
    print("adder", fresh_each, json.dumps(rows), flush=True)
    for row in rows.values():
        assert row["correct"] == row["completed"] == 8
        assert row["wrong"] == row["faults"] == row["timeouts"] == 0


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("DROSO_DALE_SURVEY") != "1", reason="requires DROSO_DALE_SURVEY=1")
def test_larger_surveys(monkeypatch):
    print("stable larger", json.dumps(stable_surveys(monkeypatch, copies=2000)), flush=True)
    for seed in (108, 109, 110):
        for policy in ("autapse", "dale_private"):
            print("kernel larger", seed, policy, kernel_survey(policy, seed=seed, copies=2), flush=True)
            for fresh_each in (True, False):
                print("adder larger", seed, policy, fresh_each,
                      adder_survey(policy, fresh_each, count=50, seed=seed), flush=True)
