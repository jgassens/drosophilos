"""One bounded round: three fixed Dale-consistent constructions, no grid search."""

import json
import os

import numpy as np
import pytest

from drosophilos.lib.netlist import Netlist
from drosophilos.protocol.rate import rate_tap
from test_fast_latch import MixProbe
from test_dale_feedback import (
    D, F, P, adder, adder_survey, corner_probe, digest, feedback, kernel,
    noisy_reload_probe,
)


def _members(net, drive):
    edges = {(s, d): q for s, d, q in zip(net.src, net.dst, net.quanta)}
    return [m for (u, v), q in edges.items()
            if q == drive.loop and edges.get((v, u)) == drive.loop
            and net.roles[u].endswith(".u")
            and net.roles[v] == net.roles[u].removesuffix(".u") + ".v"
            for m in (u, v)]


def biased_candidate(net, drive):
    # 1.4 mV below threshold: even opposing .6-mV corners stay subthreshold.
    # Scale input with the reduced gap to avoid an intentionally bursting relay.
    mirrors, net.inhibition_mirrors = net.inhibition_mirrors, {}
    try:
        for m in _members(net, drive):
            inh = net.neuron(net.roles[m] + ".fb", bias=5.6)
            net.synapse(m, inh, round(.2 * drive.loop), 0)
            net.synapse(inh, m, -round(.2 * drive.loop), 0)
    finally:
        net.inhibition_mirrors = mirrors


def partner_candidate(net, drive):
    mirrors, net.inhibition_mirrors = net.inhibition_mirrors, {}
    try:
        members = _members(net, drive)
        for u, v in zip(members[::2], members[1::2]):
            for m, partner in ((u, v), (v, u)):
                inh = net.neuron(net.roles[m] + ".fb")
                net.synapse(partner, inh, drive.loop, 0)
                net.synapse(inh, m, -round(.2 * drive.loop), 0)
    finally:
        net.inhibition_mirrors = mirrors


def twin_candidate(net, drive):
    # Clone each member's incoming synapses, including real resets. A negative
    # self-edge on an inhibitory neuron is Dale-consistent and matches the
    # member's inhibitory return. Host ignition into a member is not cloned.
    members = _members(net, drive)
    incoming = {m: [(net.src[e], net.quanta[e], net.delay[e]) for e in net.incoming[m]]
                for m in members}
    mirrors, net.inhibition_mirrors = net.inhibition_mirrors, {}
    try:
        for m in members:
            inh = net.neuron(net.roles[m] + ".fb", bias=net.bias[m])
            for source, q, delay in incoming[m]:
                net.synapse(source, inh, q, delay)
            net.synapse(inh, inh, -round(.2 * drive.loop), 0)
            net.synapse(inh, m, -round(.2 * drive.loop), 0)
    finally:
        net.inhibition_mirrors = mirrors


CANDIDATES = dict(dale_biased="dale_biased", partner=partner_candidate, twin=twin_candidate)


@pytest.mark.parametrize("build,expected", [
    (kernel, "894d6d70ec86691e1c215a2f3826e9e4d1c6e003a860cc0c211a58067b8189c4"),
    (adder, "133acfac07d2c027890b8b8024991fba615b4fdafb1fc98b05f7a24c1b6fe8f1"),
])
def test_private_v1_frozen(build, expected):
    # Captured before this round's production edits; freeze v1 independently
    # of the existing autapse/default digests, which remain untouched.
    assert digest(build(**F, latch_feedback="dale_private").net) == expected


@pytest.mark.parametrize("build", [kernel, adder])
def test_biased_structure_cost_and_versions(build):
    old = build(**F)
    private = build(**F, latch_feedback="dale_private")
    new = build(**F, latch_feedback="dale_biased")
    assert (new.net.n, new.net.nnz) == (private.net.n, private.net.nnz)
    assert new.net.n - old.net.n == new.net.nnz - old.net.nnz == len(_members(old.net, D))
    assert new.build_options["latch_feedback"] == "dale_biased"
    assert new.build_options["dale_biased_version"] == feedback.DALE_BIASED_VERSION == 1
    assert "dale_feedback_version" not in new.build_options
    net = new.net
    for m in _members(net, D):
        inh = net.roles.index(net.roles[m] + ".fb")
        assert net.bias[inh] == feedback.DALE_BIASED_BIAS_MV == 5.6
        assert [(net.src[e], net.quanta[e], net.delay[e]) for e in net.incoming[inh]] == [
            (m, round(.2 * D.loop), 0)]
        assert [(d, q, dl) for s, d, q, dl in zip(net.src, net.dst, net.quanta, net.delay)
                if s == inh] == [(m, -round(.2 * D.loop), 0)]
        # Some pre-existing control sources have mixed signs. This transform
        # removes their latch autapse and adds only an excitatory relay drive.
        assert [(d, q, dl) for s, d, q, dl in zip(net.src, net.dst, net.quanta, net.delay)
                if s == m and d != inh] == [
                    (d, q, dl) for s, d, q, dl in zip(old.net.src, old.net.dst, old.net.quanta, old.net.delay)
                    if s == m and d != m]
    with pytest.raises(ValueError, match="requires experimental_autapse"):
        build(latch_feedback="dale_biased")
    with pytest.raises(ValueError, match="requires rate_robust"):
        build(experimental_autapse=True, latch_feedback="dale_biased")
    for version in (0, 2, None, "1", True):
        with pytest.raises(ValueError, match="unsupported Dale-biased circuit version"):
            build(**F, latch_feedback="dale_biased", dale_biased_version=version)
    build(**F, dale_biased_version=None)  # irrelevant to historical autapse
    build(**F, latch_feedback="dale_private", dale_biased_version=None)


@pytest.mark.parametrize("policy", ["dale_private", "dale_biased"])
@pytest.mark.parametrize("option", ["master_ignition", "operand_xor_oneshot", "f_width16_experimental"])
def test_unsurveyed_options_rejected(policy, option):
    with pytest.raises(ValueError, match="not surveyed with " + policy):
        kernel(**F, latch_feedback=policy, **{option: True})


def test_biased_stage_d_replay():
    from drosophilos.bench import stage_d
    from drosophilos.bench.stall_diag import build_tick_pipeline

    pl = stage_d.build(stage_d.load_kernel(), P, **F, latch_feedback="dale_biased")
    record = {"stage": "D", "build_options": dict(pl.build_options)}
    rebuilt = build_tick_pipeline(record)[2]
    assert digest(rebuilt.net) == digest(pl.net)
    assert rebuilt.net.bias == pl.net.bias  # digest's historical format omits biases
    stage_d.recheck_record(record)
    for version in (0, 2, None, "1", True):
        record["build_options"]["dale_biased_version"] = version
        with pytest.raises(ValueError, match="unsupported Dale-biased circuit version"):
            build_tick_pipeline(record)
        with pytest.raises(ValueError, match="unsupported Dale-biased circuit version"):
            stage_d.recheck_record(record)
    del record["build_options"]["dale_biased_version"]
    with pytest.raises(ValueError, match="unsupported Dale-biased circuit version"):
        build_tick_pipeline(record)


def test_biased_transform_preserves_mirrors_and_is_atomic():
    from dataclasses import replace

    net = Netlist(P)
    latch = feedback.add_latch(net, D, "L")
    tap = rate_tap(net, replace(D, rate_robust=True), latch.u)
    mirrors = net.inhibition_mirrors
    candidate = Netlist(P)
    feedback.add_latch(candidate, D, "L")
    rate_tap(candidate, replace(D, rate_robust=True), 0)
    biased_candidate(candidate, D)
    feedback.add_dale_latch_feedback(net, D, "biased")
    assert digest(net) == digest(candidate) and net.bias == candidate.bias
    assert net.inhibition_mirrors is mirrors
    assert all(q > 0 for q in [net.quanta[e] for e in net.incoming[tap]])
    clear = net.neuron("clear")
    net.synapse(clear, latch.u, -D.loop)
    assert any(net.src[e] == clear and net.quanta[e] == -16 * D.loop for e in net.incoming[tap])
    before, biases = digest(net), list(net.bias)
    for variant in ("biased", "private", "unknown"):
        with pytest.raises(ValueError):
            feedback.add_dale_latch_feedback(net, D, variant)
        assert digest(net) == before and net.bias == biases


@pytest.mark.parametrize("transform", [biased_candidate, partner_candidate, twin_candidate])
def test_candidates_obey_dale(transform):
    net = Netlist(P)
    latch = feedback.add_latch(net, D, "L")
    feedback.add_reset(net, D, "clear", [latch])
    transform(net, D)
    for source in range(net.n):
        signs = {int(np.sign(q)) for s, q in zip(net.src, net.quanta) if s == source}
        assert len(signs) <= 1


def test_biased_timing_is_not_autapse_equivalent():
    net = Netlist(P)
    latch = feedback.add_latch(net, D, "L")
    dark = feedback.add_latch(net, D, "dark")
    feedback.add_dale_latch_feedback(net, D, "biased")
    sim = MixProbe(net, 1, 0, range(net.n), noisy=False, stray=False)
    sim.ignite([latch.u])
    sim.run(6000)
    u, inh = sim.spikes(latch.u)[0], sim.spikes(4)[0]
    assert u[:4].tolist() == [34, 114, 175, 235]
    assert inh[:4].tolist() == [177, 311, 438, 562]
    assert inh[0] - u[0] == 143
    assert np.median(np.diff(u[u >= 2000])) == 52
    assert np.median(np.diff(inh[inh >= 2000])) == 122
    assert all(len(sim.spikes(m)[0]) == 0 for m in (*dark.members, 6, 7))


def test_small_corners():
    row = corner_probe("dale_biased", patterns=16)
    assert row["non_ignition"] == row["loss_of_hold"] == row["not_holding_at_end"] == 0


@pytest.mark.parametrize("fresh_each", [True, False])
def test_bounded_biased_f_adder(fresh_each):
    row = adder_survey("dale_biased", fresh_each)
    print("dale_biased adder", fresh_each, json.dumps(row), flush=True)
    assert row["correct"] == row["completed"] == 8
    assert all(row[key] == 0 for key in ("wrong", "faults", "timeouts", "no_accept", "missing_completion"))


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("DROSO_DALE_FAST_SURVEY") != "1",
                    reason="requires DROSO_DALE_FAST_SURVEY=1")
def test_design_round():
    expected = dict(autapse=(0, 0, 0, 114), dale_private=(41, 43, 84, 102),
                    dale_biased=(0, 0, 0, 118), partner=(112, 60, 172, 3), twin=(67, 34, 101, 71))
    for name, policy in {"autapse": "autapse", "dale_private": "dale_private", **CANDIDATES}.items():
        row = dict(corners=corner_probe(policy), reload=noisy_reload_probe(128, policy))
        print("design round", name, json.dumps(row), flush=True)
        assert tuple(row["corners"][k] for k in ("non_ignition", "loss_of_hold", "not_holding_at_end")) + (
            row["reload"]["700"],) == expected[name]
        assert row["reload"]["None"] == 0
