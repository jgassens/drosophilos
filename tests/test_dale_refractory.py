"""One bounded bias/gain round; three settings, no further tuning.

The netlist carries bias but no per-neuron refractory override. These settings
therefore use ordinary model refractory dynamics and change only relay bias,
drive and inhibition. Full measurements retain the original shared fixtures.
"""

import hashlib
import json
import os

import numpy as np
import pytest

from test_dale_fast import _members
from test_dale_feedback import (D, F, P, adder, adder_survey, corner_probe, digest,
                                feedback, kernel, noisy_reload_probe)


SETTINGS = {"gain_02": (5.0, .4, .2), "gain_03": (5.0, .4, .3),
            "gain_04": (5.0, .4, .4)}


def candidate(name):
    bias, drive_gain, gain = SETTINGS[name]

    def install(net, drive):
        mirrors, net.inhibition_mirrors = net.inhibition_mirrors, {}
        try:
            for member in _members(net, drive):
                inh = net.neuron(net.roles[member] + ".fb", bias=bias)
                net.synapse(member, inh, round(drive_gain * drive.loop), 0)
                net.synapse(inh, member, -round(gain * drive.loop), 0)
        finally:
            net.inhibition_mirrors = mirrors

    return install


def scores(policy):
    return dict(corners=corner_probe(policy), reload=noisy_reload_probe(128, policy))


EXPECTED = {"autapse": (0, 0, 0, 36, 114, 128),
            "dale_biased": (0, 0, 0, 33, 118, 128),
            "gain_02": (4, 11, 15, 33, 114, 128),
            "gain_03": (76, 83, 159, 35, 108, 128),
            "gain_04": (217, 70, 287, 37, 37, 65)}


def assert_score(row, expected):
    corners = row["corners"]
    assert tuple(corners[key] for key in (
        "non_ignition", "loss_of_hold", "not_holding_at_end")) + (
        corners["interval"][0], row["reload"]["700"], row["reload"]["1000"]) == expected
    assert row["reload"]["None"] == 0


@pytest.mark.parametrize("policy", ["autapse", "dale_biased", "dale_refractory"])
def test_full_bounded_scores(policy):
    row = scores(policy)
    print("bounded scores", policy, json.dumps(row), flush=True)
    assert_score(row, EXPECTED["gain_02" if policy == "dale_refractory" else policy])
    # Negative results stay explicit: no policy here meets the full three-bar goal.
    corners = row["corners"]
    assert not (all(corners[k] == 0 for k in (
        "non_ignition", "loss_of_hold", "not_holding_at_end"))
        and corners["interval"][0] >= 44 and row["reload"]["700"] >= 114)


@pytest.mark.parametrize("build", [kernel, adder])
def test_refractory_structure_and_versions(build):
    old = build(**F, latch_feedback="dale_biased")
    new = build(**F, latch_feedback="dale_refractory")
    expected = build(**F, latch_feedback="autapse")
    # The independent candidate transform must match the released opt-in exactly.
    from unittest.mock import patch
    module = "drosophilos.lib.kernel" if build is kernel else "drosophilos.lib.adder"
    with patch(module + ".add_dale_latch_feedback", side_effect=lambda net, drive, **kw:
               candidate("gain_02")(net, drive)):
        screened = build(**F, latch_feedback="dale_refractory")
    assert digest(new.net) == digest(screened.net)
    assert new.net.bias == screened.net.bias
    assert (new.net.n, new.net.nnz) == (old.net.n, old.net.nnz)
    assert new.net.n - expected.net.n == new.net.nnz - expected.net.nnz == len(_members(old.net, D))
    assert new.build_options["latch_feedback"] == "dale_refractory"
    assert new.build_options["dale_refractory_version"] == feedback.DALE_REFRACTORY_VERSION == 1
    assert "dale_biased_version" not in new.build_options
    assert "dale_feedback_version" not in new.build_options
    for member in _members(new.net, D):
        inh = new.net.roles.index(new.net.roles[member] + ".fb")
        assert new.net.bias[inh] == feedback.DALE_REFRACTORY_BIAS_MV == 5.0
        assert [(new.net.src[e], new.net.quanta[e], new.net.delay[e])
                for e in new.net.incoming[inh]] == [(member, round(.4 * D.loop), 0)]
        assert [(d, q, dl) for s, d, q, dl in zip(
            new.net.src, new.net.dst, new.net.quanta, new.net.delay) if s == inh] == [
                (member, -round(.2 * D.loop), 0)]
    with pytest.raises(ValueError, match="requires experimental_autapse"):
        build(latch_feedback="dale_refractory")
    with pytest.raises(ValueError, match="requires rate_robust"):
        build(experimental_autapse=True, latch_feedback="dale_refractory")
    for version in (0, 2, None, "1", True):
        with pytest.raises(ValueError, match="unsupported Dale-refractory circuit version"):
            build(**F, latch_feedback="dale_refractory", dale_refractory_version=version)
    for policy in ("autapse", "dale_private", "dale_biased"):
        build(**F, latch_feedback=policy, dale_refractory_version=None)


@pytest.mark.parametrize("build,expected", [
    (kernel, "88eab29d4e817312ead92ede0560c6cb4103d312bc6d22be04b17eb922cf1078"),
    (adder, "847e799023d3ba993a8c36cd21d048bf978ba56886e83c09473fdc5b9479e5fe"),
])
def test_biased_v1_frozen_including_bias(build, expected):
    # Captured by executing HEAD's unedited latch transform, independently of
    # this round's production implementation. Older digest fixtures omit bias.
    net = build(**F, latch_feedback="dale_biased").net
    data = [getattr(net, k) for k in ("src", "dst", "quanta", "delay", "roles", "bias")]
    assert hashlib.sha256(json.dumps(data, separators=(",", ":")).encode()).hexdigest() == expected


def test_refractory_dale_mirroring_atomicity_and_dark_control():
    from dataclasses import replace
    from drosophilos.lib.netlist import Netlist
    from drosophilos.protocol.rate import rate_tap
    from test_fast_latch import MixProbe

    net = Netlist(P)
    latch = feedback.add_latch(net, D, "L")
    tap = rate_tap(net, replace(D, rate_robust=True), latch.u)
    mirrors = net.inhibition_mirrors
    feedback.add_dale_latch_feedback(net, D, "refractory")
    assert net.inhibition_mirrors is mirrors
    for member in latch.members:
        inh = net.roles.index(net.roles[member] + ".fb")
        for source in (member, inh):
            assert len({int(np.sign(q)) for s, q in zip(net.src, net.quanta) if s == source}) == 1
        assert all(net.src[e] != inh for e in net.incoming[tap])
    sim = MixProbe(net, 1, 0, range(net.n), noisy=False, stray=False)
    sim.run(6000)
    assert all(len(sim.spikes(n)[0]) == 0 for n in range(net.n))
    clear = net.neuron("clear")
    net.synapse(clear, latch.u, -D.loop)
    assert any(net.src[e] == clear and net.quanta[e] == -16 * D.loop for e in net.incoming[tap])
    before, biases = digest(net), list(net.bias)
    for variant in ("refractory", "biased", "private", "unknown"):
        with pytest.raises(ValueError):
            feedback.add_dale_latch_feedback(net, D, variant)
        assert digest(net) == before and net.bias == biases


@pytest.mark.parametrize("fresh_each", [True, False])
def test_bounded_refractory_f_adder(fresh_each):
    row = adder_survey("dale_refractory", fresh_each)
    print("dale_refractory adder", fresh_each, json.dumps(row), flush=True)
    assert row["correct"] == row["completed"] == 8
    assert all(row[key] == 0 for key in (
        "wrong", "faults", "timeouts", "no_accept", "missing_completion"))


def test_small_noisy_fanout():
    from drosophilos.bench import kernel_campaign
    from drosophilos.compiler.kernel import kernel_outputs
    from drosophilos.lib.kernel import run_pipeline_batched
    from test_experimental_autapse import _NoisyRefSim

    ks, pl, tokens, _ = kernel_campaign.block("fanout", P, **F, latch_feedback="dale_refractory")
    tokens = tokens * 2
    reference = kernel_outputs(ks, tokens)
    expected = {name: [row[j] for row in reference] for j, name in enumerate(ks.outputs)}
    outputs, _, stats = run_pipeline_batched(
        pl, P, [tokens] * 2, max_ms=50000, progress=0,
        sim=_NoisyRefSim(pl.net, copies=2, seed=110),
        expect_outputs=[sum(map(len, expected.values()))] * 2)
    got = [{name: [v for _, v in values] for name, values in row.items()} for row in outputs]
    row = dict(correct=sum(g == expected for g in got), copies=2, repeats=2,
               outputs=got, faults=stats["faults"], timeouts=stats["timeouts"],
               bad_outputs=stats["bad_outputs"], neural_ms=stats["neural_ms"],
               last_output_ms=[max((step * P.dt for values in row.values() for step, _ in values),
                                   default=None) for row in outputs])
    print("dale_refractory fanout", json.dumps(row), flush=True)
    assert row["correct"] == 2
    assert row["faults"] == row["timeouts"] == row["bad_outputs"] == 0


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("DROSO_DALE_REFRACTORY_SURVEY") != "1",
                    reason="requires DROSO_DALE_REFRACTORY_SURVEY=1")
def test_design_round():
    for name, policy in {"autapse": "autapse", "dale_biased": "dale_biased",
                         **{name: candidate(name) for name in SETTINGS}}.items():
        row = scores(policy)
        print("refractory round", name, json.dumps(row), flush=True)
        assert_score(row, EXPECTED[name])
