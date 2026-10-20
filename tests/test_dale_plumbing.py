"""Replay and mapping plumbing for the versioned Dale feedback option."""

import json

import numpy as np
import pandas as pd
import pytest

from drosophilos.bench import kernel_campaign
from drosophilos.bench.h1_fullgraph import adder_build_kwargs
from drosophilos.bench.h_stage_place import mapping_meta
from drosophilos.bench.stall_diag import build_tick_pipeline
from drosophilos.bench.repro import circuit_hash
from drosophilos.connectome.embed_image import load_mapping_meta, save_mapping
from drosophilos.connectome.embed_netlist import Placement
from drosophilos.connectome.mcns import MCNS
from drosophilos.lib.adder import build_adder_channel
from drosophilos.lib.kernel import build_pipeline
from drosophilos.sim.model import Params


F = dict(rate_robust=True, experimental_autapse=True, ordered_xor_oneshot=True)


def _record(latch_feedback="dale_private"):
    _ks, pl, _tokens, _reference = kernel_campaign.block("tick", Params(), **F,
                                                           latch_feedback=latch_feedback)
    return pl, {"block": "tick", **pl.build_options}


def _adder_digest(channel):
    net = channel.net
    return (net.n, net.nnz, tuple(net.roles),
            tuple(zip(net.src, net.dst, net.quanta, net.delay)))


def test_dale_private_campaign_replays_identically():
    pl, record = _record()
    assert circuit_hash(kernel_campaign.rebuild_record(record)[1]) == circuit_hash(pl)
    assert circuit_hash(build_tick_pipeline(record)[2]) == circuit_hash(pl)


@pytest.mark.parametrize("policy", ["dale_biased", "dale_refractory"])
def test_versioned_dale_campaign_replays_identically(policy):
    pl, record = _record(policy)
    assert circuit_hash(kernel_campaign.rebuild_record(record)[1]) == circuit_hash(pl)
    assert circuit_hash(build_tick_pipeline(record)[2]) == circuit_hash(pl)


@pytest.mark.parametrize("change", [
    {"latch_feedback": "unknown"},
    {"dale_feedback_version": 0},
])
def test_dale_private_replay_refuses_unknown_policy_or_version(change):
    _pl, record = _record()
    record.update(change)
    with pytest.raises(ValueError, match="latch_feedback|Dale-feedback"):
        kernel_campaign.rebuild_record(record)
    with pytest.raises(ValueError, match="latch_feedback|Dale-feedback"):
        build_tick_pipeline(record)


@pytest.mark.parametrize("policy,label", [("dale_biased", "Dale-biased"),
                                          ("dale_refractory", "Dale-refractory")])
@pytest.mark.parametrize("version", [0, 2, None, "1", True, "missing"])
def test_versioned_dale_replay_refuses_missing_or_wrong_version(policy, label, version):
    _pl, record = _record(policy)
    key = policy + "_version"
    if version == "missing":
        del record[key]
    else:
        record[key] = version
    with pytest.raises(ValueError, match=label):
        kernel_campaign.rebuild_record(record)
    with pytest.raises(ValueError, match=label):
        build_tick_pipeline(record)


def test_h_stage_mapping_rebuilds_dale_private_adder():
    meta = mapping_meta("all", 0, 1, 0, latch_feedback="dale_private")
    options = adder_build_kwargs(meta)
    assert options["latch_feedback"] == "dale_private"
    assert options["dale_feedback_version"] == meta["dale_feedback_version"]
    assert _adder_digest(build_adder_channel(Params(), **options)) == _adder_digest(
        build_adder_channel(Params(), width=4, ordered=True, watchdog_hops=100, storage="flipflop",
                            **F, latch_feedback="dale_private"))


@pytest.mark.parametrize("policy", ["dale_biased", "dale_refractory"])
def test_h_stage_mapping_rebuilds_versioned_dale_adder(policy):
    meta = mapping_meta("all", 0, 1, 0, latch_feedback=policy)
    options = adder_build_kwargs(meta)
    assert options["latch_feedback"] == policy
    assert options[policy + "_version"] == meta[policy + "_version"]
    rebuilt = build_adder_channel(Params(), **options)
    original = build_adder_channel(Params(), width=4, ordered=True, watchdog_hops=100,
                                   storage="flipflop", **F, latch_feedback=policy)
    assert _adder_digest(rebuilt) == _adder_digest(original)
    assert rebuilt.net.bias == original.net.bias


@pytest.mark.parametrize("latch_feedback", ("autapse", "dale_private", "dale_biased", "dale_refractory"))
def test_saved_h_stage_mapping_rebuilds_its_exact_roles(tmp_path, latch_feedback):
    """The Stage-H file round trip keeps optional feedback metadata in its builder contract."""
    params = Params()
    meta = mapping_meta("all", 0, 1, 0, latch_feedback=latch_feedback)
    feedback_version = (
        {"dale_feedback_version": meta["dale_feedback_version"]}
        if latch_feedback == "dale_private" else
        {"dale_biased_version": meta["dale_biased_version"]}
        if latch_feedback == "dale_biased" else
        {"dale_refractory_version": meta["dale_refractory_version"]}
        if latch_feedback == "dale_refractory" else {})
    saved_channel = build_adder_channel(
        params, 4, ordered=True, watchdog_hops=100, storage="flipflop", **F,
        latch_feedback=latch_feedback, **feedback_version)
    net = saved_channel.net
    # save_mapping only needs body IDs; a trivial one-host-per-designed-neuron MCNS makes
    # the file manifest cover every saved role without invoking the placement search.
    neurons = pd.DataFrame({"bodyId": 10_000 + np.arange(net.n)})
    m = MCNS(neurons, np.array([], np.int32), np.array([], np.int32), np.array([], np.int32), min_syn=1)
    placement = Placement(dict(enumerate(range(net.n))), [], 0, [], 0, 0)
    path = tmp_path / f"{latch_feedback}.json"
    save_mapping(path, net, m, placement, meta)

    loaded_meta = load_mapping_meta(path)
    assert loaded_meta == meta
    rebuilt = build_adder_channel(params, **adder_build_kwargs(loaded_meta)).net
    saved = json.loads(path.read_text())
    assert set(rebuilt.roles) == set(saved["mapping"]) | set(saved["unplaced"])
    assert rebuilt.n == net.n
    assert rebuilt.bias == net.bias
    assert (rebuilt.src, rebuilt.dst, rebuilt.quanta, rebuilt.delay, rebuilt.roles) == (
        net.src, net.dst, net.quanta, net.delay, net.roles)
    if latch_feedback == "autapse":
        assert rebuilt.n == 740
    else:
        assert rebuilt.n > 740


@pytest.mark.parametrize("version", [999, None])
def test_h_stage_mapping_refuses_unversioned_or_wrong_dale_feedback(version):
    meta = mapping_meta("all", 0, 1, 0, latch_feedback="dale_private")
    if version is None:
        del meta["dale_feedback_version"]
    else:
        meta["dale_feedback_version"] = version
    with pytest.raises(ValueError, match="Dale-feedback"):
        adder_build_kwargs(meta)


@pytest.mark.parametrize("policy,label", [("dale_biased", "Dale-biased"),
                                          ("dale_refractory", "Dale-refractory")])
@pytest.mark.parametrize("version", [999, None, "1", True, "missing"])
def test_h_stage_mapping_refuses_unversioned_or_wrong_dale(policy, label, version):
    meta = mapping_meta("all", 0, 1, 0, latch_feedback=policy)
    if version == "missing":
        del meta[policy + "_version"]
    else:
        meta[policy + "_version"] = version
    with pytest.raises(ValueError, match=label):
        adder_build_kwargs(meta)


@pytest.mark.parametrize("policy", ["dale_private", "dale_biased", "dale_refractory"])
@pytest.mark.parametrize("option", ["master_ignition", "operand_xor_oneshot", "f_width16_experimental"])
def test_dale_rejects_unsurveyed_kernel_options(policy, option):
    kwargs = dict(F, latch_feedback=policy)
    kwargs[option] = True
    with pytest.raises(ValueError, match="not surveyed with " + policy):
        build_pipeline(Params(), 1, [{"name": "x", "op": "MOV", "a": "input"}], **kwargs)


def test_missing_feedback_policy_rebuilds_historical_autapse():
    _ks, pl, _tokens, _reference = kernel_campaign.block("tick", Params(), **F)
    record = {"block": "tick", **pl.build_options}
    assert circuit_hash(kernel_campaign.rebuild_record(record)[1]) == circuit_hash(pl)
    assert circuit_hash(build_tick_pipeline(record)[2]) == circuit_hash(pl)


def test_refractory_nested_stage_d_replay_and_recheck():
    from drosophilos.bench import stage_d

    pl = stage_d.build(stage_d.load_kernel(), Params(), **F, latch_feedback="dale_refractory")
    record = {"stage": "D", "build_options": dict(pl.build_options)}
    assert circuit_hash(build_tick_pipeline(record)[2]) == circuit_hash(pl)
    stage_d.recheck_record(record)
    for version in (0, 2, None, "1", True, "missing"):
        if version == "missing":
            record["build_options"].pop("dale_refractory_version")
        else:
            record["build_options"]["dale_refractory_version"] = version
        with pytest.raises(ValueError, match="Dale-refractory"):
            build_tick_pipeline(record)
        if version != "missing":
            with pytest.raises(ValueError, match="Dale-refractory"):
                stage_d.recheck_record(record)


def test_refractory_non_tick_campaign_replays():
    _ks, pl, _tokens, _reference = kernel_campaign.block(
        "fanout", Params(), **F, latch_feedback="dale_refractory")
    record = {"block": "fanout", **pl.build_options}
    assert circuit_hash(kernel_campaign.rebuild_record(record)[1]) == circuit_hash(pl)
    assert circuit_hash(build_tick_pipeline(record)[2]) == circuit_hash(pl)


def test_refractory_campaign_cli_and_help():
    parser = kernel_campaign.parser()
    args = parser.parse_args(["fanout", "--latch-feedback", "dale_refractory"])
    assert args.latch_feedback == "dale_refractory"
    help_text = " ".join(parser.format_help().split())
    assert "dale_refractory is REJECTED" in help_text
    assert "4/11/15" in help_text and "33 < 44" in help_text


def test_refractory_mapping_requires_f():
    with pytest.raises(ValueError, match="requires configuration F"):
        mapping_meta("all", 0, 1, 0, config_f=False, latch_feedback="dale_refractory")
    with pytest.raises(ValueError, match="legacy mapping"):
        adder_build_kwargs({"latch_feedback": "dale_refractory"})
