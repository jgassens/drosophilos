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


def test_dale_biased_campaign_replays_identically():
    pl, record = _record("dale_biased")
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


@pytest.mark.parametrize("version", [0, None])
def test_dale_biased_replay_refuses_missing_or_wrong_version(version):
    _pl, record = _record("dale_biased")
    if version is None:
        del record["dale_biased_version"]
    else:
        record["dale_biased_version"] = version
    with pytest.raises(ValueError, match="latch_feedback|Dale-biased"):
        kernel_campaign.rebuild_record(record)
    with pytest.raises(ValueError, match="latch_feedback|Dale-biased"):
        build_tick_pipeline(record)


def test_h_stage_mapping_rebuilds_dale_private_adder():
    meta = mapping_meta("all", 0, 1, 0, latch_feedback="dale_private")
    options = adder_build_kwargs(meta)
    assert options["latch_feedback"] == "dale_private"
    assert options["dale_feedback_version"] == meta["dale_feedback_version"]
    assert _adder_digest(build_adder_channel(Params(), **options)) == _adder_digest(
        build_adder_channel(Params(), width=4, ordered=True, watchdog_hops=100, storage="flipflop",
                            **F, latch_feedback="dale_private"))


def test_h_stage_mapping_rebuilds_dale_biased_adder():
    meta = mapping_meta("all", 0, 1, 0, latch_feedback="dale_biased")
    options = adder_build_kwargs(meta)
    assert options["latch_feedback"] == "dale_biased"
    assert options["dale_biased_version"] == meta["dale_biased_version"]
    assert _adder_digest(build_adder_channel(Params(), **options)) == _adder_digest(
        build_adder_channel(Params(), width=4, ordered=True, watchdog_hops=100, storage="flipflop",
                            **F, latch_feedback="dale_biased"))


@pytest.mark.parametrize("latch_feedback", ("autapse", "dale_private", "dale_biased"))
def test_saved_h_stage_mapping_rebuilds_its_exact_roles(tmp_path, latch_feedback):
    """The Stage-H file round trip keeps optional feedback metadata in its builder contract."""
    params = Params()
    meta = mapping_meta("all", 0, 1, 0, latch_feedback=latch_feedback)
    feedback_version = (
        {"dale_feedback_version": meta["dale_feedback_version"]}
        if latch_feedback == "dale_private" else
        {"dale_biased_version": meta["dale_biased_version"]}
        if latch_feedback == "dale_biased" else {})
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


@pytest.mark.parametrize("version", [999, None])
def test_h_stage_mapping_refuses_unversioned_or_wrong_dale_biased(version):
    meta = mapping_meta("all", 0, 1, 0, latch_feedback="dale_biased")
    if version is None:
        del meta["dale_biased_version"]
    else:
        meta["dale_biased_version"] = version
    with pytest.raises(ValueError, match="Dale-biased"):
        adder_build_kwargs(meta)


@pytest.mark.parametrize("option", ["master_ignition", "operand_xor_oneshot", "f_width16_experimental"])
def test_dale_private_rejects_unsurveyed_kernel_options(option):
    kwargs = dict(F, latch_feedback="dale_private")
    kwargs[option] = True
    with pytest.raises(ValueError, match="not surveyed with dale_private"):
        build_pipeline(Params(), 1, [{"name": "x", "op": "MOV", "a": "input"}], **kwargs)


def test_missing_feedback_policy_rebuilds_historical_autapse():
    _ks, pl, _tokens, _reference = kernel_campaign.block("tick", Params(), **F)
    record = {"block": "tick", **pl.build_options}
    assert circuit_hash(kernel_campaign.rebuild_record(record)[1]) == circuit_hash(pl)
    assert circuit_hash(build_tick_pipeline(record)[2]) == circuit_hash(pl)
