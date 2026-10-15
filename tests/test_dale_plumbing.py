"""Replay and mapping plumbing for the versioned Dale feedback option."""

import pytest

from drosophilos.bench import kernel_campaign
from drosophilos.bench.h1_fullgraph import adder_build_kwargs
from drosophilos.bench.h_stage_place import mapping_meta
from drosophilos.bench.stall_diag import build_tick_pipeline
from drosophilos.bench.repro import circuit_hash
from drosophilos.lib.adder import build_adder_channel
from drosophilos.lib.kernel import build_pipeline
from drosophilos.sim.model import Params


F = dict(rate_robust=True, experimental_autapse=True, ordered_xor_oneshot=True)


def _record():
    _ks, pl, _tokens, _reference = kernel_campaign.block("tick", Params(), **F,
                                                           latch_feedback="dale_private")
    return pl, {"block": "tick", **pl.build_options}


def _adder_digest(channel):
    net = channel.net
    return (net.n, net.nnz, tuple(net.roles),
            tuple(zip(net.src, net.dst, net.quanta, net.delay)))


def test_dale_private_campaign_replays_identically():
    pl, record = _record()
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


def test_h_stage_mapping_rebuilds_dale_private_adder():
    meta = mapping_meta("all", 0, 1, 0, latch_feedback="dale_private")
    options = adder_build_kwargs(meta)
    assert options["latch_feedback"] == "dale_private"
    assert options["dale_feedback_version"] == meta["dale_feedback_version"]
    assert _adder_digest(build_adder_channel(Params(), **options)) == _adder_digest(
        build_adder_channel(Params(), width=4, ordered=True, watchdog_hops=100, storage="flipflop",
                            **F, latch_feedback="dale_private"))


@pytest.mark.parametrize("version", [999, None])
def test_h_stage_mapping_refuses_unversioned_or_wrong_dale_feedback(version):
    meta = mapping_meta("all", 0, 1, 0, latch_feedback="dale_private")
    if version is None:
        del meta["dale_feedback_version"]
    else:
        meta["dale_feedback_version"] = version
    with pytest.raises(ValueError, match="Dale-feedback"):
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
