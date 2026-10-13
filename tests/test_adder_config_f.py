"""Configuration F on the H1 flip-flop adder, using the netlist-level H1 harness."""

from dataclasses import replace
import hashlib
import json
import os
from types import SimpleNamespace

import pytest

from drosophilos.connectome.embed_image import adder_cases, simulate_channel
from drosophilos.lib.adder import build_adder_channel
from drosophilos.lib.netlist import Drive
from drosophilos.protocol.run import run_transactions
from drosophilos.sim.model import Params


P = Params()
F = dict(rate_robust=True, experimental_autapse=True, ordered_xor_oneshot=True)
# Captured from the unedited builder: 4 bits, default watchdog and other defaults.
OFF_DIGESTS = {
    (False, "latch"): "eed4a4a2b40375ff802d81eebd041d4151354e9557ac01776cface380fea6c5c",
    (False, "flipflop"): "6f57ee7b1811da9fed01be77d9a3844d0c0b2c90a140b0b2704e6c9610ee59b5",
    (True, "latch"): "3fad6be9343690fd50cc872955c33e904a511aad9731532bb6f8271cb93513b5",
    (True, "flipflop"): "f93b6aff6ab77192f1985e30dbbede653a3be20ff7228921f38cbb533b93e17a",
}


def build(**options):
    return build_adder_channel(P, 4, ordered=True, storage="flipflop", watchdog_hops=100, **options)


@pytest.mark.parametrize("ordered,storage", OFF_DIGESTS)
def test_defaults_are_frozen(ordered, storage):
    net = build_adder_channel(P, 4, ordered=ordered, storage=storage).net
    data = [getattr(net, k) for k in ("src", "dst", "quanta", "delay", "roles")]
    assert hashlib.sha256(json.dumps(data, separators=(",", ":")).encode()).hexdigest() == OFF_DIGESTS[ordered, storage]


@pytest.mark.parametrize("options,message", [
    (dict(ordered_xor_oneshot=True), "requires rate_robust"),
    (dict(rate_robust=True, ordered_xor_oneshot=True), "experimental_autapse=True"),
    (dict(experimental_autapse=True), "requires rate_robust"),
])
def test_invalid_flags(options, message):
    with pytest.raises(ValueError, match=message):
        build(**options)


def test_oneshot_requires_ordered_and_kernel_physics():
    with pytest.raises(ValueError, match="requires ordered=True"):
        build_adder_channel(P, 4, **F)
    for params, width, drive in ((replace(P, dt=.2), 4, None), (P, 9, None),
                                 (P, 4, replace(Drive.from_params(P), kill_pulses=4))):
        with pytest.raises(ValueError, match="default physics/drive"):
            build_adder_channel(params, width, drive=drive, ordered=True, **F)


def test_feedback_counts_and_reset_ownership():
    ch = build(**F)
    net = ch.net
    feedback = [(s, q, dl) for s, d, q, dl in zip(net.src, net.dst, net.quanta, net.delay) if s == d]
    local = [(s, q, dl) for s, q, dl in feedback if q == -round(.2 * ch.drive.loop)]
    once = [(s, q, dl) for s, q, dl in feedback if q == -round(4.4 * ch.drive.loop)]
    edges = {(s, d): q for s, d, q in zip(net.src, net.dst, net.quanta)}
    pairs = [(u, v) for (u, v), q in edges.items() if q == ch.drive.loop
             and edges.get((v, u)) == ch.drive.loop and net.roles[u].endswith(".u")
             and net.roles[v] == net.roles[u].removesuffix(".u") + ".v"]
    assert len(pairs) == 70
    assert len(local) == 140 and len(once) == 8 and len(feedback) == 148
    assert all(dl == 0 for _, _, dl in feedback)
    assert all(net.roles[s].endswith((".u", ".v")) for s, _, _ in local)
    ff_members = {x for pair in ch.consumer.rails for rail in pair for x in (rail.u, rail.v)}
    assert not ff_members.intersection(s for s, _, _ in local)
    reset_targets = {d for s, d, q in zip(net.src, net.dst, net.quanta) if s == ch.consumer.reset_inh and q < 0}
    assert {s for s, _, _ in local if ".spent." in net.roles[s]} <= reset_targets
    for source, mirrors in net.inhibition_mirrors.items():
        for target, _ in mirrors:
            assert not any(net.src[e] == source and net.quanta[e] < 0 for e in net.incoming[target])
    assert ch.drive.rate_robust and ch.drive.kill_pulses == Drive.from_params(P).kill_pulses


def check_additions(count, fresh_each):
    ch = build(**F)
    _, words, expected = adder_cases(4, count, seed=110)
    result = simulate_channel(ch, SimpleNamespace(topology=ch.net.topology()), words, expected, fresh_each=fresh_each)
    assert result["correct"] == count, {k: v for k, v in result.items() if k != "records"}
    assert result["faults"] == result["timeouts"] == result["wrong"] == 0
    return result


@pytest.mark.parametrize("fresh_each", [True, False])
def test_random_additions(fresh_each):
    result = check_additions(8, fresh_each)
    print("F adder", "fresh" if fresh_each else "chained", result["accept_ms"])


def test_accept_latency_against_non_f():
    _, words, expected = adder_cases(4, 4, seed=0)
    results = []
    for options in ({}, F):
        ch = build(**options)
        result = simulate_channel(ch, SimpleNamespace(topology=ch.net.topology()), words, expected)
        assert result["correct"] == len(words)
        results.append(result["accept_ms"])
    print("accept ms: non-F, F", results)


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("DROSO_F_ADDER") != "1", reason="50+50 survey requires DROSO_F_ADDER=1")
@pytest.mark.parametrize("fresh_each", [True, False])
def test_larger_random_survey(fresh_each):
    check_additions(50, fresh_each)


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("DROSO_F_ADDER_NOISE") != "1", reason="mix-B probe requires DROSO_F_ADDER_NOISE=1")
def test_mix_b_additions():
    from test_experimental_autapse import _NoisyRefSim

    ch = build(**F)
    sim = _NoisyRefSim(ch.net, copies=1, seed=110)
    for step, neuron, q in ch.power_on_events(0):
        sim.add_events(0, [step], [neuron], [q])
    _, words, expected = adder_cases(4, 8, seed=110)
    offsets = sim.rng.integers(0, 101, (len(words), ch.width)).tolist()
    records, _, stats = run_transactions(ch, P, words, expected=expected, sim=sim,
                                         bit_offsets=offsets, max_steps_per_tx=12000)
    assert stats["completed"] == len(words), stats
    assert [r.decoded for r in records] == expected
    assert all(r.fault_spikes == r.timeout_spikes == 0 for r in records)
