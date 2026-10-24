"""Fault attribution on tiny synthetic circuits only; never load real MCNS data."""

import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
from test_embed_image import _placed_latch_relay

from drosophilos.bench.h_fault_attr import IncomingAttribution, failure_point, validate_attribution_rules
from drosophilos.connectome.embed_image import build_image, full_graph_topology
from drosophilos.connectome.mcns import MCNS
from drosophilos.protocol.handshake import build_channel
from drosophilos.protocol.run import TxRecord
from drosophilos.sim.model import ModelRules, Params
from drosophilos.sim.ref64 import RefSim
from drosophilos.sim.trace import SpikeTrace


def trace(*events):
    return SpikeTrace.from_arrays([s for s, _ in events], np.zeros(len(events), np.int64),
                                  [n for _, n in events])


def boundary_fixture():
    net, _, ids, policy, m, _, pl = _placed_latch_relay()
    hosts = set(pl.mapping.values())
    a, b, c, unused = [i for i in range(m.n) if i not in hosts][:4]
    u, v = pl.mapping[ids["u"]], pl.mapping[ids["v"]]
    # Remove all random boundary edges; preserve the planted circuit unchanged.
    keep = np.isin(m.pre, list(hosts)) & np.isin(m.post, list(hosts))
    neurons = m.neurons.copy()
    neurons.loc[[a, b], "type"] = "shared"
    neurons.loc[c, "type"] = "inhibitor"
    neurons.loc[[a, b], "superclass"] = "sensory"
    neurons.loc[c, "superclass"] = "central"
    neurons.loc[[a, b, unused], "sign"] = 1
    neurons.loc[c, "sign"] = -1
    pairs = [(a, u), (a, v), (b, u), (c, u), (unused, u)]
    m = MCNS(neurons, np.append(m.pre[keep], [s for s, _ in pairs]).astype(np.int32),
             np.append(m.post[keep], [d for _, d in pairs]).astype(np.int32),
             np.append(m.count[keep], [2, 3, 4, 5, 6]).astype(np.int32), min_syn=1)
    params = Params()
    image = build_image(net, m, pl, policy)
    fg = full_graph_topology(m, params, image)
    return m, fg, net.roles, params, (a, b, c, unused), (u, v)


@pytest.mark.parametrize("changes", [{}, {"clear_g_on_spike": True},
                                   {"freeze_g_in_refractory": True}, {"stim_target": "v"},
                                   {"stim_disables_refractory": True}])
def test_saved_rule_values_round_trip_and_each_switch_is_checked(tmp_path, changes):
    m, fg, roles, params, _, _ = boundary_fixture()
    rules = ModelRules(**changes)
    attr = IncomingAttribution(m, fg, roles, 2, params, rules=rules)
    path = tmp_path / "scores.npz"
    attr.save(path)
    with np.load(path, allow_pickle=False) as saved:
        assert saved["model_rules"].shape == ()
        assert json.loads(saved["model_rules"].item()) == asdict(rules)
        assert saved["score"].tolist() == attr.score.tolist()
    validate_attribution_rules(path, rules)
    if changes:
        with pytest.raises(ValueError, match="model rules mismatch") as exc:
            validate_attribution_rules(path, ModelRules())
        assert str(asdict(rules)) in str(exc.value)
        assert str(asdict(ModelRules())) in str(exc.value)


@pytest.mark.parametrize("metadata", ["not JSON", "{}", "null",
                                      json.dumps(asdict(ModelRules()) | {"clear_g_on_spike": 1})])
def test_malformed_recorded_rules_are_rejected(tmp_path, metadata):
    path = tmp_path / "scores.npz"
    np.savez_compressed(path, model_rules=metadata)
    with pytest.raises(ValueError, match="Invalid attribution model_rules metadata"):
        validate_attribution_rules(path, ModelRules())


def test_scores_only_delivered_in_window_and_aggregate_types_once_per_failure(tmp_path):
    m, fg, roles, params, (a, b, c, unused), (u, v) = boundary_fixture()
    attr = IncomingAttribution(m, fg, roles, 2, params)  # twenty steps
    # Arrivals: a at 79 (outside), 80 (inclusive start), 99 (inside), 100 (exclusive end).
    # c at 90; b only at 101, although its source spike itself is before the failure.
    events = trace((61, a), (62, a), (81, a), (82, a), (72, c), (83, b), (99, u))
    one = attr.observe(events, {"step": 100, "reason": "fault_spike"}, end_step=120, fresh=True)
    assert one["active_edges"] == 3 and one["deliveries"] == 5
    assert one["score"] == 2 * (32 + 48) + 80
    expected = {(a, u): 64, (a, v): 96, (c, u): 80, (b, u): 0, (unused, u): 0}
    assert dict(zip(zip(attr.pre, attr.post), attr.score)) == expected
    # A second failed addition contains b and c, with b sharing a's type/superclass.
    attr.observe(trace((72, b), (72, c)), {"step": 100, "reason": "stall_deadline"},
                 end_step=120, fresh=True)
    report = attr.report(2)
    types = {r["type"]: r for r in report["by_type"]}
    assert types["shared"] == {"type": "shared", "score": 224, "exc_score": 224, "inh_score": 0, "failures": 2}
    assert types["inhibitor"]["score"] == types["inhibitor"]["inh_score"] == 160
    assert types["inhibitor"]["failures"] == 2
    classes = {r["superclass"]: r for r in report["by_superclass"]}
    assert classes["sensory"]["score"] == 224 and classes["sensory"]["failures"] == 2
    assert classes["central"]["inh_score"] == 160 and classes["central"]["failures"] == 2
    assert report["score"] == report["exc_score"] + report["inh_score"] == 384
    assert len(report["top_edges"]) == 2 and report["top_edges"][0]["weight"] == -80
    assert report["top_edges"][0]["post_role"] == roles[fg.designed_of()[u]]
    path = tmp_path / "scores.npz"
    attr.save(path)
    with np.load(path, allow_pickle=False) as saved:
        assert saved["score"].tolist() == attr.score.tolist()
        assert saved["weight"].tolist() == attr.weight.tolist()
        assert saved["synapses"].tolist() == attr.synapses.tolist()
        assert saved["pre_bodyId"].tolist() == m.neurons["bodyId"].to_numpy()[attr.pre].tolist()
        assert not any("trace" in key or "step" in key for key in saved.files)


def test_mixed_delays_zero_weights_and_chained_tail():
    m, fg, roles, params, (a, b, c, _), (u, v) = boundary_fixture()
    # One edge from a has a different delay; the other is silenced, but keeps its anatomy.
    delay = fg.topology.delay.copy()
    delay[(fg.topology.src == a) & (fg.topology.dst == v)] = 0
    weight = fg.topology.quanta.copy()
    weight[(fg.topology.src == a) & (fg.topology.dst == u)] = 0
    fg = replace(fg, topology=replace(fg.topology, delay=delay, quanta=weight))
    attr = IncomingAttribution(m, fg, roles, 2, params)
    # Successful prior addition: a and b emit at 90, before the next load at step 100.
    assert attr.observe(trace((90, a), (90, b)), None, end_step=100, fresh=True) is None
    result = attr.observe(trace((102, c)), {"step": 110, "reason": "stall_deadline"},
                          end_step=120, fresh=False)
    # a->v arrives at 90 (inclusive); b->u at 108; c arrives after failure, at 120.
    assert result["deliveries"] == 2 and result["score"] == 48 + 64
    assert attr.synapses[(attr.pre == a) & (attr.post == u)].tolist() == [2]
    # A fresh simulator must not inherit that prior source history.
    previous = attr.score.copy()
    attr.observe(SpikeTrace.empty(), {"step": 110, "reason": "stall_deadline"}, end_step=120, fresh=True)
    assert np.array_equal(previous, attr.score)


def test_failure_point_uses_earliest_signal_or_terminal_stall_and_ignores_success():
    ch = build_channel(Params(), 2, watchdog_hops=40)
    expected, wrong = ch.consumer.rail_taps[0]
    fault, timeout = ch.consumer.fault[0], ch.producer.watchdog.timeout.u
    rec = TxRecord(0, 100, None, None, None, None, "incomplete", 1)
    ref_rec = TxRecord(0, 0, 80, 90, 100, 0, "valid", 0)
    reference = trace((40, expected))
    assert failure_point(ch, rec, trace((140, expected), (150, fault)), reference, ref_rec, 200) == {
        "step": 150, "reason": "fault_spike"}
    assert failure_point(ch, rec, trace((120, wrong), (150, fault)), reference, ref_rec, 200) == {
        "step": 120, "reason": "wrong_rail"}
    assert failure_point(ch, rec, trace((139, expected), (150, fault)), reference, ref_rec, 200) == {
        "step": 139, "reason": "early_rail"}
    assert failure_point(ch, rec, trace((119, timeout), (120, wrong)), reference, ref_rec, 200) == {
        "step": 119, "reason": "timeout_spike"}
    assert failure_point(ch, rec, trace((99, fault)), reference, ref_rec, 200) == {
        "step": 200, "reason": "stall_deadline"}
    terminal = replace(rec, accept_step=160, reset_step=170, ready_step=190)
    assert failure_point(ch, terminal, trace((180, wrong)), reference, ref_rec, 190) == {
        "step": 190, "reason": "stall_terminal"}
    assert failure_point(ch, replace(terminal, status="valid", decoded=0), trace((150, fault)),
                         reference, ref_rec, 190) is None
    # A correct decode without READY is still a failed (unfinished) addition.
    assert failure_point(ch, replace(rec, status="valid", decoded=0), SpikeTrace.empty(),
                         reference, ref_rec, 200)["reason"] == "stall_deadline"


def test_real_synthetic_spikes_before_forced_fault_are_attributed():
    m, fg, roles, params, (a, b, c, _), _ = boundary_fixture()
    sim = RefSim(fg.topology, params)
    # A host stands in for the harness's fault output; force it at step 100.
    fault = int(fg.index_map[0])
    sim.add_events(0, [61, 71, 82, 99], [a, c, b, fault], [100_000] * 4)
    sim.run(110)
    ch = SimpleNamespace(consumer=SimpleNamespace(fault=[fault], rail_taps=[]),
                         producer=SimpleNamespace(watchdog=None))
    rec = TxRecord(0, 0, None, None, None, None, "fault", 1)
    failure = failure_point(ch, rec, sim.trace, SpikeTrace.empty(), rec, 110)
    assert failure == {"step": 100, "reason": "fault_spike"}
    attr = IncomingAttribution(m, fg, roles, 2, params)
    row = attr.observe(sim.trace, failure, end_step=110, fresh=True)
    assert row["score"] == 32 + 48 + 80
    assert attr.score[attr.pre == b].tolist() == [0]  # delivery occurs AFTER the fault


@pytest.mark.parametrize("window", [0, -1, float("nan"), float("inf")])
def test_invalid_window(window):
    m, fg, roles, params, _, _ = boundary_fixture()
    with pytest.raises(ValueError, match="finite and positive"):
        IncomingAttribution(m, fg, roles, window, params)


def test_empty_incoming_boundary():
    m, fg, roles, params, _, _ = boundary_fixture()
    # An empty whole graph is sufficient for accumulator/report plumbing.
    fg = replace(fg, topology=fg.topology.empty(fg.topology.n))
    attr = IncomingAttribution(m, fg, roles, 20, params)
    attr.observe(SpikeTrace.empty(), {"step": 20, "reason": "stall_deadline"}, end_step=20, fresh=True)
    assert attr.report()["top_edges"] == []
    assert attr.report()["score"] == 0 and attr.report()["failures"] == 1
