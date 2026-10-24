"""Rules-aware H0 recertification; no real MCNS files are needed by these tests."""

import hashlib
import itertools
import json
from dataclasses import asdict, replace

import numpy as np
import pytest

from drosophilos._loop_probe import probe_loop
from drosophilos.bench import h0_retune as bench
from drosophilos.connectome.embed_h0 import (
    Policy, Targets, best_embedding, find_embeddings, needed_quanta, peak_response,
)
from drosophilos.connectome.h0_run import Transaction, isolated_topology, run_transaction
from drosophilos.sim.model import ModelRules, Params
from test_embed_h0 import synthetic_mcns


# Executed against unchanged HEAD 927d6b4 before implementing rules-aware tuning.
LEGACY_TARGETS = {
    "needed_single": 2586.0434707264194, "needed_rate": 382.8363636363636,
    "period_steps": 47, "loop": 3621, "and_in": 249, "or_in": 766,
    "completion": 3621, "reset": -5432,
}
LEGACY_TRACES = [
    "e23cb25121ac77b79883b1dfdc48ec090f8e99d73e12d047603253584eabca06",
    "ac3e4d36750cea36b3e23af6fbcb631117da1725d1a6f96a7d382851eabff383",
    "fe0240dc3201dbe55add87c44dcf7368c75622dc606f5178a0cfa5263680281b",
    "dd658989cdf4a12a77f768266f606da259c50a7d4939c35ea62c92a91a86bed6",
]


@pytest.fixture
def planted():
    m = synthetic_mcns()
    solutions, _ = find_embeddings(m, max_solutions=5, verbose=False)
    return m, best_embedding(solutions)


def test_default_targets_and_complete_transaction_traces_are_identical(planted):
    p, policy = Params(), Policy()
    implicit = Targets.from_policy(p, policy)
    explicit = Targets.from_policy(p, policy, rules=ModelRules())
    assert asdict(implicit) == asdict(explicit) == LEGACY_TARGETS
    assert peak_response(p) == (0.15748873980000203, 9.200000000000001)
    m, emb = planted
    topo, idx = isolated_topology(m, emb, p, policy)
    for (a, b), expected in zip(bench.INPUTS, LEGACY_TRACES):
        for kwargs in ({}, {"rules": ModelRules()}):
            row, trace = run_transaction(topo, p, emb, idx, Transaction(a, b), implicit, **kwargs)
            assert row["correct"] and row["reset_ok"]
            assert hashlib.sha256(trace.events.tobytes()).hexdigest() == expected


def test_subthreshold_synaptic_response_is_independent_of_all_rules():
    p = Params()
    for clear, freeze, target, exempt in itertools.product((False, True), (False, True), ("g", "v"), (False, True)):
        rules = ModelRules(clear, freeze, target, exempt)
        assert peak_response(p, rules=rules) == peak_response(p)
        assert needed_quanta(p, rules=rules) == LEGACY_TARGETS["needed_single"]


@pytest.mark.parametrize("params,expected", [
    (Params().with_dt(0.02), (2586.020672645222, 384.4654545454545, 236, 3621, 250, 769, 3621, -5432)),
    (replace(Params(), tau_m=25), (3045.094302926618, 407.27272727272725, 50, 4264, 265, 815, 4264, -6396)),
    (replace(Params(), tau_s=4), (3045.1164638176665, 519.2727272727273, 51, 4264, 338, 1039, 4264, -6396)),
])
def test_additional_historical_target_snapshots(params, expected):
    # Captured from the unchanged module, including the timestep-refinement case.
    assert tuple(asdict(Targets.from_policy(params, Policy())).values()) == expected


def test_shiu_loop_probe_and_targets_use_changed_dynamics():
    p = Params()
    assert probe_loop(p, 3621) == (47, 3.5)
    assert probe_loop(p, 3621, rules=bench.SHIU) == (106, 0.1)
    assert probe_loop(p, 3621, rules=ModelRules(clear_g_on_spike=True))[0] == 106
    tg = Targets.from_policy(p, Policy(), rules=bench.SHIU)
    assert tg.period_steps == 106
    assert tg.loop == 3621 and tg.completion == 3621 and tg.reset == -5432
    assert tg.and_in == 562 and tg.or_in == 1727
    with pytest.raises(RuntimeError, match="did not self-sustain"):
        probe_loop(p, 2300, rules=bench.SHIU)


def test_retuned_fixture_computes_and_without_changing_anatomy(planted):
    m, original = planted
    tg = Targets.from_policy(Params(), bench.RETUNED_POLICY, rules=bench.SHIU)
    emb = bench.retune_embedding(original, tg)
    assert original.neurons == emb.neurons
    assert {(e.pre, e.post) for e in original.edges} == {(e.pre, e.post) for e in emb.edges}
    audit = bench.audit_embedding(m, emb, bench.RETUNED_POLICY)
    assert audit["valid"] and audit["required_multiplier"] <= 4.0
    assert len(emb.edges) == audit["physical_edges"]
    results = bench.transactions(m, emb, tg, bench.RETUNED_POLICY, bench.SHIU,
                                 drive_quanta=bench.DRIVE_QUANTA)
    assert bench.all_pass(results)
    assert [r["completion_latency_ms"] for r in results] == [15.8, 29.8, 29.8, 33.2]
    assert all(r["loop_arrivals"]["discarded"] == 0 for r in results)
    topo, idx = isolated_topology(m, emb, Params(), bench.RETUNED_POLICY)
    # Below-threshold V ignition cannot start the circuit; the drive override is real.
    row, _ = run_transaction(topo, Params(), emb, idx, Transaction(1, 1, drive_quanta=400),
                             tg, rules=bench.SHIU)
    assert not row["completion"]


def test_audit_rejects_effective_parallel_weight_and_sign_flip(planted):
    m, emb = planted
    e = emb.edges[0]
    too_strong = replace(e, quanta=e.count * 16 * 5)
    bad = replace(emb, edges=[too_strong] + emb.edges[1:])
    audit = bench.audit_embedding(m, bad, Policy())
    assert not audit["valid"] and audit["required_multiplier"] == 5
    flipped = replace(emb, edges=[replace(e, quanta=-e.quanta)] + emb.edges[1:])
    assert not bench.audit_embedding(m, flipped, Policy())["valid"]
    duplicate = replace(emb, edges=[replace(e, quanta=e.count * 16 * 3)] * 2)
    assert bench.audit_embedding(m, duplicate, Policy())["required_multiplier"] == 6


def test_real_body_id_record_can_be_resolved_without_real_data(tmp_path):
    m = synthetic_mcns(n_distractors=0)
    solutions, _ = find_embeddings(m, max_solutions=5, verbose=False)
    original = best_embedding(solutions)
    m.neurons["bodyId"] = bench.BODY_IDS
    record = {"embedding": {
        "circuit": [{"bodyId": int(m.bodies_of([x])[0]), "role": role}
                    for x, role in original.roles().items()],
        "designed_edges": [{**asdict(e), "pre": int(m.bodies_of([e.pre])[0]),
                            "post": int(m.bodies_of([e.post])[0])} for e in original.edges],
    }}
    path = tmp_path / "published.json"
    path.write_text(json.dumps(record))
    restored = bench.published_embedding(m, path)
    assert restored.loops == original.loops
    assert restored.resets == original.resets
    assert restored.edges == original.edges
    record["embedding"]["designed_edges"][0]["count"] += 1
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="count mismatch"):
        bench.published_embedding(m, path)


def test_arrivals_include_spike_step_and_refractory_boundary(planted):
    from drosophilos.sim.trace import SpikeTrace

    _, emb = planted
    e = emb.edges[0]
    # Presynaptic spikes at 0,1,21,22 -> arrivals at 18,19,39,40.
    # Postsynaptic spike at 18: ages 0,1,21 discarded; age 22 accepted.
    trace = SpikeTrace.from_arrays(np.array([0, 1, 18, 21, 22]), np.zeros(5, dtype=int),
                                   np.array([0, 0, 1, 0, 0]))
    result = bench.arrival_stats(None, trace, [e], {e.pre: 0, e.post: 1}, Params(), bench.SHIU, set(), 100)
    assert result == {"total": 4, "discarded": 3, "discarded_fraction": 0.75}
    exempt = bench.arrival_stats(None, trace, [e], {e.pre: 0, e.post: 1}, Params(), bench.SHIU, {1}, 100)
    assert exempt["discarded"] == 1  # own spike step still rejects g writes


def test_bench_skips_tier_b_on_tier_a_pass(planted, monkeypatch):
    m, emb = planted
    monkeypatch.setattr(bench, "find_embeddings", lambda *a, **k: pytest.fail("Tier B should not run"))
    result = bench.run(m, emb, source="unit fixture", probe_steps=1000)
    assert result["tier_a"]["pass"]
    assert result["tier_b"]["status"] == "not_run"
    assert [r["passes_existing_checks"] for r in result["tier_a"]["M5_stale"]] == [True, False, False, False]
    assert result["retuned_policy"]["k_max"] == 4.0
    json.dumps(result, allow_nan=False)


def test_tier_b_search_receives_shiu_rules_and_caps(planted, monkeypatch):
    m, emb = planted
    original_audit = bench.audit_embedding
    calls = []

    def fail_a_only(*args):
        result = original_audit(*args)
        if not calls:
            result["valid"] = False
        calls.append(1)
        return result

    def search(*args, **kwargs):
        assert kwargs["rules"] == bench.SHIU
        assert kwargs["max_solutions"] == 2 and kwargs["time_limit_s"] == 3
        assert args[2].k_max == 4
        return [emb], {"n_solutions": 1}

    monkeypatch.setattr(bench, "audit_embedding", fail_a_only)
    monkeypatch.setattr(bench, "find_embeddings", search)
    result = bench.run(m, emb, source="unit fixture", max_solutions=2,
                       time_limit_s=3, test_best=1, probe_steps=1000)
    assert not result["tier_a"]["pass"]
    assert result["tier_b"]["status"] == "pass"
    assert len(result["tier_b"]["candidates"]) == 1


def test_cli_unavailable_data_records_not_run_without_fallback(tmp_path, monkeypatch):
    def denied(**kwargs):
        raise PermissionError("MCNS read denied")

    monkeypatch.setattr(bench, "load_mcns", denied)
    path = tmp_path / "not_run.json"
    result = bench.main(["--mcns-dir", str(tmp_path / "missing"), "--out", str(path)])
    assert result["tier_a"]["status"] == "not_run"
    assert "read denied" in json.loads(path.read_text())["error"]


def test_search_uses_m5_target_counts(planted):
    m, _ = planted
    solutions, stats = find_embeddings(m, policy=bench.RETUNED_POLICY, rules=bench.SHIU,
                                       max_solutions=1, verbose=False)
    assert solutions
    assert stats["targets"]["period_steps"] == 106
    assert stats["required_counts"]["and_in"] == 9
    assert stats["required_counts"]["or_in"] == 21


def test_dynamics_probe_measures_sustained_latch_and_and_discrimination():
    targets = Targets.from_policy(Params(), bench.RETUNED_POLICY, rules=bench.SHIU)
    result = bench.dynamics_probe(targets, n_steps=1500)
    one, two = result["inputs"]["1"], result["inputs"]["2"]
    assert one["steady_period_steps"] == 106 and one["driver_final_50ms_spikes"] > 0
    assert one["gate_spikes"] == 0 and one["gate_max_recorded_v_mv"] < -45
    assert two["gate_spikes"] > 0 and two["first_gate_latency_ms"] < 60
    assert one["or_gate_spikes"] > 0 and one["or_first_gate_latency_ms"] < 60
    old = Targets.from_policy(Params(), Policy())
    stale = bench.dynamics_probe(old, n_steps=1500)["inputs"]
    assert stale["1"]["or_gate_spikes"] == 0
    assert stale["1"]["or_gate_max_recorded_v_mv"] < -45
    assert stale["2"]["gate_spikes"] == 0
