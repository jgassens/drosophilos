"""Rule semantics, frozen pre-change regression and bounded experimental plumbing."""

import hashlib
import itertools
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from drosophilos.bench.model_compare import (
    CONFIGS, Fixture, calibrate, h0_fixture, h0_transactions, periodic_steps,
    probe_metrics, run_protocol, small_fixtures,
)
from drosophilos.connectome.embed_h0 import Policy, Targets
from drosophilos.connectome.h0_run import Transaction, run_transaction
from drosophilos.sim import FastSim, ModelRules, Params, RefSim, Topology, TorchSim
from test_embed_h0 import synthetic_mcns


def test_default_bit_identity_frozen_before_rule_changes():
    rng = np.random.default_rng(482)
    topo = Topology.from_edges(15, np.repeat(np.arange(15), 3), rng.integers(0, 15, 45),
                               rng.integers(-9000, 9001, 45), rng.integers(0, 30, 45))
    sim = RefSim(topo, n_nodes=2, record=[(0, 0), (1, 14)], silenced=np.arange(15) == 13)
    for b in range(2):
        steps, neurons = np.nonzero(rng.random((2100, 15)) < .002)
        sim.add_events(b, steps, neurons, np.full(len(steps), 100000))
    sim.run(2100)
    digest = hashlib.sha256()
    for array in (sim.V, sim.g, sim.r, sim.ring, sim.trace.events, *sim.recorded()):
        digest.update(array.tobytes())
    assert len(sim.trace) == 921
    assert digest.hexdigest() == "a526bf81bba989b095073c8d5a9307c6ca21e25e02eac9c2f37c88aee5f806b6"


@pytest.mark.parametrize("backend", [TorchSim, FastSim])
@pytest.mark.parametrize("field,value", [("clear_g_on_spike", True), ("freeze_g_in_refractory", True),
                                         ("stim_target", "v"), ("stim_disables_refractory", True)])
def test_fast_backends_refuse_each_nondefault_rule(backend, field, value):
    with pytest.raises(NotImplementedError, match=field):
        backend(Topology.empty(1), rules=replace(ModelRules(), **{field: value}))


def test_freeze_holds_g_and_discards_arrivals_even_on_last_held_step():
    topo = Topology.empty(1)
    held = RefSim(topo, rules=ModelRules(freeze_g_in_refractory=True))
    held.g[:] = 20
    held.r[:] = 1
    held.add_events(0, [0], [0], [4000])
    held.step()
    assert held.g[0, 0] == 20
    assert held.r[0, 0] == 0
    held.step()
    assert held.g[0, 0] == 20 * held.c
    default = RefSim(topo)
    default.g[:] = 20
    default.r[:] = 1
    default.add_events(0, [0], [0], [4000])
    default.step()
    assert default.g[0, 0] == 20 * default.c + 68.75


def test_clear_occurs_after_deliver_and_voltage_stimulus_preserves_synaptic_target():
    topo = Topology.from_edges(2, [0, 0], [0, 1], [1000, 1000], [0, 0])
    sim = RefSim(topo, rules=ModelRules(clear_g_on_spike=True, stim_target="v"))
    sim.V[0, 0] = -40
    sim.g[0, 0] = 20
    sim.add_events(0, [0], [1], [4000])
    sim.step()
    assert sim.g[0, 0] == 0
    assert sim.V[0, 0] == sim.params.V_reset
    assert sim.g[0, 1] == 1000 * sim.w_unit
    assert sim.V[0, 1] == sim.params.E_L + 68.75


def test_voltage_delivery_is_gated_during_hold():
    sim = RefSim(Topology.empty(1), rules=ModelRules(stim_target="v"))
    sim.r[:] = 1
    sim.add_events(0, [0], [0], [4000])
    sim.step()
    assert sim.V[0, 0] == sim.params.V_reset
    assert sim.g[0, 0] == 0


def test_refractory_exemption_is_per_replica_per_neuron_and_snapshot_restores_it():
    rules = ModelRules(stim_disables_refractory=True)
    sim = RefSim(Topology.empty(2), n_nodes=2, rules=rules, bias=10)
    sim.r[:] = 17
    sim.add_events(0, [500], [0], [4000])
    assert sim.r.tolist() == [[0, 17], [17, 17]]
    sim.run(400)
    assert sim.stimulated.tolist() == [[True, False], [False, False]]
    snap = sim.snapshot()
    sim.run(500)
    expected = sim.trace.events[sim.trace.events["step"] >= 400].copy()
    restored = RefSim(sim.topo, n_nodes=2, rules=rules, bias=10)
    restored.restore(snap)
    restored.run(500)
    np.testing.assert_array_equal(expected, restored.trace.events)
    with pytest.raises(ValueError, match="ModelRules"):
        RefSim(sim.topo, n_nodes=2).restore(snap)


def test_periodic_phase_ensemble_has_exact_two_hz_not_one_event_per_trial():
    counts = [len(periodic_steps(2, 300, (b + .5) / 50, Params())) for b in range(50)]
    assert sum(counts) == 30
    assert set(counts) == {0, 1}
    with pytest.raises(ValueError, match="15"):
        Fixture("too_big", Topology.empty(16))


def test_single_event_transfer_and_probe_is_causal_in_silent_single_neuron():
    fixture = small_fixtures()[0]
    m0, _ = run_protocol(fixture, CONFIGS["M0_current"], replicas=5)
    m5, baseline = run_protocol(fixture, CONFIGS["M5_shiu"], replicas=5)
    _, probed = run_protocol(fixture, CONFIGS["M5_shiu"], replicas=5, probe=True)
    assert m0["target_drive_rate_hz"] > m5["target_drive_rate_hz"]
    assert m5["target_drive_rate_hz"] == 2
    response = probe_metrics(baseline, probed, 0)
    assert response["extra_spike_fraction"] == 1
    assert response["extra_spike_latency_ms_mean"] == pytest.approx(.1)


@pytest.mark.parametrize("name", ["M0_current", "M5_shiu"])
def test_matched_firing_calibrates_amplitude_and_records_measurements(name):
    result = calibrate(small_fixtures()[0], CONFIGS[name], 10, replicas=2)
    assert result["matched_within_5_percent"]
    assert result["achieved_hz"] == 10
    assert 0 < result["calibrated_quanta"] < 100000
    assert result["measurement"]["target_drive_rate_hz"] == 10
    probe = result["measurement"]["probe_trial"]["probe"]
    assert probe["probe_ms"] == 1050
    assert probe["extra_spike_fraction"] == 1


def test_pair_is_near_m0_sustain_boundary():
    pair = small_fixtures()[1]
    metrics, _ = run_protocol(pair, ModelRules(), drive_ms=5.1, off_ms=994.9,
                              single=True, replicas=1)
    assert metrics["tail_spikes_final_50ms_mean"] > 0
    weaker = Fixture("weaker", Topology.from_edges(2, [0, 1], [1, 0], [2250] * 2, [18] * 2))
    metrics, _ = run_protocol(weaker, ModelRules(), drive_ms=5.1, off_ms=994.9,
                              single=True, replicas=1)
    assert metrics["tail_spikes_final_50ms_mean"] == 0


def test_h0_pipeline_matches_existing_decoder_and_records_all_models():
    fixture = h0_fixture(synthetic_mcns(), source="synthetic test")
    assert fixture.topo.n == 15
    current = h0_transactions(fixture, ModelRules())
    targets = Targets.from_policy(Params(), Policy())
    for row in current:
        expected, _ = run_transaction(fixture.topo, Params(), fixture.embedding, fixture.index,
                                      Transaction(row["a"], row["b"]), targets)
        for key in ("correct", "reset_ok", "completion_latency_ms", "spikes_per_role"):
            assert row[key] == expected[key]
        assert row["passes_existing_checks"]
    for rules in CONFIGS.values():
        rows = h0_transactions(fixture, rules)
        assert len(rows) == 4
        assert {(r["a"], r["b"]) for r in rows} == {(0, 0), (0, 1), (1, 0), (1, 1)}
    shiu = h0_transactions(fixture, CONFIGS["M5_shiu"])
    assert [r["passes_existing_checks"] for r in shiu] == [True, False, False, False]


def test_report_inventory_and_matching_claims():
    path = Path(__file__).resolve().parents[1] / "docs/stage_h/model_comparison.json"
    report = json.loads(path.read_text())
    rows = report["conditions"]
    assert len(rows) == 18
    assert {(r["fixture"], r["configuration"]) for r in rows} == set(itertools.product(
        ("F1_single", "F2_pair", "F3_h0"), CONFIGS))
    assert all(f["n"] <= 15 for f in report["fixtures"])
    for row in rows:
        assert {"P_A", "P_B", "P_C", "P_D", "P_B_68_75mv"} <= row.keys()
        assert row["P_B"]["observed_event_rate_hz"] == 2
        for trial in row["P_D"]:
            error = abs(trial["achieved_hz"] - trial["target_hz"])
            assert trial["matched_within_5_percent"] == (error <= .05 * trial["target_hz"])
            assert trial["measurement"]["target_drive_rate_hz"] == trial["achieved_hz"]
            for key in ("measurement", "F1_matched_input_measurement"):
                assert trial[key]["probe_trial"]["probe"]["probe_ms"] == 1050
            if row["fixture"] == "F1_single":
                assert trial["achieved_hz"] == trial["target_hz"]
        if row["fixture"] == "F3_h0":
            assert len(row["and_transactions"]) == 4


def test_cli_passes_mcns_dir_and_reports_unavailable_real_conditions(tmp_path, monkeypatch):
    from drosophilos.bench import model_compare

    attempted = []

    def denied(*, data_dir):
        attempted.append(data_dir)
        raise PermissionError("test sandbox denies MCNS read")

    monkeypatch.setattr(model_compare, "load_mcns", denied)
    monkeypatch.setattr(model_compare, "campaign", lambda fixtures, **kw: {
        "conditions": [{"fixture": f.name, "configuration": name} for f in fixtures for name in CONFIGS]})
    out = tmp_path / "report.json"
    model_compare.main(["--mcns-dir", str(tmp_path / "mcns"), "--out", str(out)])
    report = json.loads(out.read_text())
    assert attempted == [tmp_path / "mcns"]
    assert len(report["conditions"]) == 18
    missing = [r for r in report["conditions"] if r["fixture"] == "F3_h0"]
    assert len(missing) == 6
    assert all(r["status"] == "not_run" and "PermissionError" in r["reason"] for r in missing)
