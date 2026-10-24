"""Rule selection and historical-record regressions on tiny synthetic brains only."""

import hashlib
import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from drosophilos.bench import h1_fullgraph as h1
from drosophilos.bench import h_surround_calib as calib
from drosophilos.connectome.embed_image import CircuitView, underlying
from drosophilos.connectome.embed_netlist import Placement
from drosophilos.connectome.mcns import MCNS
from drosophilos.lib.adder import build_adder_channel
from drosophilos.sim.model import ModelRules, Params, Topology
from test_h_surround_calib import p4_mcns


CALIB_FLAGS = ["--grid", "exc=0.5;inh=2;cap=500", "--seeds", "0,3", "--rate", "200",
               "--p4", "--p4-ms", "20", "--p4-rate", "200"]
H1_FLAGS = ["--mapping", "synthetic-mapping.json", "--width", "1", "--n-cases", "2",
            "--max-ms", "20", "--surround", "poisson", "--rate", "1000"]

CURRENT = {"clear_g_on_spike": False, "freeze_g_in_refractory": False,
           "stim_target": "g", "stim_disables_refractory": False}
SHIU = {"clear_g_on_spike": True, "freeze_g_in_refractory": True,
        "stim_target": "v", "stim_disables_refractory": True}

# Captured by running the unchanged bench modules at 8b8f987 with the fixtures and
# flags above, before adding rule selection. H1's clock was fixed by setup_h1.
# Required provenance necessarily adds bytes. Remove ONLY those additions below;
# every pre-existing field, its order, and its serialized bytes must stay identical.
LEGACY_CALIB_SHA256 = "03453db5e49e842a6cb6cecc81f2cbd7c6582418b1dd8df6c44fc75248215fdb"
LEGACY_H1_SHA256 = {
    False: "acd65872e85ecb6c2fcb169bc77de5c6d01a5ff891e2b3fe123c846c3aa9d84a",
    True: "25d34faa6fd96e5634efccef8da38142486d6c330622ef22395d4aae46cde9cd",
}


def setup_h1(monkeypatch):
    """One host per circuit neuron and two outside neurons; no on-disk MCNS data."""
    channel = build_adder_channel(Params(), **h1.adder_build_kwargs({}, width_override=1))
    n = channel.net.n
    neurons = pd.DataFrame({"bodyId": 10_000 + np.arange(n + 2), "type": "synthetic",
                            "superclass": "central", "nt": "acetylcholine",
                            "sign": np.ones(n + 2, np.int8)})
    neurons.loc[n, "superclass"] = "sensory"
    brain = MCNS(neurons, np.array([0, n, n + 1], np.int32),
                 np.array([n, 0, 1], np.int32), np.array([1, 2, 3], np.int32), min_syn=1)
    placement = Placement(dict(enumerate(range(n))), [], 0, [], 0, 0)
    monkeypatch.setattr(h1, "load_mcns", lambda: brain)
    monkeypatch.setattr(h1, "load_mapping_meta", lambda path: {})
    monkeypatch.setattr(h1, "load_mapping", lambda *args: placement)
    monkeypatch.setattr(h1.time, "time", lambda: 1.0)


def digest(raw):
    return hashlib.sha256(raw.encode()).hexdigest()


@pytest.mark.parametrize("bench,required", [(calib, ["--out", "unused.jsonl"]), (h1, [])])
def test_preset_parsing_and_rejection(bench, required, capsys):
    assert bench.parse(required).model_rules == "current"
    for name, expected in (("current", CURRENT), ("shiu", SHIU)):
        args = bench.parse(required + ["--model-rules", name])
        assert args.model_rules == name
        assert asdict(bench.MODEL_RULE_PRESETS[args.model_rules]) == expected
    with pytest.raises(SystemExit) as exc:
        bench.parse(required + ["--model-rules", "shui"])
    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_calibration_default_matches_unchanged_records(tmp_path, monkeypatch):
    monkeypatch.setattr(calib, "load_mcns", p4_mcns)
    path = tmp_path / "calib.jsonl"
    default_raw = None
    for flags in ([], ["--model-rules", "current"]):
        calib.run(calib.parse(CALIB_FLAGS + flags + ["--out", str(path)]))
        raw = path.read_text()
        if default_raw is None:
            default_raw = raw
        assert raw == default_raw  # implicit and explicit current are identical, including metadata
        record = json.loads(raw)
        assert record.pop("model_rules") == CURRENT
        legacy_raw = json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n"
        assert digest(legacy_raw) == LEGACY_CALIB_SHA256


@pytest.mark.parametrize("chained", [False, True])
def test_h1_default_matches_unchanged_records(monkeypatch, chained):
    setup_h1(monkeypatch)
    common = H1_FLAGS + (["--chained"] if chained else [])
    default_raw = None
    for flags in ([], ["--model-rules", "current"]):
        record = h1.run(h1.parse(common + flags))
        raw = json.dumps(record, indent=1, default=str)
        if default_raw is None:
            default_raw = raw
        assert raw == default_raw
        assert record.pop("model_rules") == CURRENT
        assert record["args"].pop("model_rules") == "current"
        for row in record["additions"]:
            assert row.pop("model_rules") == CURRENT
        assert digest(json.dumps(record, indent=1, default=str)) == LEGACY_H1_SHA256[chained]


@pytest.mark.parametrize("p4_only", [False, True])
def test_calibration_shiu_reaches_every_protocol_and_record(tmp_path, monkeypatch, p4_only):
    monkeypatch.setattr(calib, "load_mcns", p4_mcns)
    original = calib.RefSim
    sims = []

    def capture(*args, **kwargs):
        sim = original(*args, **kwargs)
        sims.append(sim)
        return sim

    monkeypatch.setattr(calib, "RefSim", capture)
    path = tmp_path / "calib.jsonl"
    common = CALIB_FLAGS + (["--p4-only"] if p4_only else []) + ["--out", str(path)]
    records = []
    for name, expected in (("current", CURRENT), ("shiu", SHIU)):
        sims.clear()
        calib.run(calib.parse(common + ["--model-rules", name]))
        record = json.loads(path.read_text())
        assert record["model_rules"] == expected
        assert len(sims) == (3 if p4_only else 7)  # three P4 drives + P1, two P2 seeds, P3
        assert all(asdict(sim.rules) == expected for sim in sims)
        records.append(record)
    if not p4_only:
        assert records[0]["P1_sensory"]["total_spikes"] != records[1]["P1_sensory"]["total_spikes"]
    assert records[0]["P4_responsiveness"] != records[1]["P4_responsiveness"]


@pytest.mark.parametrize("chained", [False, True])
def test_h1_shiu_records_and_both_simulators(tmp_path, monkeypatch, chained):
    setup_h1(monkeypatch)
    original = h1.RefSim
    sims = []

    def capture(*args, **kwargs):
        sim = original(*args, **kwargs)
        sims.append(sim)
        return sim

    monkeypatch.setattr(h1, "RefSim", capture)
    path = tmp_path / "h1.json"
    common = H1_FLAGS + (["--chained"] if chained else []) + ["--out", str(path)]
    records = []
    for name, expected in (("current", CURRENT), ("shiu", SHIU)):
        sims.clear()
        record = h1.run(h1.parse(common + ["--model-rules", name]))
        saved = json.loads(path.read_text())
        assert saved["model_rules"] == record["model_rules"] == expected
        assert saved["args"]["model_rules"] == name
        assert all(row["model_rules"] == expected for row in saved["additions"])
        assert len(sims) == (2 if chained else 4)
        assert all(asdict(sim.rules) == expected for sim in sims)
        records.append(saved)
    assert [row["circuit_spikes"] for row in records[0]["additions"]] != [
        row["circuit_spikes"] for row in records[1]["additions"]]


@pytest.mark.parametrize("index_map", [None, [0, 1], [2, 0]])
def test_h1_rules_apply_before_power_on_in_designed_indices(index_map):
    # A delayed pulse must register refractory exemption immediately, including
    # through a nonidentity CircuitView. Changing rules after power-on would fail.
    ch = SimpleNamespace(power_on_events=lambda start: [(start + 5, 0, 100_000)])
    rules = ModelRules(**SHIU)
    n = 3 if index_map == [2, 0] else 2
    iso, full = h1.make_sims(ch, Topology.empty(2), Topology.empty(n), index_map, Params(), rules=rules)
    assert isinstance(full, CircuitView) == (index_map == [2, 0])
    mapped = 0 if index_map is None else index_map[0]
    for sim, target in ((iso, 0), (full, mapped)):
        actual = underlying(sim)
        assert actual.rules == rules
        assert np.flatnonzero(actual.stimulated[0]).tolist() == [target]
        sim.run(10)
        assert actual.trace.events["neuron"].tolist() == [target]
        assert sim.trace.events["neuron"].tolist() == [0]
        assert not actual.g.any()  # the power-on event entered V, never g


@pytest.mark.parametrize("preset", ["current", "shiu"])
@pytest.mark.parametrize("load_case", ["matching", "mismatched", "absent"])
def test_h1_attribution_npz_provenance(tmp_path, monkeypatch, recwarn, preset, load_case):
    setup_h1(monkeypatch)
    path = tmp_path / "scores.npz"
    # No --out: the NPZ must carry provenance even when it is the only saved result.
    h1.run(h1.parse(H1_FLAGS + ["--model-rules", preset, "--attribute", "20",
                               "--attribution-out", str(path)]))
    expected = CURRENT if preset == "current" else SHIU
    with np.load(path, allow_pickle=False) as saved:
        assert json.loads(saved["model_rules"].item()) == expected
        legacy_arrays = {key: saved[key] for key in saved.files if key != "model_rules"}
    if load_case == "absent":
        np.savez_compressed(path, **legacy_arrays)
    run_preset = ("shiu" if preset == "current" else "current") if load_case == "mismatched" else preset
    args = h1.parse(H1_FLAGS + ["--model-rules", run_preset, "--cut-inputs-by", "attribution",
                               "--attribution-in", str(path), "--cut-inputs-frac", "1"])
    if load_case == "mismatched":
        # Refuse before any dataset load or simulation.
        monkeypatch.setattr(h1, "load_mcns", lambda: pytest.fail("loaded brain before checking rules"))
        with pytest.raises(ValueError, match="model rules mismatch") as exc:
            h1.run(args)
        message = str(exc.value)
        assert str(path) in message
        assert f"recorded rules {expected}" in message
        assert f"current run rules {SHIU if run_preset == 'shiu' else CURRENT}" in message
        assert not recwarn.list
        return
    if load_case == "absent":
        with pytest.warns(RuntimeWarning, match="provenance is unknown") as caught:
            record = h1.run(args)
        assert str(path) in str(caught[0].message)
    else:
        record = h1.run(args)
        assert not recwarn.list
    assert record["model_rules"] == expected
    assert record["full_graph_counts"]["inputs_cut_edges"] == 2
    assert record["full_graph_counts"]["inputs_cut_synapses"] == 5


@pytest.mark.parametrize("legacy", [False, True])
def test_h1_attribution_is_read_once_and_legacy_warns_once(tmp_path, monkeypatch, recwarn, legacy):
    setup_h1(monkeypatch)
    path = tmp_path / "scores.npz"
    metadata = {} if legacy else {"model_rules": json.dumps(CURRENT)}
    np.savez_compressed(path, pre_bodyId=np.empty(0, np.int64), post_bodyId=np.empty(0, np.int64),
                        score=np.empty(0, np.int64), **metadata)
    original_load = np.load
    reads = []

    def counted_load(file, *args, **kwargs):
        reads.append(file)
        return original_load(file, *args, **kwargs)

    monkeypatch.setattr(np, "load", counted_load)
    record = h1.run(h1.parse(H1_FLAGS + ["--cut-inputs-by", "attribution", "--attribution-in", str(path),
                                       "--cut-inputs-frac", "1"]))
    assert reads == [path]
    assert record["full_graph_counts"]["inputs_cut_edges"] == 2
    assert len(recwarn.list) == int(legacy)
    if legacy:
        assert "provenance is unknown" in str(recwarn.list[0].message)
