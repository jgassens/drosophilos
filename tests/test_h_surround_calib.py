"""Synthetic-only surround calibration tests; never load the real MCNS brain."""

import json

import numpy as np
import pandas as pd
import pytest

from drosophilos.bench import h_surround_calib as bench
from drosophilos.connectome.embed_image import full_graph_topology
from drosophilos.connectome.mcns import MCNS
from drosophilos.sim.model import Params, Topology


def small_mcns():
    neurons = pd.DataFrame({"bodyId": [100, 101, 102], "type": "test",
                            "superclass": ["sensory", "central", "central"],
                            "nt": ["acetylcholine", "acetylcholine", "gaba"],
                            "sign": np.array([1, 1, -1], np.int8)})
    return MCNS(neurons, np.array([0, 1, 2], np.int32), np.array([1, 2, 1], np.int32),
                np.array([1, 1, 1], np.int32), min_syn=1)


def test_grid_cartesian_defaults_and_errors():
    configs = bench.parse_grid("exc=1,0.75,0.5,0.35,0.25;inh=1,2,4;cap=none,2000,500")
    assert len(configs) == 45
    assert configs[0] == {"surround_scale_exc": 1, "surround_scale_inh": 1, "surround_input_cap": None}
    assert configs[-1] == {"surround_scale_exc": .25, "surround_scale_inh": 4, "surround_input_cap": 500}
    assert bench.parse_grid("exc=1,1") == [configs[0]]
    for invalid in ("exc=-1", "inh=nan", "cap=inf", "cap=-1", "foo=1", "exc=1;exc=2", "exc=", "", "exc=none"):
        with pytest.raises(ValueError, match="grid"):
            bench.parse_grid(invalid)


def test_empty_image_is_exact_anatomy_and_accepts_edgeless_brain():
    m = small_mcns()
    base = m.topology()
    fg = full_graph_topology(m, Params(), bench.empty_image(), base=base)
    assert fg.index_map.size == 0 and fg.topology.n == m.n
    for field in ("src", "dst", "quanta", "delay", "indptr"):
        assert np.array_equal(getattr(base, field), getattr(fg.topology, field))
    empty = MCNS(m.neurons, np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.int32), min_syn=1)
    assert full_graph_topology(empty, Params(), bench.empty_image(), surround_input_cap=0).topology.nnz == 0


def test_hand_built_time_course_rates_decay_and_persistence():
    metrics = bench.ActivityMetrics(3, Params(), driven_sensory=[0])
    # 100-spike peak; mean of active neuron rates uses only the two drive-active cells.
    metrics.observe(100, np.zeros(100, np.int64))  # t=10 ms
    metrics.observe(200, np.ones(20, np.int64))
    metrics.observe(3000, np.zeros(5, np.int64))  # off bin 0
    metrics.observe(3200, np.array([1]))  # 1% exactly is not BELOW threshold
    metrics.observe(5000, np.array([2]))  # rebound in last 100ms delays sustained decay
    out = metrics.finish(["sensory", "central", "central"])
    assert out["active_neurons"] == 3 and out["active_fraction_mcns"] == 3 / 166_700
    assert out["recruited_nonsensory"] == 2
    assert out["off_tail_spikes_per_bin"] == pytest.approx(.1)
    assert out["decays_to_zero"] and not out["self_sustained"]
    assert out["drive_active_neurons"] == 2
    assert out["mean_active_drive_hz"] == pytest.approx(200)
    assert out["median_active_drive_hz"] == pytest.approx(200)
    assert out["decay_ms"] == 210  # rebound at off+200ms; next bin starts at +210ms
    assert out["last_100ms"] == {"spikes": 1, "active_neurons": 1, "population_hz": 10,
                                  "mean_per_neuron_hz": 10 / 3}
    assert out["by_superclass"]["central"]["active_neurons"] == 2
    assert out["spikes_per_bin"][1:3] == [100, 20]
    assert not out["runaway"]


def test_runaway_detects_drive_rise_and_lingering_separately():
    rise = bench.ActivityMetrics(1, Params())
    for bin_index, count in zip(range(25, 30), [1, 2, 3, 4, 5]):
        rise.observe(bin_index * 100, np.zeros(count, np.int64))
    rising = rise.finish()
    assert rising["runaway"] and rising["rising_end_drive"]
    assert rising["decays_to_zero"] and not rising["self_sustained"]
    assert not rising["above_10pct_peak_at_100ms_off"]
    lingering = bench.ActivityMetrics(1, Params())
    lingering.observe(0, np.zeros(100, np.int64))
    lingering.observe(4000, np.zeros(11, np.int64))
    out = lingering.finish()
    assert out["runaway"] and out["above_10pct_peak_at_100ms_off"]
    assert not out["rising_end_drive"]
    silent = bench.ActivityMetrics(1, Params()).finish()
    assert silent["decay_ms"] == 0 and not silent["runaway"]
    assert silent["mean_active_drive_hz"] is None
    assert silent["decays_to_zero"] and not silent["self_sustained"]


@pytest.mark.parametrize("tail_count,decays,self_sustained", [(0, True, False),
                                                            (1, False, False),
                                                            (2, False, True),
                                                            (3, False, True)])
def test_off_tail_thresholds_use_mean_drive_not_peak(tail_count, decays, self_sustained):
    metrics = bench.ActivityMetrics(2, Params(), driven_sensory=[0])
    # Mean drive = 20 spikes/bin, peak = 600; 10% of mean is exactly 2.
    metrics.observe(0, np.zeros(600, np.int64))
    for bin_index in range(50, 60):
        metrics.observe(bin_index * 100, np.ones(tail_count, np.int64))
    out = metrics.finish()
    assert out["off_tail_spikes_per_bin"] == tail_count
    assert out["decays_to_zero"] is decays
    assert out["self_sustained"] is self_sustained
    assert out["recruited_nonsensory"] == int(tail_count > 0)


def test_point_source_exact_rate_bounded_relay_and_recurrent_runaway():
    params = Params()
    bounded = Topology.from_edges(2, [0], [1], [6000], [18])
    recurrent = Topology.from_edges(2, [0, 1], [1, 1], [6000, 6000], [18, 18])
    quiet = bench.simulate_protocol(bounded, params, drive_ms=300, total_ms=600, sources=[0])
    runaway = bench.simulate_protocol(recurrent, params, drive_ms=300, total_ms=600, sources=[0])
    assert quiet["source_spikes_during_drive"] == [64]
    assert quiet["recruited_non_sources"] == 1
    assert quiet["last_100ms"]["spikes"] == 0 and quiet["decay_ms"] is not None
    assert quiet["recruited_nonsensory"] == 2  # P2 has no driven sensory set
    assert quiet["decays_to_zero"] and not quiet["self_sustained"]
    assert not quiet["above_10pct_peak_at_100ms_off"]
    assert runaway["last_100ms"]["spikes"] > 0 and runaway["decay_ms"] is None
    assert runaway["above_10pct_peak_at_100ms_off"] and runaway["runaway"]
    assert runaway["self_sustained"] and not runaway["decays_to_zero"]
    rest = bench.simulate_protocol(recurrent, params, drive_ms=0, total_ms=200)
    assert rest["total_spikes"] == 0 and rest["decay_ms"] is None
    assert rest["off_tail_spikes_per_bin"] is None and rest["decays_to_zero"] is None
    assert not rest["self_sustained"]


def test_grid_job_loads_once_is_reproducible_and_writes_compact_records(tmp_path, monkeypatch):
    loads = []

    def load():
        loads.append(1)
        return small_mcns()

    monkeypatch.setattr(bench, "load_mcns", load)
    path = tmp_path / "nested" / "grid.jsonl"
    args = bench.parse(["--grid", "exc=1,0.5;inh=2;cap=none", "--seeds", "0,3", "--rate", "20", "--out", str(path)])
    assert bench.run(args) == 2 and len(loads) == 1
    raw = path.read_text()
    records = [json.loads(line) for line in raw.splitlines()]
    assert len(records) == 2 and len(raw) < 20_000
    for record in records:
        assert record["full_graph_counts"]["n"] == 3
        assert record["P1_sensory"]["sensory_neurons"] == 1
        p1 = record["P1_sensory"]
        assert p1["recruited_nonsensory"] == p1["by_superclass"]["central"]["active_neurons"]
        assert record["P1_sensory"]["forcing_events"] > 0
        assert record["P1_sensory"]["total_ms"] == 600
        assert len(record["P1_sensory"]["spikes_per_bin"]) == 60
        assert "by_superclass" in record["P1_sensory"]
        assert len(record["P2_point_sources"]) == 2
        assert [r["seed"] for r in record["P2_point_sources"]] == [0, 3]
        assert all(r["source_body_ids"][0] != 100 for r in record["P2_point_sources"])
        assert all(r["source_spikes_during_drive"] == [64] for r in record["P2_point_sources"])
        assert record["P3_rest"]["total_ms"] == 200 and not record["P3_rest"]["any_activity"]
    assert '"events":' not in raw and '"spike_trace":' not in raw
    bench.run(args)
    assert path.read_text() == raw


@pytest.mark.parametrize("flags", [["--rate", "nan"], ["--rate", "-1"], ["--rate", "10001"],
                                    ["--n-sources", "0"], ["--seeds", "-1"], ["--seeds", ""],
                                    ["--sensory-seed", "-1"], ["--grid", "oops=1"],
                                    ["--p4-rate", "nan"], ["--p4-rate", "inf"],
                                    ["--p4-rate", "-1"], ["--p4-rate", "10001"],
                                    ["--p4-ms", "0"], ["--p4-ms", "-10"], ["--p4-ms", "11"]])
def test_cli_validation(flags):
    with pytest.raises(SystemExit):
        bench.parse(["--out", "unused.jsonl", *flags])


def test_point_source_count_cannot_exceed_candidates():
    m = small_mcns()
    with pytest.raises(ValueError, match="non-sensory"):
        bench.calibrate_config(m, m.topology(), Params(), bench.parse_grid("exc=1")[0], n_sources=3)


def p4_mcns():
    neurons = pd.DataFrame({
        "bodyId": np.arange(100, 111),
        "type": ["LB1e", "LB1e", "LB3c", "LB1e", "MN9", "MN9", "DN", "M", "M", "C", "MN9_like"],
        "subclass": ["labellar bristle"] * 3 + ["other"] * 8,
        "superclass": ["sensory"] * 4 + ["cb_motor", "vnc_motor", "descending_neuron",
                                            "cb_motor", "vnc_motor", "central", "central"],
        "nt": "acetylcholine", "sign": np.ones(11, np.int8)})
    # Strong feedforward relays: LB1e -> MN9/DN -> VNC motor; LB3c -> MN9/CB motor.
    return MCNS(neurons, np.array([0, 0, 2, 2, 6], np.int32),
                np.array([4, 6, 5, 7, 8], np.int32), np.full(5, 375, np.int32), min_syn=1)


def test_p4_exact_drive_selection_readouts_and_reproducibility(monkeypatch):
    m = p4_mcns()
    calls = []
    original = bench.surround_events

    def capture(kind, neurons, *args, **kwargs):
        events = original(kind, neurons, *args, **kwargs)
        calls.append((neurons.tolist(), events, kwargs["rate_hz"]))
        assert set(events[1]) == set(neurons)  # all selected cells receive input, no others
        assert events[0].min() >= 0 and events[0].max() < 3000
        return events

    monkeypatch.setattr(bench, "surround_events", capture)
    result = bench.responsiveness_protocol(m, m.topology(), Params())
    assert [call[0] for call in calls] == [[0, 1], [2], [0, 1, 2]]
    assert all(call[2] == 100 for call in calls)
    assert result["seed"] == 0 and result["drive_ms"] == 300 and result["off_ms"] == 200
    assert result["driven_type_counts"] == {"LB1e": 2, "LB3c": 1}
    assert result["mn9_neurons"] == 2 and result["descending_neurons"] == 1
    assert result["motor_neurons"] == 4
    for drive, mask, dn, motors, recruited in zip(result["drives"], [(True, False), (False, True), (True, True)],
                                                [1, 0, 1], [2, 2, 4], [3, 2, 5]):
        assert [cell["body_id"] for cell in drive["mn9"]] == [104, 105]
        assert [cell["spikes"] > 0 for cell in drive["mn9"]] == list(mask)
        for cell in drive["mn9"]:
            assert cell["rate_hz"] == pytest.approx(cell["spikes"] / .5)
            assert cell["drive_rate_hz"] == pytest.approx(cell["drive_spikes"] / .3)
        assert drive["descending_active_neurons"] == dn
        assert drive["motor_active_neurons"] == motors
        assert drive["recruited_nonsensory"] == recruited
        assert drive["decays_to_zero"] and not drive["self_sustained"]
    assert [drive["type"] for drive in result["drives"]] == ["LB1e", "LB3c", None]
    assert [drive["driven_neurons"] for drive in result["drives"]] == [2, 1, 3]
    assert len(json.dumps(result)) < 4000
    repeat = bench.responsiveness_protocol(m, m.topology(), Params())
    assert repeat == result
    for first, second in zip(calls[:3], calls[3:]):
        assert all(np.array_equal(a, b) for a, b in zip(first[1], second[1]))


@pytest.mark.parametrize("missing", [["subclass"], ["type", "subclass"]])
def test_p4_annotation_fallback_aligns_by_body_id(tmp_path, monkeypatch, missing):
    m = p4_mcns()
    path = tmp_path / "annotations.feather"
    m.neurons.iloc[::-1].reset_index(drop=True).to_feather(path)
    expected = bench.p4_populations(m)
    m.neurons = m.neurons.drop(columns=missing)
    monkeypatch.setattr(bench.mcns_download, "path_of", lambda key: path if key == "annotations" else None)
    actual = bench.p4_populations(m)
    for first, second in ((expected[0], actual[0]), (expected[2], actual[2])):
        assert first.keys() == second.keys()
        assert all(np.array_equal(first[key], second[key]) for key in first)
    assert np.array_equal(expected[1], actual[1])
    assert all(column not in m.neurons for column in missing)


def test_p4_requires_labeled_drive_and_readout_populations():
    m = p4_mcns()
    m.neurons.loc[0, "type"] = None
    with pytest.raises(ValueError, match="type annotations"):
        bench.p4_populations(m)
    m.neurons["subclass"] = "other"
    with pytest.raises(ValueError, match="labellar bristle"):
        bench.p4_populations(m)
    m = p4_mcns()
    m.neurons.loc[m.neurons["type"] == "MN9", "type"] = "other"
    with pytest.raises(ValueError, match="MN9"):
        bench.p4_populations(m)


def test_p4_grid_job_applies_knobs_and_only_is_compact(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "load_mcns", p4_mcns)
    path = tmp_path / "p4.jsonl"
    args = bench.parse(["--p4-only", "--grid", "exc=1,0;inh=2;cap=500", "--out", str(path)])
    assert bench.run(args) == 2
    raw = path.read_text()
    records = [json.loads(line) for line in raw.splitlines()]
    assert all("P4_responsiveness" in record and "P1_sensory" not in record for record in records)
    p4 = records[1]["P4_responsiveness"]
    assert all(not any(cell["spikes"] for cell in drive["mn9"]) for drive in p4["drives"])
    assert all(drive["recruited_nonsensory"] == 0 for drive in p4["drives"])
    assert len(raw) < 12_000 and '"spikes_per_bin"' not in raw
    bench.run(args)
    assert path.read_text() == raw
    # --p4 adds the protocol without replacing the original P1/P2/P3 keys.
    args = bench.parse(["--p4", "--p4-ms", "20", "--p4-rate", "200", "--out", str(path)])
    bench.run(args)
    record = json.loads(path.read_text())
    assert {"P1_sensory", "P2_point_sources", "P3_rest", "P4_responsiveness"} <= record.keys()
    assert record["P4_responsiveness"]["drive_ms"] == 20
    assert record["P4_responsiveness"]["rate_hz"] == 200
