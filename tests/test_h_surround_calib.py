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
    metrics = bench.ActivityMetrics(3, Params())
    # 100-spike peak; mean of active neuron rates uses only the two drive-active cells.
    metrics.observe(100, np.zeros(100, np.int64))  # t=10 ms
    metrics.observe(200, np.ones(20, np.int64))
    metrics.observe(3000, np.zeros(5, np.int64))  # off bin 0
    metrics.observe(3200, np.array([1]))  # 1% exactly is not BELOW threshold
    metrics.observe(5000, np.array([2]))  # rebound in last 100ms delays sustained decay
    out = metrics.finish(["sensory", "central", "central"])
    assert out["active_neurons"] == 3 and out["active_fraction_mcns"] == 3 / 166_700
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


def test_point_source_exact_rate_bounded_relay_and_recurrent_runaway():
    params = Params()
    bounded = Topology.from_edges(2, [0], [1], [6000], [18])
    recurrent = Topology.from_edges(2, [0, 1], [1, 1], [6000, 6000], [18, 18])
    quiet = bench.simulate_protocol(bounded, params, drive_ms=300, total_ms=600, sources=[0])
    runaway = bench.simulate_protocol(recurrent, params, drive_ms=300, total_ms=600, sources=[0])
    assert quiet["source_spikes_during_drive"] == [64]
    assert quiet["recruited_non_sources"] == 1
    assert quiet["last_100ms"]["spikes"] == 0 and quiet["decay_ms"] is not None
    assert not quiet["above_10pct_peak_at_100ms_off"]
    assert runaway["last_100ms"]["spikes"] > 0 and runaway["decay_ms"] is None
    assert runaway["above_10pct_peak_at_100ms_off"] and runaway["runaway"]
    rest = bench.simulate_protocol(recurrent, params, drive_ms=0, total_ms=200)
    assert rest["total_spikes"] == 0 and rest["decay_ms"] is None


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
                                    ["--sensory-seed", "-1"], ["--grid", "oops=1"]])
def test_cli_validation(flags):
    with pytest.raises(SystemExit):
        bench.parse(["--out", "unused.jsonl", *flags])


def test_point_source_count_cannot_exceed_candidates():
    m = small_mcns()
    with pytest.raises(ValueError, match="non-sensory"):
        bench.calibrate_config(m, m.topology(), Params(), bench.parse_grid("exc=1")[0], n_sources=3)
