import json

import numpy as np
import pytest

from drosophilos.bench.h_echo_report import build_report, lag_analysis, main, summarize_live


def addition(outgoing, incoming):
    return {"echo_trace": {"bin_ms": 1, "per_ms": {
        "out": {u: list(outgoing) for u in ("count", "abs_quanta")},
        "in_exc": {u: list(incoming) for u in ("count", "abs_quanta")},
        "in_inh": {u: [0] * len(incoming) for u in ("count", "abs_quanta")},
    }}}


def load_fixture(tmp_path, additions):
    path = tmp_path / "synthetic.json"
    path.write_text(json.dumps({"additions": additions}))
    return json.loads(path.read_text())["additions"]


def test_delayed_copy_peak_and_onset(tmp_path):
    delay = 7
    outgoing = [0] * 4 + np.random.default_rng(9).integers(1, 20, 160).tolist()
    incoming = [0] * delay + outgoing[:-delay]
    traces = load_fixture(tmp_path, [addition(outgoing, incoming)])
    for unit in lag_analysis(traces).values():
        correlation = unit["correlation"]
        assert correlation["peak_lag_ms"] == delay
        assert correlation["peak_r"] == pytest.approx(1)
        assert correlation["n_additions_used"] == 1
        assert unit["onset_latency_ms"] == {"n": 1, "median": 7.0, "min": 7.0, "max": 7.0}
        assert unit["early_20_ms_return_ratio"]["median"] == pytest.approx(
            sum(incoming[4:24]) / sum(outgoing[4:24]))


def test_zero_variance_and_short_additions_skipped(tmp_path):
    traces = load_fixture(tmp_path, [addition([1] * 80, [2] * 80),
                                     addition([0] * 80, [0] * 80),
                                     addition([0, 1], [1, 0])])
    result = lag_analysis(traces)
    for unit in result.values():
        assert unit["correlation"]["n_additions_used"] == 0
        assert unit["correlation"]["peak_lag_ms"] is None
        assert all(r is None for r in unit["correlation"]["r_at_lag_ms"].values())
    json.dumps(result, allow_nan=False)


def test_mean_subtracted_and_equal_addition_average(tmp_path):
    outgoing = [0, 3, 1, 2, 5, 4]
    incoming = [5, 2, 4, 3, 0, 1]
    traces = load_fixture(tmp_path, [addition(outgoing, incoming),
                                     addition(outgoing, [v * 100 + 30 for v in outgoing])])
    c = lag_analysis(traces)["abs_quanta"]["correlation"]
    assert c["r_at_lag_ms"]["0"] == pytest.approx(0, abs=1e-15)
    assert c["n_at_lag_ms"]["3"] == 2
    assert c["n_at_lag_ms"]["5"] == 0


def test_present_live_totals_and_inhibition(tmp_path):
    trace = addition([0, 4, 2, 1], [0, 0, 2, 1])
    trace["echo_trace"]["per_ms"]["in_inh"] = {"count": [0, 0, 0, 3],
                                                          "abs_quanta": [0, 0, 0, 6]}
    path = tmp_path / "live.json"
    path.write_text(json.dumps({
        "summary": {"correct": 0, "n": 1, "outcomes": {"wrong": 1}},
        "args": {}, "params": {}, "additions": [trace],
        "echo_trace": {"out": {"count": 7, "abs_quanta": 7},
                       "in_exc": {"count": 3, "abs_quanta": 3},
                       "in_inh": {"count": 3, "abs_quanta": 6}, "metrics": {}},
    }))
    result = summarize_live(json.loads(path.read_text()))
    assert result["totals"]["in"] == {"count": 6, "abs_quanta": 9}
    assert result["return_ratio"]["abs_quanta"] == pytest.approx(9 / 7)
    assert result["inhibitory_share_in_abs_quanta"] == pytest.approx(2 / 3)
    assert result["derived_lag"]["count"]["whole_window_return_ratio"] == pytest.approx(6 / 7)


def test_absent_live_files_are_null(tmp_path, monkeypatch):
    place = tmp_path / "place"
    fullgraph = tmp_path / "fullgraph"
    place.mkdir()
    fullgraph.mkdir()
    (place / "all_w0_rescore.json").write_text(json.dumps({
        "placement": {"carried": 2, "edges": 4, "host_echo_paths": 10,
                      "host_echo_neurons": 3, "neurons": 5},
        "region": "all", "git_commit": "fixture",
        "conditions": {"A": {"fresh": {"correct": 1, "n": 1}}},
    }))
    report = build_report(place, fullgraph)
    baseline = report["placements"][0]
    assert baseline["carried_fraction"] == 0.5
    assert baseline["live"] == {"pub": None, "cal": None}
    assert baseline["live_status"]["pub"] == "not finished"
    assert baseline["live_job_ids"]["pub"] == 4048871
    assert report["placements"][2]["live_status"]["pub"] == "not measured"
    output = tmp_path / "report.json"
    monkeypatch.setattr("sys.argv", ["h_echo_report", "--place-dir", str(place),
                                    "--fullgraph-dir", str(fullgraph), "--out", str(output)])
    main()
    first = output.read_text()
    main()
    assert output.read_text() == first
    assert json.loads(first) == report
