"""Small, no-MCNS regressions for the H1 full-graph bench's mapping contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from drosophilos.bench.h1_fullgraph import adder_build_kwargs, make_sims
from drosophilos.connectome.embed_h0 import Policy
from drosophilos.connectome.embed_image import all_missing, build_image, full_graph_topology, load_mapping_meta, run_words
from drosophilos.connectome.embed_netlist import Placement
from drosophilos.connectome.mcns import MCNS
from drosophilos.lib.adder import build_adder_channel
from drosophilos.protocol.handshake import build_channel
from drosophilos.sim.model import Params


def test_stage_h_mapping_meta_rebuilds_configuration_f_and_legacy_is_latch(tmp_path):
    meta = {
        "builder": "build_adder_channel", "width": 4, "ordered": True, "watchdog_hops": 100,
        "storage": "flipflop", "rate_robust": True, "experimental_autapse": True,
        "ordered_xor_oneshot": True, "k_max": 4.0, "isolation_weight": 2.0,
        "region": "whole_brain", "restarts": 4, "seed": 0,
    }
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(meta | {"mapping": {}}))
    loaded = load_mapping_meta(path)
    assert loaded == meta
    f_options = adder_build_kwargs(loaded)
    assert f_options == {key: meta[key] for key in (
        "width", "ordered", "watchdog_hops", "storage", "rate_robust", "experimental_autapse", "ordered_xor_oneshot")}
    f = build_adder_channel(Params(), **f_options)
    assert f.net.n == 740 and f.drive.rate_robust

    legacy_path = tmp_path / "legacy-mapping.json"
    legacy_path.write_text(json.dumps({"mapping": {}}))
    legacy_options = adder_build_kwargs(load_mapping_meta(legacy_path))
    legacy = build_adder_channel(Params(), **legacy_options)
    assert legacy_options == {"width": 4, "ordered": True, "watchdog_hops": 100, "storage": "latch",
                              "rate_robust": False, "experimental_autapse": False, "ordered_xor_oneshot": False}
    assert legacy.net.n == 614
    assert adder_build_kwargs({"storage": "flipflop"})["storage"] == "flipflop"
    with pytest.raises(ValueError, match="disagrees"):
        adder_build_kwargs(loaded, storage_override="latch")
    with pytest.raises(ValueError, match="disagrees"):
        adder_build_kwargs(loaded, width_override=3)


@pytest.mark.parametrize("latch_feedback", ["dale_private", "dale_biased", "bogus"])
def test_legacy_mapping_refuses_explicit_nondefault_latch_feedback(latch_feedback):
    with pytest.raises(ValueError, match="legacy mapping.*latch_feedback"):
        adder_build_kwargs({"latch_feedback": latch_feedback})


def test_saved_docs_mappings_rebuild_their_exact_role_sets():
    """Legacy mappings still identify every role after their builder is reconstructed."""
    docs = Path(__file__).resolve().parents[1] / "docs"
    for name in ("h1_placement_mapping.json", "h1_placement_mapping_isolated.json", "h1_ffadder_mapping.json"):
        path = docs / name
        saved = json.loads(path.read_text())
        options = adder_build_kwargs(load_mapping_meta(path))
        channel = build_adder_channel(Params(), **options)
        assert set(channel.net.roles) == set(saved["mapping"]) | set(saved["unplaced"])


def test_mapping_metadata_reader_observes_a_rewritten_path(tmp_path):
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps({"storage": "latch"}))
    assert load_mapping_meta(path) == {"storage": "latch"}
    path.write_text(json.dumps({"storage": "flipflop"}))
    assert load_mapping_meta(path) == {"storage": "flipflop"}


def test_h1_simulators_power_on_a_flipflop_channel_on_a_cut_synthetic_connectome():
    """The paired H1 setup matches its isolated reference after both graph boundaries are cut."""
    params = Params()
    ch = build_channel(params, 2, storage="flipflop", watchdog_hops=40)
    assert ch.power_on_events(0)  # This regression is meaningful only for the flip-flop store.
    n = ch.net.n
    # One host for each designed neuron, plus a brain-only neuron with edges across both sides of
    # the boundary.  The image supplies every designed edge as Profile 3.
    neurons = pd.DataFrame({"bodyId": 10_000 + np.arange(n + 1), "type": "synthetic",
                            "superclass": "central", "class": None, "somaSide": "R",
                            "nt": "acetylcholine", "nt_conf": .9, "sign": np.ones(n + 1, np.int8)})
    brain = n
    m = MCNS(neurons, np.array([0, brain], np.int32), np.array([brain, 0], np.int32),
             np.array([1, 1], np.int32), min_syn=1)
    placement = Placement(dict(enumerate(range(n))), [], 0, [], 0, 0)
    image = build_image(ch.net, m, placement, Policy(), profile3=all_missing)
    full = full_graph_topology(m, params, image, zero_inputs=True, zero_outputs=True)
    assert full.counts["inputs_cut_edges"] and full.counts["outputs_cut_edges"]

    # Fresh H1 mode: the factory is called once for each word.
    for word in (1, 2):
        iso, whole = make_sims(ch, image.topology, full.topology, full.index_map, params)
        ri, _, _ = run_words(ch, iso, [word], [word], params, 8000)
        rf, _, _ = run_words(ch, whole, [word], [word], params, 8000)
        assert (rf[0].status, rf[0].decoded) == ("valid", word)
        assert whole.trace == iso.trace
        assert (ri[0].status, ri[0].decoded) == (rf[0].status, rf[0].decoded)

    # Chained H1 mode: the same pair is retained for all words, so its pulse must be injected
    # once at creation rather than once per transaction.
    iso, whole = make_sims(ch, image.topology, full.topology, full.index_map, params)
    ri, _, _ = run_words(ch, iso, [1, 2], [1, 2], params, 8000)
    rf, _, _ = run_words(ch, whole, [1, 2], [1, 2], params, 8000)
    assert [(r.status, r.decoded) for r in rf] == [("valid", 1), ("valid", 2)]
    assert whole.trace == iso.trace
    assert [(r.status, r.decoded) for r in ri] == [(r.status, r.decoded) for r in rf]


def test_cli_defaults_omit_attribution_and_validate_opt_in():
    from drosophilos.bench.h1_fullgraph import parse

    defaults = vars(parse([]))
    assert not any(key in defaults for key in ("attribute", "attribution_in", "attribution_out",
                                                "attribution_top", "cut_inputs_by", "cut_inputs_from_types", "echo_trace"))
    assert parse(["--echo-trace"]).echo_trace is True
    assert defaults["cut_inputs_frac"] == defaults["cut_outputs_frac"] == 0
    args = parse(["--attribute", "20", "--attribution-out", "scores.npz", "--attribution-top", "10"])
    assert args.attribute == 20 and args.attribution_out == Path("scores.npz") and args.attribution_top == 10
    typed = parse(["--cut-inputs-from-types", "lLN1_bc, lLN2"])
    assert typed.cut_inputs_from_types == ("lLN1_bc", "lLN2")
    for flags in (["--attribute", "0"], ["--attribute", "nan"], ["--attribute", "inf"],
                  ["--attribution-out", "scores.npz"], ["--cut-inputs-by", "attribution"],
                  ["--attribution-in", "scores.npz"], ["--attribute", "20", "--attribution-top", "-1"],
                  ["--cut-inputs-from-types", "lLN1_bc", "--inputs-zeroed"],
                  ["--cut-inputs-from-types", "lLN1_bc", "--cut-inputs-frac", ".1"]):
        with pytest.raises(SystemExit):
            parse(flags)


def test_cli_surround_knobs_are_opt_in_and_validated():
    from drosophilos.bench.h1_fullgraph import parse

    defaults = vars(parse([]))
    assert not any(key in defaults for key in ("surround_scale_exc", "surround_scale_inh", "surround_input_cap"))
    selected = parse(["--surround-scale-exc", ".35", "--surround-scale-inh", "4", "--surround-input-cap", "500"])
    assert (selected.surround_scale_exc, selected.surround_scale_inh, selected.surround_input_cap) == (.35, 4, 500)
    for flags in (["--surround-scale-exc", "-1"], ["--surround-scale-inh", "nan"],
                  ["--surround-input-cap", "inf"], ["--surround-input-cap", "-1"]):
        with pytest.raises(SystemExit):
            parse(flags)


def test_h1_records_and_applies_selected_surround_knobs(tmp_path, monkeypatch):
    from drosophilos.bench import h1_fullgraph as bench
    from drosophilos.connectome.embed_image import save_mapping

    params = Params()
    channel = build_adder_channel(params, **adder_build_kwargs({}, width_override=1))
    n = channel.net.n
    neurons = pd.DataFrame({"bodyId": 10_000 + np.arange(n + 2), "type": "synthetic",
                            "superclass": "central", "nt": "acetylcholine",
                            "sign": np.ones(n + 2, np.int8)})
    neurons.loc[n + 1, ["nt", "sign"]] = ["gaba", -1]
    m = MCNS(neurons, np.array([0, n, n + 1], np.int32), np.array([n, 0, n], np.int32),
             np.array([4, 6, 3], np.int32), min_syn=1)
    mapping = tmp_path / "mapping.json"
    save_mapping(mapping, channel.net, m, Placement(dict(enumerate(range(n))), [], 0, [], 0, 0))
    monkeypatch.setattr(bench, "load_mcns", lambda: m)
    original = bench.full_graph_topology
    topologies = []

    def capture(*args, **kwargs):
        fg = original(*args, **kwargs)
        topologies.append(fg.topology)
        return fg

    monkeypatch.setattr(bench, "full_graph_topology", capture)
    path = tmp_path / "result.json"
    result = bench.run(bench.parse(["--mapping", str(mapping), "--width", "1", "--n-cases", "1", "--max-ms", "1",
                                    "--surround-scale-exc", ".5", "--surround-scale-inh", "2",
                                    "--surround-input-cap", "2", "--out", str(path)]))
    saved = json.loads(path.read_text())
    for key, value in (("surround_scale_exc", .5), ("surround_scale_inh", 2), ("surround_input_cap", 2)):
        assert result["args"][key] == saved["full_graph_counts"][key] == value
    topo = topologies[0]
    weights = {(int(s), int(d)): int(q) for s, d, q in zip(topo.src, topo.dst, topo.quanta)}
    assert weights[0, n] == 32 and weights[n, 0] == 32 and weights[n + 1, n] == -96


def test_bench_attribution_is_opt_in_and_can_feed_cut_on_synthetic_mcns(tmp_path, monkeypatch):
    from drosophilos.bench import h1_fullgraph as bench
    from drosophilos.connectome.embed_image import save_mapping

    params = Params()
    channel = build_adder_channel(params, **adder_build_kwargs({}, width_override=1))
    n = channel.net.n
    neurons = pd.DataFrame({"bodyId": 10_000 + np.arange(n + 2), "type": "synthetic",
                            "superclass": "sensory", "class": None, "somaSide": "R",
                            "nt": "acetylcholine", "nt_conf": .9, "sign": np.ones(n + 2, np.int8)})
    m = MCNS(neurons, np.array([0, n, n + 1], np.int32), np.array([n, 0, 1], np.int32),
             np.array([1, 2, 3], np.int32), min_syn=1)
    placement = Placement(dict(enumerate(range(n))), [], 0, [], 0, 0)
    mapping = tmp_path / "mapping.json"
    save_mapping(mapping, channel.net, m, placement)
    monkeypatch.setattr(bench, "load_mcns", lambda: m)
    monkeypatch.setattr(bench.time, "time", lambda: 1.0)  # exact JSON comparison excludes wall noise
    common = ["--mapping", str(mapping), "--width", "1", "--n-cases", "2", "--max-ms", "10", "--chained"]
    plain = bench.run(bench.parse(common))
    # The measured metric is observational, with no default JSON or simulation changes.
    measured = bench.run(bench.parse(common + ["--echo-trace"]))
    assert measured.pop("echo_trace")["out"]["count"] >= 0
    for row in measured["additions"]:
        assert len(row.pop("echo_trace")["per_ms"]["out"]["count"]) == 10
    measured["args"].pop("echo_trace")
    assert json.dumps(measured, default=str) == json.dumps(plain, default=str)
    path = tmp_path / "score.npz"
    attributed = bench.run(bench.parse(common + ["--attribute", "20", "--attribution-out", str(path)]))
    assert attributed["attribution"]["failures"] == 2 and path.exists()
    assert all(r["attribution"]["step"] == r["end_step"] for r in attributed["additions"])
    # Strip precisely the opt-in additions; all preexisting output is byte-identical.
    attributed.pop("attribution")
    for row in attributed["additions"]:
        row.pop("attribution")
    attributed["args"].pop("attribute")
    attributed["args"].pop("attribution_out")
    assert json.dumps(attributed, default=str) == json.dumps(plain, default=str)
    cut = bench.run(bench.parse(common + ["--cut-inputs-by", "attribution", "--attribution-in", str(path),
                                         "--cut-inputs-frac", "1"]))
    assert cut["full_graph_counts"]["inputs_cut_edges"] == 2
    assert cut["full_graph_counts"]["inputs_cut_synapses"] == 5
    assert "attribution" not in cut


@pytest.mark.parametrize("hops, expected_lag", [(2, 2), (3, 4)])
@pytest.mark.parametrize("return_sign", [1, -1])
def test_measured_echo_on_known_anatomical_loops(hops, expected_lag, return_sign):
    """One forced host spike returns through two/three anatomical edges in silence."""
    from drosophilos.bench.h1_fullgraph import EchoTrace
    from drosophilos.sim.model import Topology
    from drosophilos.sim.ref64 import RefSim

    params = Params(default_delay_ms=2, t_ref=100)
    neurons = pd.DataFrame({"bodyId": np.arange(hops), "sign": np.ones(hops, np.int8)})
    neurons.loc[hops - 1, "sign"] = return_sign
    m = MCNS(neurons, np.arange(hops, dtype=np.int32),
             np.roll(np.arange(hops, dtype=np.int32), -1),
             np.full(hops, 10_000, np.int32), min_syn=1)
    topo = m.topology(params)
    for outputs_zeroed in (False, True):
        q = topo.quanta.copy()
        if outputs_zeroed:
            q[topo.src == 0] = 0
        active = Topology.from_edges(hops, topo.src, topo.dst, q, topo.delay)
        sim = RefSim(active, params)
        sim.add_events(0, [0], [0], [100_000])
        sim.run(150)  # 15 ms, tiny synthetic graph only
        metric = EchoTrace(active, [0], m.n, params)
        result = metric.observe(sim.trace, 0, 150)
        incoming = result["in"]
        if outputs_zeroed:
            assert incoming == result["out"] == {"count": 0, "abs_quanta": 0}
            assert result["metrics"]["count"] == {
                "return_ratio": 0, "within_20_ms_fraction": 0, "peak_lag_ms": None}
        else:
            assert incoming == result["out"] == {"count": 1, "abs_quanta": 160_000}
            assert result["in_exc" if return_sign > 0 else "in_inh"] == incoming
            for unit in ("count", "abs_quanta"):
                assert result["metrics"][unit] == {
                    "return_ratio": 1, "within_20_ms_fraction": 1, "peak_lag_ms": expected_lag}
                assert metric.report()["metrics"][unit] == result["metrics"][unit]


def test_echo_arrival_window_pending_deliveries_and_synthetic_exclusion():
    from drosophilos.bench.h1_fullgraph import EchoTrace
    from drosophilos.sim.model import Topology
    from drosophilos.sim.trace import SpikeTrace

    # 0 is a host, 1 a real surround neuron, 2 a synthetic circuit neuron.
    topo = Topology.from_edges(3, [0, 1, 0, 2], [1, 0, 2, 0], [16, -32, 99, 99], [20] * 4)
    metric = EchoTrace(topo, [0], 2, Params())
    trace = SpikeTrace.from_arrays([9, 9, 9], [0] * 3, [0, 1, 2])
    first = metric.observe(trace, 0, 20)
    assert first["out"]["count"] == first["in"]["count"] == 0  # arrivals at 29, not clipped
    second = metric.observe(SpikeTrace.empty(), 20, 40, fresh=False)
    assert second["out"] == {"count": 1, "abs_quanta": 16}
    assert second["in_inh"] == {"count": 1, "abs_quanta": 32}
    assert second["per_ms"]["out"]["count"] == [1, 0]
    third = metric.observe(SpikeTrace.empty(), 40, 60, fresh=False)
    assert third["out"]["count"] == third["in"]["count"] == 0


def test_echo_lags_span_chained_windows_and_exclude_out_at_20_ms():
    from drosophilos.bench.h1_fullgraph import EchoTrace
    from drosophilos.sim.model import Topology
    from drosophilos.sim.trace import SpikeTrace

    topo = Topology.from_edges(2, [0, 1], [1, 0], [16, 32], [0, 0])
    metric = EchoTrace(topo, [0], 2, Params())
    metric.observe(SpikeTrace.from_arrays([10], [0], [0]), 0, 20)
    second = metric.observe(SpikeTrace.from_arrays([30, 210], [0, 0], [1, 1]), 20, 220, fresh=False)
    for unit in ("count", "abs_quanta"):
        assert second["metrics"][unit]["within_20_ms_fraction"] == .5
        assert second["metrics"][unit]["peak_lag_ms"] == 2  # ties at 2 and 20 ms choose earliest
    # Fresh runs must not associate IN with OUT from an earlier simulator.
    fresh = metric.observe(SpikeTrace.from_arrays([30], [0], [1]), 20, 40, fresh=True)
    assert fresh["metrics"]["count"]["within_20_ms_fraction"] == 0
    assert fresh["metrics"]["count"]["peak_lag_ms"] is None
