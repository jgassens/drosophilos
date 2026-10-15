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


@pytest.mark.parametrize("latch_feedback", ["dale_private", "bogus"])
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
