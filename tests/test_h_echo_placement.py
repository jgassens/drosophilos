"""Output/echo placement regressions; every graph here is synthetic."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from test_embed_netlist import latch_relay_netlist, synthetic_mcns

from drosophilos.bench.h1_fullgraph import adder_build_kwargs
from drosophilos.bench.h_stage_place import host_exposure, main, mapping_meta, run_one
from drosophilos.connectome.embed_image import load_mapping, load_mapping_meta, save_mapping
from drosophilos.connectome.embed_netlist import (Placement, Policy, _Anatomy, _Design, audit,
                                                host_echo, isolation_penalty, penalty_cap, place_netlist)
from drosophilos.connectome.mcns import MCNS
from drosophilos.lib.adder import build_adder_channel
from drosophilos.lib.netlist import Netlist
from drosophilos.sim.model import Params


@pytest.mark.parametrize("weight,digest", [
    (None, "0c989626a1f2e215877b6e44db39f512b901275206dc55a7db2651c200dbfdaf"),
    (0.0, "0c989626a1f2e215877b6e44db39f512b901275206dc55a7db2651c200dbfdaf"),
    (1.0, "60321dedd1f20005f2fd187679aefa654cafceaa7f05a3f89e439f908ca5c6a3"),
    (2.0, "414804a41b6db11175ca6c391e79457e840a2db8067836fdc68a130e983c7690"),
])
def test_frozen_prechange_placement_digest(weight, digest):
    # Captured BEFORE the implementation changed: all original Placement fields except
    # wall-clock seconds/order_note. Covers mapping, repair, restart choice, exact float
    # objective, exposure and the complete edge/motif audit, not just carried coverage.
    net, drive, _ = latch_relay_netlist()
    m, _ = synthetic_mcns(Policy(), drive, twin_latch=True)
    kwargs = {} if weight is None else {"isolation_weight": weight}
    pl = place_netlist(net, m, restarts=3, time_limit_s=30, seed=7, **kwargs)
    original = dict(vars(pl))
    for key in ("seconds", "order_note", "isolation_in", "isolation_out", "echo_weight",
                "self_echo", "host_echo_paths", "host_echo_neurons"):
        original.pop(key)
    raw = json.dumps(original, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(raw.encode()).hexdigest() == digest
    assert pl.summary(net)["isolation_in"] == (weight or 0.0)
    assert pl.summary(net)["isolation_out"] == 0.25 * (weight or 0.0)


@pytest.mark.parametrize("key", ["isolation_in", "isolation_out", "echo_weight"])
@pytest.mark.parametrize("legacy", [0.0, 2.0])
def test_legacy_conflicts_even_with_explicit_new_zero(key, legacy):
    net, drive, _ = latch_relay_netlist()
    m, _ = synthetic_mcns(Policy(), drive, n_distractors=0, n_weak=0)
    with pytest.raises(ValueError, match="isolation_weight cannot be combined"):
        place_netlist(net, m, isolation_weight=legacy, **{key: 0.0})


def graph(n, edges):
    neurons = pd.DataFrame({"bodyId": 1000 + np.arange(n), "type": "synthetic",
                            "superclass": "cb_intrinsic", "nt": "acetylcholine", "sign": 1})
    pre, post, count = np.asarray(edges, np.int32).T
    return MCNS(neurons, pre, post, count, min_syn=1)


def echo_graph():
    # Includes duplicate rows, an autapse, cross-host returns, one-way input and products
    # larger than int32. Counts must be merged before computing two-step paths.
    return graph(6, [(0, 0, 11), (0, 2, 2), (0, 2, 3), (2, 0, 7), (1, 2, 3),
                     (2, 1, 4), (0, 1, 13), (1, 0, 17), (1, 3, 19), (3, 0, 23),
                     (4, 1, 29), (3, 5, 50000), (5, 3, 60000)])


def test_self_echo_is_weighted_excludes_autapses_and_is_cached():
    anatomy = _Anatomy(echo_graph(), Policy())
    expected = [256, 233, 47, 3000000000, 0, 3000000000]
    assert anatomy.self_echo.dtype == np.int64
    assert anatomy.self_echo.tolist() == expected
    assert anatomy.self_echo is anatomy.self_echo
    penalty = isolation_penalty(anatomy, isolation_in=2, isolation_out=3, echo_weight=4)
    np.testing.assert_array_equal(penalty, 2 * np.log1p(anatomy.in_syn)
                                  + 3 * np.log1p(anatomy.out_syn) + 4 * np.log1p(expected))
    assert penalty_cap(anatomy, isolation_in=2, isolation_out=3, echo_weight=4) >= penalty.max()
    # Old helper call shapes and their exact arithmetic remain supported.
    np.testing.assert_array_equal(isolation_penalty(anatomy, 1.7),
                                  1.7 * (np.log1p(anatomy.in_syn) + 0.25 * np.log1p(anatomy.out_syn)))


def test_host_echo_counts_cross_host_paths_and_excludes_host_intermediates():
    m = echo_graph()
    anatomy = _Anatomy(m, Policy())
    hosts = np.array([0, 1])
    assert host_echo(anatomy, hosts) == {"host_echo_paths": 525, "host_echo_neurons": 2}
    assert host_echo(anatomy, np.arange(m.n)) == {"host_echo_paths": 0, "host_echo_neurons": 0}
    assert host_echo(anatomy, []) == {"host_echo_paths": 0, "host_echo_neurons": 0}
    net = Netlist(Params())
    net.neuron("a")
    net.neuron("b")
    pl = audit(net, _Design(net, Policy(), 20), anatomy, hosts, 0., {}, "hand", Policy(),
               isolation_in=2, isolation_out=3, echo_weight=4)
    summary = pl.summary(net)
    assert summary["host_echo_paths"] == 525 and summary["host_echo_neurons"] == 2
    assert summary["self_echo"] == {0: 256, 1: 233}
    assert summary["external_out_synapses"] == 27
    assert summary["external_out_mean"] == 13.5 and summary["external_out_p90"] == 20.3
    cost = (2 * np.log1p([30, 33]) + 3 * np.log1p([5, 22])).sum()
    cost += 4 * np.log1p([256, 233]).sum()
    assert pl.objective == -cost
    bench = host_exposure(m, hosts, np.ones(m.n, bool), Policy())
    assert bench["host_echo_paths"] == 525 and bench["host_echo_neurons"] == 2
    assert bench["outside_hosts"]["output_synapses"]["total"] == 27


@pytest.mark.parametrize("weights", [{"isolation_out": 100.}, {"echo_weight": 100.}])
def test_output_or_echo_weight_prefers_quieter_equally_carriable_host(weights):
    net = Netlist(Params())
    a, b = net.neuron("source"), net.neuron("sink")
    net.synapse(a, b, 100)
    req = Policy().req_count(100)
    # Both 0 and 1 can supply the same designed edge to 2. Only 0 excites the surround
    # and has reciprocal paths; the host pool excludes the surrounding neurons.
    m = graph(8, [(0, 2, req), (1, 2, req)]
              + [(0, x, 2) for x in range(3, 8)] + [(x, 0, 2) for x in range(3, 8)])
    for seed in range(3):
        pl = place_netlist(net, m, allowed_hosts=[0, 1, 2], restarts=2,
                           time_limit_s=10, seed=seed, isolation_in=0., **weights)
        assert pl.carried == 1 and pl.mapping == {a: 1, b: 2}
        assert pl.host_echo_paths == 0


def test_optional_meta_roundtrip_preserves_rebuilt_adder_and_legacy(tmp_path):
    meta = mapping_meta("all", None, 4, 0, isolation_in=0., isolation_out=8., echo_weight=2.)
    original_meta = {k: v for k, v in meta.items() if k not in ("isolation_in", "isolation_out", "echo_weight")}
    opts = adder_build_kwargs(meta)
    assert opts == adder_build_kwargs(original_meta)
    ch = build_adder_channel(Params(), **opts)
    m = echo_graph()
    pl = Placement({0: 0}, list(range(1, ch.net.n)), 0, [], 0, 0.)
    path = tmp_path / "mapping.json"
    save_mapping(path, ch.net, m, pl, meta)
    loaded = load_mapping_meta(path)
    assert loaded == meta
    rebuilt = build_adder_channel(Params(), **adder_build_kwargs(loaded))
    assert vars(ch.net) == vars(rebuilt.net)
    assert load_mapping(path, rebuilt.net, m).mapping == pl.mapping
    save_mapping(path, ch.net, m, pl, original_meta)
    assert load_mapping_meta(path) == original_meta
    assert adder_build_kwargs(load_mapping_meta(path)) == opts


def test_bench_independent_weights_cli_end_to_end(tmp_path, monkeypatch):
    _, drive, _ = latch_relay_netlist()
    m, _ = synthetic_mcns(Policy(k_max=4), drive, n_distractors=8, n_weak=30)
    m.neurons["superclass"] = "cb_intrinsic"
    path = tmp_path / "mapping.json"
    result = tmp_path / "result.json"
    monkeypatch.setattr("drosophilos.bench.h_stage_place.load_mcns", lambda: m)
    monkeypatch.setattr("drosophilos.bench.h_stage_place.run_one",
                        lambda *args, **kwargs: run_one(*args, **kwargs, width=1))
    monkeypatch.setattr("sys.argv", ["h_stage_place", "--region", "all", "--cases", "1",
                                    "--restarts", "1", "--time-limit", "20", "--conditions", "A",
                                    "--isolation-in", "0", "--isolation-out", "2", "--echo-weight", "1",
                                    "--out", str(result), "--mapping-out", str(path)])
    main()
    out = json.loads(result.read_text())
    for key, expected in (("isolation_in", 0.), ("isolation_out", 2.), ("echo_weight", 1.)):
        assert out[key] == out["placement"][key] == load_mapping_meta(path)[key] == expected
    assert out["placement"]["host_echo_paths"] == out["exposure"]["host_echo_paths"]
    assert out["conditions"]["A"]["fresh"]["correct"] == 1


def test_cli_conflict_rejected_before_loading_anatomy(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["h_stage_place", "--isolation-weight", "0", "--echo-weight", "0",
                                    "--out", str(tmp_path / "out.json"),
                                    "--mapping-out", str(tmp_path / "map.json")])
    with pytest.raises(SystemExit, match="2"):
        main()
