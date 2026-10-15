"""Regional Stage H placement and reporting on synthetic data only."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from test_embed_netlist import latch_relay_netlist, synthetic_mcns

from drosophilos.bench.h_stage_place import (REGIONS, host_exposure, main, mapping_meta, region_mask,
                                           run_one, stats, summarize)
from drosophilos.connectome.embed_netlist import Policy


def test_region_masks_are_disjoint_and_cover_the_declared_superclasses():
    names = [s for values in REGIONS.values() for s in values]
    assert len(names) == len(set(names))
    neurons = pd.DataFrame({"superclass": names + ["ol_intrinsic", "cb_intrinsic", "vnc_intrinsic"]})
    masks = {k: region_mask(neurons, k) for k in REGIONS}
    for region, mask in masks.items():
        assert mask.dtype == bool and mask.shape == (len(neurons),)
        assert set(neurons.loc[mask, "superclass"]) == set(REGIONS[region])
    assert (sum(masks.values()) == 1).all()
    assert region_mask(neurons, "all").all()


def test_region_mask_warns_on_unknown_values_and_all_keeps_them():
    neurons = pd.DataFrame({"superclass": ["ol_intrinsic", "new_class", None]})
    with pytest.warns(UserWarning, match="<null>, new_class"):
        assert region_mask(neurons, "optic_lobe").tolist() == [True, False, False]
    with pytest.warns(UserWarning, match="new_class"):
        assert region_mask(neurons, "all").all()
    with pytest.raises(ValueError, match="unknown region"):
        region_mask(neurons, "imaginary")


@pytest.mark.parametrize("config_f", [True, False])
def test_mapping_meta_contract_keys_and_exact_types(config_f):
    meta = mapping_meta("optic_lobe", 1, 4, 0, config_f)
    expected = {"builder": str, "width": int, "ordered": bool, "watchdog_hops": int,
                "storage": str, "rate_robust": bool, "experimental_autapse": bool,
                "ordered_xor_oneshot": bool, "latch_feedback": str, "k_max": float, "isolation_weight": float,
                "region": str, "restarts": int, "seed": int}
    assert set(meta) == set(expected)
    assert all(type(meta[k]) is t for k, t in expected.items())
    assert meta["builder"] == "build_adder_channel" and meta["width"] == 4
    assert meta["ordered"] and meta["watchdog_hops"] == 100 and meta["storage"] == "flipflop"
    assert meta["k_max"] == 4.0 and meta["isolation_weight"] == 1.0
    assert meta["latch_feedback"] == "autapse"
    assert all(meta[k] is config_f for k in ("rate_robust", "experimental_autapse", "ordered_xor_oneshot"))


def test_exposure_separates_nonhosts_from_outside_region():
    _, drive, _ = latch_relay_netlist()
    policy = Policy(k_max=4)
    m, idx = synthetic_mcns(policy, drive, n_distractors=4, n_weak=15)
    hosts = [idx["U"], idx["V"]]
    mask = np.zeros(m.n, bool)
    mask[hosts + [idx["E"]]] = True
    result = host_exposure(m, hosts, mask, policy)
    for group, partners in (("outside_hosts", ~np.isin(np.arange(m.n), hosts)), ("outside_region", ~mask)):
        for direction in ("input", "output"):
            synapses, edges = [], []
            for host in sorted(hosts):
                keep = ((m.post == host) & partners[m.pre]) if direction == "input" else ((m.pre == host) & partners[m.post])
                synapses.append(m.count[keep].sum())
                edges.append(keep.sum())
            assert result[group][f"{direction}_synapses"] == stats(synapses)
            assert result[group][f"{direction}_edges"] == stats(edges)
    assert result["outside_hosts"]["input_synapses"]["total"] > result["outside_region"]["input_synapses"]["total"]
    empty = host_exposure(m, [], mask, policy)
    assert empty["hosts"] == 0 and empty["outside_hosts"]["input_edges"] == stats([])


def _result(correct=2):
    return {"correct": correct, "n": 2, "wrong": 2 - correct, "faults": 0, "timeouts": 0,
            "no_accept": 0, "accept_ms": [12, 14]}


def _row(region, carried, profile3, mean, correct=2):
    metrics = {"input_synapses": stats([mean]), "output_synapses": stats([2]),
               "input_edges": stats([3]), "output_edges": stats([4])}
    return {"region": region, "isolation_weight": 1.0, "config_f": True, "n_cases": 2,
            "region_neurons": 10, "neurons_placed": 4, "neurons_unplaced": 0,
            "carried_edge_pairs": carried, "designed_edge_pairs": 10,
            "profile3": {"total": profile3, "autapses": 1, "non_autapses": profile3 - 1},
            "parasitic_edges_zeroed": 2, "exposure": {"outside_hosts": metrics, "outside_region": metrics},
            "conditions": {"A": {"fresh": _result(correct), "chained": _result(correct)},
                           "B": {"fresh": _result(0), "chained": _result(0)}},
            "search_seconds": 1.2, "restarts": 4, "seed": 0, "git_commit": "abc123"}


def test_summarize_ranking_and_mapping_glob(tmp_path):
    # Each successive row loses at exactly one successive ranking criterion.
    rows = [_row("lowest_input", 7, 3, 10), _row("higher_input", 7, 3, 20),
            _row("more_p3", 7, 4, 1), _row("less_coverage", 6, 2, 1),
            _row("A_failed", 10, 1, 0, correct=1)]
    paths = []
    for i, row in enumerate(reversed(rows)):
        path = tmp_path / f"{i}.json"
        path.write_text(json.dumps(row))
        paths.append(path)
    mapping = tmp_path / "example_mapping.json"
    mapping.write_text(json.dumps({"mapping": {}, "summary": {}}))
    table = tmp_path / "table.md"
    ranked = summarize(paths + [mapping], table)
    assert [r["region"] for r in ranked] == [r["region"] for r in rows]
    text = table.read_text()
    assert "P3 total / autapse / other" in text and "Beyond-region" in text and "mean ACCEPT latency" in text
    assert "2/2 (w/f/t/noA" in text and "abc123" in text and "example_mapping" not in text
    # Full fresh success alone cannot promote a configuration whose reset/chained run fails.
    rows[0]["conditions"]["A"]["chained"] = _result(1)
    paths[-1].write_text(json.dumps(rows[0]))
    assert summarize(paths, table)[0]["region"] == "higher_input"


def test_summarize_cli_accepts_quoted_glob(tmp_path, monkeypatch):
    path = tmp_path / "one.json"
    path.write_text(json.dumps(_row("all", 7, 3, 10)))
    table = tmp_path / "ranking.md"
    monkeypatch.setattr("sys.argv", ["h_stage_place", "--summarize", str(tmp_path / "*.json"), "--table-out", str(table)])
    main()
    assert table.exists() and "| 1 | all |" in table.read_text()


@pytest.mark.parametrize("config_f", [True, False])
def test_bench_end_to_end_synthetic_connectome(tmp_path, config_f):
    # A width-one real adder is small enough to run the real image and H1 harness here.
    # The connectome is the planted five-neuron motif plus weak distractors from embed tests.
    _, drive, _ = latch_relay_netlist()
    m, _ = synthetic_mcns(Policy(k_max=4), drive, n_distractors=8, n_weak=30)
    m.neurons["superclass"] = "ol_intrinsic"
    m.neurons.loc[m.n - 2:, "superclass"] = "cb_intrinsic"
    mapping = tmp_path / "mapping.json"
    out = run_one(m, "optic_lobe", 0.0, n_cases=1, restarts=1, time_limit_s=20,
                  seed=0, mapping_path=mapping, width=1, config_f=config_f)
    saved = json.loads(mapping.read_text())
    for key, value in mapping_meta("optic_lobe", 0.0, 1, 0, config_f=config_f, width=1).items():
        assert saved[key] == value and type(saved[key]) is type(value)
    assert saved["mapping"]
    assert all(v["neuron_index"] < m.n - 2 for v in saved["mapping"].values())
    assert out["region_superclasses"] == list(REGIONS["optic_lobe"])
    assert out["neurons_placed"] + out["neurons_unplaced"] == out["placement"]["neurons"]
    assert out["profile3"]["total"] == out["designed_edge_pairs"] - out["carried_edge_pairs"]
    assert (out["profile3"]["autapses"] > 0) == config_f
    assert out["profile3"]["total"] == out["profile3"]["autapses"] + out["profile3"]["non_autapses"]
    assert len(out["profile3"]["top_non_autapse_classes"]) <= 10
    assert all(not k.startswith("autapse ") for k in out["profile3"]["top_non_autapse_classes"])
    assert out["parasitic_edges_zeroed"] == out["placement"]["parasitic_to_zero"]
    assert out["conditions"]["A"]["fresh"]["correct"] == out["conditions"]["A"]["chained"]["correct"] == 1
    assert out["conditions"]["B"]["fresh"]["correct"] == 0  # failure must still produce a chained tally
    for condition in ("A", "B"):
        for mode in ("fresh", "chained"):
            assert {"correct", "wrong", "faults", "timeouts", "no_accept", "accept_ms"} <= out["conditions"][condition][mode].keys()
            assert "records" not in out["conditions"][condition][mode]
    path = tmp_path / "result.json"
    path.write_text(json.dumps(out))  # no numpy scalars or transaction records escape into JSON
    assert summarize([path, mapping], tmp_path / "ranking.md")[0]["region"] == "optic_lobe"
