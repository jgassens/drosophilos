import numpy as np
import pandas as pd

from drosophilos.bench.h_autapse_census import (autapse_demands, count_bins, fit_summary,
                                                self_edge_table, supply_summary)


def _data():
    neurons = pd.DataFrame({"bodyId": [10, 11, 12, 13], "type": ["a", "b", "c", "d"],
                            "superclass": ["x", "x", "y", "z"], "class": ["C"] * 4,
                            "nt": ["gaba", "acetylcholine", "glutamate", "unclear"],
                            "predicted_nt": ["gaba"] * 4, "nt_conf": [.9] * 4,
                            "sign": [-1, 1, -1, 0]})
    return np.array([0, 1, 2, 3, 0]), np.array([0, 1, 2, 3, 1]), np.array([1, 5, 50, 10, 9]), neurons


def test_self_edge_counting_and_bins():
    pre, post, count, neurons = _data()
    rows = self_edge_table(pre, post, count, neurons)
    assert rows["bodyId"].tolist() == [10, 11, 12, 13]
    assert rows["sign_label"].tolist() == ["inhibitory", "excitatory", "inhibitory", "unknown"]
    assert count_bins(rows.synapse_count) == {"1": 1, "2-4": 0, "5-9": 1, "10-19": 1, "20-49": 0, ">=50": 1}
    summary = supply_summary(rows)
    assert summary["by_sign"] == {"inhibitory": 2, "excitatory": 1, "unknown": 1}


def test_fit_uses_inhibitory_edges_and_thresholds():
    pre, post, count, neurons = _data()
    rows = self_edge_table(pre, post, count, neurons)
    demand = autapse_demands()
    fit = fit_summary(rows, demand)
    assert fit["inhibitory_self_edges"] == 2
    assert fit["excitatory_self_edges"] == 1
    assert fit["latch_autapse"]["carriable_self_edges"] == 1
    assert fit["oneshot_transaction"]["carriable_self_edges"] == 0
    assert fit["latch_autapse"]["carriable_body_ids"] == [12]
