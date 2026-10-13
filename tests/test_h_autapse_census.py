import numpy as np
import pandas as pd

from drosophilos.bench.h_autapse_census import (adder_demand, autapse_demands, count_bins, fit_summary,
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
    assert rows["nt_class"].tolist() == ["inhibitory", "excitatory", "inhibitory", "unknown"]
    assert rows["placement_sign"].tolist() == ["inhibitory", "excitatory", "inhibitory", "excitatory"]
    assert count_bins(rows.synapse_count) == {"1": 1, "2-4": 0, "5-9": 1, "10-19": 1, "20-49": 0, ">=50": 1}
    summary = supply_summary(rows)
    assert summary["by_nt_class"] == {"inhibitory": 2, "excitatory": 1, "unknown": 1}


def test_fit_uses_inhibitory_edges_and_thresholds():
    pre, post, count, neurons = _data()
    rows = self_edge_table(pre, post, count, neurons)
    demand = autapse_demands()
    fit = fit_summary(rows, demand)
    assert fit["inhibitory_self_edges"] == 2
    assert fit["excitatory_self_edges"] == 2
    assert fit["latch_autapse"]["full_rule_carriable_self_edges"] == 0
    assert fit["oneshot_transaction"]["full_rule_carriable_self_edges"] == 0
    assert fit["latch_autapse"]["standalone_body_ids"] == [12]


def test_adder_demand_comes_from_configuration_f_build():
    demand = adder_demand()
    assert demand["netlist_neurons"] == 740
    assert demand["non_f_flipflop_netlist_neurons"] == 664
    assert demand["latch_autapse"]["self_edges"] == 140
    assert demand["latch_autapse"]["quanta"] == -724
    assert demand["oneshot_transaction"]["self_edges"] == 8
    assert demand["oneshot_transaction"]["quanta"] == -15_932
