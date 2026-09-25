from types import SimpleNamespace

from drosophilos.bench.repro import circuit_hash
from drosophilos.lib.netlist import Netlist
from drosophilos.sim.model import Params


def _pipeline_with_bias(bias):
    net = Netlist(Params())
    net.neuron("first", bias=bias[0])
    net.neuron("second", bias=bias[1])
    return SimpleNamespace(net=net, const_values={})


def test_circuit_hash_includes_per_neuron_bias_in_neuron_order():
    baseline = _pipeline_with_bias([0.0, 0.0])
    changed = _pipeline_with_bias([0.0, -0.002])
    assert circuit_hash(baseline) != circuit_hash(changed)


def test_circuit_hash_is_identical_for_identical_biases():
    assert circuit_hash(_pipeline_with_bias([0.0, -0.002])) == circuit_hash(
        _pipeline_with_bias([0.0, -0.002]))
