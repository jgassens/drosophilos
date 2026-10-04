"""Certify the robust request clear's zero-delay links on every delivery path."""

import numpy as np
import pytest
import torch

from drosophilos.lib.control import add_request_clear
from drosophilos.lib.netlist import Drive, Netlist
from drosophilos.protocol.latch import add_latch
from drosophilos.sim import FastSim, Params, RefSim, TorchSim


STEPS = 2203  # crosses ring wraps and ends with a partial graph_steps block


def _run(cls, topo, params, events, **kwargs):
    sim = cls(topo, params, n_nodes=len(events), **kwargs)
    for node, event in enumerate(events):
        sim.add_events(node, *event)
    sim.run(STEPS)
    return sim


@pytest.fixture(scope="module", params=["nominal", "fast30"])
def request_clear_case(request):
    params = Params()
    drive = Drive.from_params(params)
    net = Netlist(params)
    latch = add_latch(net, drive, "request")
    received = net.neuron("received")
    inhibitor = add_request_clear(net, drive, "clear", received, latch)
    thresholds = np.full(net.n, params.V_th)
    if request.param == "fast30":
        # The fast-loop corner that escapes the ordinary clear train.
        thresholds[list(latch.members)] -= 1.2
        for edge, (src, dst) in enumerate(zip(net.src, net.dst)):
            if src in latch.members and dst in latch.members:
                net.quanta[edge] = round(1.3 * drive.loop)
    topo = net.topology()
    assert set(topo.delay) == {0, params.default_delay_steps}
    taps = [net.roles.index(f"clear.h{k}") for k in (1, 2, 3)]
    assert sorted(topo.dst[topo.delay == 0].tolist()) == sorted(taps)
    events = [([10 + offset, 500 + offset, 1600 + offset],
               [latch.u, received, latch.u], [drive.ignite] * 3)
              for offset in (0, 37)]
    ref = _run(RefSim, topo, params, events, V_th=thresholds, record=[(0, taps[0])])
    for node, offset in enumerate((0, 37)):
        train = ref.trace.neuron_steps(inhibitor, node)
        assert (train - train[0]).tolist() == [0, 43, 81, 138]
        for tap in taps:
            assert len(ref.trace.neuron_steps(tap, node)) > 0
        for member in latch.members:
            spikes = ref.trace.neuron_steps(member, node)
            assert np.any((spikes > 100 + offset) & (spikes < 500 + offset))
            assert not np.any((spikes > 1000 + offset) & (spikes < 1600 + offset))
            assert np.any(spikes > 1900 + offset)  # clear, then same-rail reload
    # A zero-delay tap receives conductance on the source's spike step, but
    # integrates it on the next step. Pin the semantics as well as parity.
    relay = net.roles.index("clear.start.edge")
    first = int(ref.trace.neuron_steps(relay, 0)[0])
    voltage, conductance = ref.recorded()
    assert conductance[first - 1, 0] == 0 < conductance[first, 0]
    assert voltage[first, 0] == params.E_L < voltage[first + 1, 0]
    torch_sim = _run(TorchSim, topo, params, events, V_th=thresholds,
                     device="cpu", dtype=torch.float64)
    assert torch_sim.trace == ref.trace
    return topo, params, events, thresholds, ref.trace


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA unavailable")),
])
@pytest.mark.parametrize("delivery", ["scatter", "sparse"])
@pytest.mark.parametrize("graph_steps", [0, 50], ids=["single_steps", "blocks"])
def test_zero_delay_request_clear_is_spike_identical(request_clear_case, device, delivery, graph_steps):
    topo, params, events, thresholds, reference = request_clear_case
    fast = _run(FastSim, topo, params, events, V_th=thresholds,
                device=device, dtype=torch.float64, delivery=delivery, graph_steps=graph_steps)
    assert fast.delays == (0, params.default_delay_steps)
    assert ("sparse CSR" if delivery == "sparse" else "edge-wise") in fast.delivery_path
    assert fast.trace == reference
    if graph_steps and device == "cpu":
        # CPU executes the same K-step function that CUDA captures.
        assert not fast.graph_active and "CUDA" in fast.graph_fallback_reason
