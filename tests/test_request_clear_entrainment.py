"""Copy-28 request clear probes using the actual latch and kill-train primitives.

Static draws below reconstruct seed 108, rr copy 28 (100 copies, topology 30643 /
55936), in campaign NumPy draw order. The device stray stream is unavailable:
the 300-q kick constructs its fast excursion, rather than replaying the stall.
The added fifth tap has nominal draws; no capture contains that hypothetical
controller. Nominal-controller reload probes include its residual-charge spike.
These isolated tests do not validate kernel ordering or a new clear policy.
"""

import numpy as np
import pytest

from drosophilos.lib.control import add_kill_train
from drosophilos.lib.netlist import Drive, Netlist
from drosophilos.protocol.latch import add_latch, add_reset
from drosophilos.sim.model import Params
from drosophilos.sim.ref64 import RefSim


PARAMS = Params()
DRIVE = Drive.from_params(PARAMS)
COPY28_PARAMS = {
    "L.u": (-44.89665616330281, -0.24203759753412155),
    "L.v": (-45.50361062035792, 0.07441539314730637),
    "received": (-44.80853838789764, 0.1807660227950656),
    "kill.start.edge": (-45.126441870931245, 0.11233053804317268),
    "kill.start.edge_inh": (-45.02899101677538, 0.012026691559468108),
    "kill.inh": (-44.737115840289526, 0.035445395982901425),
    "kill.h1": (-44.74102529345646, -0.008153303900237377),
    "kill.h2": (-44.82321738212695, 0.053441154131745376),
    "kill.h3": (-44.938364386617074, -0.12538286299600274),
}
COPY28_EDGES = {
    ("L.u", "L.v"): 4050, ("L.v", "L.u"): 3672,
    ("received", "kill.start.edge"): 4728,
    ("received", "kill.start.edge_inh"): 4739,
    ("kill.start.edge", "kill.inh"): 3741,
    ("kill.start.edge", "kill.h1"): 3488,
    ("kill.start.edge_inh", "kill.start.edge"): -8136,
    ("kill.inh", "L.u"): -2893, ("kill.inh", "L.v"): -2741,
    ("kill.h1", "kill.inh"): 3441, ("kill.h1", "kill.h2"): 3474,
    ("kill.h2", "kill.inh"): 3422, ("kill.h2", "kill.h3"): 3661,
    ("kill.h3", "kill.inh"): 3469,
}


def _circuit(pulses, *, copy_controller=True, corner="copy28"):
    net = Netlist(PARAMS)
    latch = add_latch(net, DRIVE, "L")
    source = net.neuron("received")
    inh = add_kill_train(net, DRIVE, "kill", source, [latch], pulses=pulses, strength=0.75)
    for e, (s, d) in enumerate(zip(net.src, net.dst)):
        roles = net.roles[s], net.roles[d]
        if roles in COPY28_EDGES and (copy_controller or roles in (("L.u", "L.v"), ("L.v", "L.u"))):
            net.quanta[e] = COPY28_EDGES[roles]
    vth = np.full(net.n, PARAMS.V_th)
    bias = np.zeros(net.n)
    for i, role in enumerate(net.roles):
        if role in COPY28_PARAMS and (copy_controller or role.startswith("L.")):
            vth[i], bias[i] = COPY28_PARAMS[role]
    if corner != "copy28":
        scale, threshold = corner
        for e, (s, d) in enumerate(zip(net.src, net.dst)):
            if s in latch.members and d in latch.members:
                net.quanta[e] = round(DRIVE.loop * scale)
        vth[list(latch.members)] = PARAMS.V_th + threshold
        bias[list(latch.members)] = 0
    return net, latch, source, inh, vth, bias


def _sweep(pulses, kick, *, copy_controller=True, clears=None):
    net, latch, source, inh, vth, bias = _circuit(pulses, copy_controller=copy_controller)
    clears = np.arange(5000, 6500) if clears is None else np.asarray(clears)
    sim = RefSim(net.topology(), PARAMS, n_nodes=len(clears), V_th=vth, bias=bias)
    for node, clear in enumerate(clears):
        steps, neurons, quanta = [10, int(clear)], [latch.u, source], [4522, DRIVE.ignite]
        if kick:
            steps.append(5020); neurons.append(latch.v); quanta.append(300)
        sim.add_events(node, steps, neurons, quanta)
    sim.run(int(clears.max()) + 1800)
    return sim.trace, latch, inh, clears


@pytest.mark.parametrize("primitive", ["kill", "reset"])
@pytest.mark.parametrize("taps", [4, 5])
def test_nominal_taps_emit_an_extra_inhibitor_spike(primitive, taps):
    net = Netlist(PARAMS)
    latch = add_latch(net, DRIVE, "L")
    if primitive == "kill":
        source = net.neuron("received")
        inh = add_kill_train(net, DRIVE, "kill", source, [latch], pulses=taps, strength=0.75)
    else:
        source, inh, _ = add_reset(net, DRIVE, "reset", [latch], pulses=taps)
    sim = RefSim(net.topology(), PARAMS)
    sim.add_events(0, [100], [source], [DRIVE.ignite])
    sim.run(800)
    spikes = sim.trace.neuron_steps(inh)
    assert len(spikes) == taps + 1, spikes
    assert np.diff(spikes)[-1] > np.diff(spikes)[:-1].max(), spikes


@pytest.fixture(scope="module")
def phase_sweeps():
    return {(taps, kick): _sweep(taps, kick)
            for taps, kick in ((4, False), (4, True), (5, True))}


def test_copy28_real_clear_can_entrain_only_the_kicked_orbit(phase_sweeps):
    for (taps, kick), (trace, latch, inh, clears) in phase_sweeps.items():
        survivors = []
        for node, clear in enumerate(clears):
            u = trace.neuron_steps(latch.u, node)
            v = trace.neuron_steps(latch.v, node)
            train = trace.neuron_steps(inh, node)
            assert len(train) == (4 if taps == 4 else 6), (taps, node, train)
            if np.any(u > clear + 1500):
                survivors.append(node)
                arrival = train[0] + PARAMS.default_delay_steps
                last_u = u[u < arrival][-1]
                # The partner may fire just *after* the first arrival (as in
                # the capture); taking only earlier v spikes wraps the phase.
                assert np.min(np.abs(v - last_u)) <= 6
        print({"taps": taps, "kick": kick, "survivors": survivors})
        if taps == 4 and kick:
            assert survivors
        else:
            assert survivors == []


@pytest.mark.parametrize("taps", [4, 5])
def test_nominal_controller_clears_the_kicked_copy28_latch(taps):
    trace, latch, inh, clears = _sweep(taps, True, copy_controller=False)
    for node, clear in enumerate(clears):
        assert len(trace.neuron_steps(inh, node)) == taps + 1
        assert not np.any(trace.neuron_steps(latch.u, node) > clear + 1500)


def _reload_probe(taps, corner, offsets=range(0, 851, 50)):
    """Offsets are from the last *actual arrival*, not the last configured tap.

    All 11 phases must sustain a train 150 ms after ignition; a transient spike
    is insufficient. The 50-step grid is conservative, not an exact threshold.
    """
    net, latch, source, inh, vth, bias = _circuit(taps, copy_controller=False, corner=corner)
    probe = RefSim(net.topology(), PARAMS, V_th=vth, bias=bias)
    probe.add_events(0, [10, 2000], [latch.u, source], [4522, DRIVE.ignite])
    probe.run(2800)
    baseline = probe.trace
    arrivals = baseline.neuron_steps(inh) + PARAMS.default_delay_steps
    for member in latch.members:
        spikes = baseline.neuron_steps(member)
        assert np.any((spikes > 1500) & (spikes < 2000))
        assert not np.any(spikes > arrivals[-1] + 200)
    cases = [(offset, phase) for offset in offsets for phase in range(0, 44, 4)]
    sim = RefSim(net.topology(), PARAMS, n_nodes=len(cases), V_th=vth, bias=bias)
    for node, (offset, phase) in enumerate(cases):
        sim.add_events(node, [10, 2000 + phase, int(arrivals[-1]) + phase + offset],
                       [latch.u, source, latch.u], [4522, DRIVE.ignite, 4655])
    sim.run(int(arrivals[-1]) + max(offsets) + 1800)
    trace = sim.trace
    success = np.array([np.any(trace.neuron_steps(latch.u, node) > arrivals[-1] + phase + offset + 1500)
                        for node, (offset, phase) in enumerate(cases)]).reshape(-1, 11)
    return arrivals - arrivals[0], np.asarray(list(offsets)), success


@pytest.mark.parametrize("corner", ["copy28", (1.0, 0.0), (0.92, 0.4), (0.88, 0.6)])
@pytest.mark.parametrize("taps", [4, 5])
def test_real_train_reload_margin(taps, corner):
    arrivals, offsets, success = _reload_probe(taps, corner)
    reliable = offsets[np.all(success, axis=1)]
    assert len(reliable), (taps, corner, success)
    minimum = int(reliable[0])
    print({"taps": taps, "corner": corner, "arrivals_from_first": arrivals.tolist(),
           "all_phases_reload_after_last": minimum,
           "all_phases_reload_after_first": int(arrivals[-1] + minimum)})
    row = {"copy28": (250, 300), (1.0, 0.0): (350, 400),
           (0.92, 0.4): (500, 550), (0.88, 0.6): (600, 650)}
    assert minimum == row[corner][taps - 4]
    if taps == 5 and corner == (0.88, 0.6):
        assert arrivals[-1] + minimum > 862  # earliest START ignition budget
        assert np.all(success[offsets == 650])


def test_slow_corner_can_miss_earliest_start_after_five_taps():
    arrivals, offsets, success = _reload_probe(5, (0.88, 0.6), offsets=(610, 650))
    assert arrivals[-1] + offsets[0] == 862
    assert not np.all(success[0])
    assert np.all(success[1])


def _strayed_survival(taps, copies=40000):
    rng = np.random.default_rng(108)
    survived = 0
    for start in range(0, copies, 1000):
        batch = min(1000, copies - start)
        net, latch, source, _, vth, bias = _circuit(taps)
        sim = RefSim(net.topology(), PARAMS, n_nodes=batch, V_th=vth, bias=bias)
        clears = rng.integers(3000, 5000, batch)
        for node, clear in enumerate(clears):
            steps, neurons, quanta = [10, int(clear)], [latch.u, source], [4522, DRIVE.ignite]
            for member in latch.members:
                times = np.flatnonzero(rng.random(6800) < 5 * PARAMS.dt / 1000)
                steps.extend(times)
                neurons.extend([member] * len(times))
                quanta.extend([150] * len(times))
            sim.add_events(node, steps, neurons, quanta)
        sim.run(6800)
        trace = sim.trace
        survived += sum(np.any(trace.neuron_steps(latch.u, node) > clear + 1500)
                        for node, clear in enumerate(clears))
    return int(survived)


@pytest.mark.slow
def test_strayed_copy28_real_train_survival():
    """40,000 clears per setting, chunked to bound RefSim's delivery ring.

    Static controller draws are copy 28's, with a nominal added tap; 5-Hz x
    150-q strays hit both latch members. This is conditional on that controller,
    not a survey of all independently perturbed controllers in the kernel.
    """
    for taps in (4, 5):
        survived = _strayed_survival(taps)
        print({"taps": taps, "survivors": survived, "clears": 40000})
        if taps == 4:
            assert survived > 0
        else:
            assert survived == 0
