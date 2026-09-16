"""Flip-flop latch (drosophilos/protocol/flipflop.py): bistability, switching, tonic rate,
hold, the edge and veto relays on its rail, and the per-neuron bias round trip
Netlist -> Topology -> RefSim / TorchSim. Every test is a tiny circuit, < 3 s of neural time."""

import numpy as np
import pytest

from drosophilos.lib.netlist import Drive, Netlist
from drosophilos.protocol.celement import add_veto_relay
from drosophilos.protocol.flipflop import (add_clear_chain, add_flipflop, add_set_chain, clear_pulse,
                                           power_on_pulse, set_pulse)
from drosophilos.protocol.latch import add_edge_relay, add_latch
from drosophilos.sim.lif_torch import TorchSim
from drosophilos.sim.model import Params, Topology
from drosophilos.sim.ref64 import RefSim

P = Params()
D = Drive.from_params(P)


def _sim(net, ms, events=(), cls=RefSim, **kw):
    topo = net.topology()
    sim = cls(topo, P, bias=topo.sim_bias(), **kw)
    for step, neuron, q in events:
        sim.add_events(0, [step], [neuron], [q])
    sim.run(int(ms / P.dt))
    return sim


def _spikes(sim, i, lo=0, hi=10**9):
    e = sim.trace.events
    st = e["step"][e["neuron"] == i]
    return st[(st >= lo) & (st < hi)]


def _at(events, step):
    return [(step + off, n, q) for off, n, q in events]


def _state(sim, ff, lo, hi):
    su, sv = len(_spikes(sim, ff.u, lo, hi)), len(_spikes(sim, ff.v, lo, hi))
    return "SET" if su and not sv else "CLEAR" if sv and not su else "BOTH" if su and sv else "SILENT"


def test_bistable_and_switches_at_every_phase():
    net = Netlist(P)
    ff = add_flipflop(net, D, "ff")
    # power-on starts CLEAR with u never firing; SET and CLEAR each take at every phase of the running member
    for ph in range(0, 47, 6):
        ev = power_on_pulse(ff, D) + _at(set_pulse(ff, D), 1000 + ph) + _at(clear_pulse(ff, D), 2000 + ph)
        sim = _sim(net, 300, ev)
        assert len(_spikes(sim, ff.u, 0, 1000)) == 0 and _state(sim, ff, 200, 1000) == "CLEAR", ph
        assert _state(sim, ff, 1300 + ph, 2000 + ph) == "SET", ph
        assert _state(sim, ff, 2300 + ph, 3000) == "CLEAR", ph
    # the single-pulse forms switch too
    ev = power_on_pulse(ff, D) + _at(set_pulse(ff, D, pulses=1), 1000) + _at(clear_pulse(ff, D, pulses=1), 2000)
    sim = _sim(net, 300, ev)
    assert _state(sim, ff, 1300, 2000) == "SET" and _state(sim, ff, 2300, 3000) == "CLEAR"


def test_one_ignite_pulse_does_not_set_it():
    """The parked member sits ~15 mV below threshold; the latch's 12.6 mV ignition pulse does nothing."""
    net = Netlist(P)
    ff = add_flipflop(net, D, "ff")
    for ph in range(0, 47, 6):
        sim = _sim(net, 250, power_on_pulse(ff, D) + [(1000 + ph, ff.u, D.ignite)])
        assert _state(sim, ff, 1100 + ph, 2500) == "CLEAR" and len(_spikes(sim, ff.u)) == 0


def test_tonic_rate_matches_the_latch():
    net = Netlist(P)
    ff = add_flipflop(net, D, "ff")
    L = add_latch(net, D, "L")
    sim = _sim(net, 1000, power_on_pulse(ff, D) + _at(set_pulse(ff, D), 500) + [(500, L.u, D.ignite)])
    u, l = _spikes(sim, ff.u, 3000, 10000), _spikes(sim, L.u, 3000, 10000)
    rate_u, rate_l = len(u) / 0.7, len(l) / 0.7
    assert abs(rate_u - 213) / 213 < 0.15, rate_u
    assert abs(rate_u - rate_l) / rate_l < 0.05, (rate_u, rate_l)
    assert np.all(np.diff(u)[5:] == D.loop_period_steps)  # the latch's period, step for step
    assert np.diff(_spikes(sim, ff.u, 500)).min() > P.n_ref  # no doublet at ignition


def test_holds_both_states_for_2s():
    net = Netlist(P)
    ff = add_flipflop(net, D, "ff")
    sim = _sim(net, 2100, power_on_pulse(ff, D) + _at(set_pulse(ff, D), 500))
    assert _state(sim, ff, 1000, 21000) == "SET"
    sim = _sim(net, 2100, power_on_pulse(ff, D))
    assert _state(sim, ff, 1000, 21000) == "CLEAR"


def test_edge_relay_fires_once_per_set():
    net = Netlist(P)
    ff = add_flipflop(net, D, "ff")
    relay = add_edge_relay(net, D, "r", ff.u)
    ev = list(power_on_pulse(ff, D))
    for k in range(3):
        ev += _at(set_pulse(ff, D), 1000 + 3000 * k) + _at(clear_pulse(ff, D), 2500 + 3000 * k)
    sim = _sim(net, 1000, ev)
    r = _spikes(sim, relay)
    assert len(r) == 3, r
    assert all(1000 + 3000 * k < r[k] < 1300 + 3000 * k for k in range(3)), r
    assert len(_spikes(sim, ff.u)) > 80  # the train itself did not re-fire the relay


def test_set_and_clear_chains_from_one_shot_sources():
    """An edge relay's single ignite pulse sets it through add_set_chain; an edge-detected
    trigger clears it through add_clear_chain (the protocol's existing one-shot sources)."""
    net = Netlist(P)
    ff = add_flipflop(net, D, "ff")
    src = add_latch(net, D, "src")
    relay = add_edge_relay(net, D, "r", src.u)
    net.synapse(relay, add_set_chain(net, D, "ff", ff), D.ignite)
    ctrig, _ = add_clear_chain(net, D, "ff", [ff])
    sim = _sim(net, 500, power_on_pulse(ff, D) + [(1000, src.u, D.ignite), (3000, ctrig, D.pulse)])
    assert _state(sim, ff, 1500, 3000) == "SET" and _state(sim, ff, 3300, 5000) == "CLEAR"


def test_veto_relay_driven_by_flipflop_rail():
    """ff.u as a veto rail holds the relay while SET; the relay is drivable again 55 ms after CLEAR."""
    def trial(redrive_ms):
        net = Netlist(P)
        ff = add_flipflop(net, D, "ff")
        drv, tgt = add_latch(net, D, "drv"), add_latch(net, D, "tgt")
        vr = add_veto_relay(net, D, "vr", drv.u, [ff.u], tgt)
        ev = power_on_pulse(ff, D) + _at(set_pulse(ff, D), 100) + [(1000, drv.u, D.ignite)]
        ev += [(2000 + o, n, D.reset) for o in (0, 53, 106, 159) for n in drv.members]  # driver off
        ev += _at(clear_pulse(ff, D), 3000) + [(3000 + redrive_ms * 10, drv.u, D.ignite)]
        sim = _sim(net, 300 + redrive_ms + 150, ev)
        held = len(_spikes(sim, vr, 0, 3000)) == 0 and len(_spikes(sim, tgt.u, 0, 3000)) == 0
        released = len(_spikes(sim, vr, 3000)) == 1 and len(_spikes(sim, tgt.u, 3000 + redrive_ms * 10 + 200)) > 10
        return held, released
    assert trial(55) == (True, True)
    assert trial(30) == (True, False)  # still inside the veto's recovery, as for a latch veto


def test_train_into_both_members_does_not_clear():
    """What add_reset would do to a flip-flop's members: nothing."""
    net = Netlist(P)
    ff = add_flipflop(net, D, "ff")
    train = [(53 * i, n, -int(0.75 * D.loop)) for i in range(4) for n in ff.members]
    sim = _sim(net, 400, power_on_pulse(ff, D) + _at(set_pulse(ff, D), 500) + _at(train, 2000))
    assert _state(sim, ff, 2500, 4000) == "SET"


# ---- per-neuron bias plumbing ------------------------------------------------------

def test_bias_round_trip_netlist_topology_sims():
    net = Netlist(P)
    a = net.neuron("a", bias=58.0)
    b = net.neuron("b")
    net.set_bias(b, 20.0)
    net.neuron("c")
    assert net.bias == [58.0, 20.0, 0.0] and net.summary()["biased_neurons"] == 2
    topo = net.topology()
    assert np.array_equal(topo.bias, [58.0, 20.0, 0.0]) and list(topo.biased_neurons()) == [0, 1]
    assert np.array_equal(topo.sim_bias(), topo.bias) and topo.sim_bias(1.5) == 1.5
    assert np.array_equal(topo.with_added([a], [b], [16], [18]).bias, topo.bias)
    ref = RefSim(topo, P, bias=topo.sim_bias()); ref.run(2000)
    prod = TorchSim(topo, P, bias=topo.sim_bias()); prod.run(2000)
    explicit = RefSim(topo, P, bias=np.array([58.0, 20.0, 0.0])); explicit.run(2000)
    assert ref.trace == prod.trace == explicit.trace
    rates = [len(_spikes(ref, i)) / 0.2 for i in range(3)]
    assert 180 < rates[0] < 245 and 60 < rates[1] < 120 and rates[2] == 0, rates


def test_bias_zero_default_changes_nothing():
    """A netlist without biases yields the topology it always did (bias None), and the two
    simulators produce the same latch spike train from it whether or not they are handed
    topo.sim_bias()."""
    net = Netlist(P)
    L = add_latch(net, D, "L")
    add_edge_relay(net, D, "r", L.u)
    topo = net.topology()
    assert topo.bias is None and topo.sim_bias() == 0.0 and len(topo.biased_neurons()) == 0
    assert net.summary()["biased_neurons"] == 0
    assert Topology.from_edges(net.n, net.src, net.dst, net.quanta, net.delay).bias is None
    assert Topology.from_edges(net.n, net.src, net.dst, net.quanta, net.delay, bias=[0.0] * net.n).bias is None
    traces = []
    for cls in (RefSim, TorchSim):
        for kw in ({}, {"bias": topo.sim_bias()}):
            sim = cls(topo, P, **kw)
            sim.add_events(0, [100], [L.u], [D.ignite])
            sim.run(3000)
            traces.append(sim.trace)
    assert len(traces[0]) > 100 and all(t == traces[0] for t in traces[1:])


def test_split_hubs_copies_carry_the_bias():
    net = Netlist(P)
    h = net.neuron("h", bias=58.0)
    for k in range(5):
        net.synapse(h, net.neuron(f"t{k}"), 16)
    copies = net.split_hubs(2)
    assert copies and all(net.bias[c] == 58.0 for c in copies[h])
    assert len(net.bias) == net.n
