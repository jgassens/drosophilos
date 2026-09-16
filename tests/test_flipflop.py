"""Flip-flop latch (drosophilos/protocol/flipflop.py): bistability, switching, tonic rate,
hold, the edge and veto relays on its rail, and the per-neuron bias round trip
Netlist -> Topology -> RefSim / TorchSim. Every test is a tiny circuit, < 3 s of neural time."""

import numpy as np
import pytest

from drosophilos.lib.netlist import Drive, Netlist
from drosophilos.protocol.celement import add_veto_relay
from drosophilos.protocol.flipflop import (add_clear_chain, add_flipflop, add_set_chain, clear_pulse,
                                           power_on_events, power_on_pulse, set_pulse)
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


# ---- the excitatory proxy p (add_flipflop(proxy=True)) ------------------------------

def _ff_proxy(**kw):
    net = Netlist(P)
    return net, add_flipflop(net, D, "ff", proxy=True, **kw)


def test_proxy_is_excitatory_u_and_default_build_is_unchanged():
    net, ff = _ff_proxy(clear_proxy=True)
    assert (ff.p, ff.q) == (2, 3) and ff.rail == ff.p and ff.proxies == (2, 3) and ff.proxy_quanta == D.loop
    assert net.roles[2:] == ["ff.p", "ff.q"] and net.bias[2:] == [58.0, 58.0]
    assert set(zip(net.src, net.dst, net.quanta)) == {(ff.u, ff.v, -D.loop), (ff.v, ff.u, -D.loop),
                                                      (ff.v, ff.p, -D.loop), (ff.u, ff.q, -D.loop)}
    assert not any(q > 0 for q in net.quanta)  # p and q have no outputs of their own: readers add those
    # SET: p runs with u, step for step, at the latch's rate; q silent. CLEAR: the reverse. Both hold 2 s.
    sim = _sim(net, 2100, power_on_pulse(ff, D) + _at(set_pulse(ff, D), 500))
    pp, uu = _spikes(sim, ff.p, 3000, 21000), _spikes(sim, ff.u, 3000, 21000)
    assert abs(len(pp) / 1.8 - 213) / 213 < 0.05 and np.all(np.diff(pp) == D.loop_period_steps) and len(pp) == len(uu)
    assert len(_spikes(sim, ff.q, 1000)) == 0
    sim = _sim(net, 2100, power_on_pulse(ff, D))
    assert len(_spikes(sim, ff.p)) == 0 and len(_spikes(sim, ff.u)) == 0
    assert abs(len(_spikes(sim, ff.q, 3000, 21000)) / 1.8 - 213) / 213 < 0.05
    # the default build carries no proxy: two neurons, two synapses, rail == u
    net0 = Netlist(P)
    ff0 = add_flipflop(net0, D, "ff")
    assert net0.n == 2 and net0.nnz == 2 and ff0.p is None and ff0.q is None and ff0.rail == ff0.u and ff0.proxies == ()
    assert power_on_pulse(ff0, D) == [(0, ff0.u, -D.loop)]


def test_proxy_switching_time_at_every_phase():
    """p rises 18.5-23.1 ms after the first SET pulse (10.5-14.0 after v's last spike: the parked
    neuron climbs 15 mV with tau_m, as v does on CLEAR) and its last spike is <= 17.8 ms after the
    first CLEAR pulse (v's first spike 9.5-14.0, then one or two p spikes cross before the park)."""
    net, ff = _ff_proxy()
    for ph in range(0, 47, 6):
        ev = power_on_pulse(ff, D) + _at(set_pulse(ff, D), 1000 + ph) + _at(clear_pulse(ff, D), 2500 + ph)
        sim = _sim(net, 400, ev)
        t0, t1 = 1000 + ph, 2500 + ph
        p_on, v_last = _spikes(sim, ff.p, t0)[0], _spikes(sim, ff.v, 0, t1)[-1]
        assert 180 <= p_on - t0 <= 235 and 100 <= p_on - v_last <= 145, (ph, p_on - t0, p_on - v_last)
        assert len(_spikes(sim, ff.p, 0, t0)) == 0 and _state(sim, ff, t0 + 300, t1) == "SET"
        p_after = _spikes(sim, ff.p, t1)
        assert len(p_after) and p_after[-1] - t1 <= 180 and len(_spikes(sim, ff.p, t1 + 180)) == 0, (ph, p_after - t1)


def test_proxy_noise_margins():
    """A stray pulse into p touches only p. While CLEAR: 1.1x ignite never fires it (the same
    margin as u's lockstep threshold); from 1.15x it fires once and is re-parked, and the pair is
    untouched. While SET: a stray 2.25x loop (u's own margin) pauses p for <= 16 ms and it resumes;
    an edge relay on p does not re-fire on the pause."""
    net, ff = _ff_proxy()
    relay = add_edge_relay(net, D, "r", ff.p)
    for ph in range(0, 47, 6):
        sim = _sim(net, 150, power_on_pulse(ff, D) + [(1000 + ph, ff.p, int(round(1.1 * D.ignite)))])
        assert len(_spikes(sim, ff.p)) == 0 and len(_spikes(sim, relay)) == 0, ph
        sim = _sim(net, 150, power_on_pulse(ff, D) + [(1000 + ph, ff.p, int(round(1.15 * D.ignite)))])
        assert len(_spikes(sim, ff.p)) == 1 and _state(sim, ff, 1000 + ph, 1500) == "CLEAR", ph
        assert len(_spikes(sim, relay)) == 1  # one stray p spike is one relay pulse: the reader's real margin is 1.1x
        ev = power_on_pulse(ff, D) + _at(set_pulse(ff, D), 500) + [(1500 + ph, ff.p, -int(round(2.25 * D.loop)))]
        sim = _sim(net, 300, ev)
        pp = _spikes(sim, ff.p, 1400)
        assert 100 <= np.diff(pp).max() <= 160 and len(_spikes(sim, ff.p, 2500)) >= 9, ph
        assert len(_spikes(sim, relay)) == 1 and _state(sim, ff, 1500, 3000) == "SET", ph


def test_edge_relay_on_proxy_fires_once_per_set():
    """The reader's contract redone through p: once per SET (3 of 3), 4.2 ms after p's first
    spike, 22.6-27.3 ms after the first SET pulse over the 47 phases (u: ~11); never on the
    train; once at power-on if the power-on pulse omits p."""
    net, ff = _ff_proxy()
    relay = add_edge_relay(net, D, "r", ff.p)
    ev = list(power_on_pulse(ff, D))
    for k in range(3):
        ev += _at(set_pulse(ff, D), 1000 + 3000 * k) + _at(clear_pulse(ff, D), 2500 + 3000 * k)
    sim = _sim(net, 1000, ev)
    r = _spikes(sim, relay)
    assert len(r) == 3, r
    for k in range(3):
        assert 1220 + 3000 * k <= r[k] <= 1275 + 3000 * k, r
        assert 35 <= r[k] - _spikes(sim, ff.p, 1000 + 3000 * k)[0] <= 50
    assert len(_spikes(sim, ff.p)) > 80
    sim = _sim(net, 100, [(0, ff.u, -D.loop)])  # u's pulse alone: p fires at 2.5 ms and the relay follows
    assert len(_spikes(sim, ff.p)) == 1 and len(_spikes(sim, relay)) == 1


def test_veto_relay_vetoed_by_proxy():
    """ff.p as a veto rail holds the relay while SET when the SET train leads the driver by >= 22
    ms (u: 10; p rises ~12 ms later); after CLEAR the relay is drivable again 60 ms after the
    first clear pulse (u: 45): p's last spike is <= 18 ms after it, and the veto's 55 ms recovery
    runs from there."""
    def trial(redrive_ms, lead_ms=90):
        net, ff = _ff_proxy()
        drv, tgt = add_latch(net, D, "drv"), add_latch(net, D, "tgt")
        vr = add_veto_relay(net, D, "vr", drv.u, [ff.p], tgt)
        ev = power_on_pulse(ff, D) + _at(set_pulse(ff, D), 1000 - lead_ms * 10) + [(1000, drv.u, D.ignite)]
        ev += [(2000 + o, n, D.reset) for o in (0, 53, 106, 159) for n in drv.members]  # driver off
        ev += _at(clear_pulse(ff, D), 3000) + [(3000 + redrive_ms * 10, drv.u, D.ignite)]
        sim = _sim(net, 300 + redrive_ms + 150, ev)
        held = len(_spikes(sim, vr, 0, 3000)) == 0 and len(_spikes(sim, tgt.u, 0, 3000)) == 0
        released = len(_spikes(sim, vr, 3000)) == 1 and len(_spikes(sim, tgt.u, 3000 + redrive_ms * 10 + 200)) > 10
        return held, released
    assert trial(60) == (True, True)
    assert trial(50) == (True, False)
    assert trial(80, lead_ms=22) == (True, True)
    assert trial(80, lead_ms=15)[0] is False  # u's own lead would hold here; p is not up yet


def test_chains_and_kill_train_through_proxy():
    """The set chain and the clear chain (and a kill train into u) work unchanged with a proxy
    built; p follows u with its park delay and power_on_events covers p."""
    net, ff = _ff_proxy()
    src = add_latch(net, D, "src")
    relay = add_edge_relay(net, D, "r", src.u)
    net.synapse(relay, add_set_chain(net, D, "ff", ff), D.ignite)
    ctrig, _ = add_clear_chain(net, D, "ff", [ff])
    assert power_on_events(net, D) == [(0, ff.u, -D.loop), (0, ff.p, -D.loop)]
    sim = _sim(net, 500, power_on_events(net, D) + [(1000, src.u, D.ignite), (3000, ctrig, D.pulse)])
    assert _state(sim, ff, 1500, 3000) == "SET" and _state(sim, ff, 3300, 5000) == "CLEAR"
    assert len(_spikes(sim, ff.p, 1500, 3000)) > 25 and len(_spikes(sim, ff.p, 3300)) == 0
    net, ff = _ff_proxy()
    train = [(53 * i, ff.u, -int(0.75 * D.loop)) for i in range(4)]
    sim = _sim(net, 400, power_on_pulse(ff, D) + _at(set_pulse(ff, D), 500) + _at(train, 2000))
    assert _state(sim, ff, 2500, 4000) == "CLEAR" and len(_spikes(sim, ff.p, 2300)) == 0
    assert _spikes(sim, ff.p, 2000)[-1] - 2000 <= 240


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


# ---- lockstep under mix B (docs/a1_flipflop.md, Lockstep) ----------------------------

def _mix_b_flipflops(pulses: int, n_pert: int = 500, n_phase: int = 4, seed: int = 0):
    """`n_pert` mix-B-perturbed copies of a proxied flip-flop with its set chain (`pulses`) and
    clear chain, each SET and CLEARed twice through the chains at `n_phase` random phases of
    v, with 5 Hz x 150 quanta stray input, as one TorchSim on the CPU. The recipe is
    tests/test_ffregister.py::run_ff_campaign's: log-normal weights on every synapse (the
    chains' too), threshold drift, bias drift on top of the flip-flop's 58 mV. Returns the
    per-node (u, v, p) spike counts in the four hold windows."""
    from drosophilos.lib.campaign import Perturbation

    pert = Perturbation(0.04, 0.2, 0.2, 5.0, 150, 100)  # mix B
    net = Netlist(P)
    ff = add_flipflop(net, D, "ff", proxy=True)
    trig = add_set_chain(net, D, "ff", ff, pulses=pulses)
    ctrig, _ = add_clear_chain(net, D, "ff", [ff])
    topo = net.topology()
    rng = np.random.default_rng(seed)
    B = n_pert * n_phase
    q = np.rint(topo.quanta[None, :] * np.exp(rng.normal(0, pert.weight_sigma, size=(n_pert, topo.nnz)))).astype(np.int32)
    vth = P.V_th + rng.normal(0, pert.th_sigma_mv, size=(n_pert, topo.n))
    bias = topo.sim_bias()[None, :] + rng.normal(0, pert.bias_sigma_mv, size=(n_pert, topo.n))
    q, vth, bias = (np.repeat(a, n_phase, 0) for a in (q, vth, bias))
    sim = TorchSim(topo, P, n_nodes=B, V_th=vth, bias=bias, quanta=q, stray_rate_hz=pert.stray_rate_hz,
                   stray_quanta=pert.stray_quanta, stray_seed=seed)
    t_set, t_clr, t_set2, t_clr2, t_end = 1000, 2500, 4000, 5500, 7000
    for b in range(B):
        ph = int(rng.integers(0, 47))
        ev = power_on_pulse(ff, D) + [(t_set + ph, trig, D.ignite), (t_clr + ph, ctrig, D.pulse),
                                      (t_set2 + ph, trig, D.ignite), (t_clr2 + ph, ctrig, D.pulse)]
        sim.add_events(b, [s for s, _, _ in ev], [n for _, n, _ in ev], [x for _, _, x in ev])
    sim.run(t_end)
    e = sim.trace.events
    out = {}
    for name, lo, hi in (("set1", t_set + 400, t_clr), ("clear1", t_clr + 400, t_set2),
                         ("set2", t_set2 + 400, t_clr2), ("clear2", t_clr2 + 400, t_end)):
        m = (e["step"] >= lo) & (e["step"] < hi)
        out[name] = tuple(np.bincount(e["node"][m & (e["neuron"] == n)], minlength=B) for n in (ff.u, ff.v, ff.p))
    return out


def _classify(cu, cv, thresh=3):
    return np.where((cu >= thresh) & (cv >= thresh), "LOCK", np.where(cu >= thresh, "SET", np.where(cv >= thresh, "CLEAR", "SILENT")))


def test_500_mix_b_flipflops_set_and_clear_without_lockstep():
    """500 mix-B-perturbed flip-flops (x 4 phases = 2,000 copies) SET and CLEAR twice through
    the chains: with the four-pulse default no copy locks step (measured 0 of 16,000 SETs on
    the bench, 0 of 16,000 at 1.5x mix B; the bound here allows 2 of 4,000), every SET holds
    with p running and every CLEAR with p silent. The three-pulse train the same copies were
    built with before locks a few of them (0.1-0.5 % of SETs on the bench), which is the
    register campaign's fail-stop (docs/contracts/ffregister.yaml)."""
    r = _mix_b_flipflops(pulses=4)
    n = len(r["set1"][0])
    for name in ("set1", "set2"):
        cu, cv, cp = r[name]
        st = _classify(cu, cv)
        assert (st == "LOCK").sum() <= 2 and ((st != "SET") & (st != "LOCK")).sum() == 0, (name, np.unique(st, return_counts=True))
        assert (cp[st == "SET"] >= 0.9 * cu[st == "SET"]).all()  # p runs with u
    for name in ("clear1", "clear2"):
        cu, cv, cp = r[name]
        st = _classify(cu, cv)
        assert (st == "CLEAR").all(), (name, np.unique(st, return_counts=True))
        assert (cp <= 2).all()  # p parked (a stray may fire it once)
    r3 = _mix_b_flipflops(pulses=3)
    locks3 = sum(int((_classify(*r3[k][:2]) == "LOCK").sum()) for k in ("set1", "set2"))
    assert locks3 >= 1, locks3  # the design this replaces, on the same perturbations
    assert all((_classify(*r3[k][:2]) == "CLEAR").all() for k in ("clear1", "clear2"))  # the clear train resolves every lockstep
