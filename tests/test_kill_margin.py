"""Kill-train margin against a fast latch (docs/tick_stalls.md, seed-108 copy 8).

A latch is a two-neuron loop; a kill train is a few inhibitory pulses on both members, started
by another latch's rise (request pairs, idle pairs, rings, `lib/control.py add_kill_train`).
Mix-B noise (weights +-4 %, threshold drift) makes some loops run fast, and a fast loop can
slip between the pulses of a weak train. Copy 8 stalled when a request's false rail at 33-41
steps survived its third clear. Measured here on one latch with perturbed loop weights and
threshold, the kill started at 12 phases of the loop."""

import numpy as np
import pytest

from drosophilos.lib.control import KILL_PULSES, KILL_STRENGTH, add_kill_train
from drosophilos.lib.netlist import Drive, Netlist
from drosophilos.protocol.latch import add_latch
from drosophilos.sim.model import Params
from drosophilos.sim.ref64 import RefSim

P = Params()
D = Drive.from_params(P)
PHASES = list(range(0, D.loop_period_steps, 4))


def _spikes(sim):
    steps = np.concatenate([np.full(len(nn), int(st[0])) for st, nn in zip(sim._spk_step, sim._spk_neuron)])
    return steps, np.concatenate(sim._spk_node), np.concatenate(sim._spk_neuron)


def _survivors(pulses, strength, loop_scale, dvth, kill_scale=1.0):
    """(mean loop period in steps, phases at which the latch survived the kill)."""
    net = Netlist(P)
    L = add_latch(net, D, "L")
    src = net.neuron("src")
    inh = add_kill_train(net, D, "kill", src, [L], pulses=pulses, strength=strength)
    topo = net.topology()
    n = len(PHASES)
    quanta = np.broadcast_to(topo.quanta, (n, topo.nnz)).copy().astype(np.int64)
    loop_e = ((topo.src == L.u) & (topo.dst == L.v)) | ((topo.src == L.v) & (topo.dst == L.u))
    quanta[:, loop_e] = np.round(topo.quanta[loop_e] * loop_scale)
    quanta[:, topo.src == inh] = np.round(topo.quanta[topo.src == inh] * kill_scale)
    vth = np.full((n, topo.n), P.V_th)
    vth[:, [L.u, L.v]] += dvth
    sim = RefSim(topo, P, n_nodes=n, quanta=quanta, V_th=vth)
    for k, phase in enumerate(PHASES):
        sim.add_events(k, [10, 2000 + phase], [L.u, src], [D.ignite] * 2)
    sim.run(4000)
    steps, nodes, neus = _spikes(sim)
    periods, survived = [], 0
    for k in range(n):
        u = np.sort(steps[(nodes == k) & (neus == L.u)])
        pre = u[(u > 500) & (u < 2000)]
        periods.append(np.mean(np.diff(pre)))
        survived += int(np.any(u > 3200))
    return float(np.mean(periods)), survived


def test_the_default_train_lets_a_fast_latch_through():
    # the margin problem behind seed-108 copy 8: the default 3 x 0.75 train against a
    # +20 % / -0.8 mV loop (38 steps). Stronger trains kill it but broke the kernel under noise
    # (3 x 1.5: 85 of 100 mix-B copies stalled, Juno 413672) or the control machine (4 x 1.5),
    # so the default stays and this test records the open margin; a fix must turn it around.
    from drosophilos.lib.kernel import REQUEST_CLEAR_PULSES
    assert (KILL_PULSES, KILL_STRENGTH) == (3, 0.75) and REQUEST_CLEAR_PULSES == 4
    period, survived = _survivors(3, 0.75, 1.2, -0.8)  # the general train, and the request clear until 2026-09-20
    assert 36 <= period <= 40, period
    assert survived == len(PHASES), survived  # every phase: the clear never works on this latch
    period, survived = _survivors(4, 0.75, 1.12, -0.8)  # 40 steps, copy 73's latch: the fourth pulse kills it at every phase
    assert survived == 0, survived


@pytest.mark.parametrize("loop_scale,dvth", [(1.0, 0.0), (1.08, 0.0)])
def test_the_default_train_kills_a_nominal_latch_at_every_phase(loop_scale, dvth):
    period, survived = _survivors(KILL_PULSES, KILL_STRENGTH, loop_scale, dvth)
    assert survived == 0, (loop_scale, dvth, period, survived)


def test_a_stronger_train_kills_the_fast_latch_in_isolation():
    # 3 x 1.5 kills the +20 % / -0.8 mV loop at every phase (and +30 % / -1.2 mV, 33 steps);
    # the reason it is not the default is the campaign, not this measurement
    for loop_scale, dvth in ((1.2, -0.8), (1.3, -1.2)):
        period, survived = _survivors(3, 1.5, loop_scale, dvth, kill_scale=0.92)
        assert survived == 0, (loop_scale, dvth, period, survived)


def test_a_killed_rail_reloads_150_ms_later():
    # kill pair semantics: r0 lit; r1 ignited (kills r0); r0 ignited again 1,500 steps later
    # (kills r1). Both orders of the pair must hold under fast, nominal and slow loops.
    net = Netlist(P)
    r0, r1 = add_latch(net, D, "r0"), add_latch(net, D, "r1")
    add_kill_train(net, D, "k0", r0.u, [r1])
    add_kill_train(net, D, "k1", r1.u, [r0])
    topo = net.topology()
    # nominal and slow loops; a fast loop (+16 % / -0.8 mV) is the open margin problem above
    # and is not killed by the default train at any distance
    perturb = [(1.0, 0.0), (0.9, 0.8)]
    quanta = np.broadcast_to(topo.quanta, (len(perturb), topo.nnz)).copy().astype(np.int64)
    vth = np.full((len(perturb), topo.n), P.V_th)
    members = [r0.u, r0.v, r1.u, r1.v]
    loop_e = np.isin(topo.src, members) & np.isin(topo.dst, members)
    for k, (ls, dv) in enumerate(perturb):
        quanta[k, loop_e] = np.round(topo.quanta[loop_e] * ls)
        vth[k, members] += dv
    sim = RefSim(topo, P, n_nodes=len(perturb), quanta=quanta, V_th=vth)
    for k in range(len(perturb)):
        sim.add_events(k, [10, 2000, 3500], [r0.u, r1.u, r0.u], [D.ignite] * 3)
    sim.run(6000)
    steps, nodes, neus = _spikes(sim)
    for k in range(len(perturb)):
        alive = {r: bool(np.any(steps[(nodes == k) & (neus == r.u)] > 5200)) for r in (r0, r1)}
        killed = {r: bool(np.any((steps > 2300) & (steps < 3400) & (nodes == k) & (neus == r.u))) for r in (r0,)}
        assert alive[r0] and not alive[r1], (perturb[k], alive)
        assert not killed[r0], perturb[k]  # r1's train did kill r0 before the reload
