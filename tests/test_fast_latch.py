"""Fast-latch regressions and reproducible laptop-sized mix-B measurements.

The slow surveys use RefSim, the campaign's independent per-edge/per-neuron draws,
and exact sparse Bernoulli background events on every neuron (including dark rails).
"""

from dataclasses import replace

import numpy as np
import pytest

from drosophilos.bench.a2_campaigns import MIXES
from drosophilos.lib.netlist import Drive, Netlist
from drosophilos.lib.control import add_kill_train
from drosophilos.lib.gates import add_maj_latched
from drosophilos.protocol.celement import (
    add_and_gate, add_and_latched, add_or_latched, add_veto_neuron, add_veto_relay,
)
from drosophilos.protocol.latch import add_latch
from drosophilos.protocol.rate import condition_rate_gate, rate_tap
from drosophilos.sim.model import Params
from drosophilos.sim.ref64 import RefSim


P = Params()
D = Drive.from_params(P)


class MixProbe(RefSim):
    """Discard unrelated spikes as we run; retain only the requested tap records."""

    def __init__(self, net, copies, seed, watch, *, noisy=True, stray=True, **kwargs):
        topo = net.topology()
        rng = np.random.default_rng(seed)
        mix = MIXES["B"]
        if noisy:
            kwargs.update(
                quanta=np.rint(topo.quanta * np.exp(rng.normal(0, mix.weight_sigma, (copies, topo.nnz)))),
                V_th=P.V_th + rng.normal(0, mix.th_sigma_mv, (copies, topo.n)),
                bias=np.asarray(net.bias) + rng.normal(0, mix.bias_sigma_mv, (copies, topo.n)),
            )
        super().__init__(topo, P, n_nodes=copies, **kwargs)
        self.rng = np.random.default_rng(int(rng.integers(2**31 - 1)))
        self.stray = stray
        self.watch = tuple(watch)
        self.events = {n: ([], []) for n in watch}

    def step(self):
        if self.stray:
            mix = MIXES["B"]
            count = self.rng.binomial(self.B * self.n, mix.stray_rate_hz * P.dt / 1000)
            flat = self.rng.choice(self.B * self.n, count, replace=False)
            self._events[self.step_index].append(
                (flat // self.n, flat % self.n, np.full(count, mix.stray_quanta)))
        super().step()
        if self._spk_neuron:
            neurons, nodes = self._spk_neuron[-1], self._spk_node[-1]
            for neuron in self.watch:
                hit = nodes[neurons == neuron]
                if len(hit):
                    steps, copies = self.events[neuron]
                    steps.append(np.full(len(hit), self.step_index - 1, dtype=np.int32))
                    copies.append(hit.astype(np.int32))
        self._spk_step.clear()
        self._spk_node.clear()
        self._spk_neuron.clear()

    def spikes(self, neuron):
        steps, nodes = self.events[neuron]
        if not steps:
            return np.empty(0, dtype=int), np.empty(0, dtype=int)
        return np.concatenate(steps), np.concatenate(nodes)

    def counts(self, neuron, since=0):
        steps, nodes = self.spikes(neuron)
        return np.bincount(nodes[steps >= since], minlength=self.B)

    def ignite(self, neurons, step=10, quanta=None):
        nodes = np.arange(self.B)
        for neuron in neurons:
            self._events[step].append((nodes, np.full(self.B, neuron),
                                      np.full(self.B, D.ignite if quanta is None else quanta)))


def period_survey(drive=D, copies=10_000, seed=109, steps=6000, stray=True):
    net = Netlist(P)
    latch = add_latch(net, drive, "L")
    tap = rate_tap(net, drive, latch.u)
    sim = MixProbe(net, copies, seed, set([*latch.members, tap]), stray=stray)
    sim.ignite([latch.u])
    sim.run(steps)
    raw = _periods(sim, latch.u)
    if tap != latch.u:
        raw["readout"] = _periods(sim, tap)
    return {"copies": copies, "seed": seed, "steps": steps, "stray": stray, **raw}


def _periods(sim, neuron):
    st, nd = sim.spikes(neuron)
    order = np.lexsort((st, nd))
    st, nd = st[order], nd[order]
    valid = (nd[1:] == nd[:-1]) & (st[:-1] >= 2000)
    periods, owners = np.diff(st)[valid], nd[1:][valid]
    count = np.bincount(owners, minlength=sim.B)
    means = np.divide(np.bincount(owners, weights=periods, minlength=sim.B), count,
                      out=np.full(sim.B, np.nan), where=count > 0)
    percentiles = [0, 1, 5, 50, 95, 99, 100]
    return {
        "silent": int((count == 0).sum()),
        "percentiles": percentiles,
        "mean_period": np.nanpercentile(means, percentiles).tolist(),
        "interval": np.percentile(periods, percentiles).tolist(),
        "interval_count": len(periods),
    }


def _prescribed_fault(robust, period, live, *, fraction=.55, scale=1., corner=False):
    """Prescribed spike arrivals, including all 34 phases of the copy-73 stray."""
    drive = replace(D, rate_robust=robust)
    net = Netlist(P)
    sources = [net.neuron("a"), net.neuron("b")]
    if fraction == .65:
        gate, _ = add_and_latched(net, drive, "completion", sources)
    else:
        gate = add_and_gate(net, drive, "fault", sources, fraction=fraction)
    topo = net.topology()
    copies = max(34, period)
    q = np.broadcast_to(topo.quanta, (copies, topo.nnz)).copy()
    q[:] = np.rint(q * scale)
    vth = np.full((copies, topo.n), P.V_th)
    bias = np.broadcast_to(net.bias, (copies, topo.n)).copy()
    if corner:
        # The recapture did not record the gate's Vth/bias draws. This explicit 4-sigma
        # threshold / 2.5-sigma bias corner reproduces a false spike with its recorded
        # 211-q tap, 34-step rail and 150-q stray; those three values alone do not make
        # an unperturbed 7-mV-gap gate fire.
        vth[:, gate] -= .8
        bias[:, gate] += .5
    sim = MixProbe(net, copies, 0, [gate, *net.rate_readouts.values()], noisy=False,
                   stray=False, quanta=q, V_th=vth, bias=bias)
    for edge in np.flatnonzero(np.isin(topo.src, sources[:live])):
        phases = range(period) if topo.src[edge] == sources[1] else [0]
        for phase in phases:
            nodes = np.flatnonzero(np.arange(copies) % period == phase) if len(phases) > 1 else np.arange(copies)
            for step in range(100 + phase, 4000, period):
                sim._events[step + int(topo.delay[edge])].append(
                    (nodes, np.full(len(nodes), topo.dst[edge]), q[nodes, edge]))
    if corner:
        for node in range(sim.B):
            sim.add_events(node, [2500 + node], [gate], [150])
    sim.run(4200)
    return net, sim, gate


def test_copy73_fast_single_rail_and_one_stray():
    legacy, before, gate = _prescribed_fault(False, 34, 1, corner=True)
    assert legacy.quanta == [211, 211]
    assert np.all(before.counts(gate) > 0)
    fixed, after, gate = _prescribed_fault(True, 34, 1, corner=True)
    assert not after.counts(gate).any()
    # The readout also has an observable bounded rate, rather than clipping simulated
    # spikes or substituting a nominal-period source in the regression.
    tap = next(iter(fixed.rate_readouts.values()))
    steps, nodes = after.spikes(tap)
    gaps = np.diff(steps[(nodes == 0) & (steps > 1000) & (steps < 3900)])
    assert P.n_ref <= gaps.min() <= gaps.max() <= P.n_ref + 5


@pytest.mark.parametrize("fraction", [.55, .65])
@pytest.mark.parametrize("scale", [1., .9])
def test_true_double_rail_at_measured_slowest_period(fraction, scale):
    # The two 10,000-latch surveys reached 55 steps without readouts and 56 with them
    # (adding an edge changes the RNG assignment, not the latch's dynamics).
    _net, sim, gate = _prescribed_fault(True, 56, 2, fraction=fraction, scale=scale)
    assert np.all(sim.counts(gate) > 0)
    assert np.all(sim.counts(gate, since=2000) > 0)


def test_existing_four_pulse_kill_clears_fast_rail_at_every_phase():
    drive = replace(D, rate_robust=True, kill_pulses=4)
    net = Netlist(P)
    latch = add_latch(net, drive, "fast")
    dark = add_latch(net, drive, "dark")
    gate = add_and_gate(net, drive, "fault", [latch.u, dark.u], fraction=.55)
    source = net.neuron("clear")
    add_kill_train(net, drive, "kill", source, [latch])
    topo = net.topology()
    phases = np.arange(0, D.loop_period_steps, 4)
    q = np.broadcast_to(topo.quanta, (len(phases), topo.nnz)).copy()
    loops = ((topo.src == latch.u) & (topo.dst == latch.v)) | (
        (topo.src == latch.v) & (topo.dst == latch.u))
    q[:, loops] = np.rint(q[:, loops] * 1.2)
    vth = np.full((len(phases), topo.n), P.V_th)
    vth[:, list(latch.members)] -= .8
    sim = MixProbe(net, len(phases), 0, [latch.u, gate], noisy=False, stray=False,
                   quanta=q, V_th=vth)
    for node, phase in enumerate(phases):
        sim.add_events(node, [10, 2000 + phase, 3500],
                       [latch.u, source, latch.u], [D.ignite] * 3)
    sim.run(5200)
    steps, nodes = sim.spikes(latch.u)
    assert not np.any((steps > 3000) & (steps < 3500))
    assert np.all(sim.counts(latch.u, since=4800) > 0)  # reload, no extra inhibition hangover
    assert not sim.counts(gate).any()
    pre = steps[(nodes == 0) & (steps > 500) & (steps < 1900)]
    assert 36 <= np.mean(np.diff(pre)) <= 40


GATE_KINDS = ("fault", "completion", "majority", "valid_or", "veto", "require", "retry")


def gate_survey(kind, robust, copies=2000, seed=109, steps=10_000):
    """One second per case; empirical probabilities per exposure, not per spike.

    AND/majority: 1 live is false, 2 live is true. OR/veto: both are true (also
    measure a dark case). Require: required rail alone, driver alone, and both;
    the driver rises after the required rail settles, as its ordering contract says.
    """
    drive = replace(D, rate_robust=robust)
    net = Netlist(P)
    rails = [add_latch(net, drive, f"input{k}") for k in range(3)]
    taps = [r.u for r in rails]
    if kind == "fault":
        gate = add_and_gate(net, drive, kind, taps[:2], fraction=.55)
    elif kind == "completion":
        gate, _ = add_and_latched(net, drive, kind, taps[:2])
    elif kind == "majority":
        gate, _ = add_maj_latched(net, drive, kind, taps)
    elif kind == "valid_or":
        gate, _ = add_or_latched(net, drive, kind, taps[:2])
    elif kind == "veto":
        gate = add_veto_neuron(net, drive, kind, taps[:2])
    elif kind == "require":
        target = add_latch(net, drive, "target")
        gate = add_veto_relay(net, drive, kind, taps[1], [], target, require=[taps[0]])
    elif kind == "retry":
        # Optional kernel retry-clear: a delayed pulse and BOTH request rails.
        driver = net.neuron("driver")
        gate = net.neuron("retry")
        net.synapse(driver, gate, int(round(.5 * D.single_need)))
        for tap in taps[:2]:
            net.synapse(tap, gate, int(round(.35 / .65 * D.and_in)))
        condition_rate_gate(net, drive, gate)
    else:
        raise ValueError(kind)
    sim = MixProbe(net, copies * 3, seed, [gate, *taps[:2]])
    for case in range(3):
        nodes = np.arange(case * copies, (case + 1) * copies)
        live = (taps[:1] if case == 1 else taps[:2] if case == 2 else [])
        if kind == "require" and case == 0:
            live = taps[1:2]  # driver alone instead of an all-dark case
        for tap in live:
            step = 2500 if kind == "require" and tap == taps[1] else 10
            sim._events[step].append((nodes, np.full(copies, tap), np.full(copies, D.ignite)))
        if kind == "retry":
            sim._events[2500].append((nodes, np.full(copies, driver), np.full(copies, D.ignite)))
    sim.run(steps)
    counts = sim.counts(gate).reshape(3, copies)
    fired = counts > 0
    return {"kind": kind, "robust": robust, "copies_per_case": copies,
            "seed": seed, "steps": steps,
            "zero_or_driver_only_fire": int(fired[0].sum()),
            "one_live_fire": int(fired[1].sum()),
            "two_live_miss": int((~fired[2]).sum()),
            "two_live_multiple": int((counts[2] > 1).sum()) if kind in ("require", "retry") else None}


@pytest.mark.slow
def test_mix_b_period_distribution():
    result = period_survey()
    print(result)
    assert result["silent"] == 0
    assert result["interval"][0] < 34 < D.loop_period_steps < result["interval"][-1]
    conditioned = period_survey(replace(D, rate_robust=True))
    print(conditioned)
    assert conditioned["silent"] == conditioned["readout"]["silent"] == 0
    assert conditioned["readout"]["interval"][0] >= P.n_ref
    assert conditioned["readout"]["interval"][-1] <= P.n_ref + 6


@pytest.mark.slow
@pytest.mark.parametrize("kind", GATE_KINDS)
def test_mix_b_gate_probabilities(kind):
    before = gate_survey(kind, False)
    after = gate_survey(kind, True)
    print(before, after)
    assert after["zero_or_driver_only_fire"] == 0
    if kind in ("fault", "completion", "majority", "require", "retry"):
        assert after["one_live_fire"] == 0
    else:
        assert after["one_live_fire"] == after["copies_per_case"]
    assert after["two_live_miss"] == 0
    if kind in ("require", "retry"):
        assert after["two_live_multiple"] == 0


@pytest.mark.slow
def test_conditioned_guards_keep_ordering_with_veto_history():
    from drosophilos.lib.kernel import add_kill_pair, guarded_pulse
    from drosophilos.protocol.celement import add_delay_chain

    drive = replace(D, rate_robust=True, kill_pulses=4)
    net = Netlist(P)
    a = add_kill_pair(net, drive, "a", pulses=4)
    b = add_kill_pair(net, drive, "b", pulses=4)
    target = net.neuron("target")
    guarded_pulse(net, drive, "guard", a, b, target, true_guards=True)
    add_kill_train(net, drive, "consume", target, [a[1], b[1]], pulses=4)
    relight = add_delay_chain(net, drive, "relight", target, 5)
    for pair in (a, b):
        net.synapse(relight, pair[0].u, drive.ignite)
    sim = MixProbe(net, 256, 7, [target])
    for node, offset in enumerate(np.rint(np.linspace(-600, 600, sim.B)).astype(int)):
        sim.add_events(node, [1, 1, 2500, 2500 + offset],
                       [a[0].u, b[0].u, a[1].u, b[1].u], [drive.ignite] * 4)
    sim.run(5200)
    assert np.all(sim.counts(target) == 1), sim.counts(target)


def kill_survey(copies=10_000, seed=109):
    """Existing kernel kill policy under all mix-B draws, including the clear circuit."""
    drive = replace(D, rate_robust=True, kill_pulses=4)
    net = Netlist(P)
    latch = add_latch(net, drive, "L")
    rate_tap(net, drive, latch.u)
    source = net.neuron("clear")
    add_kill_train(net, drive, "kill", source, [latch])
    sim = MixProbe(net, copies, seed, [latch.u])
    sim.ignite([latch.u])
    for node in range(copies):
        sim.add_events(node, [2000 + node % D.loop_period_steps], [source], [D.ignite])
    sim.run(3400)
    survived = int((sim.counts(latch.u, since=3000) > 0).sum())
    sim.ignite([latch.u], step=3600)
    sim.run(1800)
    missed_reload = int((sim.counts(latch.u, since=4800) == 0).sum())
    return {"copies": copies, "seed": seed, "survived": survived,
            "missed_reload": missed_reload}


@pytest.mark.slow
def test_mix_b_existing_kill_train():
    result = kill_survey()
    print(result)
    assert result["survived"] == result["missed_reload"] == 0
