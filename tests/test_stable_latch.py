"""Rate-stable storage qualification, including reproducible rejected candidates.

No candidate passed. These tests assert the measured frontier, not an invented
success contract. Production stable_latch=True must therefore fail closed.
Every trajectory is integrated by the unchanged float64 RefSim; no prescribed
latch spikes, clipped rates, altered refractory periods or synthetic clears.
"""

from dataclasses import replace
from itertools import product

import numpy as np
import pytest

from drosophilos.lib.kernel import build_pipeline, run_pipeline
from drosophilos.lib.netlist import Drive, Netlist
from drosophilos.protocol.celement import add_and_gate, add_and_latched, add_or_latched, add_veto_neuron
from drosophilos.protocol.latch import Latch, add_latch, add_ready, add_reset
from drosophilos.sim.model import Params
from drosophilos.sim.ref64 import RefSim
from test_fast_latch import MixProbe, _periods


P = Params()
D = Drive.from_params(P)
# family, loop/drive.loop, inhibition/drive.loop, u->v delay, v->u delay, bias
LEGACY = ("none", 1., 0., 18, 18, 0.)
AUTAPSE = ("auto0", 1., .2, 18, 18, 0.)
ZERO_DELAY = ("auto8", .95, .1, 0, 0, 0.)
SHARED = ("shared18", 1.2, .1, 18, 18, 0.)
ASYMMETRIC = ("none", .95, 0., 24, 60, 0.)
HIGH_GAIN = ("auto22", 8., 6., 22, 22, 0.)
REPRESENTATIVES = (LEGACY, AUTAPSE, ZERO_DELAY, SHARED, ASYMMETRIC, HIGH_GAIN)
# Correlated corners are only an initial screen. Independent signs follow below.
SCREEN_CORNERS = (
    (1., 1., 0., 0.),
    (np.exp(.12), np.exp(-.12), -.6, .6),
    (np.exp(-.12), np.exp(.12), .6, -.6),
    (np.exp(.12), np.exp(.12), -.6, .6),
    (np.exp(-.12), np.exp(-.12), .6, -.6),
)


def _modify(net, latch, config):
    """Test-only candidate, applied after constructing the real storage primitive."""
    family, gain, strength, uv, vu, bias = config
    for e in list(net.incoming[latch.v]):
        if net.src[e] == latch.u and net.quanta[e] > 0:
            net.quanta[e], net.delay[e] = round(D.loop * gain), uv
    for e in list(net.incoming[latch.u]):
        if net.src[e] == latch.v and net.quanta[e] > 0:
            net.quanta[e], net.delay[e] = round(D.loop * gain), vu
    for member in latch.members:
        net.set_bias(member, bias)
    if family.startswith("auto"):
        for member in latch.members:
            net.synapse(member, member, -round(D.loop * strength), int(family[4:]))
    elif family.startswith("shared"):
        inh = net.neuron(net.roles[latch.u].removesuffix(".u") + ".adapt")
        for member in latch.members:
            net.synapse(member, inh, D.pulse, 0)
            net.synapse(inh, member, -round(D.loop * strength), int(family[6:]))


def _candidate(net, name, config):
    latch = add_latch(net, D, name)
    _modify(net, latch, config)
    return latch


def period_survey(config, copies=10_000, steps=6000, seed=109):
    net = Netlist(P)
    latch = _candidate(net, "L", config)
    sim = MixProbe(net, copies, seed, latch.members)
    sim.ignite([latch.u])
    sim.run(steps)
    return {"copies": copies, "steps": steps, **_periods(sim, latch.u),
            "not_holding_at_end": int((sim.counts(latch.u, since=steps - 200) == 0).sum())}


def independent_corners(config, *, steps=6000):
    """Every independent ±3-sigma edge/threshold/bias sign, both initial orbits."""
    net = Netlist(P)
    latch = _candidate(net, "L", config)
    topo = net.topology()
    signs = np.repeat(np.array(list(product((-1, 1), repeat=topo.nnz + 2 * net.n))), 2, axis=0)
    copies = len(signs)
    sim = MixProbe(net, copies, 109, latch.members, noisy=False, stray=False,
                   quanta=np.rint(topo.quanta * np.exp(.12 * signs[:, :topo.nnz])),
                   V_th=P.V_th + .6 * signs[:, topo.nnz:topo.nnz + net.n],
                   bias=np.asarray(net.bias) + .6 * signs[:, -net.n:])
    sim.ignite([latch.u], quanta=round(D.ignite * np.exp(-.12)))
    for step in (11, 14, 23):
        sim._events[step].append((np.arange(1, copies, 2), np.full(copies // 2, latch.u),
                                  np.full(copies // 2, D.ignite)))
    for node in range(copies):
        sim.add_events(node, [2520], [latch.v if node % 2 else latch.u], [300])
    sim.run(steps)
    return {"copies": copies, **_periods(sim, latch.u),
            "not_holding_at_end": int((sim.counts(latch.u, since=steps - 200) == 0).sum())}


def candidate_grid(family):
    if family == "delay":
        return [("none", g, 0., a, b, 0.) for g, a, b in product(
            (.85, .9, .95, 1., 1.05, 1.1, 1.2, 1.4, 1.8), range(0, 61, 6), range(0, 61, 6))]
    if family == "inhibit":
        return [(k, g, h, 18, 18, 0.) for k, g, h in product(
            ("auto0", "auto18", "auto35", "auto45", "auto55", "shared0", "shared18", "shared35"),
            (1., 1.2, 1.4, 1.6, 2., 2.5, 3., 4.), (.025, .05, .1, .2, .3, .5, .75, 1., 1.5, 2., 3.))]
    if family == "bias":
        return [("none", g, 0., a, b, bias) for g, a, b, bias in product(
            (.85, 1., 1.2, 1.5, 2., 3.), (0, 10, 18, 30), (0, 10, 18, 30), (-14., -7., -3.5, 3.5))]
    if family == "high":
        return [(f"auto{ad}", g, g * ratio, d, d, 0.) for g, ratio, d, ad in product(
            (2., 4., 8., 16., 32.), (.75, .9, 1., 1.1, 1.25), (12, 18, 22, 26, 32, 44), (0, 10, 22, 35))]
    if family == "refine":
        return [(f"auto{ad}", g, h, d, d, 0.) for g, h, d, ad in product(
            (.95, 1., 1.05, 1.1, 1.2), (.1, .15, .2, .25, .3, .4), (0, 8, 18, 24, 30), (0, 8, 18))]
    raise ValueError(family)


def screen(configs, steps=5000):
    """Batch circuits, observe intervals online, leave RefSim dynamics untouched."""
    net = Netlist(P)
    taps = np.array([_candidate(net, f"L{i}", cfg).u for i, cfg in enumerate(configs)])
    topo = net.topology()
    copies = 2 * len(SCREEN_CORNERS)
    q = np.broadcast_to(topo.quanta, (copies, topo.nnz)).copy()
    vth = np.full((copies, topo.n), P.V_th)
    bias = np.broadcast_to(net.bias, (copies, topo.n)).copy()
    for b, (exc, inh, th, bs) in enumerate(np.repeat(SCREEN_CORNERS, 2, axis=0)):
        q[b] = np.rint(q[b] * np.where(q[b] > 0, exc, inh))
        vth[b] += th
        bias[b] += bs
    sim = RefSim(topo, P, n_nodes=copies, quanta=q, V_th=vth, bias=bias)
    for b in range(copies):
        for step in ([10] if b % 2 == 0 else [10, 11, 14, 23]):
            sim.add_events(b, [step] * len(taps), taps,
                           [round(D.ignite * SCREEN_CORNERS[b // 2][0])] * len(taps))
        sim.add_events(b, [2520] * len(taps), taps + b % 2, [300] * len(taps))
    last = np.full((copies, net.n), -1)
    minimum = np.full_like(last, 100_000)
    first = np.full_like(last, -1)
    count = np.zeros_like(last)
    for step in range(steps):
        sim.step()
        if sim._spk_neuron:
            neurons, nodes = sim._spk_neuron[-1], sim._spk_node[-1]
            if step >= 3000:
                prior = last[nodes, neurons]
                valid = prior >= 3000
                b, n = nodes[valid], neurons[valid]
                minimum[b, n] = np.minimum(minimum[b, n], step - prior[valid])
                count[nodes, neurons] += 1
                new = first[nodes, neurons] < 0
                first[nodes[new], neurons[new]] = step
            last[nodes, neurons] = step
        sim._spk_step.clear()
        sim._spk_node.clear()
        sim._spk_neuron.clear()
    return [{"config": config, "alive": (last[:, u] > steps - 200).tolist(),
             "minimum": minimum[:, u].tolist(),
             "mean": np.divide(last[:, u] - first[:, u], count[:, u] - 1,
                               where=count[:, u] > 1, out=np.full(copies, np.nan)).tolist()}
            for config, u in zip(configs, taps)]


def gate_probe(config, *, corner=None, copies=256, seed=109, steps=10_000):
    """Dark, one-live and two-live inputs; actual noisy source latches and gates."""
    net = Netlist(P)
    rails = [_candidate(net, f"r{k}", config) for k in range(2)]
    taps = [l.u for l in rails]
    fault = add_and_gate(net, D, "fault", taps, fraction=.55)
    completion, _ = add_and_latched(net, D, "completion", taps)
    valid, _ = add_or_latched(net, D, "valid", taps)
    veto = add_veto_neuron(net, D, "veto", taps)
    kwargs = {}
    if corner is not None:
        exc, inh, th, bs = SCREEN_CORNERS[corner]
        topo = net.topology()
        q = topo.quanta.copy()
        storage = [member for latch in rails for member in latch.members]
        internal = np.isin(topo.src, storage) & np.isin(topo.dst, storage)
        q[internal] = np.rint(q[internal] * np.where(q[internal] > 0, exc, inh))
        vth, bias = np.full(net.n, P.V_th), np.asarray(net.bias).copy()
        vth[storage] += th
        bias[storage] += bs
        kwargs = dict(noisy=False, stray=False, quanta=np.broadcast_to(q, (copies * 3, topo.nnz)),
                      V_th=vth, bias=bias)
    sim = MixProbe(net, copies * 3, seed, [fault, completion, valid, veto, *taps], **kwargs)
    for case in range(3):
        nodes = np.arange(case * copies, (case + 1) * copies)
        for tap in taps[:case]:
            sim._events[10].append((nodes, np.full(copies, tap), np.full(copies, D.ignite)))
    sim.run(steps)
    return {name: (sim.counts(neuron).reshape(3, copies) > 0).sum(axis=1).tolist()
            for name, neuron in (("fault", fault), ("completion", completion), ("or", valid),
                                 ("veto", veto), ("holding", taps[0]))}


CAPTURES = ("request28", "master74", "master17", "master24", "stage18")


def clear_fixture(config, case, *, ready=False):
    """Keep reconstructed OLD-edge draws; new feedback edges have no capture draw."""
    from test_completion_stall import _map
    from test_request_clear_entrainment import _circuit
    from test_robust_reset import CAPTURES as REGISTERS

    if case == "request28":
        net, latch, source, inh, vth, bias = _circuit(4)
        before = net.topology()
        q = before.quanta.copy()
    else:
        copy, rate, name, rail = REGISTERS.get(case, (None, False, "R", "L"))
        net = Netlist(P)
        latch = add_latch(net, D, f"{name}.{rail}")
        source, inh, _ = add_reset(net, D, name, [latch], pulses=4, strength=.75)
        before = net.topology()
        if copy is None:
            q, vth, bias = before.quanta.copy(), np.full(net.n, P.V_th), np.asarray(net.bias).copy()
        else:
            q, vth, bias, _ = _map(copy, net, rate_robust=rate)
    factors = {(int(s), int(d)): observed / nominal
               for s, d, observed, nominal in zip(before.src, before.dst, q, before.quanta)}
    if case == "request28":
        # _circuit stores its reconstructed weights IN the netlist, whereas
        # _map returns them separately from the nominal register topology.
        for e, (s, d) in enumerate(zip(before.src, before.dst)):
            if s in latch.members and d in latch.members:
                factors[(int(s), int(d))] = q[e] / D.loop
    _modify(net, latch, config)
    # AUTAPSE, HIGH_GAIN and delay-only candidates preserve neuron membership.
    assert net.n == len(vth)
    ready_id = add_ready(net, D, "probe", source, hops=15) if ready else None
    if ready:
        vth = np.r_[vth, np.full(net.n - len(vth), P.V_th)]
        bias = np.r_[bias, np.zeros(net.n - len(bias))]
    topo = net.topology()
    q = np.array([round(weight * factors.get((int(s), int(d)), 1.))
                  for s, d, weight in zip(topo.src, topo.dst, topo.quanta)])
    new = np.array([(int(s), int(d)) not in factors for s, d in zip(topo.src, topo.dst)])
    if case in ("fast3sigma", "slow3sigma"):
        exc, neg, th, bs = SCREEN_CORNERS[1 if case == "fast3sigma" else 2]
        storage = np.isin(topo.dst, latch.members)
        q[storage] = np.rint(q[storage] * np.where(q[storage] > 0, exc, neg))
        vth[list(latch.members)] += th
        bias[list(latch.members)] += bs
    return net, latch, source, inh, ready_id, q, vth, bias, new


def clear_survey(config, case, copies=40_000, seed=108):
    """Actual four-tap controller, independent strays on ALL fixture neurons.

    Captures preserve the campaign's original edge/threshold/bias draws. Added
    feedback edges receive independent mix-B draws; they cannot be reconstructed
    from a capture of a circuit that did not contain them. Corner feedback stays
    at its adverse fixed value. This is not a replay of CUDA's stray bitstream.
    """
    net, latch, source, inh, _, q, vth, bias, new = clear_fixture(config, case)
    survivors = not_live = 0
    for lo in range(0, copies, 1000):
        count = min(1000, copies - lo)
        rng = np.random.default_rng(seed + lo)
        quanta = np.broadcast_to(q, (count, len(q))).copy()
        if case in CAPTURES:
            quanta[:, new] = np.rint(quanta[:, new] * np.exp(rng.normal(0, .04, (count, int(new.sum())))))
        sim = MixProbe(net, count, seed + lo, [*latch.members, inh], noisy=(case == "mix_b"),
                       quanta=quanta, V_th=vth, bias=bias)
        clears = rng.integers(3000, 4500, count)
        for b, clear in enumerate(clears):
            sim.add_events(b, [10, 11, 14, 23, 2520, int(clear)],
                           [latch.u] * 4 + [latch.v if b % 2 else latch.u, source],
                           [D.ignite] * 4 + [300, D.relay_in])
        sim.run(6200)
        survived = np.zeros(count, bool)
        live = np.zeros(count, bool)
        for member in latch.members:
            steps, nodes = sim.spikes(member)
            survived[nodes[steps > clears[nodes] + 1500]] = True
            live[nodes[(steps > clears[nodes] - 200) & (steps < clears[nodes])]] = True
        survivors += int(survived.sum())
        not_live += int((~live).sum())
    return {"copies": copies, "survivors": survivors, "not_live_before_clear": not_live}


def reload_survey(config, case, offsets=range(400, 1601, 25)):
    """First offset (from trigger input) that holds at every phase and later offset.

    Same ordinary controller, weak ignition at the slow combined corner, no
    extra recovery hops. Spontaneous survivors do NOT count as successful reloads.
    """
    net, latch, source, inh, ready, q, vth, bias, _ = clear_fixture(config, case, ready=True)
    offsets = list(offsets)
    phases = list(range(0, 60, 4))
    cases = list(product(offsets, phases))
    copies = len(cases)
    sim = MixProbe(net, copies, 0, [*latch.members, ready], noisy=False, stray=False,
                   quanta=np.broadcast_to(q, (copies, len(q))), V_th=vth, bias=bias)
    ignition = round(D.ignite * np.exp(-.12)) if case == "slow3sigma" else D.ignite
    reloads = np.array([2000 + phase + offset for offset, phase in cases])
    for b, ((_, phase), reload) in enumerate(zip(cases, reloads)):
        sim.add_events(b, [10, 2000 + phase, int(reload)], [latch.u, source, latch.u],
                       [D.ignite, D.relay_in, ignition])
    sim.run(int(reloads.max()) + 1600)
    steps, nodes = sim.spikes(latch.u)
    success = np.zeros(copies, bool)
    success[nodes[steps > reloads[nodes] + 1500]] = True
    success = success.reshape(len(offsets), len(phases)).all(axis=1)
    reliable = np.logical_and.accumulate(success[::-1])[::-1]
    recovery = int(np.array(offsets)[reliable][0]) if reliable.any() else None
    ready_steps, ready_nodes = sim.spikes(ready)
    first_ready = int(ready_steps[ready_nodes == 0][0]) - 2000
    # Separate no-reload controls prove that the controller stopped the old orbit.
    control = MixProbe(net, len(phases), 0, latch.members, noisy=False, stray=False,
                       quanta=np.broadcast_to(q, (len(phases), len(q))), V_th=vth, bias=bias)
    for b, phase in enumerate(phases):
        control.add_events(b, [10, 2000 + phase], [latch.u, source], [D.ignite, D.relay_in])
    control.run(4000)
    survivors = sum(int((control.counts(member, since=3500) > 0).sum()) for member in latch.members)
    return {"recovery": recovery, "ready": first_ready,
            "margin": None if recovery is None else first_ready - recovery,
            "control_survivors": survivors}


def kernel_probe(config, *, rate=False, other_options=False, max_ms=6000):
    """Test-only transform on EVERY actual kernel latch, including image state.

    Both directions and their roles identify each two-neuron primitive, without
    relying on globally unique role names. Conditioning's inhibition mirrors are
    deliberately included: the build with the options combined would add them.
    """
    pl = build_pipeline(P, 2,
                        [{"name": "sum", "op": "ADD", "a": "input", "b": ("const", "one")},
                         {"name": "out", "op": "XOR", "a": "sum", "b": ("const", "one")}],
                        consts={"one": 1}, rate_robust=rate, zero_once=other_options,
                        robust_request_clear=other_options,
                        experimental_register_reset=other_options)
    edges = {(s, d): q for s, d, q in zip(pl.net.src, pl.net.dst, pl.net.quanta)}
    latches = [Latch(u, v) for (u, v), q in edges.items()
               if q == D.loop and edges.get((v, u)) == D.loop
               and pl.net.roles[u].endswith(".u")
               and pl.net.roles[v] == pl.net.roles[u].removesuffix(".u") + ".v"]
    before = (pl.net.n, pl.net.nnz)
    for latch in latches:
        _modify(pl.net, latch, config)
    outputs, sim, stats = run_pipeline(pl, P, [0, 1, 3], max_ms=max_ms)
    assert isinstance(sim, RefSim)
    return {"values": [v for _, v in outputs], "steps": [int(s) for s, _ in outputs],
            "faults": stats["faults"], "timeouts": stats["timeouts"],
            "bad_outputs": stats["bad_outputs"], "latches": len(latches),
            "extra_neurons": pl.net.n - before[0], "extra_edges": pl.net.nnz - before[1]}


def false_fault_probe(config):
    """The fast_latch copy-73 gate probe, driven by a real fast30 source latch.

    Its ±.8 mV/+.5 mV gate offsets are the earlier synthetic reproduction,
    not captured gate draws and not a combined three-sigma corner.
    """
    net = Netlist(P)
    latch = _candidate(net, "live", config)
    dark = _candidate(net, "dark", config)
    gate = add_and_gate(net, D, "fault", [latch.u, dark.u], fraction=.55)
    topo = net.topology()
    copies = 47
    q = topo.quanta.copy()
    loop = ((topo.src == latch.u) & (topo.dst == latch.v)) | (
        (topo.src == latch.v) & (topo.dst == latch.u))
    q[loop] = np.rint(q[loop] * 1.3)
    vth, bias = np.full(net.n, P.V_th), np.asarray(net.bias).copy()
    vth[list(latch.members)] -= 1.2
    bias[list(latch.members)] += .6
    vth[gate] -= .8
    bias[gate] += .5
    sim = MixProbe(net, copies, 0, [*latch.members, gate, dark.u], noisy=False, stray=False,
                   quanta=np.broadcast_to(q, (copies, len(q))), V_th=vth, bias=bias)
    sim.ignite([latch.u])
    for node in range(copies):
        sim.add_events(node, [2500 + node], [gate], [150])
    sim.run(4200)
    return {"false_faults": int((sim.counts(gate) > 0).sum()),
            "dark_fires": int((sim.counts(dark.u) > 0).sum()), **_periods(sim, latch.u)}


@pytest.mark.parametrize("config,neurons,edges", [(LEGACY, 2, 2), (AUTAPSE, 2, 4),
                                                  (ZERO_DELAY, 2, 4), (SHARED, 3, 6)])
def test_candidate_costs(config, neurons, edges):
    net = Netlist(P)
    _candidate(net, "L", config)
    assert (net.n, net.nnz) == (neurons, edges)


def test_unqualified_primitive_rejects_before_mutation():
    net = Netlist(P)
    with pytest.raises(ValueError, match="stable_latch is not qualified"):
        add_latch(net, replace(D, stable_latch=True), "L")
    assert net.n == net.nnz == 0


def test_correlated_screen_does_not_qualify_zero_delay_storage():
    result, = screen([ZERO_DELAY])
    assert all(result["alive"]) and min(result["minimum"]) == 44
    actual = independent_corners(ZERO_DELAY)
    assert actual["copies"] == 512
    assert actual["silent"] == 173 and actual["interval"][0] == 34


def test_autapse_holds_but_has_a_fast_independent_corner():
    result = independent_corners(AUTAPSE)
    assert result["silent"] == result["not_holding_at_end"] == 0
    assert result["interval"][0] == 36 < 44


def test_slow_autapse_holds_but_stops_completion_gates():
    result = gate_probe(AUTAPSE, corner=2, copies=1)
    assert result["holding"] == [0, 1, 1]
    assert result["fault"] == result["completion"] == [0, 0, 0]
    assert result["or"] == [0, 0, 1]  # One actual live rail is no longer sufficient.
    assert result["veto"] == [0, 1, 1]


@pytest.mark.parametrize("case,old,new", [("nominal", 675, 700), ("slow3sigma", 1075, 1100)])
def test_autapse_recovery_margin_is_worse_than_legacy(case, old, new):
    legacy = reload_survey(LEGACY, case)
    candidate = reload_survey(AUTAPSE, case)
    assert legacy["control_survivors"] == candidate["control_survivors"] == 0
    assert legacy["recovery"] == old and candidate["recovery"] == new
    assert candidate["margin"] < legacy["margin"]


@pytest.mark.parametrize("case", CAPTURES)
def test_autapse_captured_clear_smoke(case):
    result = clear_survey(AUTAPSE, case, copies=256)
    assert result == {"copies": 256, "survivors": 0, "not_live_before_clear": 0}


def test_high_gain_rate_cap_does_not_make_the_ordinary_clear_reliable():
    periods = independent_corners(HIGH_GAIN)
    assert periods["silent"] == 0 and periods["interval"][0] == 46
    assert clear_survey(HIGH_GAIN, "fast3sigma", copies=256) == {
        "copies": 256, "survivors": 256, "not_live_before_clear": 0}


@pytest.mark.parametrize("config,expected", [(LEGACY, 47), (AUTAPSE, 0), (HIGH_GAIN, 47)])
def test_fast_source_false_fault_probe(config, expected):
    result = false_fault_probe(config)
    assert result["dark_fires"] == result["silent"] == 0
    assert result["false_faults"] == expected


@pytest.mark.slow
@pytest.mark.parametrize("config", [LEGACY, AUTAPSE])
def test_ten_second_holding_exposures(config):
    random = period_survey(config, copies=1000, steps=100_000)
    corners = independent_corners(config, steps=100_000)
    print({"config": config, "random": random, "corners": corners}, flush=True)
    assert random["silent"] == random["not_holding_at_end"] == 0
    assert corners["silent"] == corners["not_holding_at_end"] == 0


@pytest.mark.slow
def test_random_gate_frontier_has_no_false_faults_but_misses_true_inputs():
    result = gate_probe(AUTAPSE, copies=2000)
    print(result, flush=True)
    assert result["fault"] == [0, 0, 964]
    assert result["completion"] == [0, 0, 1986]
    assert result["or"] == result["veto"] == [0, 2000, 2000]


@pytest.mark.slow
@pytest.mark.parametrize("config,minimum,silent", [(LEGACY, 32, 0), (AUTAPSE, 47, 0),
                                                    (ZERO_DELAY, 36, 3370), (SHARED, 31, 0),
                                                    (ASYMMETRIC, 34, 0), (HIGH_GAIN, 54, 0)])
def test_mix_b_full_period_distribution(config, minimum, silent):
    result = period_survey(config)
    print(config, result, flush=True)
    assert result["interval"][0] == minimum and result["silent"] == silent


@pytest.mark.slow
@pytest.mark.parametrize("case", [*CAPTURES, "fast3sigma", "slow3sigma", "mix_b"])
def test_40000_autapse_ordinary_clears(case):
    result = clear_survey(AUTAPSE, case)
    print(case, result, flush=True)
    assert result == {"copies": 40_000, "survivors": 0, "not_live_before_clear": 0}


@pytest.mark.slow
@pytest.mark.parametrize("rate", [False, True])
@pytest.mark.parametrize("other", [False, True])
def test_multi_cell_candidate_frontier_with_and_without_other_options(rate, other):
    baseline = kernel_probe(LEGACY, rate=rate, other_options=other)
    result = kernel_probe(AUTAPSE, rate=rate, other_options=other)
    print({"rate": rate, "other_options": other, "baseline": baseline, "candidate": result}, flush=True)
    assert baseline["values"] == [0, 3, 1]
    assert result["faults"] == result["timeouts"] == result["bad_outputs"] == 0
    assert result["extra_neurons"] == 0
    assert result["extra_edges"] == (630 if rate else 498)
    if rate:
        assert result["values"] == []  # The combination, including feedback mirrors, fails.
    else:
        assert result["values"] == [0, 3, 1]
        assert result["steps"][-1] > baseline["steps"][-1]


@pytest.mark.slow
@pytest.mark.parametrize("family,expected", [("delay", (1089, 847, 0)), ("inhibit", (704, 415, 5)),
                                             ("bias", (384, 208, 0)), ("high", (600, 242, 39)),
                                             ("refine", (450, 295, 35))])
def test_design_screen(family, expected):
    configs = candidate_grid(family)
    rows = [row for start in range(0, len(configs), 128) for row in screen(configs[start:start + 128])]
    holding = [r for r in rows if all(r["alive"])]
    capped = [r for r in holding if min(r["minimum"]) >= 44]
    assert (len(rows), len(holding), len(capped)) == expected
    if family == "refine":
        shortlist = [r for r in capped if 44 <= r["mean"][0] <= 56]
        assert len(shortlist) == 20
        for row in shortlist:
            result = independent_corners(row["config"])
            random = period_survey(row["config"], copies=2000, steps=10_000)
            print(row["config"], {"corners": result, "random": random}, flush=True)
            assert result["silent"] > 0 and result["interval"][0] < 44
            assert random["silent"] > 0 and random["interval"][0] < 44
