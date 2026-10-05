"""Rate-stable storage qualification, including reproducible rejected candidates.

These tests assert the measured frontier, not an invented success contract.
Every trajectory is integrated by the unchanged float64 RefSim; no prescribed
latch spikes, clipped rates, altered refractory periods or synthetic clears.
"""

from dataclasses import replace
from itertools import product

import numpy as np
import pytest

from drosophilos.lib.kernel import build_pipeline, run_pipeline
from drosophilos.lib.gates import add_maj_latched
from drosophilos.lib.netlist import Drive, Netlist
from drosophilos.protocol.celement import (
    add_and_gate, add_and_latched, add_or_latched, add_veto_neuron, add_veto_relay,
)
from drosophilos.protocol.latch import Latch, add_latch, add_ready, add_reset
from drosophilos.protocol.rate import rate_tap
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
HIGH_MARGIN = ("auto22", 8., 6., 26, 26, 0.)
COMBINATIONS = (AUTAPSE, HIGH_GAIN, HIGH_MARGIN)
# Correlated corners are only an initial screen. Independent signs follow below.
SCREEN_CORNERS = (
    (1., 1., 0., 0.),
    (np.exp(.12), np.exp(-.12), -.6, .6),
    (np.exp(-.12), np.exp(.12), .6, -.6),
    (np.exp(.12), np.exp(.12), -.6, .6),
    (np.exp(-.12), np.exp(-.12), .6, -.6),
)


def _modify(net, latch, config):
    """Test-only transform, AFTER building all readers/qualifiers and controllers.

    Local feedback is not a reset/kill signal. Suppress automatic mirroring only
    while adding that feedback; existing and future real clears keep their mirrors.
    No reader may be constructed after this transform: mirror_inhibition would
    otherwise mistake pre-existing feedback for a kill edge as well.
    """
    family, gain, strength, uv, vu, bias = config
    for e in list(net.incoming[latch.v]):
        if net.src[e] == latch.u and net.quanta[e] > 0:
            net.quanta[e], net.delay[e] = round(D.loop * gain), uv
    for e in list(net.incoming[latch.u]):
        if net.src[e] == latch.v and net.quanta[e] > 0:
            net.quanta[e], net.delay[e] = round(D.loop * gain), vu
    for member in latch.members:
        net.set_bias(member, bias)
    mirrors, net.inhibition_mirrors = net.inhibition_mirrors, {}
    try:
        if family.startswith("auto"):
            for member in latch.members:
                net.synapse(member, member, -round(D.loop * strength), int(family[4:]))
        elif family.startswith("shared"):
            inh = net.neuron(net.roles[latch.u].removesuffix(".u") + ".adapt")
            for member in latch.members:
                net.synapse(member, inh, D.pulse, 0)
                net.synapse(inh, member, -round(D.loop * strength), int(family[6:]))
    finally:
        net.inhibition_mirrors = mirrors


def _candidate(net, name, config, *, rate=False):
    latch = add_latch(net, D, name)
    rate_tap(net, replace(D, rate_robust=rate), latch.u)
    _modify(net, latch, config)
    return latch


def _holding(sim, tap, steps):
    """Separate failure to establish a train from a subsequent >200-step gap.

    Two spikes in [500, 1000) establish hold, after the ignition transient and
    before the kick. A late rescue does not erase an earlier loss of hold.
    """
    st, nd = sim.spikes(tap)
    established = np.bincount(nd[(st >= 500) & (st < 1000)], minlength=sim.B) >= 2
    order = np.lexsort((st, nd))
    st, nd = st[order], nd[order]
    gaps = (nd[1:] == nd[:-1]) & (st[:-1] >= 500) & (np.diff(st) > 200)
    lost = np.zeros(sim.B, bool)
    lost[nd[1:][gaps]] = True
    terminal = sim.counts(tap, since=steps - 200) == 0
    return {"non_ignition": int((~established).sum()),
            "loss_of_hold": int((established & (lost | terminal)).sum()),
            "not_holding_at_end": int(terminal.sum())}


def period_survey(config, copies=10_000, steps=6000, seed=109, *, rate=False):
    net = Netlist(P)
    latch = _candidate(net, "L", config, rate=rate)
    sim = MixProbe(net, copies, seed, latch.members)
    sim.ignite([latch.u])
    sim.run(steps)
    return {"copies": copies, "steps": steps, **_periods(sim, latch.u),
            **_holding(sim, latch.u, steps)}


def independent_corners(config, *, steps=6000, rate=False):
    """Every independent ±3-sigma edge/threshold/bias sign, both initial orbits."""
    net = Netlist(P)
    latch = _candidate(net, "L", config, rate=rate)
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
            **_holding(sim, latch.u, steps)}


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


def gate_probe(config, *, rate=False, corner=None, copies=256, seed=109, steps=10_000):
    """Dark, one-live and two-live inputs; actual noisy source latches and gates."""
    net = Netlist(P)
    drive = replace(D, rate_robust=rate)
    rails = [add_latch(net, drive, f"r{k}") for k in range(2)]
    taps = [l.u for l in rails]
    fault = add_and_gate(net, drive, "fault", taps, fraction=.55)
    completion, completion_latch = add_and_latched(net, drive, "completion", taps)
    valid, valid_latch = add_or_latched(net, drive, "valid", taps)
    veto = add_veto_neuron(net, drive, "veto", taps)
    for latch in [*rails, completion_latch, valid_latch]:
        _modify(net, latch, config)
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


def additional_gate_probe(config, kind, *, rate=True, copies=2000, seed=109):
    """Majority and required-rail one-shot, including each absent-input control."""
    net = Netlist(P)
    drive = replace(D, rate_robust=rate)
    rails = [add_latch(net, drive, f"r{k}") for k in range(3)]
    taps = [l.u for l in rails]
    if kind == "majority":
        gate, output = add_maj_latched(net, drive, kind, taps)
    elif kind == "require":
        output = add_latch(net, drive, "out")
        gate = add_veto_relay(net, drive, kind, taps[1], [], output, require=[taps[0]])
    else:
        raise ValueError(kind)
    for latch in [*rails, output]:
        _modify(net, latch, config)
    sim = MixProbe(net, copies * 3, seed, [gate, *taps])
    for case in range(3):
        nodes = np.arange(case * copies, (case + 1) * copies)
        live = taps[:case]
        if kind == "require" and case == 0:
            live = taps[1:2]  # Driver only; case 1 is required rail only.
        for tap in live:
            step = 2500 if kind == "require" and tap == taps[1] else 10
            sim._events[step].append((nodes, np.full(copies, tap), np.full(copies, D.ignite)))
    sim.run(10_000)
    counts = sim.counts(gate).reshape(3, copies)
    return {"fires": (counts > 0).sum(axis=1).tolist(),
            "multiple": (counts > 1).sum(axis=1).tolist() if kind == "require" else None}


CAPTURES = ("request28", "master74", "master17", "master24", "stage18")
CLEAR_CASES = (*CAPTURES, "fast3sigma", "slow3sigma", "mix_b")
RELOAD_ROWS = (("nominal", 675, 700), ("fast3sigma", 425, 500),
               ("slow3sigma", 1075, 1100), ("request28", 500, 600),
               ("master74", 525, 575), ("master17", 475, 525),
               ("master24", 550, 625), ("stage18", 600, 650))


def clear_fixture(config, case, *, ready=False, rate=False):
    """Keep reconstructed storage/controller draws; sample feedback/readers afresh.

    Even master24, captured with rate readers, uses a rebuilt reader with fresh
    independent parameters here. This is a storage-conditioned probe, not replay.
    """
    from test_completion_stall import _map
    from test_request_clear_entrainment import _circuit
    from test_robust_reset import CAPTURES as REGISTERS

    if case == "request28":
        net, latch, source, inh, vth, bias = _circuit(4)
        before = net.topology()
        q = before.quanta.copy()
    else:
        copy, captured_rate, name, rail = REGISTERS.get(case, (None, False, "R", "L"))
        net = Netlist(P)
        latch = add_latch(net, D, f"{name}.{rail}")
        source, inh, _ = add_reset(net, D, name, [latch], pulses=4, strength=.75)
        before = net.topology()
        if copy is None:
            q, vth, bias = before.quanta.copy(), np.full(net.n, P.V_th), np.asarray(net.bias).copy()
        else:
            q, vth, bias, _ = _map(copy, net, rate_robust=captured_rate)
    factors = {(int(s), int(d)): observed / nominal
               for s, d, observed, nominal in zip(before.src, before.dst, q, before.quanta)}
    if case == "request28":
        # _circuit stores its reconstructed weights IN the netlist, whereas
        # _map returns them separately from the nominal register topology.
        for e, (s, d) in enumerate(zip(before.src, before.dst)):
            if s in latch.members and d in latch.members:
                factors[(int(s), int(d))] = q[e] / D.loop
            elif s == inh and d in latch.members:
                # Build new reader mirrors from NOMINAL clear weights. Reapply
                # the captured storage-edge draw below, without copying that
                # draw into a newly added edge that needs independent noise.
                nominal = -round(.75 * D.loop)
                for edge in net.incoming[int(d)]:
                    if net.src[edge] == s:
                        net.quanta[edge] = nominal
                factors[(int(s), int(d))] = q[e] / nominal
    rate_tap(net, replace(D, rate_robust=rate), latch.u)
    assert (latch.u in net.rate_readouts) == rate
    ready_id = add_ready(net, D, "probe", source, hops=15) if ready else None
    _modify(net, latch, config)
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


def clear_survey(config, case, copies=40_000, seed=108, *, rate=False):
    """Actual four-tap controller, independent strays on ALL fixture neurons.

    Captures preserve the campaign's original edge/threshold/bias draws. Added
    feedback edges receive independent mix-B draws; they cannot be reconstructed
    from a capture of a circuit that did not contain them. Corner feedback stays
    at its adverse fixed value. This is not a replay of CUDA's stray bitstream.
    """
    net, latch, source, inh, _, q, vth, bias, new = clear_fixture(config, case, rate=rate)
    readers = list(net.rate_readouts.values())
    survivors = not_live = 0
    reader_survivors = 0
    for lo in range(0, copies, 1000):
        count = min(1000, copies - lo)
        rng = np.random.default_rng(seed + lo)
        quanta = np.broadcast_to(q, (count, len(q))).copy()
        thresholds = np.broadcast_to(vth, (count, net.n)).copy()
        biases = np.broadcast_to(bias, (count, net.n)).copy()
        if case in CAPTURES:
            quanta[:, new] = np.rint(quanta[:, new] * np.exp(rng.normal(0, .04, (count, int(new.sum())))))
            thresholds[:, readers] += rng.normal(0, .2, (count, len(readers)))
            biases[:, readers] += rng.normal(0, .2, (count, len(readers)))
        sim = MixProbe(net, count, seed + lo, [*latch.members, inh, *readers], noisy=(case == "mix_b"),
                       quanta=quanta, V_th=thresholds, bias=biases)
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
        for reader in readers:
            steps, nodes = sim.spikes(reader)
            reader_survivors += len(np.unique(nodes[steps > clears[nodes] + 1500]))
    result = {"copies": copies, "survivors": survivors, "not_live_before_clear": not_live}
    if rate:
        result["reader_survivors"] = reader_survivors
    return result


def reload_survey(config, case, offsets=range(400, 1601, 25), *, rate=False):
    """First offset (from trigger input) that holds at every phase and later offset.

    Same ordinary controller, weak ignition at the slow combined corner, no
    extra recovery hops. Spontaneous survivors do NOT count as successful reloads.
    """
    net, latch, source, inh, ready, q, vth, bias, _ = clear_fixture(config, case, ready=True, rate=rate)
    offsets = list(offsets)
    phases = list(range(0, 60, 4))
    cases = list(product(offsets, phases))
    copies = len(cases)
    readers = list(net.rate_readouts.values())
    sim = MixProbe(net, copies, 0, [*latch.members, ready, *readers], noisy=False, stray=False,
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
    for reader in readers:
        steps, nodes = sim.spikes(reader)
        reading = np.zeros(copies, bool)
        reading[nodes[steps > reloads[nodes] + 1500]] = True
        success &= reading
    success = success.reshape(len(offsets), len(phases)).all(axis=1)
    reliable = np.logical_and.accumulate(success[::-1])[::-1]
    recovery = int(np.array(offsets)[reliable][0]) if reliable.any() else None
    ready_steps, ready_nodes = sim.spikes(ready)
    first_ready = int(ready_steps[ready_nodes == 0][0]) - 2000
    # Separate no-reload controls prove that the controller stopped the old orbit.
    control = MixProbe(net, len(phases), 0, [*latch.members, *readers], noisy=False, stray=False,
                       quanta=np.broadcast_to(q, (len(phases), len(q))), V_th=vth, bias=bias)
    for b, phase in enumerate(phases):
        control.add_events(b, [10, 2000 + phase], [latch.u, source], [D.ignite, D.relay_in])
    control.run(4000)
    survived = np.zeros(len(phases), bool)
    for member in [*latch.members, *readers]:
        survived |= control.counts(member, since=3500) > 0
    survivors = int(survived.sum())
    # A surviving old orbit is not evidence of successful re-ignition.
    recovery = None if survivors else recovery
    return {"recovery": recovery, "ready": first_ready,
            "margin": None if recovery is None else first_ready - recovery,
            "control_survivors": survivors}


def kernel_probe(config, *, rate=False, other_options=False, max_ms=6000):
    """Test-only transform on EVERY actual kernel latch, including image state.

    Both directions and their roles identify each two-neuron primitive, without
    relying on globally unique role names. Local feedback never reaches readers;
    the production reset/kill mirrors remain intact.
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


def false_fault_probe(config, *, rate=False):
    """The synthetic copy-73 gate stress, driven by real candidate source latches.

    Its loop x1.3, -1.2/+0.6 mV source and -0.8/+0.5 mV gate offsets
    reproduce the earlier synthetic stress,
    not captured gate draws and not a combined three-sigma corner.
    """
    net = Netlist(P)
    drive = replace(D, rate_robust=rate)
    latch = add_latch(net, drive, "live")
    dark = add_latch(net, drive, "dark")
    gate = add_and_gate(net, drive, "fault", [latch.u, dark.u], fraction=.55)
    for rail in (latch, dark):
        _modify(net, rail, config)
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


def test_local_feedback_does_not_inhibit_readers_but_real_clears_do():
    net = Netlist(P)
    drive = replace(D, rate_robust=True)
    latch = add_latch(net, drive, "L")
    # Cover a reset already wired when the reader is installed, and one added
    # after feedback. Qualifiers use the same mirror registry as rate readers.
    _, first, _ = add_reset(net, drive, "first", [latch])
    reader = rate_tap(net, drive, latch.u)
    qualifier = net.neuron("qualifier")
    net.mirror_inhibition(latch.u, qualifier, 1)
    before = list(zip(net.src, net.dst, net.quanta, net.delay))
    _modify(net, latch, AUTAPSE)
    assert list(zip(net.src, net.dst, net.quanta, net.delay))[:len(before)] == before
    assert net.nnz == len(before) + 2
    _, last, _ = add_reset(net, drive, "last", [latch])
    for target, gain in ((reader, 16), (qualifier, 1)):
        inhibitory = [(net.src[e], net.quanta[e]) for e in net.incoming[target] if net.quanta[e] < 0]
        assert inhibitory == [(inh, -round(.75 * D.loop) * gain) for inh in (first, last)]


def test_request_capture_keeps_old_draws_and_gives_new_mirrors_nominal_weights():
    from test_request_clear_entrainment import _circuit

    captured, *_ = _circuit(4)
    net, latch, _, inh, _, q, _, _, new = clear_fixture(AUTAPSE, "request28", rate=True)
    topo = net.topology()
    observed = {(int(s), int(d)): weight for s, d, weight in zip(topo.src, topo.dst, q)}
    assert all(observed[s, d] == weight for s, d, weight in
               zip(captured.src, captured.dst, captured.quanta))
    reader = net.rate_readouts[latch.u]
    mirrors = np.flatnonzero((topo.dst == reader) & (topo.src == inh))
    assert len(mirrors) == 1
    assert new[mirrors[0]]
    assert q[mirrors[0]] == -round(.75 * D.loop) * 16


def test_correlated_screen_does_not_qualify_zero_delay_storage():
    result, = screen([ZERO_DELAY])
    assert all(result["alive"]) and min(result["minimum"]) == 44
    actual = independent_corners(ZERO_DELAY)
    assert actual["copies"] == 512
    assert actual["silent"] == 173 and actual["interval"][0] == 34
    assert actual["non_ignition"] == 98 and actual["loss_of_hold"] == 77
    assert actual["not_holding_at_end"] == 175


def test_autapse_holds_but_has_a_fast_independent_corner():
    result = independent_corners(AUTAPSE)
    assert result["silent"] == result["not_holding_at_end"] == 0
    assert result["interval"][0] == 36 < 44


def test_slow_autapse_holds_but_stops_completion_gates():
    legacy = gate_probe(LEGACY, corner=2, copies=1)
    result = gate_probe(AUTAPSE, corner=2, copies=1)
    assert legacy["holding"] == result["holding"] == [0, 1, 1]
    assert legacy["fault"] == [0, 0, 0]  # Not a discriminating failure at this corner.
    assert legacy["completion"] == [0, 0, 1]
    assert legacy["or"] == [0, 1, 1]
    assert result["holding"] == [0, 1, 1]
    assert result["fault"] == result["completion"] == [0, 0, 0]
    assert result["or"] == [0, 0, 1]  # One actual live rail is no longer sufficient.
    assert result["veto"] == [0, 1, 1]


def test_nominal_raw_fault_gate_misses_two_autapse_inputs():
    legacy = gate_probe(LEGACY, corner=0, copies=1)
    candidate = gate_probe(AUTAPSE, corner=0, copies=1)
    assert legacy["fault"] == [0, 0, 1]
    assert candidate["fault"] == [0, 0, 0]


@pytest.mark.parametrize("case,old,new", RELOAD_ROWS)
def test_autapse_recovery_margin_is_worse_than_legacy(case, old, new):
    legacy = reload_survey(LEGACY, case, rate=True)
    candidate = reload_survey(AUTAPSE, case, rate=True)
    assert legacy["control_survivors"] == candidate["control_survivors"] == 0
    assert legacy["recovery"] == old and candidate["recovery"] == new
    assert candidate["margin"] < legacy["margin"]
    assert candidate["ready"] == legacy["ready"] == {"master24": 871, "stage18": 870}.get(case, 872)


@pytest.mark.parametrize("case", CAPTURES)
def test_autapse_captured_clear_smoke(case):
    result = clear_survey(AUTAPSE, case, copies=256)
    assert result == {"copies": 256, "survivors": 0, "not_live_before_clear": 0}


@pytest.mark.parametrize("config,minimum", [(LEGACY, 32), (AUTAPSE, 36),
                                          (HIGH_GAIN, 46), (HIGH_MARGIN, 54)])
def test_independent_corners_with_readers(config, minimum):
    periods = independent_corners(config, rate=True)
    assert periods["copies"] == (1024 if config == LEGACY else 4096)
    assert periods["silent"] == periods["non_ignition"] == periods["loss_of_hold"] == 0
    assert periods["not_holding_at_end"] == 0
    assert periods["interval"][0] == minimum


@pytest.mark.parametrize("config", [HIGH_GAIN, HIGH_MARGIN])
def test_high_gain_rate_floor_does_not_make_the_ordinary_clear_reliable(config):
    legacy = clear_survey(LEGACY, "fast3sigma", copies=256, rate=True)
    result = clear_survey(config, "fast3sigma", copies=256, rate=True)
    assert legacy["survivors"] == result["not_live_before_clear"] == legacy["not_live_before_clear"] == 0
    assert result["survivors"] == 256


@pytest.mark.parametrize("config", [LEGACY, *COMBINATIONS])
@pytest.mark.parametrize("corner", [0, 1, 2])
def test_reader_gate_corner_controls(config, corner):
    result = gate_probe(config, rate=True, corner=corner, copies=1)
    conjunction = 0 if config in (HIGH_GAIN, HIGH_MARGIN) and corner == 2 else 1
    assert result["fault"] == result["completion"] == [0, 0, conjunction]
    assert result["or"] == result["veto"] == result["holding"] == [0, 1, 1]


@pytest.mark.parametrize("config,expected", [(LEGACY, 47), (AUTAPSE, 0), (HIGH_GAIN, 47)])
def test_fast_source_false_fault_probe(config, expected):
    result = false_fault_probe(config)
    assert result["dark_fires"] == result["silent"] == 0
    assert result["false_faults"] == expected


@pytest.mark.parametrize("config", [LEGACY, *COMBINATIONS])
def test_readers_reject_the_fast_source_false_fault(config):
    result = false_fault_probe(config, rate=True)
    assert result["dark_fires"] == result["silent"] == result["false_faults"] == 0


@pytest.mark.slow
@pytest.mark.parametrize("config", [HIGH_GAIN, HIGH_MARGIN])
@pytest.mark.parametrize("case,old,_new", RELOAD_ROWS)
def test_high_gain_reload_requires_a_successful_clear(config, case, old, _new):
    baseline = reload_survey(LEGACY, case, rate=True)
    candidate = reload_survey(config, case, rate=True)
    assert baseline["control_survivors"] == 0 and baseline["recovery"] == old
    if case == "slow3sigma":
        assert candidate["control_survivors"] == 0
        assert candidate["recovery"] == (900 if config == HIGH_GAIN else 950)
        assert candidate["margin"] >= baseline["margin"]
    else:
        assert candidate["control_survivors"] == 15
        assert candidate["recovery"] is candidate["margin"] is None


@pytest.mark.slow
@pytest.mark.parametrize("config", [LEGACY, *COMBINATIONS])
@pytest.mark.parametrize("kind", ["majority", "require"])
def test_additional_reader_gate_inputs(config, kind):
    result = additional_gate_probe(config, kind)
    assert result["fires"] == [0, 0, 2000]
    if kind == "require":
        assert result["multiple"] == [0, 0, 0]


@pytest.mark.slow
@pytest.mark.parametrize("config", [LEGACY, *COMBINATIONS])
def test_ten_second_holding_exposures(config):
    random = period_survey(config, copies=1000, steps=100_000, rate=True)
    # Readers have no path back to storage. The short independent-corner test
    # above enumerates their draws too; this long exposure covers storage signs.
    corners = independent_corners(config, steps=100_000)
    print({"config": config, "random": random, "corners": corners}, flush=True)
    assert random["silent"] == random["not_holding_at_end"] == 0
    assert corners["silent"] == corners["not_holding_at_end"] == 0
    assert random["non_ignition"] == random["loss_of_hold"] == 0
    assert corners["non_ignition"] == corners["loss_of_hold"] == 0


@pytest.mark.slow
def test_random_gate_frontier_has_no_false_faults_but_misses_true_inputs():
    legacy = gate_probe(LEGACY, copies=2000)
    result = gate_probe(AUTAPSE, copies=2000)
    print({"legacy": legacy, "autapse": result}, flush=True)
    assert legacy["fault"] == legacy["completion"] == [0, 0, 2000]
    assert result["fault"][:2] == result["completion"][:2] == [0, 0]
    assert 0 < result["fault"][2] < 2000
    assert 0 < result["completion"][2] < 2000
    assert result["or"] == result["veto"] == [0, 2000, 2000]


@pytest.mark.slow
@pytest.mark.parametrize("config", [LEGACY, *COMBINATIONS])
def test_random_reader_gates_have_no_false_faults_or_missed_inputs(config):
    result = gate_probe(config, rate=True, copies=2000)
    print(config, result, flush=True)
    assert result["fault"] == result["completion"] == [0, 0, 2000]
    assert result["or"] == result["veto"] == result["holding"] == [0, 2000, 2000]


@pytest.mark.slow
@pytest.mark.parametrize("config,minimum,silent", [(LEGACY, 32, 0), (AUTAPSE, 47, 0),
                                                    (ZERO_DELAY, 36, 3370), (SHARED, 31, 0),
                                                    (ASYMMETRIC, 34, 0), (HIGH_GAIN, 54, 0)])
def test_mix_b_full_period_distribution(config, minimum, silent):
    result = period_survey(config)
    print(config, result, flush=True)
    assert result["interval"][0] == minimum and result["silent"] == silent


@pytest.mark.slow
@pytest.mark.parametrize("config", [LEGACY, *COMBINATIONS])
def test_mix_b_storage_distribution_with_readers(config):
    result = period_survey(config, rate=True)
    print(config, result, flush=True)
    assert result["silent"] == result["non_ignition"] == result["loss_of_hold"] == 0
    assert result["not_holding_at_end"] == 0
    assert (result["interval"][0] >= 44) == (config != LEGACY)


@pytest.mark.slow
@pytest.mark.parametrize("config", [LEGACY, *COMBINATIONS])
@pytest.mark.parametrize("case", CLEAR_CASES)
def test_40000_ordinary_clears_with_readers(config, case):
    result = clear_survey(config, case, rate=True)
    print(config, case, result, flush=True)
    assert result["copies"] == 40_000 and result["not_live_before_clear"] == 0
    if config == AUTAPSE:
        assert result["survivors"] == result["reader_survivors"] == 0
    elif config == LEGACY:
        expected = dict(zip(CAPTURES, (51, 1220, 10, 39, 80)))
        expected.update(fast3sigma=14, slow3sigma=0, mix_b=0)
        assert result["survivors"] == result["reader_survivors"] == expected[case]
    elif case == "slow3sigma":
        assert result["survivors"] == result["reader_survivors"] == (6 if config == HIGH_MARGIN else 0)
    elif config in (HIGH_GAIN, HIGH_MARGIN):
        assert result["survivors"] > 0
        assert result["reader_survivors"] > 0
        if case == "fast3sigma":
            assert result["survivors"] == 40_000


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
    assert baseline["latches"] == result["latches"] == 249
    assert result["extra_edges"] == 498
    assert result["values"] == [0, 3, 1]
    assert baseline["faults"] == baseline["timeouts"] == baseline["bad_outputs"] == 0
    last_steps = {(False, False): (47304, 55616), (False, True): (50728, 59215),
                  (True, False): (45499, 46147), (True, True): (48825, 49570)}
    assert (baseline["steps"][-1], result["steps"][-1]) == last_steps[rate, other]


@pytest.mark.slow
@pytest.mark.parametrize("config", [HIGH_GAIN, HIGH_MARGIN])
@pytest.mark.parametrize("other", [False, True])
def test_high_gain_reader_kernel_stalls_after_first_output(config, other):
    result = kernel_probe(config, rate=True, other_options=other)
    print(config, other, result, flush=True)
    assert result["values"] == [0]  # Missing two legitimate outputs is failure.
    assert result["faults"] == (0 if other else 1)
    assert result["timeouts"] == result["bad_outputs"] == 0
    assert result["latches"] == 249 and result["extra_edges"] == 498
    assert result["extra_neurons"] == 0


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
    clear_refutations = []
    for row in capped:
        result = independent_corners(row["config"])
        print(row["config"], result, flush=True)
        if result["interval"][0] >= 44 and result["non_ignition"] == 0:
            # Even a circuit with >200-step gaps gets a clear test: a very slow
            # continuous orbit is not necessarily a dropout. Every screen-passer
            # is refuted by rate, non-ignition or an actual reader-enabled clear,
            # never solely by the unchanged raw gate's need for a faster train.
            raw_gate = gate_probe(row["config"], corner=0, copies=1)
            assert raw_gate["fault"] != [0, 0, 1]
            clear = clear_survey(row["config"], "fast3sigma", copies=32, rate=True)
            assert clear["not_live_before_clear"] == 0 and clear["survivors"] == 32
            clear_refutations.append(row["config"])
    if family == "high":
        ranked = sorted((row for row in capped if row["config"] in clear_refutations),
                        key=lambda row: (row["config"][1], row["mean"][0]))
        assert [row["config"] for row in ranked[:2]] == [HIGH_GAIN, HIGH_MARGIN]
    else:
        assert not clear_refutations
    print({"family": family, "clear_refutations": len(clear_refutations)}, flush=True)
