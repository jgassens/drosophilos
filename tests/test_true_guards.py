"""True-rail requirements on the kernel's go and commit guards."""

import numpy as np
import pytest

from drosophilos.bench.a2_campaigns import MIXES
from drosophilos.lib.control import add_kill_pair, add_kill_train
from drosophilos.lib.kernel import guarded_pulse, build_pipeline, run_pipeline_batched
from drosophilos.lib.netlist import Drive, Netlist
from drosophilos.sim.model import Params
from drosophilos.sim.ref64 import RefSim


P = Params()


class GuardProbe(RefSim):
    """RefSim with exact sparse Bernoulli strays and only the requested spike record.

    The distribution matches campaign.py: independent log-normal edge noise, Gaussian
    threshold/bias drift, and 5 Hz x 150-quanta input on *every* neuron. Mix B's sigmas
    are spreads, not bounds. Filtering the record keeps 10,000 trials laptop-sized.
    """

    def __init__(self, net, copies, seed, watch, *, noisy=True, stray=True, **kwargs):
        topo = net.topology()
        rng = np.random.default_rng(seed)
        mix = MIXES["B"]
        if noisy:
            kwargs.update(
                quanta=np.rint(topo.quanta * np.exp(rng.normal(0, mix.weight_sigma, (copies, topo.nnz)))),
                V_th=P.V_th + rng.normal(0, mix.th_sigma_mv, (copies, topo.n)),
                bias=rng.normal(0, mix.bias_sigma_mv, (copies, topo.n)) + np.asarray(net.bias),
            )
        super().__init__(topo, P, n_nodes=copies, **kwargs)
        self.rng = rng
        self.stray = noisy and stray
        self.watch = np.asarray(watch)
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
                    steps.extend([self.step_index - 1] * len(hit))
                    copies.extend(hit.tolist())
        self._spk_step.clear()
        self._spk_node.clear()
        self._spk_neuron.clear()

    def counts(self, neuron):
        return np.bincount(self.events[neuron][1], minlength=self.B)


def _guard_fixture(relight_hops=0, true_guards=True):
    from drosophilos.protocol.celement import add_delay_chain

    drive = Drive.from_params(P)
    net = Netlist(P)
    a = add_kill_pair(net, drive, "a")
    b = add_kill_pair(net, drive, "b")
    target = net.neuron("target")
    guarded_pulse(net, drive, "guard", a, b, target, true_guards=true_guards)
    # A real START consumes both TRUE rails immediately, then re-lights the FALSE rails.
    add_kill_train(net, drive, "consume", target, [a[1], b[1]])
    relight = add_delay_chain(net, drive, "relight", target, relight_hops)
    for pair in (a, b):
        net.synapse(relight, pair[0].u, drive.ignite)
    return net, drive, a, b, target


@pytest.mark.slow
@pytest.mark.parametrize("relight_hops", [0, 5])
def test_mix_b_nearby_rises_are_never_lost(relight_hops):
    """Reviewer near.py: prior FALSE activity, real trains, B-A in [-60, +60] ms."""
    net, drive, a, b, target = _guard_fixture(relight_hops)
    copies = 3000
    sim = GuardProbe(net, copies, 7, [target])
    offsets = np.rint(np.linspace(-600, 600, copies)).astype(int)
    for node, offset in enumerate(offsets):
        sim.add_events(node, [1, 1, 2500, 2500 + offset],
                       [a[0].u, b[0].u, a[1].u, b[1].u], [drive.ignite] * 4)
    sim.run(5200)
    counts = sim.counts(target)
    assert np.all(counts == 1), ("lost", np.flatnonzero(counts == 0),
                                "duplicate", np.flatnonzero(counts > 1))


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(10))
def test_mix_b_dark_pair_never_passes(seed):
    """10 x 1,000 actual driver trains, including strays on the whole guard."""
    net, drive, a, b, target = _guard_fixture()
    sim = GuardProbe(net, 1000, seed, [target])
    for node in range(sim.B):
        # Both orderings, with the other pair completely dark. No single-spike shortcut.
        source = a if node % 2 else b
        sim.add_events(node, [1000], [source[1].u], [drive.ignite])
    sim.run(3500)
    assert not sim.counts(target).any(), sim.events[target]


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(5))
def test_mix_b_dark_required_relay_with_213_hz_driver(seed):
    """Review mc.py's settled-rate driver: 10,000 events with a dark required rail.

    Prescribe source spikes at 213 Hz by delivering each outgoing edge's perturbed
    quanta at its synaptic delay. This exercises the first/second-spike race directly,
    without a latch's initially slower warm-up or a particular delay-chain transient.
    Both the edge inhibitor and coincidence neuron retain full RefSim dynamics.
    """
    from drosophilos.lib.kernel import _N
    from drosophilos.protocol.celement import add_veto_relay
    from drosophilos.protocol.latch import add_latch

    drive = Drive.from_params(P)
    net = Netlist(P)
    driver = net.neuron("driver")
    rail = add_latch(net, drive, "dark")
    target = net.neuron("target")
    add_veto_relay(net, drive, "guard", driver, [], _N(target), require=[rail.u])
    sim = GuardProbe(net, 2000, seed, [target])
    nodes = np.arange(sim.B)
    for edge in np.flatnonzero(sim.topo.src == driver):
        for step in range(1000, 3500, drive.loop_period_steps):
            sim._events[step + int(sim.topo.delay[edge])].append(
                (nodes, np.full(sim.B, sim.topo.dst[edge]), sim.quanta[:, edge]))
    sim.run(4000)
    assert not sim.counts(target).any(), sim.events[target]


def test_permanent_rail_and_fast_inhibitor_corners():
    """Real trains: weak coincidence/fast inhibitor, dark driver, and permanent rail.

    Include the review's +20% latch loop, +12% rail edge, -1 mV threshold corner;
    the permanent rail is watched for two seconds, with no delayed driver event.
    """
    from itertools import product
    from drosophilos.protocol.celement import add_veto_relay
    from drosophilos.protocol.latch import add_latch

    drive = Drive.from_params(P)
    net = Netlist(P)
    source = add_latch(net, drive, "source")
    rail = add_latch(net, drive, "rail")
    target = add_latch(net, drive, "target")
    relay = add_veto_relay(net, drive, "guard", source.u, [], target, require=[rail.u])
    topo = net.topology()
    cases = list(product(("both", "driver", "rail"), (0.88, 0.96, 1.04, 1.12), (-1.0, 0.0, 1.0)))
    q = np.broadcast_to(topo.quanta, (len(cases), topo.nnz)).copy()
    vth = np.full((len(cases), topo.n), P.V_th)
    inh = net.roles.index("guard.driver.edge_inh")
    for node, (mode, scale, threshold) in enumerate(cases):
        q[node, (topo.dst == relay) & (topo.quanta > 0)] = np.rint(
            q[node, (topo.dst == relay) & (topo.quanta > 0)] * scale)
        vth[node, relay] += threshold
        # Speed up the edge inhibitor independently of the coincidence neuron's drive.
        q[node, topo.dst == inh] = np.rint(q[node, topo.dst == inh] * 1.12)
        vth[node, inh] -= 1.0
        if mode == "rail":
            loop = np.isin(topo.src, rail.members) & np.isin(topo.dst, rail.members)
            q[node, loop] = np.rint(q[node, loop] * 1.20)
    sim = GuardProbe(net, len(cases), 0, [relay], noisy=False, quanta=q, V_th=vth)
    for node, (mode, _, _) in enumerate(cases):
        if mode != "driver":
            sim.add_events(node, [1], [rail.u], [drive.ignite])
        if mode != "rail":
            sim.add_events(node, [1200], [source.u], [drive.ignite])
    sim.run(20000)
    counts = sim.counts(relay)
    assert counts.tolist() == [int(mode == "both") for mode, _, _ in cases], [
        (case, count) for case, count in zip(cases, counts) if count != int(case[0] == "both")]


def test_chained_guard_rechecks_every_original_true_rail():
    """An injected stale cache cannot stand in for any dark original input."""
    from drosophilos.lib.kernel import _chain_true

    drive = Drive.from_params(P)
    net = Netlist(P)
    originals = [add_kill_pair(net, drive, f"req{k}") for k in range(3)]
    idle = add_kill_pair(net, drive, "idle")
    target = net.neuron("start")
    image = [idle[0]]
    _chain_true(net, drive, "go", originals + [idle], target, image, target)
    add_kill_train(net, drive, "consume", target, [p[1] for p in originals + [idle]])
    for pair in originals + [idle]:
        net.synapse(target, pair[0].u, drive.ignite)
    passed = net.roles.index("go.p1r1.u")
    # 0: all true. 1..3: each original dark in turn. 4..6: each original false in turn.
    sim = GuardProbe(net, 7, 0, [target], noisy=False)
    for node in range(sim.B):
        for latch in image:
            sim.add_events(node, [1], [latch.u], [drive.ignite])
        for k, pair in enumerate(originals):
            if node == k + 1:
                continue
            value = 0 if node == k + 4 else 1
            sim.add_events(node, [1], [pair[value].u], [drive.ignite])
        sim.add_events(node, [1500, 2500], [passed, idle[1].u], [drive.ignite] * 2)
    sim.run(5000)
    assert sim.counts(target).tolist() == [1, 0, 0, 0, 0, 0, 0]


def test_both_arrival_orders_keep_the_legacy_latency():
    offsets = np.arange(-1000, 1001, 100)
    latencies = []
    for true_guards in (False, True):
        net, drive, a, b, target = _guard_fixture(true_guards=true_guards)
        sim = GuardProbe(net, len(offsets), 0, [target], noisy=False)
        for node, offset in enumerate(offsets):
            sim.add_events(node, [1, 1, 2500, 2500 + offset],
                           [a[0].u, b[0].u, a[1].u, b[1].u], [drive.ignite] * 4)
        sim.run(5500)
        assert sim.counts(target).min() >= 1
        steps, nodes = (np.asarray(x) for x in sim.events[target])
        first = np.array([steps[nodes == node].min() for node in range(sim.B)])
        latencies.append((first - 2500 - np.maximum(offsets, 0)) * P.dt)
    # Recovery improvements can select an earlier path; no ordering may get >10 ms slower.
    delta = latencies[1] - latencies[0]
    assert delta.max() <= 10.0, (offsets, latencies, delta)
    print("guard latency delta (ms):", delta.min(), delta.max())


def _spikes(sim, neuron, node):
    out = []
    for steps, nodes, neurons in zip(sim._spk_step, sim._spk_node, sim._spk_neuron):
        out.extend(int(s) for s, b, n in zip(steps, nodes, neurons)
                   if int(b) == node and int(n) == neuron)
    return out


def test_guard_requires_the_other_true_rail_and_keeps_noise_margin():
    """A delayed pulse needs a live true rail; dark and false-live pairs do not pass.

    Nodes 3--5 exercise selected corners on the coincidence inputs:
    -4 % weights/+1 mV threshold for the required coincidence, and +4 %/-1 mV for
    pulse-only and rail-only leakage. Full-train and veto-history coverage is above.
    """
    drive = Drive.from_params(P)
    net = Netlist(P)
    a = add_kill_pair(net, drive, "a")
    b = add_kill_pair(net, drive, "b")
    target = net.neuron("target")
    guarded_pulse(net, drive, "guard", a, b, target)

    topo = net.topology()
    relay = net.roles.index("guard.pa.edge")
    delayed = [k for k, r in enumerate(net.roles) if r.startswith("guard.ad.d")][-1]
    quanta = np.broadcast_to(topo.quanta, (6, topo.nnz)).copy()
    positive_inputs = (topo.dst == relay) & (topo.quanta > 0)
    quanta[3, positive_inputs] = np.rint(quanta[3, positive_inputs] * 0.96)
    quanta[4:, positive_inputs] = np.rint(quanta[4:, positive_inputs] * 1.04)
    thresholds = np.full((6, topo.n), P.V_th)
    thresholds[3, relay] += 1.0
    thresholds[4:, relay] -= 1.0
    sim = RefSim(topo, P, n_nodes=6, quanta=quanta, V_th=thresholds)

    # 0 dark+pulse; 1 true+pulse; 2 false+pulse; 3 weak true+pulse;
    # 4 strong dark+pulse; 5 strong true rail without a pulse.
    for node in (1, 3, 5):
        sim.add_events(node, [1], [b[1].u], [drive.ignite])
    sim.add_events(2, [1], [b[0].u], [drive.ignite])
    for node in (0, 1, 2, 3, 4):
        # Inject the already-delayed A-path pulse; the circuit supplies the one-shot.
        sim.add_events(node, [1000], [delayed], [drive.pulse])
    sim.run(4000)

    assert [len(_spikes(sim, target, node)) for node in range(6)] == [0, 1, 0, 1, 0, 0]


def _run_dark_commit_pair(true_guards):
    """Run one producer/reader token after suppressing both commit-idle ignitions."""
    spec = [
        {"name": "a", "op": "MOV", "a": "input", "b": ("const", "zero")},
        {"name": "b", "op": "MOV", "a": "a", "b": ("const", "zero"),
         "trigger": ["input:other"]},
    ]
    pl = build_pipeline(P, 1, spec, consts={"zero": 0}, outputs=["b"],
                        streams=["input", "other"], start_relight_hops=0, true_guards=true_guards)
    producer, reader = pl.cells
    topo = pl.net.topology()
    quanta = np.broadcast_to(topo.quanta, (1, topo.nnz)).copy()
    pulse = pl.net.roles.index("a.commit_pulse")
    again = pl.net.roles.index("a.commit.idle_again.d15")
    direct = (topo.src == pulse) & (topo.dst == producer.creq[0].u)
    delayed = (topo.src == again) & (topo.dst == producer.creq[0].u)
    assert direct.sum() == delayed.sum() == 1
    quanta[:, direct | delayed] = 0

    # Hold the reader's second request back long enough for its free-rail guard path to
    # recover. When it finally STARTs, the old veto-only guard reads producer.creq dark as
    # true and commits the already-empty stage a second time.
    sim = RefSim(topo, P, n_nodes=1, quanta=quanta)
    sim.add_events(0, [30000], [pl.inputs["other"][0].done_relay], [pl.drive.ignite])
    request = reader.reqs["input:other"][1].u
    ids = [pulse, reader.start, request]
    _, _, stats = run_pipeline_batched(
        pl, P, [[1]], max_ms=4000, expect_outputs=[2], sim=sim, progress=0,
        capture_spikes=(0, ids),
    )
    steps, neurons = (np.asarray(x) for x in stats["captured_spikes"])
    commits = steps[neurons == pulse].tolist()
    starts = steps[neurons == reader.start].tolist()
    requests = steps[neurons == request].tolist()
    assert len(starts) == 1 and requests
    return commits, (starts[0] - requests[0]) * P.dt


def test_dark_commit_request_does_not_issue_a_second_commit():
    old_commits, old_start_latency = _run_dark_commit_pair(False)
    guarded_commits, guarded_start_latency = _run_dark_commit_pair(True)

    assert len(old_commits) == 2, old_commits
    assert len(guarded_commits) == 1, guarded_commits
    print("reader START after request (ms):", old_start_latency, guarded_start_latency)
    # Requiring a true train changes the qualification, not an ordinary request's timing.
    assert abs(guarded_start_latency - old_start_latency) <= 10.0
