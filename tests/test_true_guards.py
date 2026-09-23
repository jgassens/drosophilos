"""True-rail requirements on the kernel's go and commit guards."""

import numpy as np

from drosophilos.lib.control import add_kill_pair
from drosophilos.lib.kernel import guarded_pulse, build_pipeline, run_pipeline_batched
from drosophilos.lib.netlist import Drive, Netlist
from drosophilos.sim.model import Params
from drosophilos.sim.ref64 import RefSim


P = Params()


def _spikes(sim, neuron, node):
    out = []
    for steps, nodes, neurons in zip(sim._spk_step, sim._spk_node, sim._spk_neuron):
        out.extend(int(s) for s, b, n in zip(steps, nodes, neurons)
                   if int(b) == node and int(n) == neuron)
    return out


def test_guard_requires_the_other_true_rail_and_keeps_noise_margin():
    """A delayed pulse needs a live true rail; dark and false-live pairs do not pass.

    Nodes 3--5 exercise the campaign mix's stacked edge cases on the relay inputs:
    -4 % weights/+1 mV threshold for the required coincidence, and +4 %/-1 mV for
    pulse-only and rail-only leakage. This is the measured check for the 0.65 + 0.65
    threshold fractions documented by ``add_veto_relay``.
    """
    drive = Drive.from_params(P)
    net = Netlist(P)
    a = add_kill_pair(net, drive, "a")
    b = add_kill_pair(net, drive, "b")
    target = net.neuron("target")
    guarded_pulse(net, drive, "guard", a, b, target)

    topo = net.topology()
    relay = net.roles.index("guard.pa.edge")
    delayed = net.roles.index("guard.ad.d11")
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
        # Inject the already-delayed A-path pulse, isolating the relay's coincidence margin.
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
                        streams=["input", "other"], true_guards=true_guards)
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
    # Requiring a true train changes the qualification, not an ordinary request's timing.
    assert abs(guarded_start_latency - old_start_latency) <= 3.0
