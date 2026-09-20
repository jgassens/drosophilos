"""A failed re-ignition of a cell's "nothing to commit" rail (docs/tick_stalls.md, seed-110 copy 55).

A cell's commit request is a kill pair [nothing to commit, pending]. The stage's completion
lights "pending" (killing "nothing"); the commit pulse fires when pending is true and every
reader is free, kills "pending" and re-lights "nothing". When the readers were already free
the re-light comes ~50 ms after the kill, inside the rail's after-hyperpolarisation, and under
noise the single ignition can fail: one spike, no train, both rails dark. The guard
(`guarded_pulse`) vetoes on the false rail only, so a dark pair reads as true, and the
reader's next START fires a second commit of the already-emptied stage. The reader takes a
stale value and the producer runs one token behind from then on: silent wrong values, no
fault. The fix is a second ignition ~85 ms after the commit pulse. This test models the
failed first ignition by zeroing that synapse on one of two copies."""

import numpy as np

from drosophilos.lib.kernel import build_pipeline, run_pipeline_batched
from drosophilos.sim.lif_torch import TorchSim
from drosophilos.sim.model import Params

P = Params()
TOKENS = [1, 2, 3, 5]
EXPECTED = [2, 3, 4, 6]


def _kernel():
    spec = [{"name": "a", "op": "AND", "a": "input", "b": ("const", "k15")},
            {"name": "b", "op": "ADD", "a": "a", "b": ("const", "k1")}]
    return build_pipeline(P, 4, spec, consts={"k15": 15, "k1": 1}, outputs=["b"])


def test_the_idle_rail_relights_after_every_commit_even_when_the_first_ignition_is_lost():
    pl = _kernel()
    roles = pl.net.roles
    a = pl.cells[0]
    topo = pl.net.topology()
    quanta = np.broadcast_to(topo.quanta, (3, topo.nnz)).copy()
    pulse = roles.index("a.commit_pulse")
    again = roles.index("a.commit.idle_again.d15")
    direct = (topo.src == pulse) & (topo.dst == a.creq[0].u)
    delayed = (topo.src == again) & (topo.dst == a.creq[0].u)
    assert direct.sum() == 1 and delayed.sum() == 1
    quanta[1, direct] = 0  # copy 1: the immediate re-ignition never lands
    quanta[2, direct] = 0  # copy 2: neither does the delayed one — the build before the fix
    quanta[2, delayed] = 0
    sim = TorchSim(topo, P, n_nodes=3, device="cpu", quanta=quanta)
    idle_u, pending_u = a.creq[0].u, a.creq[1].u
    outs, _, st = run_pipeline_batched(pl, P, [TOKENS] * 3, max_ms=30000, device="cpu", progress=0, sim=sim,
                                       capture_spikes=None)
    # read the three copies' idle rails through a second run with capture (capture is one node)
    trains = {}
    for node in range(3):
        sim_n = TorchSim(topo, P, n_nodes=3, device="cpu", quanta=quanta)
        _, _, st_n = run_pipeline_batched(pl, P, [TOKENS] * 3, max_ms=30000, device="cpu", progress=0, sim=sim_n,
                                          capture_spikes=(node, [idle_u, pulse]))
        steps, neus = st_n["captured_spikes"]
        steps, neus = np.asarray(steps), np.asarray(neus)
        commits = np.sort(steps[neus == pulse])
        idle = np.sort(steps[neus == idle_u])
        # spikes of the idle rail in the 1,500 steps after each commit pulse (the delayed ignition
        # lands at ~850): a train (>= 8) or a lone spike
        trains[node] = [int(((idle > c) & (idle < c + 1500)).sum()) for c in commits[:4]]
    got = [[v for _, v in o["b"]] for o in outs]
    assert got[0] == EXPECTED and got[1] == EXPECTED, got
    assert got[2] == EXPECTED[: len(got[2])], got  # the unfixed copy: a prefix at best (its duplicate commit is not absorbed on every build)
    assert all(n >= 8 for n in trains[0]), trains  # nominal: the rail relights at once
    assert all(n >= 8 for n in trains[1]), trains  # lost first ignition: the delayed one relights it
    assert all(n <= 2 for n in trains[2]), trains  # before the fix: one spike, no train — a dark pair
