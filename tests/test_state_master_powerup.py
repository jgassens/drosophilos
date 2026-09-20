"""A state cell's master at power-up (docs/tick_stalls.md, seed-109 copy 61, seed-110 copy 55).

The image lights a state cell's initial value in its master but keeps the master's completion
root dark on purpose: a complete master would fire DONE and every reader would run a
transaction before the first token. The root is the last AND of the completion tree; with the
data half of the tree lit and the flag half dark it sits at 65 % of threshold for the whole
run, and one lucky stray coincidence fires it once — after which the root latch is lit for
good, DONE fires, and the readers run an extra transaction on the initial value (mx lagged a
token from then on; scored as 8 wrong values). The fix vetoes the root's ignition relay from
power-up until the master's first reset (its first commit)."""

import numpy as np
import pytest

from drosophilos.lib.kernel import build_pipeline, run_pipeline_batched
from drosophilos.sim.lif_torch import TorchSim
from drosophilos.sim.model import Params

pytestmark = pytest.mark.slow  # minutes of TorchSim on a CPU: the daily suite, not every push

P = Params()
TOKENS = [1, 2, 3]


def _state_kernel():
    spec = [{"name": "s", "op": "ADD", "a": "s", "b": "input", "init": 5}]
    return build_pipeline(P, 4, spec, consts={}, outputs=["s"])


def _root_and(pl, cell):
    roles = pl.net.roles
    root = roles[cell.master.completion.u][: -len(".L.u")]
    return roles.index(f"{root}.and"), roles.index(f"{root}.ign.edge")


def _run(stray_steps):
    pl = _state_kernel()
    cell = pl.cells[0]
    root_and, _ = _root_and(pl, cell)
    sim = TorchSim(pl.net.topology(), P, n_nodes=1, device="cpu")
    if stray_steps:
        # one spike of the root AND gate before any token: what a stray coincidence does
        sim.add_events(0, list(stray_steps), [root_and] * len(stray_steps), [pl.drive.ignite] * len(stray_steps))
    outs, _, st = run_pipeline_batched(pl, P, [TOKENS], max_ms=30000, device="cpu", progress=0, sim=sim,
                                       capture_spikes=(0, [cell.master.completion.u]))
    return pl, outs[0]["s"], st


def test_state_value_is_not_announced_before_the_first_token():
    pl, out, st = _run([2000])
    first_load = st["load_events"][0][0]["event_step"]
    steps, _ = st["captured_spikes"]
    assert not np.any(np.asarray(steps) < first_load), "the master's completion rose before the first token"
    assert [v for _, v in out] == [6, 8, 11], (out, st["refused"])
    assert all(s_ > first_load for s_, _ in out)
    assert st["faults"] == 0 and st["refusals"] == 0


def test_the_first_commit_lifts_the_veto():
    pl, out, st = _run([])
    assert [v for _, v in out] == [6, 8, 11], out
