"""A request's false rail that slips through its clear train is cleared again
(docs/tick_stalls.md: seed-108 copy 8, seed-110 copy 73, the seed-107 precedent's shape).

A cell's request from a source is a pair of rails [false = consumed / no pending request,
true = pending]. A source's DONE lights true and fires a three-pulse kill train on false;
START kills true and re-lights false. A false rail whose loop runs fast under noise survives
the train at some phases, both rails stay live, false vetoes the go chain, and the cell never
starts again — a stall with no fault. The retry: ~64 ms after the receipt, a gate that needs
the delayed pulse AND both rails' trains fires a second train at another phase of the loop.

Status: an experiment, off by default. It cures this constructed stall, but in the 100-copy
mix-B tick campaign (seed 108, Juno 413959) it produced 3 silent wrong values in 2 copies —
a SEL starting on a stale condition — plus 5 stalls, the same trade the stronger and longer
kill trains made. Every change so far that moves when a request rail can be relit has done
this; that ordering is the thing to understand before the next attempt."""

import numpy as np
import pytest

from drosophilos.lib.kernel import build_pipeline, run_pipeline_batched
from drosophilos.sim.lif_torch import TorchSim
from drosophilos.sim.model import Params

pytestmark = pytest.mark.slow  # minutes of TorchSim on a CPU: the daily suite, not every push

P = Params()
TOKENS = [1, 2, 3]


def _run(fast, retry):
    """Copies: node 0 nominal; node 1 a fast false rail (+20 % loop, -0.8 mV) with or without the retry.
    Built with the 3 x 0.75 train of the time (the experiment's baseline; the default became
    4 x 0.75 with the START re-light delay on 2026-09-20, which kills this rail by itself)."""
    from drosophilos.lib import control

    spec = [{"name": "out", "op": "MOV", "a": ("const", "zero"), "b": "input"}]
    saved = control.KILL_PULSES, control.KILL_STRENGTH
    control.KILL_PULSES, control.KILL_STRENGTH = 3, 0.75
    try:
        pl = build_pipeline(P, 4, spec, consts={"zero": 0}, retry_clear=retry, start_relight_hops=0, request_clear_pulses=3)
    finally:
        control.KILL_PULSES, control.KILL_STRENGTH = saved
    roles = pl.net.roles
    cell = pl.cells[0]
    false, true = cell.reqs["input"]
    topo = pl.net.topology()
    quanta = np.broadcast_to(topo.quanta, (2, topo.nnz)).copy().astype(np.int64)
    vth = np.full((2, topo.n), P.V_th)
    if fast:
        loop_e = ((topo.src == false.u) & (topo.dst == false.v)) | ((topo.src == false.v) & (topo.dst == false.u))
        quanta[1, loop_e] = np.round(topo.quanta[loop_e] * 1.2)
        vth[1, [false.u, false.v]] -= 0.8
    sim = TorchSim(topo, P, n_nodes=2, device="cpu", quanta=quanta, V_th=vth)
    outs, _, st = run_pipeline_batched(pl, P, [TOKENS, TOKENS], max_ms=12000, device="cpu", progress=0, sim=sim)
    return [[v for _, v in o["out"]] for o in outs], st


def test_a_fast_false_rail_stalls_the_cell_without_the_retry():
    got, st = _run(fast=True, retry=False)
    assert got[0] == TOKENS, got
    # copy 73's shape: both rails live after a DONE and no START — at most the first token gets
    # through (the train's transient slowing of the false rail can let one go fire)
    assert len(got[1]) <= 1, got
    assert st["faults"] == 0


def test_the_retry_clears_it_and_the_cell_runs():
    got, st = _run(fast=True, retry=True)
    assert got == [TOKENS, TOKENS], got
    assert st["faults"] == 0 and st["refusals"] == 0


def test_the_retry_gate_stays_silent_on_a_nominal_cell():
    spec = [{"name": "out", "op": "MOV", "a": ("const", "zero"), "b": "input"}]
    pl = build_pipeline(P, 4, spec, consts={"zero": 0}, retry_clear=True)
    gate = pl.net.roles.index("out.req.input.retry_gate")
    outs, _, st = run_pipeline_batched(pl, P, [TOKENS], max_ms=12000, device="cpu", progress=0,
                                       capture_spikes=(0, [gate]))
    assert [v for _, v in outs[0]["out"]] == TOKENS
    steps, _ = st["captured_spikes"]
    assert len(steps) == 0, steps  # never both rails live 64 ms after a receipt
