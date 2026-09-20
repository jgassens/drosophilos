"""Minimal reproducer for the §5 finding (docs/mul_diag.md): a value read by a cell far down
the chain holds its producer's commit — a producer commits its next value only once every
consumer has started on the previous one (`lib/kernel.py`, `gate_commit`) — so the whole
chain upstream of that producer runs one token at a time and the pipeline's period is its
path latency, not its slowest cell.

Chain A -> B -> C -> D with D also reading A directly ("held"), against the same chain with
A's value carried to D through copies ("relayed": A -> m1 -> m2 -> D). Same arithmetic, same
outputs; only the period differs. Measured on the reference simulator (4-bit, CPU)."""

import statistics

import pytest

from drosophilos.bench import mul_diag
from drosophilos.compiler.kernel import KernelSpec, kernel_outputs
from drosophilos.lib.kernel import build_pipeline, run_pipeline_batched
from drosophilos.sim.model import Params

pytestmark = pytest.mark.slow  # ~5 min of RefSim on a CI runner

P = Params()
TOKENS = [1, 2, 3, 5, 7, 4]


def _held_spec():
    # D = (C) XOR A, C = B + 1, B = A + 1, A = input AND 15: A is read at depth 1 and depth 3
    return [{"name": "A", "op": "AND", "a": "input", "b": ("const", "k15")},
            {"name": "B", "op": "ADD", "a": "A", "b": ("const", "k1")},
            {"name": "C", "op": "ADD", "a": "B", "b": ("const", "k1")},
            {"name": "D", "op": "XOR", "a": "C", "b": "A"}]


def _relayed_spec():
    # the same function; A reaches D through two MOV copies so every consumer of A is one hop away
    return [{"name": "A", "op": "AND", "a": "input", "b": ("const", "k15")},
            {"name": "B", "op": "ADD", "a": "A", "b": ("const", "k1")},
            {"name": "m1", "op": "MOV", "a": ("const", "k0"), "b": "A"},
            {"name": "C", "op": "ADD", "a": "B", "b": ("const", "k1")},
            {"name": "m2", "op": "MOV", "a": ("const", "k0"), "b": "m1"},
            {"name": "D", "op": "XOR", "a": "C", "b": "m2"}]


def _run(spec):
    consts = {"k15": 15, "k1": 1, "k0": 0}
    ks = KernelSpec(spec, consts, {}, "input", 4)
    ks.outputs = ["D"]
    pl = build_pipeline(P, 4, spec, consts=consts, outputs=["D"])
    cells = {c.name: c for c in pl.cells}
    probes = {n: mul_diag.probes_for(pl, cells[n]) for n in ("A", "D")}
    ids = sorted({v for p in probes.values() for v in p.values()})
    outs, _, st = run_pipeline_batched(pl, P, [TOKENS], max_ms=60000, device="cpu", expect_outputs=[len(TOKENS)],
                                       progress=0, capture_spikes=(0, ids))
    got = [v for _, v in outs[0]["D"]]
    assert got == [row[0] for row in kernel_outputs(ks, TOKENS)], got
    rep = mul_diag.analyse(st["captured_spikes"], probes, pl.drive.loop_period_steps, P.dt, cells)
    med = lambda n, key: statistics.median(r[key] for r in rep["cells"][n]["tokens"] if r.get(key) is not None)
    return {"neurons": pl.net.n, "period_A": med("A", "period_ms"), "period_D": med("D", "period_ms"),
            "compute_A": med("A", "compute_ms"), "commit_wait_A": med("A", "commit_wait_ms"),
            "first_D_ms": rep["cells"]["D"]["summary"]["first_result_ms"], "faults": st["faults"]}


@pytest.mark.parametrize("variant", ["held", "relayed"])
def test_chain_variants_compute_the_same_function(variant):
    r = _run(_held_spec() if variant == "held" else _relayed_spec())
    assert r["faults"] == 0 and r["period_A"] > 0


def test_a_late_reader_holds_the_producer_so_the_period_is_the_path_latency():
    held, relayed = _run(_held_spec()), _run(_relayed_spec())
    # held: A's value is complete after its compute but A cannot commit it until D has started
    # on the previous token, so the wait shows up as A's commit_wait and A's period becomes the
    # path latency A -> B -> C -> D; relayed: every reader of A is one hop away, the commit
    # wait is small and the period is about one cell's cycle.
    # Measured (RefSim, 4-bit, 2026-09-20): held period 2,550 ms, commit_wait 1,591 ms (compute 365 ms);
    # relayed period 1,117 ms — 2.3x the throughput for the same function (+2,928 neurons).
    assert held["commit_wait_A"] > 3.0 * relayed["commit_wait_A"], (held, relayed)
    assert held["commit_wait_A"] > held["compute_A"], held
    assert relayed["period_A"] < 0.6 * held["period_A"], (held, relayed)
    assert relayed["period_D"] < 0.6 * held["period_D"], (held, relayed)
