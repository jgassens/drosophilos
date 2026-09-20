"""The §5 multiplier diagnostic: rise detection and the per-token timeline on synthetic
events, and the probe set on real array and pipelined multiplier cells (a 4-bit MUL on a
token stream, one copy, CPU)."""

import numpy as np
import pytest

from drosophilos.bench import mul_diag
from drosophilos.compiler.kernel import KernelSpec, kernel_outputs
from drosophilos.lib.kernel import build_pipeline, run_pipeline_batched
from drosophilos.sim.model import Params

P = Params()


def test_rises_collapse_latch_trains_and_keep_pulses():
    train = np.concatenate([np.arange(1000, 1400, 47), np.arange(3000, 3300, 47), [5000]])
    assert mul_diag.rises(train, gap=141) == [1000, 3000, 5000]
    assert mul_diag.rises(np.array([], dtype=np.int64), gap=141) == []


def test_token_timeline_splits_phases_in_order():
    ev = {"req:a": [90, 2090], "idle": [0, 1500], "start": [100, 2100], "actd": [200, 2200],
          "completion": [700, 2700], "creq": [710, 2710], "commit": [900, 3400], "done": [1000, 3500],
          "consumer_start:c2": [1050, 3550]}
    rows = mul_diag.token_timeline(ev, dt_ms=0.1)
    assert [r["token"] for r in rows] == [0, 1]
    r0, r1 = rows
    assert r0["compute_ms"] == pytest.approx(60.0) and r0["commit_wait_ms"] == pytest.approx(20.0)
    assert r0["drain_ms"] == pytest.approx(10.0) and r0["period_ms"] == pytest.approx(200.0)
    assert r0["wait_input_ms"] is None  # no previous done
    assert r1["wait_input_ms"] == pytest.approx((2090 - 1000) * 0.1)
    assert r1["commit_wait_ms"] == pytest.approx(70.0)  # the second token waited on its reader
    s = mul_diag.summarize(rows)
    assert s["tokens"] == 2 and s["compute_ms"]["median"] == pytest.approx(60.0)


@pytest.mark.parametrize("op", ["MUL", "MULP"])
def test_probes_and_capture_on_a_real_multiplier_cell(op):
    spec = [{"name": "m", "op": op, "a": "input", "b": ("const", "k")}]
    ks = KernelSpec(spec, {"k": 3}, {}, "input", 4)
    ks.outputs = ["m"]
    pl = build_pipeline(P, 4, spec, consts={"k": 3}, outputs=["m"])
    ids, per_cell = mul_diag.capture_ids(pl)
    cells = mul_diag.multiplier_cells(pl)
    assert cells and all(c.op in mul_diag.MUL_OPS for c in cells)
    if op == "MULP":
        assert len(cells) == 4 and sorted(c.row for c in cells) == [0, 1, 2, 3]
    for probes in per_cell.values():
        for key in ("start", "act", "actd", "completion", "creq", "commit", "done", "idle"):
            assert key in probes, key
        assert any(k.startswith("req:") for k in probes)
    assert len(ids) == len(set(ids.tolist()))
    tokens = [5, 2, 7]
    outs, sim, stats = run_pipeline_batched(pl, P, [tokens], max_ms=20000, device="cpu",
                                            expect_outputs=[len(tokens)], progress=0, capture_spikes=(0, ids))
    got = [v for _, v in outs[0]["m"]]
    assert got == [row[0] for row in kernel_outputs(ks, tokens)]
    report = mul_diag.analyse(stats["captured_spikes"], per_cell, pl.drive.loop_period_steps, P.dt,
                              {c.name: c for c in pl.cells})
    last = [c for c in cells if c.op == "MUL" or c.row == 3][0]
    info = report["cells"][last.name]
    assert info["summary"]["tokens"] == len(tokens)
    for row in info["tokens"]:
        assert row["compute_ms"] is not None and row["compute_ms"] > 0
        assert row["done"] is not None and row["commit"] is not None and row["completion"] < row["commit"] <= row["done"]
    assert info["summary"]["first_result_ms"] is not None
    if op == "MULP":
        (base_stats,) = report["row_overlap"].values()
        assert base_stats["rows"] == 4 and base_stats["tokens"] == len(tokens)
