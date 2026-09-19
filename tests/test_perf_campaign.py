"""Fast contract tests for the fixed §3 comparison driver."""

import argparse
import json

import pytest

from drosophilos.bench.perf_campaign import (
    Variant,
    coverage_check,
    run_campaign,
)
from drosophilos.bench.render_doom import RenderConfig


def _args(**overrides):
    base = dict(levels="primitive", backend="torch", device="cpu", dtype="float64",
               mul="array", copies="1,8", pacing="host", repeats=1, tokens=16,
               primitive_blocks="cells,fanout,tick", primitive_max_s=150, profile_steps=0,
               historical=False, allow_cpu_renders=False, dry_run=False, out="data/perf/campaign")
    base.update(overrides)
    return argparse.Namespace(**base)


def test_level_two_coverage_accepts_selected_inputs_and_rejects_static_input(tmp_path):
    assert coverage_check(RenderConfig(source="examples/doom2.c", width=8, height=5, frames=3,
                                       inputs="2,258,0"), directory=tmp_path / "doom2")
    assert coverage_check(RenderConfig(source="examples/doom4.c", width=8, height=5, frames=3,
                                       inputs="6,256,0"), directory=tmp_path / "doom4")["sprite_pixels"]
    with pytest.raises(RuntimeError, match="input sequence.*adjust"):
        coverage_check(RenderConfig(source="examples/doom2.c", width=8, height=5, frames=3,
                                    inputs="0,0,0"), directory=tmp_path / "static")


def test_two_key_variant_is_refused():
    with pytest.raises(ValueError, match="at most one key"):
        Variant("not-a-comparison", {"copies": 8, "dtype": "float32"})


def test_dry_run_lists_copy_configurations(tmp_path):
    args = _args(levels="primitive,small", primitive_blocks="cells", copies="1,8", repeats=3,
                allow_cpu_renders=True, dry_run=True, out=str(tmp_path / "report"))
    result = run_campaign(args)
    assert result["dry_run"]
    assert [x["copies"] for x in result["configurations"]] == [1, 8]


def test_dry_run_prints_cost_estimate(tmp_path):
    args = _args(levels="primitive", primitive_blocks="cells,perspective", copies="1,8",
                dry_run=True, out=str(tmp_path / "report"))
    result = run_campaign(args)
    assert result["dry_run"]
    assert result["cost_estimate"], "expected one estimate per configuration"
    for entry in result["cost_estimate"]:
        assert entry["estimated_neural_s"] > 0
        assert entry["estimated_neural_s"] <= args.primitive_max_s
    assert "wall" in result["cost_estimate_note"]
    assert "not a measurement" in result["cost_estimate_note"] or "estimate" in result["cost_estimate_note"]


def test_primitive_cells_write_correct_cpu_report(tmp_path):
    args = _args(primitive_blocks="cells", tokens=3, copies="1", out=str(tmp_path / "cells"))
    report = run_campaign(args)
    assert {x["workload"] for x in report["summaries"]} == {
        "cells-add", "cells-and", "cells-xor", "cells-mov"}
    assert all(x["wrong"] == x["missing"] == 0 for x in report["summaries"])
    assert (tmp_path / "cells.json").exists()
    assert (tmp_path / "cells.md").exists()


def test_bare_defaults_run_the_primitive_smoke_set(tmp_path):
    """A bare invocation is a short smoke: primitive cells,fanout,tick at repeats 1."""
    args = _args(out=str(tmp_path / "smoke"))
    assert args.levels == "primitive"
    assert args.repeats == 1
    assert args.primitive_blocks == "cells,fanout,tick"
    assert args.copies == "1,8"
    # The assertions above pin down what a bare invocation selects; the run itself is kept to
    # the cells block (the tick kernel is 13.7k neurons: minutes per token on a CPU, and the
    # blocks are exercised for real on the cluster).
    args = _args(primitive_blocks="cells", tokens=3, copies="1", out=str(tmp_path / "smoke"))
    report = run_campaign(args)
    assert {x["workload"] for x in report["summaries"]} == {
        "cells-add", "cells-and", "cells-xor", "cells-mov"}


def test_small_level_on_cpu_is_refused_without_allow_cpu_renders(tmp_path):
    args = _args(levels="small", out=str(tmp_path / "small"))
    with pytest.raises(ValueError, match="allow-cpu-renders"):
        run_campaign(args)


def test_historical_level_on_cpu_is_refused_without_allow_cpu_renders(tmp_path):
    args = _args(levels="historical", historical=True, out=str(tmp_path / "hist"))
    with pytest.raises(ValueError, match="allow-cpu-renders"):
        run_campaign(args)


def test_primitive_max_s_caps_neural_time_and_records_it(tmp_path):
    args = _args(primitive_blocks="cells", tokens=3, primitive_max_s=0.001, out=str(tmp_path / "capped"))
    report = run_campaign(args)
    assert all(r["max_neural_s"] == 0.001 for r in report["records"])


def test_report_is_written_incrementally_and_marked_incomplete_until_done(tmp_path):
    out = tmp_path / "incremental"
    args = _args(primitive_blocks="cells", tokens=3, out=str(out))

    seen_incomplete = []
    from drosophilos.bench import perf_campaign as pc
    original_write = pc.write_report

    def spying_write(out_path, report):
        seen_incomplete.append(dict(report))
        original_write(out_path, report)

    pc.write_report = spying_write
    try:
        report = pc.run_campaign(args)
    finally:
        pc.write_report = original_write

    assert report["complete"] is True
    assert any(not r["complete"] for r in seen_incomplete[:-1])
    assert seen_incomplete[-1]["complete"] is True
    partial = json.loads((out.with_suffix(".json")).read_text())
    assert "complete" in partial
    for r in seen_incomplete[:-1]:
        assert r["configurations_done"] < r["configurations_planned"]
