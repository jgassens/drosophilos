"""Fast contract tests for the fixed §3 comparison driver."""

import argparse
import json
import os

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
               historical=False, allow_cpu_renders=False, dry_run=False, no_spike_count=False, out="data/perf/campaign")
    base.update(overrides)
    return argparse.Namespace(**base)


def _stub_primitive_run(name, cfg, *, tokens, profile_steps=0, max_neural_s=150, count_spikes=True):
    """Return the complete record shape produced by the real primitive runner."""
    return {
        "level": "primitive", "workload": name, "config": dict(cfg), "tokens": tokens,
        "reference": [], "outputs": [], "first_neural_s": 0.001, "first_wall_s": 0.002,
        "interval_neural_s": [], "interval_wall_s": [], "neural_s": 0.003, "wall_s": 0.004,
        "wall_neural": 4 / 3, "neurons": 1, "edges": 0, "spikes": 0,
        "spike_count_method": "stub", "spike_count_cost": "stub", "wrong": 0,
        "missing": 0, "duplicates": 0, "invalid": 0, "faults": 0, "timeouts": 0,
        "host_stalls": 0, "truncated": False, "max_neural_s": max_neural_s,
        "t_load": 0.0, "load_events": [], "profile": {}, "profiled": bool(profile_steps),
    }


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


@pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1", reason="real CPU kernel test; set RUN_SLOW=1")
def test_primitive_cells_write_correct_cpu_report(tmp_path):
    """Run one real CPU kernel only when RUN_SLOW=1; the contract suite uses stubs."""
    args = _args(primitive_blocks="cells", tokens=2, copies="1", out=str(tmp_path / "cells"))
    report = run_campaign(args)
    assert {x["workload"] for x in report["summaries"]} == {
        "cells-add", "cells-and", "cells-xor", "cells-mov"}
    assert all(x["wrong"] == x["missing"] == 0 for x in report["summaries"])
    assert (tmp_path / "cells.json").exists()
    assert (tmp_path / "cells.md").exists()


def test_bare_defaults_run_the_primitive_smoke_set(tmp_path, monkeypatch):
    """A bare invocation is a short smoke: primitive cells,fanout,tick at repeats 1."""
    args = _args(out=str(tmp_path / "smoke"))
    assert args.levels == "primitive"
    assert args.repeats == 1
    assert args.primitive_blocks == "cells,fanout,tick"
    assert args.copies == "1,8"
    # The assertions above pin down what a bare invocation selects; the run itself is kept to
    # the cells block (the tick kernel is 13.7k neurons: minutes per token on a CPU, and the
    # blocks are exercised for real on the cluster).
    monkeypatch.setattr("drosophilos.bench.perf_campaign._primitive_run", _stub_primitive_run)
    args = _args(primitive_blocks="cells", tokens=2, copies="1", out=str(tmp_path / "smoke"))
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


def test_primitive_max_s_caps_neural_time_and_records_it(tmp_path, monkeypatch):
    monkeypatch.setattr("drosophilos.bench.perf_campaign._primitive_run", _stub_primitive_run)
    args = _args(primitive_blocks="cells", tokens=2, primitive_max_s=0.001, out=str(tmp_path / "capped"))
    report = run_campaign(args)
    assert all(r["max_neural_s"] == 0.001 for r in report["records"])


def test_report_is_written_incrementally_and_marked_incomplete_until_done(tmp_path, monkeypatch):
    out = tmp_path / "incremental"
    args = _args(primitive_blocks="cells", tokens=3, out=str(out))
    monkeypatch.setattr("drosophilos.bench.perf_campaign._primitive_run", _stub_primitive_run)

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
    assert all(not r["complete"] for r in seen_incomplete[:-1])
    assert seen_incomplete[0]["configurations_done"] == 1
    done_counts = [r["configurations_done"] for r in seen_incomplete]
    assert done_counts == sorted(done_counts)
    assert done_counts[-1] == seen_incomplete[-1]["configurations_planned"]
    assert seen_incomplete[-1]["complete"] is True


@pytest.mark.slow  # 343 s on the CI runner
def test_no_spike_count_skips_the_all_neuron_capture(tmp_path, monkeypatch):
    """--no-spike-count times the step with the runner's own watched set only (docs/track_a.md:
    the all-neuron capture was most of a FastSim step for the larger blocks)."""
    from drosophilos.bench import perf_campaign as pc
    seen = {}
    real = pc.run_pipeline_batched

    def spy(*a, **kw):
        seen["capture"] = kw.get("capture_spikes")
        return real(*a, **kw)
    monkeypatch.setattr(pc, "run_pipeline_batched", spy)
    args = _args(primitive_blocks="cells", tokens=2, copies="1", no_spike_count=True, out=str(tmp_path / "nsc"))
    report = pc.run_campaign(args)
    assert seen["capture"] is None
    rec = [r for r in report["records"] if r["workload"] == "cells-add"][0]
    assert rec["spikes"] is None and "not captured" in rec["spike_count_method"]


def test_render_cap_covers_the_sizing_estimate_several_times_over():
    """Juno 412442: the doom4 8 x 5 x 3-frame render needed ~4,200 s of neural time and was cut
    at render_doom's 3,600 s default; the cap must scale with the estimate."""
    from drosophilos.bench import perf_campaign as pc
    spec = pc.SMALL_WORKLOADS["doom4-small"]
    est = pc._render_estimate_s(spec, 1)
    assert est > 3600
    assert pc._render_max_ms(spec, 1) >= 4 * est * 1000
    assert pc._render_max_ms(pc.SMALL_WORKLOADS["doom2-small"], 8) == 3_600_000.0  # small jobs keep the default hour
