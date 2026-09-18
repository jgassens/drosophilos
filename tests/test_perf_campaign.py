"""Fast contract tests for the fixed §3 comparison driver."""

import argparse

import pytest

from drosophilos.bench.perf_campaign import (
    Variant,
    coverage_check,
    run_campaign,
)
from drosophilos.bench.render_doom import RenderConfig


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
    args = argparse.Namespace(levels="primitive,small", backend="torch", device="cpu", dtype="float64",
                              mul="array", copies="1,8", pacing="host", repeats=3, tokens=16,
                              primitive_blocks="cells", profile_steps=0, historical=False, dry_run=True,
                              out=str(tmp_path / "report"))
    result = run_campaign(args)
    assert result["dry_run"]
    assert [x["copies"] for x in result["configurations"]] == [1, 8]


def test_primitive_cells_write_correct_cpu_report(tmp_path):
    args = argparse.Namespace(levels="primitive", backend="torch", device="cpu", dtype="float64",
                              mul="array", copies="1", pacing="host", repeats=1, tokens=3,
                              primitive_blocks="cells", profile_steps=0, historical=False, dry_run=False,
                              out=str(tmp_path / "cells"))
    report = run_campaign(args)
    assert {x["workload"] for x in report["summaries"]} == {
        "cells-add", "cells-and", "cells-xor", "cells-mov"}
    assert all(x["wrong"] == x["missing"] == 0 for x in report["summaries"])
    assert (tmp_path / "cells.json").exists()
    assert (tmp_path / "cells.md").exists()
