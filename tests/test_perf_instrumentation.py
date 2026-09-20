"""Regression tests for the §2 benchmark instrumentation."""

import numpy as np
import pytest

from drosophilos.bench.render_doom import RenderConfig, _effective_backend
from drosophilos.bench.repro import circuit_hash
from drosophilos.lib.kernel import build_pipeline, run_pipeline_batched
from drosophilos.sim.model import Params
from drosophilos.sim.profile import Profiler


PARAMS = Params()


@pytest.mark.parametrize(
    ("cfg", "reason"),
    [
        (RenderConfig(backend="ref", device="cuda"), "only --device cpu"),
        (RenderConfig(backend="ref", dtype="float32"), "only --dtype float64"),
        (RenderConfig(backend="torch", device="mps", dtype="float64"), "does not support"),
    ],
)
def test_backend_rejections_have_a_reason(cfg, reason):
    with pytest.raises(SystemExit, match=reason):
        _effective_backend(cfg)


def test_unavailable_cuda_is_rejected(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit, match="CUDA is not available"):
        _effective_backend(RenderConfig(backend="torch", device="cuda"))


def test_profiled_batched_run_observes_outputs_without_changing_spikes():
    import torch

    spec = [{"name": "out", "op": "MOV", "a": ("const", "zero"), "b": "input"}]  # MOV = PASSB: output is the token
    pl = build_pipeline(PARAMS, 1, spec, consts={"zero": 0})
    schedules = [[1], [0]]

    observed = []
    profiled, profiled_sim, stats = run_pipeline_batched(
        pl, PARAMS, schedules, max_ms=3000, device="cpu", dtype=torch.float64,
        progress=0, profile_steps=200, full_trace=True,
        on_output=lambda node, cell, step, value, wall: observed.append(
            (node, cell, step, value, wall)),
    )
    plain, plain_sim, _ = run_pipeline_batched(
        pl, PARAMS, schedules, max_ms=3000, device="cpu", dtype=torch.float64,
        progress=0, full_trace=True,
    )
    assert [[v for _, v in node["out"]] for node in plain] == [[1], [0]]  # real values, not zeros
    assert len(plain_sim.trace) > 100  # the whole trace, not the runner's retained tail

    assert len(observed) == sum(len(values) for node in profiled for values in node.values())
    assert all(b[-1] >= a[-1] for a, b in zip(observed, observed[1:]))
    assert profiled == plain
    assert np.array_equal(profiled_sim.trace.events, plain_sim.trace.events)
    assert stats["profile"]["profiled_steps"] == 200
    assert set(stats["profile"]) == {"profiled_steps", "regions"}
    for name in ("integrate", "threshold", "observe", "deliver", "reset",
                 "host_schedule", "decode"):
        region = stats["profile"]["regions"][name]
        assert set(region) == {"calls", "total_s"}
        assert region["calls"] > 0


def test_circuit_hash_is_stable_and_includes_constant_image():
    spec = [{"name": "out", "op": "ADD", "a": "input", "b": ("const", "k")}]
    one = build_pipeline(PARAMS, 4, spec, consts={"k": 1})
    same = build_pipeline(PARAMS, 4, spec, consts={"k": 1})
    changed = build_pipeline(PARAMS, 4, spec, consts={"k": 2})
    assert circuit_hash(one) == circuit_hash(same)
    assert circuit_hash(one) != circuit_hash(changed)


def test_disabled_profiler_has_no_regions():
    profiler = Profiler(False)
    with profiler.region("unused"):
        pass
    assert profiler.regions() == {}


def test_count_duplicates_maps_stream_keys_to_schedule_names():
    """Regression: Juno job 412137 crashed after 2.5 h with KeyError 'input:f' because cells
    carry stream keys and schedules carry stream names (the doom programs have several)."""
    from types import SimpleNamespace
    from drosophilos.bench.render_doom import count_duplicates

    ks = SimpleNamespace(streams=["f", "p"], stream_keys={"f": "input:f", "p": "input:p"},
                         cells=[{"name": "tick", "op": "ADD", "stream": "input:f"},
                                {"name": "pix", "op": "ADD", "stream": "input:p"},
                                {"name": "ring", "op": "RING"}])
    scheds = [[("f", 1, 0), ("p", 5, 0), ("p", 6, 0)]]
    outs = [{"tick": [(1, 1)], "pix": [(2, 5), (3, 6), (4, 6)]}]
    assert count_duplicates(ks, scheds, outs) == 1
