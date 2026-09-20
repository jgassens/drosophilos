"""Track B fixed-operation cell specialization contracts."""

import os

import numpy as np
import pytest

from drosophilos.compiler.kernel import KernelSpec, kernel_outputs
from drosophilos.lib.kernel import build_pipeline, run_pipeline
from drosophilos.protocol.token import decode_recent
from drosophilos.sim.model import Params


PARAMS = Params()
OPS = ("AND", "OR", "XOR", "MOV")


def _run(spec: KernelSpec, tokens: list[int], datapath: str, *, gap_ms: float = 0.0,
         full_trace: bool = False):
    pl = build_pipeline(PARAMS, spec.width, spec.cells, consts=spec.consts, mems=spec.mems,
                        outputs=spec.outputs, datapath=datapath)
    _first, sim, stats = run_pipeline(
        pl, PARAMS, tokens, max_ms=4000 + 1600 * len(tokens), gap_ms=gap_ms,
        expect_outputs=len(tokens) * len(spec.outputs), full_trace=full_trace)
    by_cell = stats["outputs_by_cell"]
    assert all(len(by_cell[name]) == len(tokens) for name in spec.outputs), stats
    rows = [[by_cell[name][index][1] for name in spec.outputs]
            for index in range(len(tokens))]
    return rows, pl, sim, stats


def _compare(spec: KernelSpec, tokens: list[int], *, gap_ms: float = 0.0,
             full_trace: bool = False):
    reference = kernel_outputs(spec, tokens)
    generic, generic_pl, _generic_sim, generic_stats = _run(
        spec, tokens, "generic", gap_ms=gap_ms, full_trace=full_trace)
    specialized, specialized_pl, specialized_sim, specialized_stats = _run(
        spec, tokens, "specialized", gap_ms=gap_ms, full_trace=full_trace)
    assert generic == specialized == reference
    for stats in (generic_stats, specialized_stats):
        assert stats["faults"] == stats["timeouts"] == stats["bad_outputs"] == 0
        assert not stats["host_stalls"]
    return specialized_pl, specialized_sim, specialized_stats


def _grid_spec(op: str, width: int, b_values: list[int]) -> KernelSpec:
    cells, consts, outputs = [], {}, []
    for index, b in enumerate(b_values):
        name, key = f"out{index}", f"b{index}"
        cells.append({"name": name, "op": op, "a": "input", "b": ("const", key)})
        consts[key] = b
        outputs.append(name)
    spec = KernelSpec(cells, consts, {}, "input", width)
    spec.outputs = outputs
    return spec


@pytest.mark.parametrize("b", [0, 5, 10, 15])
@pytest.mark.parametrize("op", OPS)
def test_width4_stratified_64_pairs_match_generic_and_oracle(op, b):
    # Four columns across all 16 A values: zero/one-heavy, alternating and all-one B rails.
    spec = _grid_spec(op, 4, [b])
    _compare(spec, list(range(16)))


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1", reason="set RUN_SLOW=1 for exhaustive neural run")
@pytest.mark.parametrize("b", range(16))
@pytest.mark.parametrize("op", OPS)
def test_width4_exhaustive_256_pairs_match_generic_and_oracle(op, b):
    spec = _grid_spec(op, 4, [b])
    _compare(spec, list(range(16)))


@pytest.mark.parametrize("b", [0, 1, 0x7F, 0x80, 0xFE, 0xFF])
@pytest.mark.parametrize("op", OPS)
def test_width8_boundary_pairs_match_generic_and_oracle(op, b):
    values = [0, 1, 0x7F, 0x80, 0xFE, 0xFF]
    _compare(_grid_spec(op, 8, [b]), values)


@pytest.mark.parametrize("b_index", range(4))
@pytest.mark.parametrize("op", OPS)
def test_width8_32_seeded_random_pairs_match_generic_and_oracle(op, b_index):
    rng = np.random.default_rng(0xB2)
    a_values = rng.choice(256, size=8, replace=False).tolist()
    b_values = rng.choice(256, size=4, replace=False).tolist()
    # The four parameter cases form exactly 32 seeded pairs without one oversized RefSim net.
    _compare(_grid_spec(op, 8, [b_values[b_index]]), a_values)


@pytest.mark.parametrize("width", [4, 8])
@pytest.mark.parametrize("op", OPS)
def test_repeated_alternating_and_restart_streams_match_generic_and_oracle(op, width):
    mask = (1 << width) - 1
    repeated = [mask // 3] * 4
    alternating = [1, mask - 1] * 4
    spec = _grid_spec(op, width, [mask ^ (mask // 3)])
    _compare(spec, repeated + alternating + [0, mask, 0])


@pytest.mark.parametrize("op", OPS)
def test_specialized_flags_are_valid_zero_tokens_on_every_transaction(op):
    width = 4
    tokens = [5] * 4 + [3, 12] * 3
    spec = _grid_spec(op, width, [9])
    pl, sim, stats = _compare(spec, tokens, full_trace=True)
    cell = pl.cells[0]
    window = 2 * pl.drive.loop_period_steps
    for step, _value in stats["outputs_by_cell"][cell.name]:
        for bit in (width, width + 2):  # C and V; Z remains the stage consumer's flag.
            value, status = decode_recent(sim, [cell.master.rail_taps[bit]], step + window, window)
            assert (value, status) == (0, "valid")


@pytest.mark.parametrize("width", [4, 8])
def test_specialized_rom_and_ram_loads_match_generic_and_oracle(width):
    mask = (1 << width) - 1
    contents = {address: (address * 13 + 3) & mask for address in range(16)}
    tokens = [0, 1, 15, 7, 7, 2, 14, 0]
    for memory in ((16, contents), (16, contents, "ram")):
        spec = KernelSpec([{"name": "out", "op": "LOAD", "a": "input", "mem": "table"}],
                          {}, {"table": memory}, "input", width)
        spec.outputs = ["out"]
        pl, _sim, _stats = _compare(spec, tokens)
        assert pl.cells[0].datapath == "specialized"


def test_stalled_two_cell_consumer_has_no_wrong_or_duplicate_output():
    spec = KernelSpec([
        {"name": "first", "op": "XOR", "a": "input", "b": ("const", "k3")},
        {"name": "out", "op": "AND", "a": "first", "b": ("const", "k7"),
         "trigger": ["input:gate"]},
    ], {"k3": 3, "k7": 7}, {}, "input", 4)
    spec.outputs = ["out"]
    values = [0, 5, 5, 15, 1]
    reference = kernel_outputs(spec, values)
    schedule = [item for value in values for item in (("input", value), ("gate", 0))]
    results = []
    for datapath in ("generic", "specialized"):
        pl = build_pipeline(PARAMS, 4, spec.cells, consts=spec.consts, outputs=spec.outputs,
                            streams=["input", "gate"], datapath=datapath)
        _first, _sim, stats = run_pipeline(
            pl, PARAMS, schedule, max_ms=16000, gap_ms=350, expect_outputs=len(values))
        got = [[value] for _step, value in stats["outputs_by_cell"]["out"]]
        assert got == reference
        assert stats["faults"] == stats["timeouts"] == stats["bad_outputs"] == 0
        assert len(got) == len(values)
        results.append(got)
    assert results[0] == results[1]


def test_specialized_fanout_to_two_consumers_matches_generic_and_oracle():
    spec = KernelSpec([
        {"name": "source", "op": "XOR", "a": "input", "b": ("const", "k3")},
        {"name": "left", "op": "AND", "a": "source", "b": ("const", "k7")},
        {"name": "right", "op": "OR", "a": "source", "b": ("const", "k8")},
    ], {"k3": 3, "k7": 7, "k8": 8}, {}, "input", 4)
    spec.outputs = ["left", "right"]
    _compare(spec, [0, 5, 9, 14, 14, 3])


@pytest.mark.parametrize("op", OPS)
def test_specialized_cell_contains_only_its_fixed_unit(op):
    spec = _grid_spec(op, 4, [3])
    pl = build_pipeline(PARAMS, 4, spec.cells, consts=spec.consts, outputs=spec.outputs,
                        datapath="specialized")
    roles = pl.net.roles
    assert pl.datapath == pl.cells[0].datapath == "specialized"
    assert not any(role.startswith("out0.u") for role in roles)
    assert not any(role.startswith("out0.mux") for role in roles)
    assert not any(role.startswith(tuple(f"out0.fa{i}." for i in range(4))) or
                   role.startswith("out0.vx") for role in roles)


def test_store_stays_generic_until_write_completion_is_acknowledged():
    memory = (4, {0: 0}, "ram")
    spec = [{"name": "store", "op": "STORE", "a": ("const", "address"),
             "b": "input", "mem": "ram"}]
    pl = build_pipeline(PARAMS, 4, spec, consts={"address": 0}, mems={"ram": memory},
                        datapath="specialized")
    assert pl.datapath == "specialized"
    assert pl.cells[0].datapath == "generic"
    assert any(role.startswith("store.u0r") for role in pl.net.roles)


def test_invalid_datapath_is_rejected():
    with pytest.raises(ValueError, match="datapath"):
        build_pipeline(PARAMS, 4, [{"name": "out", "op": "MOV", "a": "input",
                                    "b": ("const", "zero")}], consts={"zero": 0},
                       datapath="unknown")
