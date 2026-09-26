"""Build-policy recording and structural netlist regressions."""

from dataclasses import replace

import pytest

from drosophilos.bench import kernel_campaign
from drosophilos.bench.stall_diag import build_tick_pipeline
from drosophilos.lib.control import build_machine, load_image
from drosophilos.lib.kernel import (
    LEGACY_2026_09_20,
    TRUE_GUARD_VERSION,
    build_pipeline,
    load_pipeline_image,
)
from drosophilos.lib.netlist import Drive
from drosophilos.sim.model import Params


PARAMS = Params()
CURRENT_PIPELINE_OPTIONS = {
    "relight_requests": True,
    "datapath": "generic",
    "act_hops": 11,
    "idle_hops": 20,
    "watchdog_hops": 170,
    "in_watchdog_hops": None,
    "powerup_veto": True,
    "commit_reignite": True,
    "retry_clear": False,
    "start_relight_hops": 5,
    "relight_repair_delay": True,
    "copy_requires_rail": True,
    "request_clear_pulses": 4,
    "kernel_kill_pulses": 4,
    "true_guards": True,
}


class _ImageRecorder:
    def __init__(self):
        self.neurons = []

    def add_events(self, node, steps, neurons, quanta):
        self.neurons.extend(int(i) for i in neurons)


def _fingerprint(net, image_neurons):
    """Full role-labelled structure, independent of neuron and synapse insertion indices."""
    edges = tuple(sorted(
        (net.roles[src], net.roles[dst], int(quanta), int(delay))
        for src, dst, quanta, delay in zip(net.src, net.dst, net.quanta, net.delay)
    ))
    biases = tuple((role, float(bias)) for role, bias in zip(net.roles, net.bias))
    image_roles = tuple(sorted(net.roles[i] for i in image_neurons))
    return edges, biases, image_roles


def _pipeline_fingerprint(pl):
    image = _ImageRecorder()
    load_pipeline_image(image, pl)
    return _fingerprint(pl.net, image.neurons)


def test_current_default_netlists_match_fixed_policy_builds():
    fixed_drive = replace(Drive.from_params(PARAMS), kill_pulses=3, kill_strength=0.75)

    ks, tick_default, _, _ = kernel_campaign.block("tick", PARAMS)
    tick_fixed = build_pipeline(
        PARAMS, ks.width, ks.cells, consts=ks.consts, mems=ks.mems, outputs=ks.outputs,
        drive=fixed_drive, **CURRENT_PIPELINE_OPTIONS,
    )
    campaign = {
        "block": "tick",
        "kill_train": [tick_fixed.build_options["kernel_kill_pulses"],
                       tick_fixed.build_options["kill_strength"]],
        **tick_fixed.build_options,
    }
    _, _, tick_rebuilt = build_tick_pipeline(campaign)
    assert (tick_default.net.n, tick_default.net.nnz) == (29375, 52808)
    assert _pipeline_fingerprint(tick_default) == _pipeline_fingerprint(tick_fixed)
    assert _pipeline_fingerprint(tick_default) == _pipeline_fingerprint(tick_rebuilt)

    machine_default = build_machine(PARAMS, n=2, n_prog=4, n_data=2)
    machine_fixed = build_machine(PARAMS, n=2, n_prog=4, n_data=2, drive=fixed_drive)
    program = [("MOV", 1), ("ADD", 2), ("STORE", 0), ("HALT", 3)]
    machine_fingerprints = []
    for machine in (machine_default, machine_fixed):
        image = _ImageRecorder()
        load_image(image, machine, program, {1: 2})
        machine_fingerprints.append(_fingerprint(machine.net, image.neurons))
    assert (machine_default.net.n, machine_default.net.nnz) == (2449, 4513)
    assert machine_fingerprints[0] == machine_fingerprints[1]

    mov_spec = [{"name": "out", "op": "MOV", "a": "input", "b": ("const", "zero")}]
    mov_default = build_pipeline(PARAMS, 1, mov_spec, consts={"zero": 0})
    mov_fixed = build_pipeline(PARAMS, 1, mov_spec, consts={"zero": 0},
                               drive=fixed_drive, **CURRENT_PIPELINE_OPTIONS)
    assert (mov_default.net.n, mov_default.net.nnz) == (1029, 1752)
    assert _pipeline_fingerprint(mov_default) == _pipeline_fingerprint(mov_fixed)

    mulp_spec = [{"name": "m", "op": "MULP", "a": "input", "b": ("const", "k")}]
    mulp_default = build_pipeline(PARAMS, 4, mulp_spec, consts={"k": 3})
    mulp_fixed = build_pipeline(PARAMS, 4, mulp_spec, consts={"k": 3},
                                drive=fixed_drive, **CURRENT_PIPELINE_OPTIONS)
    assert (mulp_default.net.n, mulp_default.net.nnz) == (7414, 12906)
    assert _pipeline_fingerprint(mulp_default) == _pipeline_fingerprint(mulp_fixed)


def test_build_options_record_every_netlist_shaping_option():
    drive = replace(Drive.from_params(PARAMS), kill_strength=0.9)
    options = {
        "relight_requests": True,
        "datapath": "specialized",
        "act_hops": 9,
        "idle_hops": 18,
        "watchdog_hops": 160,
        "in_watchdog_hops": 70,
        "powerup_veto": False,
        "commit_reignite": False,
        "retry_clear": True,
        "start_relight_hops": 3,
        "relight_repair_delay": False,
        "copy_requires_rail": False,
        "request_clear_pulses": 5,
        "kernel_kill_pulses": 6,
        "true_guards": False,
    }
    pl = build_pipeline(
        PARAMS, 1,
        [{"name": "out", "op": "MOV", "a": "input", "b": ("const", "zero")}],
        consts={"zero": 0}, drive=drive, **options,
    )
    assert pl.build_options == {
        **options,
        "kill_strength": 0.9,
        "true_guard_version": TRUE_GUARD_VERSION,
    }
    assert (pl.drive.kill_pulses, pl.drive.kill_strength) == (6, 0.9)


def test_stall_diag_keeps_legacy_fallbacks_for_old_campaign_records():
    _, _, pl = build_tick_pipeline({"block": "tick", "kill_train": [3, 0.75]})
    # copy_requires_rail=False must retain the pre-fix legacy topology exactly.
    assert (pl.net.n, pl.net.nnz) == (28439, 51071)
    assert pl.build_options == {
        "relight_requests": True,
        "datapath": "generic",
        "act_hops": 11,
        "idle_hops": 20,
        "watchdog_hops": 170,
        "in_watchdog_hops": None,
        "powerup_veto": False,
        "commit_reignite": False,
        "retry_clear": False,
        **LEGACY_2026_09_20,
        "kill_strength": 0.75,
        "true_guard_version": TRUE_GUARD_VERSION,
    }


def test_true_guards_require_request_relight_repair():
    with pytest.raises(ValueError, match="relight_requests=False requires true_guards=False"):
        build_pipeline(
            PARAMS, 1,
            [{"name": "out", "op": "MOV", "a": "input", "b": ("const", "zero")}],
            consts={"zero": 0}, relight_requests=False, true_guards=True,
        )
