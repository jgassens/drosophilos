"""Build-policy recording and structural netlist regressions."""

from dataclasses import replace
import hashlib
import json
from time import perf_counter

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
from drosophilos.lib.netlist import Drive, Netlist
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
    "rate_robust": False,
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
    # Ordered roles, edges, weights, delays and biases, not only neuron/edge counts.
    # Captured from the pre-rate-conditioning machine; Drive's legacy default is vital.
    assert not machine_default.drive.rate_robust
    raw = json.dumps([machine_default.net.roles, machine_default.net.src,
                      machine_default.net.dst, machine_default.net.quanta,
                      machine_default.net.delay, machine_default.net.bias], separators=(",", ":"))
    assert hashlib.sha256(raw.encode()).hexdigest() == (
        "7175c3fcd639651dc0eee0c99a132bdb4c6f292ba0c058d5e8ae20952cb504f4")

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
        "rate_robust": False,
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


def test_conditioned_rates_are_recorded_rebuilt_and_shared():
    ks, legacy, _, _ = kernel_campaign.block("tick", PARAMS)
    pl = build_pipeline(PARAMS, ks.width, ks.cells, consts=ks.consts, mems=ks.mems,
                        outputs=ks.outputs, rate_robust=True)
    assert pl.build_options["rate_robust"] is pl.drive.rate_robust is True
    assert (pl.net.n, pl.net.nnz) == (30643, 55936)
    assert pl.net.n - legacy.net.n == len(pl.net.rate_readouts) == 1268
    assert pl.net.nnz - legacy.net.nnz == 1268 + 1318 + 542  # drives, readout clears, qualifier clears
    _, _, rebuilt = build_tick_pipeline({"block": "tick", **pl.build_options})
    assert _pipeline_fingerprint(pl) == _pipeline_fingerprint(rebuilt)
    _assert_conditioned_train_inputs(pl.net, pl.drive)
    # Include kills wired before and after readout creation (cached passed pairs too).
    for source, tap in pl.net.rate_readouts.items():
        expected = sorted((pl.net.src[k], 16 * pl.net.quanta[k], pl.net.delay[k])
                          for k in pl.net.incoming[source] if pl.net.quanta[k] < 0)
        actual = sorted((pl.net.src[k], pl.net.quanta[k], pl.net.delay[k])
                        for k in pl.net.incoming[tap] if pl.net.quanta[k] < 0)
        assert actual == expected
    # Removing the recorded option restores the exact pre-change topology, even if
    # all the other, newer build flags are present in the capture.
    recorded = {"block": "tick", **legacy.build_options}
    recorded.pop("rate_robust")
    _, _, rebuilt_legacy = build_tick_pipeline(recorded)
    assert _pipeline_fingerprint(legacy) == _pipeline_fingerprint(rebuilt_legacy)


def _assert_conditioned_train_inputs(net, drive):
    # Undo compiled gain before identifying weak inputs. This includes edges added
    # AFTER conditioning, and makes no assumptions about source role names (.u/.p/
    # delay taps). All positive inputs in the weak-drive range are train inputs.
    readouts = set(net.rate_readouts.values())
    for source, target, q in zip(net.src, net.dst, net.quanta):
        if 0 < q <= round(drive.or_in * net.input_scales.get(target, 1)):
            assert source in readouts, (
                f"unconditioned rate input: {net.roles[source]} -> {net.roles[target]}")


@pytest.mark.parametrize("role", ["latch.u", "flipflop.p", "delay.d0"])
def test_train_audit_detects_late_raw_inputs_independent_of_source_name(role):
    from drosophilos.protocol.celement import add_or_latched

    drive = replace(Drive.from_params(PARAMS), rate_robust=True)
    net = Netlist(PARAMS)
    source = net.neuron(role)
    gate, _ = add_or_latched(net, drive, "valid", [source])
    _assert_conditioned_train_inputs(net, drive)
    net.synapse(source, gate, drive.or_in)
    assert net.quanta[-1] == 2 * drive.or_in
    with pytest.raises(AssertionError, match="unconditioned rate input"):
        _assert_conditioned_train_inputs(net, drive)


# Captured from the merged builder before the review fixes: ordered bytes, including
# roles, weights, delays and biases. The image tests above additionally check loading.
DEFAULT_HASHES = {
    "tick": "ebb0657dd0c7fddce934599dffb7d989cca43a828f605de9fbba41a7baf1842d",
    "render": "936a860cbfcc091147837a986c71be185952306004fb7d8065cc324851526738",
    "perspective": "890c0b68af7eae7a7004001838b7e24b0727d03304959c207bbf2b29bb6fef31",
    "tick-specialized": "4fcdd692b2e41f182296722818d5f189b2ab31e29001bcea6a550c96fce7f3c2",
    "tick-legacy": "b86925aae14aa67dd1170f2d265a0129aea2ddf3074ef46de2db41a2fe3409e5",
    "tick-retry": "b8ec3201e26df31802324d2518df0be0e9b054e26370282721328804d9ff8fff",
    "mov": "fc6411ad1d51247db14ff0406dde0058d88d9a3865d5d0d206fae246b9d4420f",
    "mulp": "2a65ea10a80441a8393d02a886fe11cbabddce076979b9b150eba3a3af9ae1cc",
    "machine2": "7175c3fcd639651dc0eee0c99a132bdb4c6f292ba0c058d5e8ae20952cb504f4",
    "machine4": "daed6ba1056cc57a8c9a2d5a8585d4bb81c792bbabcd798a735a63a8ee8b9671",
}


@pytest.mark.parametrize("name", DEFAULT_HASHES)
def test_default_ordered_netlists_are_byte_identical(name):
    if name.startswith("machine"):
        net = build_machine(PARAMS, n=int(name[-1]), n_prog=4, n_data=2).net
    elif name in ("mov", "mulp"):
        net = build_pipeline(
            PARAMS, 1 if name == "mov" else 4,
            [{"name": "out", "op": name.upper(), "a": "input", "b": ("const", "k")}],
            consts={"k": 0 if name == "mov" else 3}).net
    else:
        ks, pl, _, _ = kernel_campaign.block(
            name.split("-")[0], PARAMS,
            datapath="specialized" if name.endswith("specialized") else "generic")
        if name.endswith(("legacy", "retry")):
            pl = build_pipeline(
                PARAMS, ks.width, ks.cells, consts=ks.consts, mems=ks.mems,
                outputs=ks.outputs, **(LEGACY_2026_09_20 if name.endswith("legacy")
                                        else {"retry_clear": True}))
        net = pl.net
    raw = json.dumps([net.roles, net.src, net.dst, net.quanta, net.delay, net.bias],
                     separators=(",", ":"))
    assert hashlib.sha256(raw.encode()).hexdigest() == DEFAULT_HASHES[name]


def build_cost_survey():
    """Print reproducible builder/circuit costs; wall time is not a CI assertion."""
    results = []
    for name in ("mov", "mov-specialized", "mulp", "tick", "tick-specialized", "tick-retry", "perspective"):
        for robust in (False, True):
            start = perf_counter()
            if name.startswith("mov") or name == "mulp":
                pl = build_pipeline(
                    PARAMS, 4 if name == "mulp" else 1,
                    [{"name": "out", "op": "MULP" if name == "mulp" else "MOV",
                      "a": "input", "b": ("const", "k")}],
                    consts={"k": 3 if name == "mulp" else 0}, rate_robust=robust,
                    datapath="specialized" if name.endswith("specialized") else "generic")
            else:
                ks, pl, _, _ = kernel_campaign.block(
                    name.split("-")[0], PARAMS, rate_robust=robust,
                    datapath="specialized" if name.endswith("specialized") else "generic")
                if name.endswith("retry"):
                    pl = build_pipeline(PARAMS, ks.width, ks.cells, consts=ks.consts,
                                        mems=ks.mems, outputs=ks.outputs,
                                        rate_robust=robust, retry_clear=True)
                if robust:
                    _assert_conditioned_train_inputs(pl.net, pl.drive)
            results.append({"build": name, "robust": robust, "neurons": pl.net.n,
                            "edges": pl.net.nnz, "readouts": len(pl.net.rate_readouts),
                            "largest_edge": max(map(abs, pl.net.quanta)),
                            "anatomical_k4": sum((abs(q) + 63) // 64 for q in pl.net.quanta),
                            "build_seconds": round(perf_counter() - start, 3)})
    return results


@pytest.mark.slow
def test_build_cost_survey():
    result = build_cost_survey()
    for row in result:
        print(row)
    tick = next(r for r in result if r["build"] == "tick" and r["robust"])
    assert (tick["neurons"], tick["edges"], tick["largest_edge"]) == (30643, 55936, 43456)
    retry = next(r for r in result if r["build"] == "tick-retry" and r["robust"])
    assert retry["largest_edge"] == 86912
