"""Build-policy recording and structural netlist regressions."""

from dataclasses import replace
import hashlib
import json
from time import perf_counter

import pytest

from drosophilos.bench import kernel_campaign
from drosophilos.bench.stall_diag import build_tick_pipeline
from drosophilos.compiler.frontend_c import compile_c
from drosophilos.compiler.kernel import compile_program
from drosophilos.lib.control import build_machine, load_image
from drosophilos.lib.kernel import (
    LEGACY_2026_09_20,
    RATE_ROBUST_VERSION,
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
    "zero_once": False,
    "robust_request_clear": False,
    "robust_register_reset": False,
    "verified_register_reset": False,
    "experimental_register_reset": False,
    "ready_interlock": False,
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
        "zero_once": True,
        "robust_request_clear": False,
        "robust_register_reset": False,
        "verified_register_reset": False,
        "experimental_register_reset": False,
        "ready_interlock": False,
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
        "zero_once": False,
        "robust_request_clear": False,
        "robust_register_reset": False,
        "verified_register_reset": False,
        "experimental_register_reset": False,
        "ready_interlock": False,
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


@pytest.mark.parametrize("rate_robust", [False, True])
def test_robust_request_clear_changes_only_request_k1_edges(rate_robust):
    _, old, _, _ = kernel_campaign.block("tick", PARAMS, rate_robust=rate_robust)
    _, new, _, _ = kernel_campaign.block("tick", PARAMS, rate_robust=rate_robust,
                                       robust_request_clear=True)
    assert (old.net.n, old.net.nnz) == (new.net.n, new.net.nnz)
    assert (old.net.roles, old.net.src, old.net.dst, old.net.bias) == (
        new.net.roles, new.net.src, new.net.dst, new.net.bias)
    changed_delays = changed_weights = 0
    for src, dst, q0, q1, d0, d1 in zip(old.net.src, old.net.dst, old.net.quanta,
                                       new.net.quanta, old.net.delay, new.net.delay):
        source, target = old.net.roles[src], old.net.roles[dst]
        if q0 != q1:
            assert ".req." in source and source.endswith(".k1.inh"), (source, target)
            assert q0 < 0 and q1 < q0
            changed_weights += 1
        if d0 != d1:
            assert ".req." in target and target.endswith((".k1.h1", ".k1.h2", ".k1.h3"))
            assert d0 == PARAMS.default_delay_steps and d1 == 0
            changed_delays += 1
    assert changed_delays == 25 * 3
    assert changed_weights >= 25 * 2  # readouts/require neurons mirror the new inhibition too


@pytest.mark.parametrize("options", [
    {"request_clear_pulses": 5}, {"retry_clear": True},
    {"relight_requests": False, "true_guards": False},
    {"drive": replace(Drive.from_params(PARAMS), kill_strength=1.0)},
])
def test_robust_request_clear_rejects_unqualified_policy_combinations(options):
    with pytest.raises(ValueError, match="robust_request_clear requires"):
        build_pipeline(PARAMS, 1,
                       [{"name": "out", "op": "MOV", "a": "input", "b": ("const", "zero")}],
                       consts={"zero": 0}, robust_request_clear=True, **options)


@pytest.mark.parametrize("rate_robust", [False, True])
@pytest.mark.parametrize("zero_once", [False, True])
def test_register_reset_covers_the_complete_domain_and_changes_nothing_else(rate_robust, zero_once):
    _, old, _, _ = kernel_campaign.block("tick", PARAMS, rate_robust=rate_robust, zero_once=zero_once)
    _, new, _, _ = kernel_campaign.block("tick", PARAMS, rate_robust=rate_robust,
                                       zero_once=zero_once, experimental_register_reset=True)
    assert (old.net.n, old.net.nnz) == (new.net.n, new.net.nnz)
    assert (old.net.roles, old.net.src, old.net.dst, old.net.bias) == (
        new.net.roles, new.net.src, new.net.dst, new.net.bias)
    regs = [r for sr, _ in old.inputs.values() for r in (sr.stage, sr.master)]
    regs += [r for c in old.cells for r in (c.stage, c.master)]
    inhibitors = {r.reset_inh for r in regs}
    scale = round(1.75 * old.drive.loop) / round(.75 * old.drive.loop)
    changed_taps = changed_ready = 0
    for src, dst, q0, q1, d0, d1 in zip(old.net.src, old.net.dst, old.net.quanta,
                                       new.net.quanta, old.net.delay, new.net.delay):
        assert q1 == (round(q0 * scale) if src in inhibitors and q0 < 0 else q0)
        if d0 != d1:
            source, target = old.net.roles[src], old.net.roles[dst]
            if target.endswith((".reset_relay1", ".reset_relay2", ".reset_relay3")):
                assert d1 == 0
                changed_taps += 1
            else:
                assert target.endswith(tuple(f".ready_delay{k}" for k in range(8, 15)) + (".ready",))
                assert d1 == 100
                changed_ready += 1
            assert source.rsplit(".", 1)[0] in {old.net.roles[r.reset_trigger].removesuffix(".reset") for r in regs}
            assert d0 == 18
    assert changed_taps == 3 * len(regs) and changed_ready == 8 * len(regs)
    targets = {new.net.roles[d] for s, d in zip(new.net.src, new.net.dst) if s in inhibitors}
    for suffix in (".act.u", ".faultL.u", ".commit.u", ".grant.L.u", ".copy.u", ".veto0.u"):
        assert any(t.endswith(suffix) for t in targets), suffix
    assert "c0_add.mux0r1.u" in targets  # extend_reset's datapath state
    if zero_once:
        assert "alu.z0.or" in targets and "alu.z0.ign.hold_inh" in targets
    if rate_robust:
        for source, tap in new.net.rate_readouts.items():
            expected = sorted((new.net.src[k], 16 * new.net.quanta[k], new.net.delay[k])
                              for k in new.net.incoming[source] if new.net.quanta[k] < 0)
            actual = sorted((new.net.src[k], new.net.quanta[k], new.net.delay[k])
                            for k in new.net.incoming[tap] if new.net.quanta[k] < 0)
            assert actual == expected


@pytest.mark.parametrize("options", [
    {"act_hops": 10}, {"idle_hops": 19}, {"watchdog_hops": 169}, {"in_watchdog_hops": 70},
    {"start_relight_hops": 4}, {"request_clear_pulses": 5}, {"kernel_kill_pulses": 3},
    {"retry_clear": True}, {"copy_requires_rail": False}, {"true_guards": False},
    {"powerup_veto": False}, {"commit_reignite": False}, {"relight_repair_delay": False},
    {"relight_requests": False, "true_guards": False}, {"streams": ["other"]},
    {"phases": [("input", None, "each")]},
    {"drive": replace(Drive.from_params(PARAMS), kill_strength=1.)},
])
def test_register_reset_rejects_unqualified_policy(options):
    with pytest.raises(ValueError, match="robust_register_reset requires"):
        build_pipeline(PARAMS, 2, [{"name": "out", "op": "ADD", "a": "input", "b": ("const", "k")}],
                       consts={"k": 1}, robust_register_reset=True, **options)


@pytest.mark.parametrize("width,op,params,mems", [
    (16, "ADD", PARAMS, None), (2, "MULP", PARAMS, None),
    (1, "ADD", PARAMS, None), (4, "ADD", PARAMS, None),
    (2, "MOV", PARAMS, None), (2, "OR", PARAMS, None),
    (2, "LOAD", PARAMS, {"rom": (2, {0: 1})}),
    (2, "ADD", replace(PARAMS, default_delay_ms=1.9), None),
    (2, "ADD", PARAMS, {"ram": (2, {0: 1}, "ram")}),
])
def test_register_reset_rejects_unqualified_physics_and_kernel(width, op, params, mems):
    with pytest.raises(ValueError, match="robust_register_reset requires"):
        build_pipeline(params, width, [{"name": "out", "op": op, "a": "input", "b": ("const", "k")}],
                       consts={"k": 1}, mems=mems, robust_register_reset=True)


def test_register_reset_rejects_parameter_sources():
    with pytest.raises(ValueError, match="robust_register_reset requires"):
        build_pipeline(PARAMS, 2, [{"name": "out", "op": "ADD", "a": "input", "b": ("param", "k")}],
                       robust_register_reset=True)


def test_recheck_rejects_unqualified_recorded_register_reset_timing():
    from drosophilos.bench.stage_d import recheck_record

    with pytest.raises(ValueError, match="robust_register_reset requires"):
        recheck_record({"robust_register_reset": True, "build_options": {"idle_hops": 1}})


@pytest.mark.parametrize("entry", ["kernel", "stage_d", "recheck", "diagnostic"])
def test_unqualified_register_reset_cannot_be_enabled(entry):
    from drosophilos.bench import stage_d

    with pytest.raises(ValueError, match="robust_register_reset is not qualified"):
        if entry == "kernel":
            kernel_campaign.block("tick", PARAMS, robust_register_reset=True)
        elif entry == "stage_d":
            stage_d.main(["--robust-register-reset", "--ticks", "1", "--c-ticks", "0"])
        elif entry == "recheck":
            stage_d.recheck_record({"build_options": {"robust_register_reset": True}})
        else:
            _, pl, _, _ = kernel_campaign.block("tick", PARAMS)
            build_tick_pipeline({"block": "tick", **pl.build_options, "robust_register_reset": True})


@pytest.mark.parametrize("verified", [False, True])
@pytest.mark.parametrize("entry", ["kernel", "stage_d", "campaign", "recheck", "diagnostic", "stage_d_diagnostic"])
def test_ready_interlock_qualification_is_enforced_at_every_entry(entry, verified, tmp_path):
    from drosophilos.bench import stage_d

    options = dict(ready_interlock=True, verified_register_reset=verified)
    flag = "verified_register_reset" if verified else "ready_interlock"
    cli = ["--ready-interlock"] + (["--verified-register-reset"] if verified else [])
    with pytest.raises(ValueError, match=flag + " is not qualified"):
        if entry == "kernel":
            kernel_campaign.block("tick", PARAMS, **options)
        elif entry == "stage_d":
            stage_d.main([*cli, "--ticks", "1", "--c-ticks", "0"])
        elif entry == "campaign":
            kernel_campaign.main(["tick", *cli, "--copies", "1"])
        elif entry == "recheck":
            path = tmp_path / "candidate.json"
            path.write_text(json.dumps({"build_options": options}))
            stage_d.main(["--recheck", str(path)])
        else:
            _, pl, _, _ = kernel_campaign.block("tick", PARAMS)
            record = {"block": "tick", **pl.build_options, **options}
            if entry == "stage_d_diagnostic":
                record.update(stage="D", build_options={**pl.build_options, **options})
                record.update(ready_interlock=False, verified_register_reset=False)
            build_tick_pipeline(record)


@pytest.mark.parametrize("option", ["ready_interlock", "verified_register_reset"])
def test_recheck_and_rebuild_preserve_authoritative_false_and_reject_top_level_true(option):
    from drosophilos.bench import stage_d

    k = stage_d.load_kernel(stage_d.PROGRAM)
    pl = stage_d.build(k, PARAMS)
    record = {"stage": "D", "build_options": dict(pl.build_options), option: True}
    _, _, rebuilt = build_tick_pipeline(record)
    assert _pipeline_fingerprint(pl) == _pipeline_fingerprint(rebuilt)
    stage_d.recheck_record(dict(record))
    record["build_options"].pop(option)
    for rebuild in (build_tick_pipeline, stage_d.recheck_record):
        with pytest.raises(ValueError, match=option + " is not qualified"):
            rebuild(dict(record))


@pytest.mark.parametrize("options", [{"idle_hops": 19}, {"copy_requires_rail": False},
                                     {"true_guards": False}, {"retry_clear": True},
                                     {"streams": ["other"]}])
def test_ready_interlock_rejects_unsupported_kernel_policies(options):
    with pytest.raises(ValueError, match="ready_interlock requires"):
        build_pipeline(PARAMS, 2, [{"name": "out", "op": "ADD", "a": "input", "b": ("const", "k")}],
                       consts={"k": 1}, ready_interlock=True, **options)


def test_conditioned_rates_are_recorded_rebuilt_and_shared():
    ks, legacy, _, _ = kernel_campaign.block("tick", PARAMS)
    pl = build_pipeline(PARAMS, ks.width, ks.cells, consts=ks.consts, mems=ks.mems,
                        outputs=ks.outputs, rate_robust=True)
    assert pl.build_options["rate_robust"] is pl.drive.rate_robust is True
    assert pl.build_options["rate_robust_version"] == RATE_ROBUST_VERSION
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


@pytest.mark.parametrize("rate_robust", [False, True])
@pytest.mark.parametrize("datapath", ["generic", "specialized"])
@pytest.mark.parametrize("option", ["zero_once", "robust_request_clear", "experimental_register_reset"])
def test_opt_in_is_recorded_and_rebuilt_with_legacy_false_fallback(rate_robust, datapath, option):
    _, pl, _, _ = kernel_campaign.block("tick", PARAMS, rate_robust=rate_robust,
                                       datapath=datapath, **{option: True})
    assert pl.build_options[option] is True
    _, _, rebuilt = build_tick_pipeline({"block": "tick", **pl.build_options})
    assert _pipeline_fingerprint(pl) == _pipeline_fingerprint(rebuilt)
    # Removing only the new option recovers the exact old build in both modes.
    record = {"block": "tick", **pl.build_options}
    record.pop(option)
    _, _, old = build_tick_pipeline(record)
    _, expected, _, _ = kernel_campaign.block("tick", PARAMS, rate_robust=rate_robust,
                                             datapath=datapath, **{option: False})
    assert old.build_options[option] is False
    assert _pipeline_fingerprint(old) == _pipeline_fingerprint(expected)
    if rate_robust:
        _assert_conditioned_train_inputs(pl.net, pl.drive)


@pytest.mark.parametrize("option", ["zero_once", "robust_request_clear", "experimental_register_reset"])
def test_kernel_campaign_cli_builds_and_records_the_option(tmp_path, monkeypatch, option):
    real_block = kernel_campaign.block
    seen = {}

    def recording_block(*args, **kwargs):
        result = real_block(*args, **kwargs)
        seen["build"] = result
        seen[option] = kwargs[option]
        return result

    def fake_run(pl, params, schedules, **kwargs):
        ks, _, _, reference = seen["build"]
        outputs = [{name: [(i, row[j]) for i, row in enumerate(reference)]
                    for j, name in enumerate(ks.outputs)} for _ in schedules]
        return outputs, None, {"simulator": "Fake", "faults": 0, "timeouts": 0,
                               "bad_outputs": 0, "neural_ms": 1.0}

    monkeypatch.setattr(kernel_campaign, "block", recording_block)
    monkeypatch.setattr(kernel_campaign, "make_perturbed_sim", lambda *args, **kwargs: None)
    monkeypatch.setattr(kernel_campaign, "run_pipeline_batched", fake_run)
    out = tmp_path / "campaign.json"
    kernel_campaign.main(["tick", "--copies", "1", "--max-ms", "1",
                          "--" + option.replace("_", "-"), "--out", str(out)])
    record = json.loads(out.read_text())
    assert seen[option] is seen["build"][1].build_options[option] is True
    assert record[option] is True
    _, _, rebuilt = build_tick_pipeline(record)
    assert _pipeline_fingerprint(rebuilt) == _pipeline_fingerprint(seen["build"][1])


@pytest.mark.parametrize("option", ["zero_once", "robust_request_clear", "experimental_register_reset"])
def test_stage_d_cli_record_recheck_and_diagnostic_rebuild(tmp_path, monkeypatch, option):
    from drosophilos.bench import stage_d

    seen = []
    real_build = stage_d.build

    def recording_build(*args, **kwargs):
        result = real_build(*args, **kwargs)
        seen.append(result)
        return result

    def fake_run(k, pl, params, tokens, *, copies=1, **kwargs):
        reference = stage_d.reference_states(k, tokens)
        outputs = [{k.cells[field]: [(1000 + 100 * i, row[field])
                                    for i, row in enumerate(reference)] for field in k.fields}
                   for _ in range(copies)]
        stats = {"load_steps": [[100]] * copies, "load_events": [[] for _ in range(copies)],
                 "refused": [[] for _ in range(copies)], "faults": 0, "timeouts": 0,
                 "bad_outputs": 0, "blocked_nodes": [], "host_stalls": False,
                 "truncated": False, "stopped_on_stall": False, "simulator": "Fake",
                 "neural_ms": 1000.0, "run_started_perf": 0.0}
        return {"outs": outputs, "stats": stats,
                "wall": [[(cell, step, 1.0) for cell, events in node.items()
                          for step, _ in events] for node in outputs]}

    monkeypatch.setattr(stage_d, "build", recording_build)
    monkeypatch.setattr(stage_d, "run_neural", fake_run)
    # Avoid C compilation: this test verifies the build and record wiring.
    monkeypatch.setattr(stage_d, "three_point_check", lambda *args: {"ir_equal": True})
    out = tmp_path / "stage_d.json"
    record = stage_d.main(["--ticks", "2", "--max-ms", "1", "--" + option.replace("_", "-"),
                           "--rate-robust", "--out", str(out)])
    original = seen[-1]
    assert record[option] is record["build_options"][option] is True
    _, _, rebuilt = build_tick_pipeline(record)
    assert _pipeline_fingerprint(original) == _pipeline_fingerprint(rebuilt)
    stage_d.main(["--recheck", str(out)])
    assert seen[-1].build_options[option] is True
    assert _pipeline_fingerprint(original) == _pipeline_fingerprint(seen[-1])
    # Nested Stage-D records also retain the option; genuinely old records default false.
    record.pop(option)
    stage_d.recheck_record(record)
    assert seen[-1].build_options[option] is True
    record["build_options"].pop(option)
    stage_d.recheck_record(record)
    assert seen[-1].build_options[option] is False


def test_stall_diag_rejects_unidentified_or_obsolete_rate_circuits():
    with pytest.raises(ValueError, match="unsupported rate-robust circuit.*missing"):
        build_tick_pipeline({"block": "tick", "rate_robust": True, "neurons": 30643})
    with pytest.raises(ValueError, match="unsupported rate-robust circuit.*version is 1"):
        build_tick_pipeline({"block": "tick", "rate_robust": True,
                             "rate_robust_version": 1})


def test_stall_diag_accepts_versionless_rate_circuit_when_counts_prove_identity():
    ks, _, _, _ = kernel_campaign.block("tick", PARAMS)
    current = build_pipeline(PARAMS, ks.width, ks.cells, consts=ks.consts,
                             mems=ks.mems, outputs=ks.outputs, rate_robust=True)
    record = {"block": "tick", **current.build_options,
              "neurons": current.net.n, "edges": current.net.nnz}
    record.pop("rate_robust_version")
    _, _, rebuilt = build_tick_pipeline(record)
    assert rebuilt.build_options["rate_robust_version"] == RATE_ROBUST_VERSION


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


# Captured from an isolated detached Git work tree at pre-rate-conditioning commit
# a6c325d: ordered bytes, including roles, weights, delays and biases. The image tests
# above additionally check loading.
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
    "doom4-host-array": "f0fb1dd6cb34620b64637666d1c164115ad54acab2d86857c9db5223087090f4",
    "doom4-host-pipelined": "52e5728946d132a3aef02938cdc74234cbe82eab322efd82908613214b879888",
    "doom4-neural-array": "04dbdb630634360dcd04950b6be4de17686e748dc45029859e4340a1266c77dc",
}


@pytest.mark.parametrize("name", DEFAULT_HASHES)
def test_default_ordered_netlists_are_byte_identical(name):
    if name.startswith("machine"):
        net = build_machine(PARAMS, n=int(name[-1]), n_prog=4, n_data=2).net
    elif name.startswith("doom4-"):
        prog = compile_c(open("examples/doom4.c").read())
        neural = name.startswith("doom4-neural-")
        ks = compile_program(
            prog, params=None, pacing="neural" if neural else "host",
            mul="pipelined" if name.endswith("pipelined") else "array",
            counts={"input": 8, "s": 8, "p": 40} if neural else None,
        )
        net = build_pipeline(
            PARAMS, prog.width, ks.cells, consts=ks.consts, mems=ks.mems,
            outputs=ks.outputs, streams=ks.streams, phases=ks.phases,
        ).net
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
