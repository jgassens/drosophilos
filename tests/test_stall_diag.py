"""Regression coverage for docs/perf_campaign.md §6 tick-stall diagnosis."""

from pathlib import Path
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from drosophilos.bench.stall_diag import analyze_dump, build_tick_pipeline, render_markdown
from drosophilos.lib.campaign import Perturbation, make_perturbed_sim
from drosophilos.lib.kernel import LEGACY_2026_09_20, build_pipeline, load_pipeline_image, run_pipeline_batched
from drosophilos.sim.model import Params


PARAMS = Params()


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("datapath,rate_robust", [("generic", False), ("specialized", True)])
def test_stage_d_record_rebuilds_three_outputs_and_both_diagnostics_read_it(tmp_path, datapath, rate_robust, compact):
    from drosophilos.bench import stage_d
    from drosophilos.bench.fault_diag import analyze_faults

    k = stage_d.load_kernel()
    # A non-default option proves the record is applied, not just the two CLI flags.
    pl = stage_d.build(k, PARAMS, datapath, rate_robust=rate_robust, idle_hops=21)
    record = {"stage": "D", "program": stage_d.PROGRAM, "datapath": datapath, "rate_robust": rate_robust,
              "build_options": pl.build_options, "neurons": pl.net.n, "copies": 2, "ticks": 3,
              "tokens": [5, 5, 250], "state_cells": k.cells,
              "per_copy": [{}, {"completed": 1, "wrong": 0, "missing": 2, "duplicates": 0, "refusals": 0}],
              "commit_events": [{}, {cell: [[300, 0]] for cell in k.cells.values()}]}
    params, ks, rebuilt = build_tick_pipeline(record)
    assert ks.outputs == ["c4_sel", "c9_sel", "c12_sel"]
    assert rebuilt.net.roles == pl.net.roles and rebuilt.build_options == pl.build_options
    np.testing.assert_array_equal(rebuilt.net.topology().quanta, pl.net.topology().quanta)
    if rate_robust:
        versionless = {**record, "build_options": dict(pl.build_options)}
        versionless["build_options"].pop("rate_robust_version")
        with pytest.raises(ValueError, match="unsupported rate-robust circuit.*missing"):
            build_tick_pipeline(versionless)
        versionless["edges"] = pl.net.nnz
        assert build_tick_pipeline(versionless)[2].build_options == pl.build_options

    cell = pl.cells[0]
    neurons = np.asarray([cell.start, cell.stage.fault[0]], dtype=np.int64)
    dump = tmp_path / "copy1.npz"
    window = {"start_step": 90, "end_step": 250}
    if compact:
        stage_d.write_compact_dump(dump, np.asarray([100, 200]), neurons, pl.net.roles,
                                   window=window, capture_ids=neurons)
    else:
        np.savez(dump, step=np.asarray([100, 200]), neuron=neurons,
                 role=np.asarray(pl.net.roles)[neurons])
    campaign = tmp_path / "stage_d.json"
    campaign.write_text(json.dumps(record))
    stall = analyze_dump(dump, campaign, node=1)
    assert stall["output_cells"] == ks.outputs
    assert stall["counts"] == {"requested_transactions": 3, "requested_outputs": 9,
                               "completed_outputs": 3, "wrong": 0, "duplicates": 0,
                               "campaign_refusals": 0, "unfinished": 6}
    fault = analyze_faults(dump, campaign, node=1)
    if compact:
        assert stall["capture_window"] == fault["capture_window"] == window
        assert fault["dump_end"] == 250
    assert fault["build_options"] == pl.build_options
    assert len(fault["faults"]) == 1 and fault["faults"][0]["step"] == 200

    # Both tools must still reject a dump whose neuron/role mapping drifted.
    np.savez(dump, step=np.asarray([100, 200]), neuron=neurons, role=np.asarray(["wrong", "wrong"]))
    for diagnose in (analyze_dump, analyze_faults):
        with pytest.raises(AssertionError, match="dump role mismatch"):
            diagnose(dump, campaign, node=1)


@pytest.mark.parametrize("wide_steps", [False, True])
def test_compact_dump_roundtrip_size_and_filtered_silence(tmp_path, wide_steps):
    from drosophilos.bench.stall_diag import load_dump, write_compact_dump, _observed
    from drosophilos.bench.fault_diag import _observed as fault_observed

    count = 100_000
    roles = ["cell.start", "cell.req.very_long_role_name_that_must_not_repeat_per_spike", "ignored"]
    steps = np.arange(count, dtype=np.int64) + (2**31 if wide_steps else 0)
    neurons = np.ones(count, dtype=np.int64)
    path = tmp_path / "compact.npz"
    window = {"start_step": int(steps[0]), "end_step": int(steps[-1]) + 20}
    write_compact_dump(path, steps, neurons, roles, window=window, capture_ids=[0, 1])
    dump = load_dump(path, roles)
    np.testing.assert_array_equal(dump.step, steps)
    np.testing.assert_array_equal(dump.neuron, neurons)
    assert dump.step.dtype == (np.int64 if wide_steps else np.int32)
    assert dump.neuron.dtype == np.int32
    assert not dump.full_capture
    assert _observed(dump, roles, 0) and fault_observed(dump, roles, 0)
    assert not _observed(dump, roles, 2) and not fault_observed(dump, roles, 2)
    assert dump.window == window and dump.end == window["end_step"]
    per_spike = 12 if wide_steps else 8
    assert per_spike <= path.stat().st_size / count < per_spike + 0.1
    with np.load(path, allow_pickle=False) as raw:
        assert "role" not in raw
        assert raw["uniq"].tolist() == [1] and raw["role_of"].tolist() == [roles[1]]


@pytest.mark.parametrize("existing", [False, True])
def test_atomic_dump_failure_never_replaces_target_or_leaves_partial(tmp_path, monkeypatch, existing):
    from drosophilos.bench.stall_diag import write_compact_dump

    target = tmp_path / "dump.npz"
    if existing:
        target.write_bytes(b"previous complete dump")

    def fail(stream, **arrays):
        stream.write(b"partial zip")
        raise OSError("injected short write")

    monkeypatch.setattr(np, "savez", fail)
    with pytest.raises(OSError, match="injected short write"):
        write_compact_dump(target, [1], [0], ["cell.start"])
    if existing:
        assert target.read_bytes() == b"previous complete dump"
    else:
        assert not target.exists()
    assert list(tmp_path.iterdir()) == ([target] if existing else [])


@pytest.mark.parametrize("fail_write", [False, True])
def test_kernel_campaign_dump_is_compact_and_json_survives_failure(tmp_path, monkeypatch, capsys, fail_write):
    from drosophilos.bench import kernel_campaign as kc

    built = kc.block("fanout", PARAMS)
    ks, pl, tokens, reference = built
    monkeypatch.setattr(kc, "block", lambda *args, **kwargs: built)
    sim = object()
    monkeypatch.setattr(kc, "make_perturbed_sim", lambda *args, **kwargs: sim)
    neuron = pl.cells[0].start

    def fake_run(*args, **kwargs):
        outs = [{name: [(i, row[j]) for i, row in enumerate(reference)] for j, name in enumerate(ks.outputs)}]
        return outs, sim, {"simulator": "Fake", "faults": 0, "timeouts": 0, "bad_outputs": 0,
                           "neural_ms": 1, "first_output_ms": None, "per_token_ms": None,
                           "captured_spikes": (np.asarray([1, 2]), np.asarray([neuron, neuron]))}

    monkeypatch.setattr(kc, "run_pipeline_batched", fake_run)
    if fail_write:
        def fail(*args, **kwargs):
            raise OSError("injected campaign write failure")
        monkeypatch.setattr(kc, "write_compact_dump", fail)
    dump = tmp_path / "copy.npz"
    record = tmp_path / "record.json"
    kc.main(["fanout", "--copies", "1", "--max-ms", "1", "--dump-node", "0",
             "--dump-roles", "start$", "--dump-out", str(dump), "--out", str(record)])
    saved = json.loads(record.read_text())
    if fail_write:
        assert not dump.exists()
        assert saved["spike_dump_errors"]["0"]["error"] == "OSError: injected campaign write failure"
        assert "Traceback (most recent call last)" in capsys.readouterr().err
    else:
        assert not saved["spike_dump_errors"]
        with np.load(dump, allow_pickle=False) as raw:
            assert raw["step"].dtype == raw["neuron"].dtype == np.int32
            assert "role" not in raw and raw["uniq"].tolist() == [neuron]


def test_stage_d_record_rejects_obsolete_nested_circuit_version():
    with pytest.raises(ValueError, match="unsupported true-guard circuit"):
        build_tick_pipeline({"stage": "D", "build_options": {"true_guards": True, "true_guard_version": "old"}})


def test_obsolete_true_guard_capture_is_not_silently_rebuilt_with_new_roles():
    # The simple guard has the same neuron count as v1, but different roles and biases.
    with pytest.raises(ValueError, match="unsupported true-guard circuit"):
        build_tick_pipeline({"true_guards": True})


def _artifact(name: str) -> Path:
    local = Path("data/a2") / name
    if local.exists():
        return local
    # The isolated worktree excludes large data, but the task's read-only precedent lives in
    # the source checkout on the laptop. CI/integration checkouts take the local branch above.
    source_checkout = Path("/Users/jeremiahgassensmith/programming/drosophilos/data/a2") / name
    if not source_checkout.exists():  # CI has no data/ (35 MB, untracked): the precedent is a laptop/cluster test
        pytest.skip(f"precedent artifact {local} not available here")
    return source_checkout


def test_precedent_dump_localizes_live_false_repair_and_failed_clear():
    report = analyze_dump(
        _artifact("tick_s107_node18_compact.npz"),
        _artifact("kc_tick_B90s107b.json"),
        node=18,
    )
    assert report["classification"] == "stuck_request"
    assert report["first_blocked_cell"] == "c2_sel"
    assert report["anomaly"]["transaction"] == 2
    assert report["stuck_source"] == "c0_add"
    assert report["stuck_latch_neuron"] == 11828
    assert report["repair_neuron"] == 11970
    assert report["repair_step"] == 37674
    assert report["clear_pulses"] == [79310, 79354, 79398]
    text = render_markdown(report)
    assert "request / repair / clear" in text
    assert "**37,674**" in text and "11828" in text


def _two_cell_request_dump(tmp_path, monkeypatch, *, live_true: bool):
    """Build a true-guard reader edge, then describe its spikes without simulation."""
    pl = build_pipeline(
        PARAMS,
        1,
        [
            {"name": "producer", "op": "MOV", "a": ("const", "zero"), "b": "input"},
            {"name": "reader", "op": "MOV", "a": ("const", "zero"), "b": "producer"},
        ],
        consts={"zero": 0},
        outputs=["reader"],
        true_guards=True,
    )
    producer, reader = pl.cells
    request = reader.reqs["producer"]
    spikes = [
        (50, request[1].u),
        (100, reader.start),
        (200, reader.stage.completion.u),
        (300, reader.commit_pulse),
        (400, reader.reg.done_relay),
        # The pair's final spike is after DONE; thereafter both rails are silent.
        (500, request[0].u),
        (1_000, producer.start),
        (2_000, producer.stage.completion.u),
        (30_000, reader.idle[1].u),
    ]
    if live_true:
        spikes.append((30_000, request[1].u))
    spikes.sort()
    step = np.asarray([item[0] for item in spikes], dtype=np.int64)
    neuron = np.asarray([item[1] for item in spikes], dtype=np.int64)
    role = np.asarray([pl.net.roles[item[1]] for item in spikes])
    dump_path = tmp_path / ("live_true.npz" if live_true else "dark_request.npz")
    np.savez(dump_path, step=step, neuron=neuron, role=role)
    campaign_path = tmp_path / "campaign.json"
    campaign_path.write_text('{"copies": 1}')
    monkeypatch.setattr(
        "drosophilos.bench.stall_diag.build_tick_pipeline",
        lambda _campaign: (PARAMS, SimpleNamespace(outputs=["reader"]), pl),
    )
    return analyze_dump(dump_path, campaign_path, node=0)


def test_dark_request_names_reader_and_producer_and_renders_evidence(tmp_path, monkeypatch):
    report = _two_cell_request_dump(tmp_path, monkeypatch, live_true=False)

    assert report["classification"] == "dark_request"
    assert report["first_blocked_cell"] == "reader"
    assert report["waiting_producer"] == "producer"
    assert report["anomaly"]["source"] == "producer"
    assert report["anomaly"]["true_last"] == 50
    assert report["anomaly"]["false_last"] == 500
    assert report["anomaly"]["producer_stage"] == 2_000
    assert any(candidate["kind"] == "consumer_holds" and candidate["cell"] == "producer"
               for candidate in report["candidates"][1:])

    text = render_markdown(report)
    assert "**dark request**" in text
    assert "Producer **`producer`**" in text
    assert "`reader.req.producer` TRUE u" in text
    assert "`reader.req.producer` FALSE u" in text
    assert "**`reader` / 2 (blocked)**" in text


def test_live_true_request_keeps_consumer_holds_classification(tmp_path, monkeypatch):
    report = _two_cell_request_dump(tmp_path, monkeypatch, live_true=True)

    assert report["classification"] == "consumer_holds"
    assert report["anomaly"]["cell"] == "producer"
    assert not any(candidate["kind"] == "dark_request" for candidate in report["candidates"])


@pytest.mark.parametrize("true_guards", [False, True])
def test_live_false_repair_is_vetoed_and_done_clear_survives(true_guards):
    """Encode the seed-107 mechanism directly; no impossible single-copy replay.

    Two RefSim copies share the measured within-token source-DONE spacing and a bounded
    weak-clear corner, with the next token aligned to the measured FALSE-rail phases.
    Copy 1 removes only the fixed false-rail -> repair-veto edge, recreating the old
    mechanism: repair of a live false rail, faster orbit, failed DONE clear, both request
    rails live, and no second START. The current copy 0 must survive and complete twice.
    """
    from drosophilos.sim.ref64 import RefSim
    # the mechanism was measured on the 3 x 0.75 kill train of the time (its weights are set
    # explicitly below); pinned so the test does not depend on the default (tests/test_kill_margin.py)
    pl = build_pipeline(
        PARAMS,
        1,
        [{"name": "out", "op": "MOV", "a": ("const", "zero"), "b": ("const", "zero"),
          "trigger": ["input", "input:other"]}],
        consts={"zero": 0},
        streams=["input", "other"],
        **{**LEGACY_2026_09_20, "true_guards": true_guards},
    )
    cell = pl.cells[0]
    req = cell.reqs["input"]
    false_u, false_v = req[0].members
    roles = pl.net.roles
    tap = roles.index("out.actd2.d2")
    repair = roles.index("out.relight.input.edge")
    veto = roles.index("out.relight.input.veto")
    clear = roles.index("out.req.input.k1.inh")
    received = roles.index("out.req.input.received")
    topo = pl.net.topology()
    quanta = np.broadcast_to(topo.quanta, (2, topo.nnz)).copy()

    def set_weight(src, dst, value):
        edge = (topo.src == src) & (topo.dst == dst)
        assert edge.sum() == 1
        quanta[:, edge] = value

    # The bounded synthetic corner used to regress the localized causal sequence. It is not
    # a claim about the campaign's unrecorded per-edge noise or membrane state.
    for src, dst, value in [
        (false_v, false_u, 4056), (false_u, false_v, 4056),
        (clear, false_u, -2631), (clear, false_v, -2608),
        (cell.start, false_u, 4507), (repair, false_u, 4408),
    ]:
        set_weight(src, dst, value)
    vth = np.full((2, topo.n), PARAMS.V_th)
    bias = np.broadcast_to(pl.net.bias, (2, topo.n)).copy()
    vth[:, [false_u, false_v]] -= 0.4
    bias[:, [false_u, false_v]] += 0.4
    false_veto = (topo.src == false_u) & (topo.dst == veto)
    assert false_veto.sum() == 1
    quanta[1, false_veto] = 0  # old circuit, before the live-false veto fix
    quanta[:, topo.dst == tap] = 0  # inject the observed relative tap below

    class RecordingRefSim(RefSim):
        def __init__(self):
            super().__init__(topo, PARAMS, n_nodes=2, quanta=quanta, V_th=vth, bias=bias)
            self.watch = {
                cell.start: "start", false_u: "false", req[1].u: "true",
                repair: "repair", tap: "tap", clear: "clear",
                cell.reg.done_relay: "done", received: "received",
                cell.stage.fault_latch.u: "fault",
            }
            self.events = [{kind: [] for kind in self.watch.values()} for _ in range(2)]
            self.false_phase = {}

        def step(self):
            before = len(self._spk_step)
            super().step()
            for chunk in range(before, len(self._spk_step)):
                step = int(self._spk_step[chunk][0])
                for node, neuron in zip(self._spk_node[chunk].tolist(), self._spk_neuron[chunk].tolist()):
                    kind = self.watch.get(neuron)
                    if kind is not None:
                        self.events[node][kind].append(step)
                    if neuron == cell.start:
                        self.add_events(node, [step + 692], [tap], [pl.drive.ignite])
                    if neuron == false_u and step >= 20000 and node not in self.false_phase:
                        self.false_phase[node] = step
                        if len(self.false_phase) == 2:
                            # Preserve both measured 41/40-step phases (20001/20002 in
                            # the legacy replay), with one shared second-token schedule.
                            shift = next(d for d in range(-820, 821)
                                         if (d - self.false_phase[0] + 20001) % 41 == 0
                                         and (d - self.false_phase[1] + 20002) % 40 == 0)
                            for copy in range(2):
                                self.add_events(copy, [58253 + shift, 69451 + shift],
                                                [pl.inputs[st][0].done_relay for st in ("input", "other")],
                                                [pl.drive.ignite] * 2)
            if self.step_index % 1000 == 0:
                self._spk_step.clear()
                self._spk_node.clear()
                self._spk_neuron.clear()

    sim = RecordingRefSim()
    for node in range(2):
        load_pipeline_image(sim, pl, node=node)
        # Seed-107 source DONE steps shifted earlier by 20,881 steps.
        sim.add_events(
            node,
            [3000, 14043],
            [pl.inputs[stream][0].done_relay for stream in ("input", "other")],
            [pl.drive.ignite] * 2,
        )
        # The healthy first clear had a fourth inhibitory spike; the failed clear had three.
        sim.add_events(node, [3367], [clear], [pl.drive.pulse // 2])
    sim.run(79000)

    assert [len(e["start"]) for e in sim.events] == [2, 1]
    assert [len(e["done"]) for e in sim.events] == [2, 1]
    assert not sim.events[0]["repair"] and len(sim.events[1]["repair"]) == 1
    for node, events in enumerate(sim.events):
        assert len(events["received"]) == 2 and not events["fault"]
        assert len([step for step in events["clear"] if step < 4000]) == 4
        assert len([step for step in events["clear"] if 58000 < step < 60000]) == 3
        assert [tap_ - start for tap_, start in zip(events["tap"], events["start"])] == [716] * len(events["start"])
        false = np.asarray(events["false"])
        train = false[(false > 20000) & (false < 57000)]
        assert np.all(np.diff(train) == (41 if node == 0 else 40))
        assert bool(np.any((false > 61000) & (false < 69000))) == (node == 1)
        assert [step for step in events["true"] if 61000 < step < 69000]


def test_make_perturbed_sim_backends_agree_on_seeded_short_cpu_run():
    pl = build_pipeline(
        PARAMS, 1, [{"name": "out", "op": "MOV", "a": ("const", "zero"), "b": "input"}],
        consts={"zero": 0},
    )
    perturbation = Perturbation(
        weight_sigma=0.001,
        th_sigma_mv=0.01,
        bias_sigma_mv=0.01,
        stray_rate_hz=5.0,
        stray_quanta=10,
        arrival_jitter_steps=0,
    )
    topology = pl.net.topology()
    torch_sim = make_perturbed_sim(
        topology, PARAMS, 2, perturbation, np.random.default_rng(2468), 30000,
        device="cpu", backend="torch",
    )
    fast_sim = make_perturbed_sim(
        topology, PARAMS, 2, perturbation, np.random.default_rng(2468), 30000,
        device="cpu", backend="torch-fast",
    )
    assert type(torch_sim).__name__ == "TorchSim"
    assert type(fast_sim).__name__ == "FastSim"
    assert torch.equal(torch_sim.t_quanta, fast_sim.t_quanta)
    assert torch.equal(torch_sim.V_th, fast_sim.V_th)
    assert torch.equal(torch_sim.bias, fast_sim.bias)
    assert torch.equal(torch_sim._stray_gen.get_state(), fast_sim._stray_gen.get_state())

    schedule = [[0], [1]]
    kwargs = dict(max_ms=3000, progress=0, observe_every=1)
    torch_out, _, torch_stats = run_pipeline_batched(pl, PARAMS, schedule, sim=torch_sim, **kwargs)
    fast_out, _, fast_stats = run_pipeline_batched(
        pl, PARAMS, schedule, sim=fast_sim, backend="torch-fast", **kwargs,
    )
    values = lambda out: [[value for _, value in node["out"]] for node in out]
    assert values(torch_out) == values(fast_out) == [[0], [1]]
    assert torch_out == fast_out
    assert torch_stats["faults"] == fast_stats["faults"] == 0


def test_kernel_campaign_torch_fast_reaches_sim_builder_and_runner(monkeypatch):
    from drosophilos.bench import kernel_campaign

    defaults = kernel_campaign.parser().parse_args(["tick"])
    assert defaults.backend == "torch" and defaults.observe_every is None
    real_block = kernel_campaign.block
    seen = {}

    def recording_block(*args, **kwargs):
        built = real_block(*args, **kwargs)
        seen["built"] = built
        return built

    sentinel = object()

    def fake_make(*args, **kwargs):
        seen["make_backend"] = kwargs["backend"]
        return sentinel

    def fake_run(pl, params, schedules, **kwargs):
        seen["run_backend"] = kwargs["backend"]
        seen["runner_sim"] = kwargs["sim"]
        seen["observe_every"] = kwargs["observe_every"]
        ks, _pl, _tokens, reference = seen["built"]
        outs = []
        for _schedule in schedules:
            outs.append({name: [(i, row[j]) for i, row in enumerate(reference)]
                         for j, name in enumerate(ks.outputs)})
        stats = {
            "simulator": "FastSim", "faults": 0, "timeouts": 0, "bad_outputs": 0,
            "neural_ms": 1.0, "first_output_ms": None, "per_token_ms": None,
        }
        return outs, sentinel, stats

    monkeypatch.setattr(kernel_campaign, "block", recording_block)
    monkeypatch.setattr(kernel_campaign, "make_perturbed_sim", fake_make)
    monkeypatch.setattr(kernel_campaign, "run_pipeline_batched", fake_run)
    kernel_campaign.main(["fanout", "--copies", "1", "--max-ms", "1", "--backend", "torch-fast",
                          "--observe-every", "1"])
    assert seen["make_backend"] == "torch-fast"
    assert seen["run_backend"] == "torch-fast"
    assert seen["runner_sim"] is sentinel
    assert seen["observe_every"] == 1
