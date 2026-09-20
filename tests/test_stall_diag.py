"""Regression coverage for docs/perf_campaign.md §6 tick-stall diagnosis."""

from pathlib import Path

import numpy as np
import torch

from drosophilos.bench.stall_diag import analyze_dump, render_markdown
from drosophilos.lib.campaign import Perturbation, make_perturbed_sim
from drosophilos.lib.kernel import build_pipeline, load_pipeline_image, run_pipeline_batched
from drosophilos.sim.model import Params


PARAMS = Params()


def _artifact(name: str) -> Path:
    local = Path("data/a2") / name
    if local.exists():
        return local
    # The isolated worktree excludes large data, but the task's read-only precedent lives in
    # the source checkout on the laptop. CI/integration checkouts take the local branch above.
    source_checkout = Path("/Users/jeremiahgassensmith/programming/drosophilos/data/a2") / name
    assert source_checkout.exists(), f"missing required precedent artifact {local}"
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


def test_live_false_repair_is_vetoed_and_done_clear_survives(monkeypatch):
    """Encode the seed-107 mechanism directly; no impossible single-copy replay.

    Two RefSim copies share the measured source-DONE spacing and a bounded weak-clear
    corner. Copy 1 removes only the fixed false-rail -> repair-veto edge, recreating the old
    mechanism: repair of a live false rail, faster orbit, failed DONE clear, both request
    rails live, and no second START. The current copy 0 must survive and complete twice.
    """
    from drosophilos.sim.ref64 import RefSim
    from drosophilos.lib import control

    # the mechanism was measured on the 3 x 0.75 kill train of the time (its weights are set
    # explicitly below); the default is 3 x 1.5 since 2026-09-20 (tests/test_kill_margin.py)
    monkeypatch.setattr(control, "KILL_PULSES", 3)
    monkeypatch.setattr(control, "KILL_STRENGTH", 0.75)
    pl = build_pipeline(
        PARAMS,
        1,
        [{"name": "out", "op": "MOV", "a": ("const", "zero"), "b": ("const", "zero"),
          "trigger": ["input", "input:other"]}],
        consts={"zero": 0},
        streams=["input", "other"],
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
    bias = np.zeros((2, topo.n))
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
            [3000, 14043, 58253, 69451],
            [pl.inputs[stream][0].done_relay for stream in ("input", "other", "input", "other")],
            [pl.drive.ignite] * 4,
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

    assert kernel_campaign.parser().parse_args(["tick"]).backend == "torch"
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
    kernel_campaign.main(["fanout", "--copies", "1", "--max-ms", "1", "--backend", "torch-fast"])
    assert seen["make_backend"] == "torch-fast"
    assert seen["run_backend"] == "torch-fast"
    assert seen["runner_sim"] is sentinel
