"""A reset stage must not be interpreted as an all-double word at COPY."""

from __future__ import annotations

import pytest

from drosophilos.lib.kernel import build_pipeline, run_pipeline
from drosophilos.protocol.token import decode_recent
from drosophilos.sim.lif_torch import TorchSim
from drosophilos.sim.model import Params
from drosophilos.sim.ref64 import RefSim


P = Params()


def _two_cell_pipeline(*, copy_requires_rail: bool):
    return build_pipeline(
        P,
        1,
        [
            {"name": "c1", "op": "MOV", "a": ("const", "zero"), "b": "input"},
            {"name": "out", "op": "ADD", "a": "c1", "b": ("const", "one")},
        ],
        consts={"zero": 0, "one": 1},
        copy_requires_rail=copy_requires_rail,
    )


def _race_sim(pl, sim_cls):
    """Time c1's opposite result rail so reset lands after COMMIT enters guardd."""
    cell = pl.cells[0]
    completion = cell.stage.completion.u
    commit = cell.commit_pulse
    opposite = cell.stage.rails[0][0].u  # token 1 through MOV legitimately holds rail 1
    watch = {
        completion: "completion",
        commit: "commit",
        cell.stage.fault[0]: "fault_gate",
        cell.stage.fault_latch.u: "fault",
        cell.stage.reset_trigger: "stage_reset",
        cell.stage.reset_inh: "stage_reset_inh",
        pl.net.roles.index("c1.guardd.d11"): "guard",
        cell.reg.granted.u: "grant",
        cell.reg.copy.u: "copy",
    }

    class InjectFault(sim_cls):
        def __init__(self):
            kwargs = {"device": "cpu"} if sim_cls is TorchSim else {}
            super().__init__(pl.net.topology(), P, **kwargs)
            self.injected_at = None
            self.first = {}
            self.events = {label: [] for label in watch.values()}

        def step(self):
            before = len(self._spk_step)
            super().step()
            for k in range(before, len(self._spk_step)):
                step = int(self._spk_step[k][0])
                for neuron in self._spk_neuron[k].tolist():
                    label = watch.get(int(neuron))
                    if label is not None:
                        self.first.setdefault(label, step)
                        self.events[label].append(step)
                    if neuron == completion and self.injected_at is None:
                        # The opposite rail needs time to become a train and trip the
                        # rate-mode fault detector. This offset puts reset just after
                        # commit_pulse and before guardd's final hop, matching the capture.
                        self.injected_at = self.step_index + 230
                        self.add_events(
                            0, [self.injected_at], [opposite], [pl.drive.ignite]
                        )

    return InjectFault()


@pytest.mark.parametrize("sim_cls", [RefSim, TorchSim], ids=["ref", "torch-cpu"])
@pytest.mark.parametrize("copy_requires_rail", [False, True], ids=["legacy", "true-rail"])
def test_fault_reset_during_guard_cannot_copy_an_empty_stage(sim_cls, copy_requires_rail):
    """Reproduce the captured guardd race and distinguish corruption from fail-stop."""
    pl = _two_cell_pipeline(copy_requires_rail=copy_requires_rail)
    sim = _race_sim(pl, sim_cls)
    outputs, sim, stats = run_pipeline(
        pl, P, [1], sim=sim, max_ms=2200, retry_refused=False
    )

    assert sim.injected_at is not None
    commit = sim.first["commit"]
    stage_reset = next(step for step in sim.events["stage_reset"] if step > commit)
    grant = next(
        (step for step in sim.events["grant"] if step > commit),
        None,
    )
    assert grant is not None, "\n".join(f"{key}: {value}" for key, value in sim.events.items())
    copy = next(step for step in sim.events["copy"] if step > grant)
    assert commit < stage_reset < grant < copy, (commit, stage_reset, grant, copy, sim.first)

    window = 2 * pl.drive.loop_period_steps
    _value, master_status = decode_recent(
        sim, pl.cells[0].master.rail_taps, sim.step_index - 1, window
    )
    bit_statuses = [
        decode_recent(sim, [pair], sim.step_index - 1, window)[1]
        for pair in pl.cells[0].master.rail_taps
    ]
    if not copy_requires_rail:
        assert master_status == "fault"
        assert set(bit_statuses) == {"fault"}  # both legacy COPY arms fired for every dark bit
    else:
        assert master_status == "incomplete"
        assert "fault" not in bit_statuses
        # This transaction fail-stops: c1's reset master is incomplete, so its consumer
        # never sees a valid word and cannot silently report a value.
        assert outputs == stats["outputs_by_cell"]["out"] == []
        assert stats["faults"] == 1 and stats["bad_outputs"] == 0
        assert stats["host_stalls"]


@pytest.mark.parametrize(
    ("op", "constant", "tokens", "expected"),
    [
        ("MOV", 0, [0, 1, 1, 0], [0, 1, 1, 0]),
        ("ADD", 1, [0, 1, 0, 1], [1, 0, 1, 0]),
    ],
)
def test_true_rail_copy_preserves_values_and_normal_latency(op, constant, tokens, expected):
    """The coincidence adds only a few milliseconds to an ordinary transaction."""
    spec = [{
        "name": "out",
        "op": op,
        "a": ("const", "k") if op == "MOV" else "input",
        "b": "input" if op == "MOV" else ("const", "k"),
    }]
    runs = {}
    for required in (False, True):
        pl = build_pipeline(
            P, 1, spec, consts={"k": constant}, copy_requires_rail=required
        )
        outputs, sim, stats = run_pipeline(pl, P, tokens, max_ms=8000, full_trace=True)
        assert [value for _step, value in outputs] == expected
        assert stats["faults"] == stats["timeouts"] == stats["bad_outputs"] == 0
        events = sim.trace.events
        grant_steps = events["step"][
            (events["node"] == 0) & (events["neuron"] == pl.output.reg.granted.u)
        ]
        starts = grant_steps[
            [True, *(grant_steps[1:] - grant_steps[:-1] > 3 * pl.drive.loop_period_steps)]
        ]
        assert len(starts) == len(outputs)
        runs[required] = {
            "stats": stats,
            "commit_ms": [
                (output_step - int(grant_step)) * P.dt
                for (output_step, _value), grant_step in zip(outputs, starts)
            ],
        }

    # One transaction crosses the input register and the result cell, so its end-to-end
    # first-output delta contains two COPY coincidences. Each individual commit remains
    # within a few milliseconds of the legacy path.
    assert max(
        abs(new - old)
        for new, old in zip(runs[True]["commit_ms"], runs[False]["commit_ms"])
    ) <= 5.0
    assert abs(
        runs[True]["stats"]["first_output_ms"]
        - runs[False]["stats"]["first_output_ms"]
    ) <= 8.0
    assert abs(
        runs[True]["stats"]["per_token_ms"]
        - runs[False]["stats"]["per_token_ms"]
    ) <= 8.0
