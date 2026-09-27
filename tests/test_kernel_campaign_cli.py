"""Fast CLI wiring checks for kernel_campaign."""

import json

from drosophilos.bench import kernel_campaign as kc
from drosophilos.sim.model import Params


def test_rate_robust_cli_reaches_the_pipeline_and_record(tmp_path, monkeypatch):
    """The campaign flag must affect the build, not merely be recorded by argparse."""
    real_block = kc.block
    seen = {}

    def recording_block(*args, **kwargs):
        result = real_block(*args, **kwargs)
        seen["build"] = result
        seen["rate_robust"] = kwargs["rate_robust"]
        return result

    sentinel = object()

    def fake_make(*args, **kwargs):
        return sentinel

    def fake_run(pl, params, schedules, **kwargs):
        ks, _pl, _tokens, reference = seen["build"]
        outs = [{name: [(i, row[j]) for i, row in enumerate(reference)]
                 for j, name in enumerate(ks.outputs)} for _ in schedules]
        return outs, sentinel, {
            "simulator": "Fake", "faults": 0, "timeouts": 0, "bad_outputs": 0,
            "neural_ms": 1.0, "first_output_ms": None, "per_token_ms": None,
        }

    monkeypatch.setattr(kc, "block", recording_block)
    monkeypatch.setattr(kc, "make_perturbed_sim", fake_make)
    monkeypatch.setattr(kc, "run_pipeline_batched", fake_run)
    out = tmp_path / "campaign.json"
    kc.main(["fanout", "--copies", "1", "--max-ms", "1", "--rate-robust", "--out", str(out)])

    _, robust, _, _ = seen["build"]
    _, default, _, _ = real_block("fanout", Params())
    rec = json.loads(out.read_text())
    assert seen["rate_robust"] is robust.build_options["rate_robust"] is True
    assert robust.net.n > default.net.n
    assert rec["rate_robust"] is True
