"""Fast CLI wiring checks for kernel_campaign."""

import json

import pytest

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


_F = dict(rate_robust=True, experimental_autapse=True, ordered_xor_oneshot=True)


def test_mul8_builds_under_f_with_array_mul():
    ks, pl, tokens, _ref = kc.block("mul8", Params(), **_F)
    assert ks.width == 8 and tokens == [0, 1, 7, 13, 51, 128, 200, 255]
    assert "ordered_xor_oneshot_mul_version" in pl.build_options


def test_mul8_reference_matches_integer_arithmetic():
    _ks, _pl, tokens, ref = kc.block("mul8", Params(), **_F)
    assert [int(row[0]) for row in ref] == [((x * 5 + 3) * x) & 255 for x in tokens]


@pytest.mark.parametrize("block", ["mul8", "perspective"])
def test_f_cli_smoke_writes_record(tmp_path, monkeypatch, block):
    seen = {}
    real_block = kc.block

    def recording_block(*args, **kwargs):
        seen["build"] = real_block(*args, **kwargs)
        return seen["build"]

    def fake_run(pl, params, schedules, **kwargs):
        ks, _pl, _tokens, reference = seen["build"]
        outs = [{name: [(i, row[j]) for i, row in enumerate(reference)]
                 for j, name in enumerate(ks.outputs)} for _ in schedules]
        return outs, object(), {"simulator": "Fake", "faults": 0, "timeouts": 0, "bad_outputs": 0,
                                "neural_ms": 1.0, "first_output_ms": None, "per_token_ms": None}

    monkeypatch.setattr(kc, "block", recording_block)
    monkeypatch.setattr(kc, "make_perturbed_sim", lambda *a, **k: object())
    monkeypatch.setattr(kc, "run_pipeline_batched", fake_run)
    out = tmp_path / f"{block}.json"
    kc.main([block, "--copies", "1", "--max-ms", "1", "--rate-robust", "--experimental-autapse",
             "--ordered-xor-oneshot", "--out", str(out),
             *(["--f-width16-experimental"] if block == "perspective" else [])])
    rec = json.loads(out.read_text())
    assert rec["block"] == block and rec["ordered_xor_oneshot"] is True
    if block == "perspective":
        assert seen["build"][1].n == 16
        assert rec["f_width16_experimental"] is True
    else:
        assert "f_width16_experimental" not in rec


@pytest.mark.slow
def test_perspective_width16_experimental_cli_real_simulator_smoke(tmp_path):
    out = tmp_path / "perspective.json"
    kc.main(["perspective", "--copies", "1", "--max-ms", "1", "--rate-robust",
             "--experimental-autapse", "--ordered-xor-oneshot", "--f-width16-experimental",
             "--out", str(out)])
    rec = json.loads(out.read_text())
    assert rec["f_width16_experimental"] is True
    assert rec["copies"] == 1 and rec["block"] == "perspective"
    assert rec["simulator"] == "TorchSim"
    assert rec["faults"] == rec["timeouts"] == rec["bad_outputs"] == 0
