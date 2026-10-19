"""Fresh-process checkpoint acceptance tests, using a tiny real perturbed circuit."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


def _small_block(name, params, **options):
    from drosophilos.compiler.kernel import KernelSpec, kernel_outputs
    from drosophilos.lib import kernel
    ks = KernelSpec([dict(name="out", op="MOV", a=("const", "zero"), b="input")],
                    {"zero": 0}, {}, "input", 2)
    ks.outputs = ["out"]
    pl = kernel.build_pipeline(params, 2, ks.cells, consts=ks.consts, outputs=ks.outputs)
    tokens = [1]
    return ks, pl, tokens, kernel_outputs(ks, tokens)


def _worker():
    """Exercise the real CLI/runner with a two-bit MOV instead of a large campaign."""
    import ast
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    from drosophilos.bench import kernel_campaign as kc
    from drosophilos.lib import kernel
    from drosophilos.lib.campaign import CampaignCheckpoint

    torch.set_num_threads(1)
    scenario, backend, mode, step, checkpoint, output = sys.argv[1:]

    kc.block = _small_block
    if mode == "legacy":
        # Pin the pre-change revision: committing this work must not silently turn
        # this regression check into a comparison of the new runner with itself.
        source = subprocess.check_output([
            "git", "show", "b176cb21de465f88b3d350c12fa42e27d16da76c:drosophilos/lib/kernel.py"
        ], text=True)
        tree = ast.parse(source)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_pipeline_batched")
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "pre_checkpoint_kernel.py", "exec"), vars(kernel))
        kc.run_pipeline_batched = kernel.run_pipeline_batched
    original_run = kc.run_pipeline_batched

    def run(*args, **kwargs):
        if mode == "no_compute":
            raise AssertionError("finished chain recomputed")
        if mode == "runner_error":
            raise ValueError("simulated runner failure")
        if scenario == "refusal":
            # Stateless: every load of word 1 loses bit zero; blocks after two
            # watchdog refusals. Checkpoint while the first refusal is queued for retry.
            kwargs.update(rail_filter=lambda b, index, st, value, rails: [(i, r) for i, r in rails if i != 0],
                          max_retries=1)
        return original_run(*args, **kwargs)

    kc.run_pipeline_batched = run
    original_poll = CampaignCheckpoint.poll

    def poll(self, sim, host, observer):
        if mode == "interrupt":
            trigger = bool(host.retry[0]) if step == "refusal" else sim.step_index >= int(step)
            if trigger:
                os.kill(os.getpid(), signal.SIGUSR1)
        if mode == "term" and sim.step_index >= 3107:
            Path(checkpoint + ".ready").touch()
            # The parent sends a real SIGTERM once a mid-token boundary is reached.
            while self.signal_received is None:
                time.sleep(0.01)
        if mode == "periodic" and sim.step_index == 3107:
            self.last_write = time.monotonic() - self.every_s - 1
        result = original_poll(self, sim, host, observer)
        if mode == "periodic" and sim.step_index == 3107:
            saved = torch.load(checkpoint, weights_only=False)
            assert not saved["finished"] and saved["run"]["simulator"]["step_index"] == 3107
        return result

    CampaignCheckpoint.poll = poll
    argv = ["fanout", "--copies", "1", "--backend", backend, "--seed", "17",
            "--max-ms", "2200", "--out", output]
    if backend == "torch-fast":
        argv += ["--observe-every", "37"]
    if mode not in ("plain", "legacy"):
        argv += ["--checkpoint", checkpoint]
    if mode == "mismatch":
        argv += ["--seed", "18"]
    kc.main(argv)


def _command(tmp_path, *, scenario="normal", backend="torch", mode="resume", step="3107", name="run"):
    return [sys.executable, str(Path(__file__).resolve()), scenario, backend, mode, str(step),
            str(tmp_path / f"{name}.pt"), str(tmp_path / f"{name}.json")]


def _run(tmp_path, **kwargs):
    result = subprocess.run(_command(tmp_path, **kwargs), capture_output=True, text=True, timeout=90)
    return result


def _record(path):
    record = json.loads(path.read_text())
    record.pop("wall_s")
    record.pop("checkpoint_fingerprint", None)
    # JSON's NaN represents the historical all-missing confidence limit.
    return json.dumps(record, sort_keys=True)


@pytest.fixture(scope="module")
def baselines(tmp_path_factory):
    directory = tmp_path_factory.mktemp("baseline")
    result = {}
    for scenario, backend in [("normal", "torch"), ("normal", "torch-fast"),
                              ("refusal", "torch"), ("refusal", "torch-fast")]:
        name = scenario + backend
        run = _run(directory, scenario=scenario, backend=backend, mode="plain", name=name)
        assert run.returncode == 0, run.stderr
        assert "checkpoint_fingerprint" not in json.loads((directory / f"{name}.json").read_text())
        result[scenario, backend] = _record(directory / f"{name}.json")
    return result


@pytest.mark.parametrize("backend,scenario,step", [
    ("torch", "normal", 53),  # image settling
    ("torch", "normal", 3107),  # loaded token, before its output
    ("torch-fast", "normal", 3107),  # also has an unflushed observation buffer
    ("torch", "refusal", "refusal"),
    ("torch-fast", "refusal", "refusal"),
])
def test_exact_fresh_process_resume(tmp_path, baselines, backend, scenario, step):
    interrupted = _run(tmp_path, backend=backend, scenario=scenario, mode="interrupt", step=step)
    assert interrupted.returncode == 75, interrupted.stderr
    assert not (tmp_path / "run.json").exists()
    import torch
    saved = torch.load(tmp_path / "run.pt", weights_only=False)["run"]
    assert "legacy_spikes" not in saved
    if backend == "torch":
        assert saved["observer"]["_legacy_seen"] == 0
    resumed = _run(tmp_path, backend=backend, scenario=scenario)
    assert resumed.returncode == 0, resumed.stderr
    assert _record(tmp_path / "run.json") == baselines[scenario, backend]
    if scenario == "refusal":
        assert json.loads((tmp_path / "run.json").read_text())["refusals"] >= 1


def test_fingerprint_mismatch_refuses(tmp_path):
    interrupted = _run(tmp_path, mode="interrupt", step=5)
    assert interrupted.returncode == 75, interrupted.stderr
    refused = _run(tmp_path, mode="mismatch")
    assert refused.returncode == 2
    assert "fingerprint mismatch: config" in refused.stderr


def test_sigterm_saves_and_exits_75(tmp_path, baselines):
    with subprocess.Popen(_command(tmp_path, mode="term"), stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True) as process:
        deadline = time.monotonic() + 30
        while not (tmp_path / "run.pt.ready").exists():
            assert process.poll() is None, process.communicate()
            assert time.monotonic() < deadline
            time.sleep(0.02)
        process.send_signal(signal.SIGTERM)
        _, stderr = process.communicate(timeout=30)
        assert process.returncode == 75, stderr
    assert (tmp_path / "run.pt").exists()
    resumed = _run(tmp_path)
    assert resumed.returncode == 0, resumed.stderr
    assert _record(tmp_path / "run.json") == baselines["normal", "torch"]


def test_finished_chain_is_noop(tmp_path):
    assert _run(tmp_path).returncode == 0
    record = json.loads((tmp_path / "run.json").read_text())
    assert "checkpoint_fingerprint" in record
    assert "commit" not in record["checkpoint_fingerprint"]
    ordinary = tmp_path / "ordinary.json"
    ordinary.write_text("{}")
    assert (tmp_path / "run.json").stat().st_mode == ordinary.stat().st_mode
    modified = (tmp_path / "run.json").stat().st_mtime_ns
    rerun = _run(tmp_path, mode="no_compute")
    assert rerun.returncode == 0, rerun.stderr
    assert (tmp_path / "run.json").stat().st_mtime_ns == modified
    # A completed atomic output is also enough if the final marker was not written.
    (tmp_path / "run.pt").unlink()
    rerun = _run(tmp_path, mode="no_compute")
    assert rerun.returncode == 0, rerun.stderr


@pytest.mark.parametrize("fingerprint", [None, {"source": "older-code"}])
def test_stale_output_is_refused_without_overwrite(tmp_path, fingerprint):
    assert _run(tmp_path, mode="plain").returncode == 0
    path = tmp_path / "run.json"
    rec = json.loads(path.read_text())
    if fingerprint is not None:
        rec["checkpoint_fingerprint"] = fingerprint
    path.write_text(json.dumps(rec))
    original = path.read_bytes()
    modified = path.stat().st_mtime_ns
    refused = _run(tmp_path, mode="no_compute")
    assert refused.returncode == 2, refused.stderr
    assert "existing --out has no matching checkpoint fingerprint" in refused.stderr
    assert path.read_bytes() == original and path.stat().st_mtime_ns == modified
    assert not (tmp_path / "run.pt").exists()


def test_fingerprint_ignores_commit_but_checks_source_and_device(monkeypatch):
    import torch
    from drosophilos.bench import kernel_campaign as kc
    from drosophilos.sim.model import Params
    params = Params()
    _, pl, _, _ = _small_block("fanout", params)
    args = kc.parser().parse_args(["fanout", "--device", "cuda"])
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "GPU model A")
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: "commit A")
    before = kc._fingerprint(args, pl, params)
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: "unrelated commit B")
    assert kc._fingerprint(args, pl, params) == before
    assert "commit" not in before
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "GPU model B")
    assert kc._fingerprint(args, pl, params)["device_name"] != before["device_name"]
    read_bytes = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda path: read_bytes(path) +
                        (b"\n# changed source" if path.name == "kernel.py" else b""))
    assert kc._fingerprint(args, pl, params)["source"] != before["source"]


def test_runner_value_error_is_not_an_argument_error(tmp_path):
    result = _run(tmp_path, mode="runner_error")
    assert result.returncode == 1, result.stderr
    assert "ValueError: simulated runner failure" in result.stderr
    assert "Traceback" in result.stderr
    assert "error: simulated runner failure" not in result.stderr


def test_checkpoint_version_error_is_an_argument_error(tmp_path):
    import torch
    torch.save({"version": -1}, tmp_path / "run.pt")
    result = _run(tmp_path)
    assert result.returncode == 2
    assert "unsupported campaign checkpoint version" in result.stderr


def test_startup_removes_only_this_checkpoints_temporary_files(tmp_path):
    from drosophilos.lib.campaign import CampaignCheckpoint
    stale = [tmp_path / "run.pt.1234abcd.tmp", tmp_path / "run.pt.ab_cd123.tmp"]
    keep = [tmp_path / name for name in ("other.pt.1234abcd.tmp", "run.pt.tmp",
                                        "run.pt.notes.tmp", "run.pt.1234abcd.tmp.backup", "run.pt")]
    for path in stale + keep:
        path.write_bytes(b"keep")
    with CampaignCheckpoint(tmp_path / "run.pt"):
        assert all(not path.exists() for path in stale)
        assert all(path.read_bytes() == b"keep" for path in keep)


def test_periodic_checkpoint_continues_exactly(tmp_path, baselines):
    result = _run(tmp_path, mode="periodic")
    assert result.returncode == 0, result.stderr
    assert "at step 3107" in result.stderr
    assert _record(tmp_path / "run.json") == baselines["normal", "torch"]


@pytest.mark.parametrize("backend", ["torch", "torch-fast"])
def test_simulator_snapshot_restores_stray_rng_and_delay_ring(backend):
    import torch
    from drosophilos.sim.lif_fast import FastSim
    from drosophilos.sim.lif_torch import TorchSim
    from drosophilos.sim.model import Params, Topology
    simulator = TorchSim if backend == "torch" else FastSim
    sim = simulator(Topology.from_edges(3, [0, 1], [1, 2], [2000, 2000], [13, 17]), Params(),
                    stray_rate_hz=2000, stray_quanta=5000, stray_seed=19)
    sim.run(25)
    saved = sim.snapshot()
    sim.run(31)
    expected = sim.snapshot()
    sim.restore(saved)
    sim.run(31)
    got = sim.snapshot()
    for key in ("V", "g", "r", "ring", "stray_rng"):
        actual, want = got[key], expected[key]
        if key == "ring":
            assert actual.is_sparse and want.is_sparse
            actual, want = actual.to_dense(), want.to_dense()
        assert torch.equal(actual, want), key


@pytest.mark.parametrize("backend", ["torch", "torch-fast"])
@pytest.mark.parametrize("empty", [False, True])
def test_sparse_ring_serialization_restores_identical_dense_tensor(backend, empty):
    import io
    import torch
    from drosophilos.sim.lif_fast import FastSim
    from drosophilos.sim.lif_torch import TorchSim
    from drosophilos.sim.model import Topology
    cls = TorchSim if backend == "torch" else FastSim
    topo = Topology.from_edges(2048, [], [], [], [])
    sim = cls(topo, n_nodes=2, dtype=torch.float32)
    if not empty:
        sim.ring[0, 0, 0] = 2**62 + 17
        sim.ring[-1, -1, -1] = -(2**62) - 19
        sim.ring[13, 1, 1024] = 42
    expected = sim.ring.clone()
    snap = sim.snapshot()
    ring = snap["ring"].coalesce()
    assert ring.is_sparse and ring.dtype == expected.dtype
    assert ring.indices().shape == (1, 0 if empty else 3)
    assert torch.equal(ring.to_dense().reshape(expected.shape), expected)
    stream = io.BytesIO()
    torch.save(snap, stream)
    assert stream.tell() < expected.numel() * expected.element_size() // 4
    stream.seek(0)
    restored = cls(topo, n_nodes=2, dtype=torch.float32)
    restored.restore(torch.load(stream, map_location="cpu", weights_only=False))
    assert restored.ring.layout == torch.strided
    assert restored.ring.dtype == expected.dtype
    assert torch.equal(restored.ring, expected)


@pytest.mark.parametrize("backend", ["torch", "torch-fast"])
def test_default_record_matches_prechange_runner(tmp_path, baselines, backend):
    result = _run(tmp_path, backend=backend, mode="legacy")
    assert result.returncode == 0, result.stderr
    assert "checkpoint_fingerprint" not in json.loads((tmp_path / "run.json").read_text())
    assert _record(tmp_path / "run.json") == baselines["normal", backend]


def test_atomic_failure_preserves_previous_file(tmp_path):
    from drosophilos.lib.campaign import atomic_write
    path = tmp_path / "checkpoint"
    path.write_bytes(b"previous")

    def fail(stream):
        stream.write(b"incomplete")
        raise OSError("simulated write failure")

    with pytest.raises(OSError, match="simulated"):
        atomic_write(path, fail)
    assert path.read_bytes() == b"previous"
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_output_preserves_existing_permissions(tmp_path):
    import stat
    from drosophilos.lib.campaign import atomic_write
    path = tmp_path / "output.json"
    path.write_text("old")
    path.chmod(0o640)
    atomic_write(path, lambda stream: stream.write(b"new"), mode=0o666)
    assert path.read_bytes() == b"new"
    assert stat.S_IMODE(path.stat().st_mode) == 0o640


@pytest.mark.parametrize("cluster", ["juno", "g2"])
def test_slurm_chain_dependencies_and_signal(tmp_path, cluster):
    import shutil
    slurm = tmp_path / "slurm"
    slurm.mkdir()
    shutil.copyfile(Path(__file__).resolve().parents[1] / "slurm/chain.sh", slurm / "chain.sh")
    submit = slurm / "submit.sh"
    submit.write_text('''#!/bin/bash
set -eu
printf '%s\\n' "$*" >> "$CHAIN_LOG"
count=0
if [ -f "$CHAIN_COUNT" ]; then read -r count < "$CHAIN_COUNT"; fi
count=$((count + 1))
echo "$count" > "$CHAIN_COUNT"
echo "Submitted batch job $count"
''')
    submit.chmod(0o755)
    env = dict(os.environ, CHAIN_LOG=str(tmp_path / "calls"), CHAIN_COUNT=str(tmp_path / "count"))
    result = subprocess.run(["/bin/bash", str(slurm / "chain.sh"), "3", "data/run.pt",
                             "--cluster", cluster, "--time=12:00:00", "--", "perspective",
                             "--copies", "100", "--out", "data/run.json"], env=env,
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    calls = (tmp_path / "calls").read_text().splitlines()
    assert len(calls) == 3 and "--dependency" not in calls[0]
    for i, call in enumerate(calls):
        assert f"--cluster {cluster}" in call
        assert "--signal=B:USR1@600" in call and "--checkpoint data/run.pt" in call
        if i:
            assert f"--dependency=afterany:{i}" in call
    refused = subprocess.run(["/bin/bash", str(slurm / "chain.sh"), "3", "lost.pt",
                              "--", "fanout"], env=env, text=True, capture_output=True)
    assert refused.returncode == 2 and "checkpoint must be under data/" in refused.stderr


if __name__ == "__main__":
    _worker()
