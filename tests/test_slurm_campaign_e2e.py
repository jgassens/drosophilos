"""Local Slurm B:USR1 acceptance: real git worktrees, CLI, simulator and resume.

Only conda activation and the GPU inventory command are stubbed. Bash xtrace
records the real EXIT cleanup so its ordering relative to the save is visible.
"""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ARGS = [
    "fanout", "--copies", "1", "--seed", "17", "--max-ms", "2200",
    "--device", "cpu", "--backend", "torch", "--mix", "B",
    "--checkpoint", "data/e2e.pt", "--checkpoint-every-s", "0.5",
    "--out", "data/e2e.json",
]
CHECKPOINT_LINE = re.compile(r"^checkpoint: data/e2e\.pt at step (\d+)$", re.MULTILINE)


def _git(directory, *args, env):
    return subprocess.run(
        ["git", *args], cwd=directory, env=env, check=True,
        capture_output=True, text=True, timeout=30,
    ).stdout.strip()


def _executable(path, contents):
    path.write_text(contents)
    path.chmod(0o755)


@pytest.fixture
def campaign_home(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    checkout = home / "drosophilos"
    env = dict(os.environ, HOME=str(home), GIT_CONFIG_GLOBAL="/dev/null",
               GIT_CONFIG_NOSYSTEM="1", PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env.pop("PYTHONHOME", None)
    sha = _git(ROOT, "rev-parse", "HEAD", env=env)
    _git(tmp_path, "clone", "--quiet", "--no-hardlinks", str(ROOT), str(checkout), env=env)
    _git(checkout, "checkout", "--quiet", "--detach", sha, env=env)
    for name in ("data", "runs"):
        (checkout / name).mkdir()
    conda = home / "miniforge3/etc/profile.d/conda.sh"
    conda.parent.mkdir(parents=True)
    conda.write_text("conda() { :; }\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # Replace any inherited editable-install PYTHONPATH with the job's worktree.
    # exec preserves the child PID that the real sbatch USR1 trap forwards to.
    _executable(bin_dir / "python", (
        '#!/bin/bash\nset -eu\n'
        'printf "e2e python cwd: %s\\n" "$PWD"\n'
        'export PYTHONPATH="$PWD"\n'
        f'exec {shlex.quote(sys.executable)} "$@"\n'
    ))
    _executable(bin_dir / "nvidia-smi", "#!/bin/bash\nexit 0\n")
    env.update(PATH=f"{bin_dir}:{env['PATH']}", DROSO_SHA=sha)
    return home, checkout, env


@contextmanager
def _job(script, home, checkout, env, job_id):
    log = home / f"{job_id}.log"
    with log.open("w") as output:
        process = subprocess.Popen(
            ["/bin/bash", "-x", str(ROOT / "slurm" / script),
             "drosophilos.bench.kernel_campaign", *CAMPAIGN_ARGS],
            cwd=checkout, env={**env, "SLURM_JOB_ID": job_id},
            stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            yield process, log, home / "drosophilos-jobs" / job_id
        finally:
            # Kill descendants even if Bash has already exited (the old tail can
            # leave its foreground Python alive). File output avoids pipe hangs.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)


def _wait_for(process, log, predicate, *, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        output = log.read_text()
        result = predicate(output)
        if result:
            return result
        assert process.poll() is None, f"sbatch exited {process.returncode}:\n{output}"
        time.sleep(0.02)
    pytest.fail(f"timed out waiting for campaign:\n{log.read_text()}")


def _wait_exit(process, log, *, timeout):
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pytest.fail(f"timed out waiting for sbatch exit:\n{log.read_text()}")


def _validated_checkpoint(path):
    """Use the campaign's validator with an independently rebuilt fingerprint."""
    import numpy as np
    from drosophilos.bench import kernel_campaign as kc
    from drosophilos.lib.campaign import CampaignCheckpoint
    from drosophilos.sim.model import Params

    args = kc.parser().parse_args(CAMPAIGN_ARGS)
    params = Params()
    _, pipeline, _, _ = kc.block(args.block, params)
    checkpoint = CampaignCheckpoint(path)
    checkpoint.configure(kc._fingerprint(args, pipeline, params), vars(args),
                         np.random.default_rng(args.seed))
    return checkpoint


def _assert_cleanup(output, worktree):
    cleanup = [line for line in output.splitlines()
               if line.startswith("+ git worktree remove --force ")]
    assert len(cleanup) == 1, output
    assert shlex.split(cleanup[0]) == ["+", "git", "worktree", "remove", "--force", str(worktree)]
    saves = list(CHECKPOINT_LINE.finditer(output))
    assert saves, output
    assert saves[-1].end() < output.index(cleanup[0]), output
    assert not worktree.exists(), output


@pytest.mark.parametrize("script", ["juno.sbatch", "g2.sbatch", "juno-cpu.sbatch"])
def test_batch_usr1_checkpoints_and_next_job_resumes(campaign_home, script):
    home, checkout, env = campaign_home
    checkpoint_path = checkout / "data/e2e.pt"
    result_path = checkout / "data/e2e.json"
    job_prefix = f"{os.getpid()}-{Path(script).stem}"

    with _job(script, home, checkout, env, job_prefix + "-1") as (process, log, worktree):
        # Past the 3000-step image settling period: a real token is in flight,
        # and the signal handler and periodic checkpoint writer are both active.
        ready_step = _wait_for(
            process, log,
            lambda text: next((int(m[1]) for m in CHECKPOINT_LINE.finditer(text)
                               if int(m[1]) >= 4000), None),
            timeout=30,
        )
        assert worktree.is_dir()
        assert (worktree / "data").is_symlink()
        assert (worktree / "data").resolve() == checkout / "data"
        assert (worktree / "runs").resolve() == checkout / "runs"
        assert f"e2e python cwd: {worktree}\n" in log.read_text()
        # Exactly Slurm --signal=B:USR1@600: signal ONLY the batch shell,
        # never its Python child or the process group.
        os.kill(process.pid, signal.SIGUSR1)
        assert _wait_exit(process, log, timeout=15) == 75, log.read_text()
        interrupted_output = log.read_text()
        assert not result_path.exists(), interrupted_output
        _assert_cleanup(interrupted_output, worktree)

    interrupted = _validated_checkpoint(checkpoint_path)
    assert interrupted.saved["finished"] is False
    run = interrupted.resume_state
    saved_step = run["simulator"]["step_index"]
    assert saved_step > ready_step
    assert saved_step == int(list(CHECKPOINT_LINE.finditer(interrupted_output))[-1][1])
    assert run["host"]["settle"] == 0
    assert run["host"]["t_start"] is not None
    assert run["host"]["load_events"][0], "no token was injected before the signal"

    with _job(script, home, checkout, env, job_prefix + "-2") as (process, log, worktree):
        _wait_for(process, log, lambda text: CHECKPOINT_LINE.search(text), timeout=30)
        resumed = _validated_checkpoint(checkpoint_path)
        assert resumed.saved["finished"] is False
        assert resumed.resume_state["simulator"]["step_index"] >= saved_step
        # The runner has no resume log. Its original start time and injected
        # event records survive only when the saved host state is restored;
        # mere checkpoint existence or a successful fresh run cannot pass this.
        resumed_host = resumed.resume_state["host"]
        assert resumed_host["t_start"] == run["host"]["t_start"]
        events = run["host"]["load_events"][0]
        assert resumed_host["load_events"][0][:len(events)] == events
        assert _wait_exit(process, log, timeout=60) == 0, log.read_text()
        resumed_output = log.read_text()
        assert f"e2e python cwd: {worktree}\n" in resumed_output
        _assert_cleanup(resumed_output, worktree)

    record = json.loads(result_path.read_text())
    finished = _validated_checkpoint(checkpoint_path)
    assert finished.saved["finished"] is True
    assert finished.resume_state is None
    # Confidence limits can be NaN when all outputs are missing; compare their
    # canonical JSON encoding rather than Python's NaN != NaN semantics.
    assert json.dumps(finished.saved["record"], sort_keys=True) == json.dumps(record, sort_keys=True)
    assert record["checkpoint_fingerprint"] == interrupted.fingerprint
    assert record["block"] == "fanout" and record["simulator"] == "TorchSim"
    assert record["tokens_list"] == [0, 5, 9, 14, 14, 200, 77, 3]
    assert record["reference"] == [[((v + 1) & 7) + ((v + 1) ^ 3)]
                                    for v in record["tokens_list"]]
    assert record["copies"] == 1 and record["outputs_expected"] == 8
    assert record["ok"] + record["wrong"] + record["missing"] == 8
    assert record["per_node"] == [[record["ok"], record["wrong"], record["missing"]]]
    assert len(record["outputs"]) == 1 and set(record["outputs"][0]) == {"c4"}
    # Reaching the neural ceiling is a completed campaign, including classified
    # missing outputs; this test does not require mix B to produce perfect logic.
    assert saved_step * 0.1 / 1000 < record["neural_s"] <= 2.2
