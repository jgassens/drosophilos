from contextlib import contextmanager
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest


SLURM = Path(__file__).resolve().parents[1] / "slurm"
MODULE = "drosophilos.bench.perf_campaign"
MARKER = "droso-submit: sbatch"


def git(path, *args, env=None, check=True):
    return subprocess.run(
        ["git", *args], cwd=path, env=env, text=True, capture_output=True, check=check
    )


@pytest.fixture
def repo(tmp_path):
    env = os.environ.copy()
    env.update(GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_NOSYSTEM="1")
    root = tmp_path / "repo"
    root.mkdir()
    origin = tmp_path / "origin.git"
    git(tmp_path, "init", "--bare", str(origin), env=env)
    git(root, "init", env=env)
    git(root, "config", "user.name", "Submit test", env=env)
    git(root, "config", "user.email", "submit@example.invalid", env=env)
    git(root, "config", "commit.gpgsign", "false", env=env)
    git(root, "remote", "add", "origin", str(origin), env=env)
    (root / "slurm").mkdir()
    script = root / "slurm" / "submit.sh"
    shutil.copyfile(SLURM / "submit.sh", script)
    script.chmod(0o755)
    git(root, "add", "slurm/submit.sh", env=env)
    git(root, "commit", "-m", "Add submit script", env=env)
    head = git(root, "rev-parse", "HEAD", env=env).stdout.strip()
    log = tmp_path / "ssh.log"
    counter = tmp_path / "ssh.count"
    fake = tmp_path / "fake-ssh"
    env.update(
        DROSO_SSH=str(fake),
        DROSO_SUBMIT_RETRY_SLEEP="0",
        DROSO_SUBMIT_ATTEMPTS="3",
        FAKE_SSH_LOG=str(log),
        FAKE_SSH_COUNTER=str(counter),
    )

    def run(behavior, *options, combined=False, module_args=None):
        fake.write_text(
            '#!/bin/bash\nset -eu\n'
            'printf "%s\\n" "$@" >> "$FAKE_SSH_LOG"\n'
            'count=0\n'
            'if [ -f "$FAKE_SSH_COUNTER" ]; then read -r count < "$FAKE_SSH_COUNTER"; fi\n'
            'count=$((count + 1))\n'
            'printf "%s\\n" "$count" > "$FAKE_SSH_COUNTER"\n'
            + behavior + "\n"
        )
        fake.chmod(0o755)
        if module_args is None:
            module_args = ["--device", "cuda"]
        return subprocess.run(
            ["/bin/bash", str(script), *options, "--", MODULE, *module_args],
            cwd=root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT if combined else subprocess.PIPE,
        )

    def calls():
        return int(counter.read_text()) if counter.exists() else 0

    return SimpleNamespace(
        root=root, origin=origin, env=env, head=head, log=log, run=run, calls=calls
    )


def test_first_attempt_success(repo):
    result = repo.run(
        "echo 'post-quantum noise openssh.com' >&2\n"
        "echo 'droso-submit: sbatch'\necho 'Submitted batch job 123'"
    )
    assert result.returncode == 0
    assert repo.calls() == 1
    assert result.stdout.splitlines()[-1] == "Submitted batch job 123"
    assert MARKER not in result.stdout
    assert "post-quantum" not in result.stdout + result.stderr


def test_transient_failure_retries(repo):
    result = repo.run(
        'if [ "$count" -eq 1 ]; then\n'
        "  echo 'fatal: Unable to read current working directory: No such file or directory' >&2\n"
        '  exit 128\nfi\n'
        "echo 'droso-submit: sbatch'\necho 'Submitted batch job 124'"
    )
    assert result.returncode == 0
    assert repo.calls() == 2
    assert result.stdout.splitlines()[-1] == "Submitted batch job 124"
    assert (result.stdout + result.stderr).count("Submitted batch job") == 1
    assert "Unable to read current working directory" in result.stderr
    assert "cluster step failed before sbatch ran (attempt 1/3); retrying in 0 s" in result.stderr


def test_attempts_exhausted(repo):
    repo.env["DROSO_SUBMIT_ATTEMPTS"] = "3"
    result = repo.run("echo 'filesystem unavailable' >&2\nexit 128", combined=True)
    assert result.returncode != 0
    assert repo.calls() == 3
    assert result.stdout.splitlines()[-1] == (
        f"submit.sh: NOT SUBMITTED: juno slurm/juno.sbatch -- {MODULE} --device cuda"
    )


@pytest.mark.parametrize("exit_status", [0, 1])
def test_unconfirmed_submission_never_retries(repo, exit_status):
    result = repo.run(
        f"echo 'droso-submit: sbatch'\necho 'submission interrupted' >&2\nexit {exit_status}",
        combined=True,
    )
    assert result.returncode != 0
    assert repo.calls() == 1
    assert "check squeue on juno before resubmitting" in result.stdout
    assert result.stdout.splitlines()[-1] == (
        f"submit.sh: NOT CONFIRMED (check squeue): juno slurm/juno.sbatch -- {MODULE} --device cuda"
    )


def test_dry_run_does_not_push_or_ssh(repo):
    result = repo.run("exit 99", "--dry-run", "--script", "juno-cpu")
    assert result.returncode == 0
    assert repo.calls() == 0
    assert not repo.log.exists()
    assert git(repo.origin, "show-ref", env=repo.env, check=False).returncode == 1
    assert f"dry run: would push HEAD ({repo.head}) to origin" in result.stdout
    assert "dry run: " + repo.env["DROSO_SSH"] + " -o ConnectTimeout=20 juno cd ~/drosophilos" in result.stdout
    assert "&& echo 'droso-submit: sbatch' && sbatch" in result.stdout
    assert "slurm/juno-cpu.sbatch" in result.stdout


@pytest.mark.parametrize("options", [[], ["--time=1:00:00", "--mem=8G"]])
def test_remote_command_contains_marker_before_sbatch(repo, options):
    result = repo.run("echo 'droso-submit: sbatch'\necho 'Submitted batch job 123'", *options)
    assert result.returncode == 0
    ssh_args = repo.log.read_text().splitlines()
    cluster, remote = ssh_args[-2:]
    assert ssh_args[:-2] == ["-o", "ConnectTimeout=20"]
    assert "ServerAliveInterval" not in " ".join(ssh_args)
    assert cluster == "juno"
    assert "git fetch -q origin" in remote
    assert f"git checkout -q --detach {repo.head}" in remote
    assert "mkdir -p runs && echo 'droso-submit: sbatch' && sbatch " in remote
    assert f"--export=ALL,DROSO_SHA={repo.head}" in shlex.split(remote)
    assert f"slurm/juno.sbatch {MODULE} --device cuda" in remote
    for option in options:
        assert option in remote


@pytest.mark.parametrize("script", ["juno", "juno cpu (test)"])
@pytest.mark.parametrize("options", [[], ["--job-name=my job"]])
def test_remote_command_quotes_arguments(repo, script, options):
    result = repo.run(
        "echo 'droso-submit: sbatch'\necho 'Submitted batch job 123'",
        "--script", script, *options,
        module_args=["--label", "a sample (trial)"],
    )
    assert result.returncode == 0
    remote = repo.log.read_text().splitlines()[-1]
    if options:
        assert r"--job-name=my\ job" in remote
    assert r"a\ sample\ \(trial\)" in remote
    home = repo.root.parent / "remote-home"
    (home / "drosophilos").mkdir(parents=True)
    harness = (
        "git() { :; }\n"
        "sbatch() { printf 'sbatch arg: %s\\n' \"$@\"; }\n"
        + remote
    )
    evaluated = subprocess.run(
        ["/bin/bash", "-c", harness],
        env={**repo.env, "HOME": str(home)},
        text=True, capture_output=True,
    )
    assert evaluated.returncode == 0, evaluated.stderr
    assert evaluated.stdout.splitlines() == [
        MARKER,
        f"sbatch arg: --export=ALL,DROSO_SHA={repo.head}",
        *(f"sbatch arg: {option}" for option in options),
        f"sbatch arg: slurm/{script}.sbatch",
        f"sbatch arg: {MODULE}",
        "sbatch arg: --label",
        "sbatch arg: a sample (trial)",
    ]


@pytest.mark.parametrize("setting, value, message", [
    ("DROSO_SUBMIT_ATTEMPTS", "0", "positive integer"),
    ("DROSO_SUBMIT_ATTEMPTS", "-1", "positive integer"),
    ("DROSO_SUBMIT_ATTEMPTS", "invalid", "positive integer"),
    ("DROSO_SUBMIT_ATTEMPTS", "1.5", "positive integer"),
    ("DROSO_SUBMIT_RETRY_SLEEP", "-1", "non-negative integer"),
    ("DROSO_SUBMIT_RETRY_SLEEP", "invalid", "non-negative integer"),
    ("DROSO_SUBMIT_RETRY_SLEEP", "0.5", "non-negative integer"),
])
def test_invalid_retry_settings_do_not_push_or_ssh(repo, setting, value, message):
    repo.env[setting] = value
    result = repo.run("exit 99", combined=True)
    assert result.returncode != 0
    assert repo.calls() == 0
    assert git(repo.origin, "show-ref", env=repo.env, check=False).returncode == 1
    assert f"{setting} must be a {message}" in result.stdout
    assert result.stdout.splitlines()[-1] == (
        f"submit.sh: NOT SUBMITTED: juno slurm/juno.sbatch -- {MODULE} --device cuda"
    )


def test_marker_with_large_output_never_retries(repo):
    result = repo.run(
        "echo 'droso-submit: sbatch'\nprintf '%070000d\\n' 0 >&2\nexit 1",
        combined=True,
    )
    assert result.returncode != 0
    assert repo.calls() == 1
    assert "NOT CONFIRMED" in result.stdout.splitlines()[-1]


def test_push_failure_has_final_summary(repo):
    hook = repo.origin / "hooks" / "pre-receive"
    hook.write_text("#!/bin/bash\necho 'push rejected for test' >&2\nexit 1\n")
    hook.chmod(0o755)
    result = repo.run("exit 99", combined=True)
    assert result.returncode != 0
    assert repo.calls() == 0
    assert "git push failed" in result.stdout
    assert result.stdout.splitlines()[-1] == (
        f"submit.sh: NOT SUBMITTED: juno slurm/juno.sbatch -- {MODULE} --device cuda"
    )


def test_empty_ssh_output_retries(repo):
    result = repo.run("exit 255", combined=True)
    assert result.returncode != 0
    assert repo.calls() == 3
    assert "NOT SUBMITTED" in result.stdout.splitlines()[-1]


def test_job_id_confirms_success_even_if_ssh_fails(repo):
    result = repo.run(
        "echo 'droso-submit: sbatch'\necho 'Submitted batch job 123'\n"
        "echo 'connection closed' >&2\nexit 255"
    )
    assert result.returncode == 0
    assert repo.calls() == 1
    assert result.stdout.splitlines()[-1] == "Submitted batch job 123"
    assert "connection closed" in result.stdout


SBATCH_SCRIPTS = ["juno.sbatch", "g2.sbatch", "juno-cpu.sbatch"]


@pytest.fixture
def sbatch_exit_job(tmp_path):
    home = tmp_path / "home"
    checkout = home / "drosophilos"
    checkout.mkdir(parents=True)
    env = os.environ.copy()
    env.update(GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_NOSYSTEM="1")
    git(checkout, "init", env=env)
    git(checkout, "config", "user.name", "Sbatch test", env=env)
    git(checkout, "config", "user.email", "sbatch@example.invalid", env=env)
    (checkout / "data").mkdir()
    (checkout / "runs").mkdir()
    (checkout / "tracked").write_text("test\n")
    git(checkout, "add", ".", env=env)
    git(checkout, "commit", "-m", "test", env=env)
    conda = home / "miniforge3/etc/profile.d/conda.sh"
    conda.parent.mkdir(parents=True)
    conda.write_text("conda() { :; }\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_git = bin_dir / "git"
    fake_git.write_text(
        "#!/bin/bash\n"
        'if [ "$1" = worktree ] && [ "$2" = add ]; then printf "%s\\n" "$*" >> "$FAKE_GIT_ADD_LOG"; fi\n'
        'if [ "${FAIL_WORKTREE_REMOVE:-0}" = 1 ] && [ "$1" = worktree ] && [ "$2" = remove ]; then exit 1; fi\n'
        f'exec "{shutil.which("git")}" "$@"\n'
    )
    fake_git.chmod(0o755)
    fake_python = bin_dir / "python"
    fake_python.write_text('#!/bin/bash\nexit "${FAKE_PYTHON_STATUS:-0}"\n')
    fake_python.chmod(0o755)
    fake_sleep = bin_dir / "sleep"
    fake_sleep.write_text("#!/bin/bash\nexit 0\n")
    fake_sleep.chmod(0o755)
    fake_nvidia = bin_dir / "nvidia-smi"
    fake_nvidia.write_text("#!/bin/bash\nexit 0\n")
    fake_nvidia.chmod(0o755)
    env.update(
        HOME=str(home),
        PATH=f"{bin_dir}:{env['PATH']}",
        DROSO_SHA=git(checkout, "rev-parse", "HEAD", env=env).stdout.strip(),
        SLURM_JOB_ID="987654",
        FAKE_GIT_ADD_LOG=str(tmp_path / "worktree-add.log"),
    )
    return SimpleNamespace(home=home, checkout=checkout, bin_dir=bin_dir, env=env)


def run_sbatch_job(sbatch_exit_job, script, *, python_status=0, fail_remove=False):
    env = {
        **sbatch_exit_job.env,
        "FAKE_PYTHON_STATUS": str(python_status),
        "FAIL_WORKTREE_REMOVE": "1" if fail_remove else "0",
    }
    result = subprocess.run(
        ["/bin/bash", str(SLURM / script), "example.module"],
        cwd=sbatch_exit_job.checkout,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )
    return result, sbatch_exit_job.home / "drosophilos-jobs" / env["SLURM_JOB_ID"]


def wait_for_sbatch_file(job, path, expected=None):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if path.exists():
            content = path.read_text()
            if content and (expected is None or content == expected):
                return content
        assert job.poll() is None, f"sbatch exited {job.returncode} waiting for {path}"
        time.sleep(0.01)
    pytest.fail(f"timed out waiting for {path}")


@pytest.fixture
def sbatch_signal_job(sbatch_exit_job):
    @contextmanager
    def launch(script, setup, body, *, python_status=0):
        started = sbatch_exit_job.home / "python-started"
        finish = sbatch_exit_job.home / "python-finish"
        fake_python = sbatch_exit_job.bin_dir / "python"
        # Use the real interpreter and time.sleep, never the harness's fake sleep.
        fake_python.write_text(
            f"#!{sys.executable}\n"
            "import os, signal, time\n"
            "from pathlib import Path\n"
            "status = int(os.environ['FAKE_PYTHON_STATUS'])\n"
            + setup + "\n"
            "Path(os.environ['FAKE_PYTHON_STARTED']).write_text(str(os.getpid()))\n"
            + body + "\n"
        )
        env = {
            **sbatch_exit_job.env,
            "FAKE_PYTHON_STATUS": str(python_status),
            "FAKE_PYTHON_STARTED": str(started),
            "FAKE_PYTHON_FINISH": str(finish),
        }
        wt = sbatch_exit_job.home / "drosophilos-jobs" / env["SLURM_JOB_ID"]
        job = subprocess.Popen(
            ["/bin/bash", str(SLURM / script), "example.module"],
            cwd=sbatch_exit_job.checkout, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
        )
        try:
            python_pid = int(wait_for_sbatch_file(job, started))
            yield SimpleNamespace(
                job=job, python_pid=python_pid, wt=wt, finish=finish,
                data=sbatch_exit_job.checkout / "data",
            )
        finally:
            try:
                os.killpg(job.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            job.communicate(timeout=5)

    return launch


@pytest.mark.parametrize("script",SBATCH_SCRIPTS)
def test_sbatch_cleanup_failure_preserves_success(sbatch_exit_job, script):
    result, wt = run_sbatch_job(sbatch_exit_job, script, fail_remove=True)
    assert result.returncode == 0, result.stderr
    assert wt.is_dir()


@pytest.mark.parametrize("script",SBATCH_SCRIPTS)
def test_sbatch_python_failure_is_preserved_and_worktree_removed(sbatch_exit_job, script):
    result, wt = run_sbatch_job(sbatch_exit_job, script, python_status=3)
    assert result.returncode == 3
    assert not wt.exists()


@pytest.mark.parametrize("script", SBATCH_SCRIPTS)
@pytest.mark.parametrize("python_status", [0, 7])
def test_sbatch_sigterm_waits_for_python_final_write(sbatch_signal_job, script, python_status):
    setup = (
        "def finish(signum, frame):\n"
        "    with open('data/signals.txt', 'a') as log:\n"
        "        log.write('TERM\\n')\n"
        "    while not Path(os.environ['FAKE_PYTHON_FINISH']).exists(): time.sleep(0.01)\n"
        "    Path('data/final.txt').write_text('final\\n')\n"
        "    raise SystemExit(status)\n"
        "signal.signal(signal.SIGTERM, finish)\n"
    )
    with sbatch_signal_job(
        script, setup, "while True: time.sleep(0.01)", python_status=python_status,
    ) as run:
        # Signal the shell alone first; Slurm delivers Python's signal separately.
        os.kill(run.job.pid, signal.SIGTERM)
        os.kill(run.python_pid, signal.SIGTERM)
        wait_for_sbatch_file(run.job, run.data / "signals.txt", "TERM\n")
        assert run.job.poll() is None
        assert run.wt.is_dir()
        run.finish.touch()
        stdout, stderr = run.job.communicate(timeout=10)
        assert run.job.returncode == python_status, stdout + stderr
        assert (run.data / "signals.txt").read_text() == "TERM\n"
        assert (run.data / "final.txt").read_text() == "final\n"
        assert not run.wt.exists()


@pytest.mark.parametrize("script", SBATCH_SCRIPTS)
def test_sbatch_usr1_checkpoints_before_worktree_cleanup(sbatch_signal_job, script):
    setup = (
        "def checkpoint(signum, frame):\n"
        "    Path('data/checkpoint-started.txt').write_text('saving\\n')\n"
        "    while not Path(os.environ['FAKE_PYTHON_FINISH']).exists(): time.sleep(0.01)\n"
        "    Path('data/checkpoint.txt').write_text('checkpoint\\n')\n"
        "    raise SystemExit(75)\n"
        "signal.signal(signal.SIGUSR1, checkpoint)\n"
    )
    with sbatch_signal_job(script, setup, "while True: time.sleep(0.01)") as run:
        # B:USR1 signals only the batch shell, never the process group.
        os.kill(run.job.pid, signal.SIGUSR1)
        wait_for_sbatch_file(run.job, run.data / "checkpoint-started.txt")
        assert run.job.poll() is None
        assert run.wt.is_dir()
        run.finish.touch()
        stdout, stderr = run.job.communicate(timeout=10)
        assert run.job.returncode == 75, stdout + stderr
        assert (run.data / "checkpoint.txt").read_text() == "checkpoint\n"
        assert not run.wt.exists()


@pytest.mark.parametrize("script", SBATCH_SCRIPTS)
@pytest.mark.parametrize("shell_signal", [signal.SIGTERM, signal.SIGHUP, signal.SIGINT])
def test_sbatch_shell_signal_keeps_waiting(sbatch_signal_job, script, shell_signal):
    setup = (
        "def unexpected_signal(signum, frame):\n"
        "    Path('data/unexpected-signal.txt').write_text(str(signum))\n"
        "    raise SystemExit(99)\n"
        "for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):\n"
        "    signal.signal(sig, unexpected_signal)\n"
    )
    body = (
        "while not Path(os.environ['FAKE_PYTHON_FINISH']).exists(): time.sleep(0.01)\n"
        "Path('data/final.txt').write_text('final\\n')\n"
    )
    with sbatch_signal_job(script, setup, body) as run:
        os.kill(run.job.pid, shell_signal)
        time.sleep(0.1)
        assert run.job.poll() is None
        assert run.wt.is_dir()
        run.finish.touch()
        stdout, stderr = run.job.communicate(timeout=10)
        assert run.job.returncode == 0, stdout + stderr
        assert (run.data / "final.txt").read_text() == "final\n"
        assert not (run.data / "unexpected-signal.txt").exists()
        assert not run.wt.exists()


@pytest.mark.parametrize("script", SBATCH_SCRIPTS)
def test_sbatch_forwards_each_usr_signal(sbatch_signal_job, script):
    setup = (
        "def record(signum, frame):\n"
        "    with open('data/signals.txt', 'a') as log:\n"
        "        log.write(signal.Signals(signum).name + '\\n')\n"
        "signal.signal(signal.SIGUSR1, record)\n"
        "signal.signal(signal.SIGUSR2, record)\n"
    )
    body = (
        "while not Path(os.environ['FAKE_PYTHON_FINISH']).exists(): time.sleep(0.01)\n"
        "raise SystemExit(status)\n"
    )
    with sbatch_signal_job(script, setup, body, python_status=7) as run:
        expected = ""
        for sig in (signal.SIGUSR1, signal.SIGUSR1, signal.SIGUSR2):
            os.kill(run.job.pid, sig)
            expected += sig.name + "\n"
            # Acknowledge each signal so standard signals cannot coalesce.
            wait_for_sbatch_file(run.job, run.data / "signals.txt", expected)
            assert run.wt.is_dir()
        run.finish.touch()
        stdout, stderr = run.job.communicate(timeout=10)
        assert run.job.returncode == 7, stdout + stderr
        assert (run.data / "signals.txt").read_text() == expected
        assert not run.wt.exists()


@pytest.mark.parametrize("script", SBATCH_SCRIPTS)
def test_sbatch_preserves_python_signal_exit_status(sbatch_signal_job, script):
    with sbatch_signal_job(script, "", "while True: time.sleep(0.01)") as run:
        os.kill(run.python_pid, signal.SIGKILL)
        stdout, stderr = run.job.communicate(timeout=10)
        assert run.job.returncode == 128 + signal.SIGKILL, stdout + stderr
        assert not run.wt.exists()


@pytest.mark.parametrize("script",SBATCH_SCRIPTS)
def test_sbatch_force_clears_stale_worktree_registration(sbatch_exit_job, script):
    stale = sbatch_exit_job.home / "drosophilos-jobs" / "987654"
    stale.parent.mkdir()
    git(sbatch_exit_job.checkout, "worktree", "add", "--detach", str(stale), "HEAD", env=sbatch_exit_job.env)
    shutil.rmtree(stale)
    add_log = Path(sbatch_exit_job.env["FAKE_GIT_ADD_LOG"])
    add_log.unlink()  # Count only the job's attempts, not the stale registration setup.
    result, wt = run_sbatch_job(sbatch_exit_job, script)
    assert result.returncode == 0, result.stderr
    assert len(add_log.read_text().splitlines()) == 1
    assert not wt.exists()
    assert MARKER not in result.stdout


@pytest.mark.parametrize("failure", ["dirty", "invalid_ref", "non_ancestor"])
def test_dry_run_checks_locally(repo, failure):
    options = ["--dry-run"]
    if failure == "dirty":
        (repo.root / "uncommitted").write_text("dirty\n")
    elif failure == "invalid_ref":
        options += ["--ref", "missing-ref"]
    else:
        git(repo.root, "commit", "--allow-empty", "-m", "Other ref", env=repo.env)
        other = git(repo.root, "rev-parse", "HEAD", env=repo.env).stdout.strip()
        git(repo.root, "reset", "--hard", repo.head, env=repo.env)
        options += ["--ref", other]
    result = repo.run("exit 99", *options, combined=True)
    assert result.returncode != 0
    assert repo.calls() == 0
    assert git(repo.origin, "show-ref", env=repo.env, check=False).returncode == 1
    assert "NOT SUBMITTED" in result.stdout.splitlines()[-1]


@pytest.fixture
def sbatch_job(tmp_path):
    real_git = shutil.which("git")
    env = os.environ.copy()
    env.update(GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_NOSYSTEM="1")
    home = tmp_path / "home"
    root = home / "drosophilos"
    root.mkdir(parents=True)
    git(root, "init", env=env)
    git(root, "config", "user.name", "Submit test", env=env)
    git(root, "config", "user.email", "submit@example.invalid", env=env)
    git(root, "config", "commit.gpgsign", "false", env=env)
    (root / "README").write_text("Test worktree\n")
    git(root, "add", "README", env=env)
    git(root, "commit", "-m", "Initial commit", env=env)
    sha = git(root, "rev-parse", "HEAD", env=env).stdout.strip()
    conda = home / "miniforge3/etc/profile.d/conda.sh"
    conda.parent.mkdir(parents=True)
    conda.write_text("conda() { :; }\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    counter = tmp_path / "worktree.count"
    wrappers = {
        "git": (
            "#!/bin/bash\nset -eu\n"
            'if [ "${1:-}" = worktree ] && [ "${2:-}" = add ]; then\n'
            '  count=0\n'
            '  if [ -f "$FAKE_GIT_COUNTER" ]; then read -r count < "$FAKE_GIT_COUNTER"; fi\n'
            '  count=$((count + 1))\n'
            '  printf "%s\\n" "$count" > "$FAKE_GIT_COUNTER"\n'
            '  if [ "$count" -eq 1 ] || [ "$FAKE_GIT_ALWAYS_FAIL" -eq 1 ]; then\n'
            "    echo 'fatal: Unable to read current working directory: No such file or directory' >&2\n"
            '    exit 128\n'
            '  fi\n'
            'fi\n'
            'exec "$REAL_GIT" "$@"\n'
        ),
        "nvidia-smi": "#!/bin/bash\nexit 0\n",
        "sleep": "#!/bin/bash\nexit 0\n",
        "python": (
            "#!/bin/bash\n"
            'printf "python arg: %s\\n" "$@"\n'
            'printf "python cwd: %s\\n" "$PWD"\n'
            'exit 0\n'
        ),
    }
    for name, content in wrappers.items():
        wrapper = bin_dir / name
        wrapper.write_text(content)
        wrapper.chmod(0o755)
    env.update(
        HOME=str(home),
        PATH=str(bin_dir) + os.pathsep + env["PATH"],
        SLURM_JOB_ID="77",
        DROSO_SHA=sha,
        REAL_GIT=real_git,
        FAKE_GIT_COUNTER=str(counter),
        FAKE_GIT_ALWAYS_FAIL="0",
    )

    def run(script):
        return subprocess.run(
            ["/bin/bash", str(script), "drosophilos.dummy"],
            cwd=home, env=env, text=True, capture_output=True,
        )

    return SimpleNamespace(
        home=home, root=root, env=env, counter=counter, run=run,
        wt=home / "drosophilos-jobs/77",
    )


@pytest.mark.parametrize("script", sorted(SLURM.glob("*.sbatch")))
def test_sbatch_worktree_retries_and_cleans_up(sbatch_job, script):
    result = sbatch_job.run(script)
    assert result.returncode == 0, result.stderr
    assert sbatch_job.counter.read_text().strip() == "2"
    assert "Unable to read current working directory" in result.stderr
    assert "python arg: -m\npython arg: drosophilos.dummy\n" in result.stdout
    assert f"python cwd: {sbatch_job.wt}" in result.stdout
    assert not sbatch_job.wt.exists()
    registered = git(
        sbatch_job.root, "worktree", "list", "--porcelain", env=sbatch_job.env
    ).stdout
    assert str(sbatch_job.wt) not in registered


@pytest.mark.parametrize("script", sorted(SLURM.glob("*.sbatch")))
def test_sbatch_worktree_attempts_exhausted(sbatch_job, script):
    sbatch_job.env["FAKE_GIT_ALWAYS_FAIL"] = "1"
    result = sbatch_job.run(script)
    assert result.returncode == 1
    assert sbatch_job.counter.read_text().strip() == "3"
    assert "could not create worktree" in result.stderr
    assert "python arg:" not in result.stdout


@pytest.mark.parametrize("script", sorted(SLURM.glob("*.sbatch")))
def test_sbatch_empty_job_id_preserves_jobs_directory(sbatch_job, script):
    sentinel = sbatch_job.home / "drosophilos-jobs/sentinel"
    sentinel.mkdir(parents=True)
    sbatch_job.env["SLURM_JOB_ID"] = ""
    result = sbatch_job.run(script)
    assert result.returncode != 0
    assert "SLURM_JOB_ID is empty" in result.stderr
    assert sentinel.is_dir()
    assert not sbatch_job.counter.exists()


@pytest.mark.parametrize("script", sorted(SLURM.glob("*.sbatch")) + [SLURM / "submit.sh"])
def test_bash_syntax(script):
    subprocess.run(["/bin/bash", "-n", str(script)], check=True, capture_output=True)
