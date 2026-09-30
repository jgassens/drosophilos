import os
from pathlib import Path
import shutil
import subprocess
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

    def run(behavior, *options, combined=False):
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
        return subprocess.run(
            ["/bin/bash", str(script), *options, "--", MODULE, "--device", "cuda"],
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
    assert "dry run: " + repo.env["DROSO_SSH"] + " juno cd ~/drosophilos" in result.stdout
    assert "&& echo 'droso-submit: sbatch' && sbatch" in result.stdout
    assert "slurm/juno-cpu.sbatch" in result.stdout


@pytest.mark.parametrize("options", [[], ["--time=1:00:00", "--mem=8G"]])
def test_remote_command_contains_marker_before_sbatch(repo, options):
    result = repo.run("echo 'droso-submit: sbatch'\necho 'Submitted batch job 123'", *options)
    assert result.returncode == 0
    cluster, remote = repo.log.read_text().splitlines()
    assert cluster == "juno"
    assert "git fetch -q origin" in remote
    assert f"git checkout -q --detach {repo.head}" in remote
    assert "mkdir -p runs && echo 'droso-submit: sbatch' && sbatch " in remote
    assert f"--export=ALL,DROSO_SHA={repo.head}" in remote
    assert f"slurm/juno.sbatch {MODULE} --device cuda" in remote
    for option in options:
        assert option in remote


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


@pytest.mark.parametrize("script", sorted(SLURM.glob("*.sbatch")) + [SLURM / "submit.sh"])
def test_bash_syntax(script):
    subprocess.run(["/bin/bash", "-n", str(script)], check=True, capture_output=True)
