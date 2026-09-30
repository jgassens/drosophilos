#!/bin/bash
# Submit a drosophilos module run to a cluster from this laptop (campus VPN must be connected).
#
#   slurm/submit.sh [--dry-run] [--cluster juno|g2] [--script juno-cpu] [--ref sha] [sbatch options...] -- drosophilos.<module> [args...]
#   slurm/submit.sh --script juno-cpu -- pytest -q tests        # the test suite on a CPU node (dev partition, 2 h)
#   slurm/submit.sh --time=6:00:00 -- drosophilos.bench.perf_campaign --device cuda --out data/perf/juno-h200
#
# Pushes HEAD to origin, fetches it on the cluster, submits slurm/<cluster>.sbatch and prints the job
# id; the job checks its commit out in a worktree of its own when it starts. Refuses a dirty tree: the cluster runs exactly the committed code.
# --dry-run checks locally and prints the push and remote command without running them; DROSO_SSH overrides ssh.
# DROSO_SUBMIT_RETRY_SLEEP: seconds between safe retries before sbatch runs (default 10).
# DROSO_SUBMIT_ATTEMPTS: maximum remote attempts before sbatch runs (default 3).
# juno (default): partition h200, 26 nodes x 2 H200 NVL, 2-day limit, MCNS data present.
# g2: partition gpu-preempt; pick a GPU with --gres=gpu:nvidia_h100_nvl:1 etc. (sinfo -o "%N %G");
#     nvidia_geforce_rtx_3090 is slow at float64. Queue waits of days are normal there.
set -euo pipefail
cd "$(dirname "$0")/.."
cluster=juno; script=""; ref=HEAD; dry_run=0
ssh_cmd=${DROSO_SSH:-ssh}
retry_sleep=${DROSO_SUBMIT_RETRY_SLEEP:-10}
attempts=${DROSO_SUBMIT_ATTEMPTS:-3}
while :; do
  case "${1:-}" in
    --cluster) cluster=$2; shift 2;;   # juno (default) | g2
    --script) script=$2; shift 2;;     # slurm/<script>.sbatch instead of slurm/<cluster>.sbatch (juno-cpu: dev partition, no GPU)
    --ref) ref=$2; shift 2;;           # run this commit instead of HEAD (an earlier build for a comparison)
    --dry-run) dry_run=1; shift;;      # check locally and print the commands without pushing or submitting
    *) break;;
  esac
done
script=${script:-$cluster}
opts=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do opts+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift
job="$cluster slurm/$script.sbatch -- $*"
not_submitted() { echo "submit.sh: NOT SUBMITTED: $job" >&2; exit 1; }
[ $# -gt 0 ] || { echo "usage: $0 [--dry-run] [--cluster c] [--script s] [--ref sha] [sbatch options] -- drosophilos.module [args]" >&2; not_submitted; }
status=$(git status --porcelain) || not_submitted
if [ -n "$status" ]; then echo "working tree is dirty; commit first" >&2; not_submitted; fi
sha=$(git rev-parse --verify "$ref^{commit}") || not_submitted
git merge-base --is-ancestor "$sha" HEAD || { echo "$ref is not an ancestor of HEAD; push it first" >&2; not_submitted; }
head=$(git rev-parse HEAD) || not_submitted
# the main checkout follows HEAD only to supply slurm/*.sbatch; the job itself runs $sha in its own worktree
remote="cd ~/drosophilos && git fetch -q origin && git checkout -q --detach $head && mkdir -p runs && echo 'droso-submit: sbatch' && sbatch --export=ALL,DROSO_SHA=$sha ${opts[@]+"${opts[@]}"} slurm/$script.sbatch $(printf '%q ' "$@")"
if [ "$dry_run" -eq 1 ]; then
  echo "dry run: would push HEAD ($head) to origin"
  echo "dry run: $ssh_cmd $cluster $remote"
  exit 0
fi
if ! git push -q origin HEAD; then echo "submit.sh: git push failed" >&2; not_submitted; fi
if ! [[ "$attempts" =~ ^[1-9][0-9]*$ ]]; then
  echo "submit.sh: DROSO_SUBMIT_ATTEMPTS must be a positive integer" >&2; not_submitted
fi
for ((attempt=1; attempt<=attempts; attempt++)); do
  if output=$("$ssh_cmd" "$cluster" "$remote" 2>&1 | grep -v "post-quantum\|store now\|openssh.com"); then
    remote_status=0
  else
    remote_status=$?
  fi
  if submitted=$(printf '%s\n' "$output" | grep -E '^Submitted batch job [0-9]+$'); then
    printf '%s\n' "$output" | grep -v -E '^(droso-submit: sbatch|Submitted batch job [0-9]+)$' || true
    printf '%s\n' "$submitted"
    exit 0
  fi
  [ -z "$output" ] || printf '%s\n' "$output" >&2
  if printf '%s\n' "$output" | grep '^droso-submit: sbatch$' >/dev/null; then
    echo "submit.sh: sbatch started but submission was not confirmed; check squeue on $cluster before resubmitting" >&2
    echo "submit.sh: NOT CONFIRMED (check squeue): $job" >&2
    [ "$remote_status" -ne 0 ] || remote_status=1
    exit "$remote_status"
  fi
  if [ "$attempt" -lt "$attempts" ]; then
    echo "submit.sh: cluster step failed before sbatch ran (attempt $attempt/$attempts); retrying in $retry_sleep s" >&2
    sleep "$retry_sleep" || not_submitted
  fi
done
not_submitted
