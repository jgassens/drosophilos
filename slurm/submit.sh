#!/bin/bash
# Submit a drosophilos module run to a cluster from this laptop (campus VPN must be connected).
#
#   slurm/submit.sh [--cluster juno|g2] [--script juno-cpu] [--ref sha] [sbatch options...] -- drosophilos.<module> [args...]
#   slurm/submit.sh --script juno-cpu -- pytest -q tests        # the test suite on a CPU node (dev partition, 2 h)
#   slurm/submit.sh --time=6:00:00 -- drosophilos.bench.perf_campaign --device cuda --out data/perf/juno-h200
#
# Pushes HEAD to origin, checks it out in ~/drosophilos on the cluster, submits slurm/<cluster>.sbatch,
# and prints the job id. Refuses a dirty tree: the cluster runs exactly the committed code.
# juno (default): partition h200, 26 nodes x 2 H200 NVL, 2-day limit, MCNS data present.
# g2: partition gpu-preempt; pick a GPU with --gres=gpu:nvidia_h100_nvl:1 etc. (sinfo -o "%N %G");
#     nvidia_geforce_rtx_3090 is slow at float64. Queue waits of days are normal there.
set -euo pipefail
cd "$(dirname "$0")/.."
cluster=juno; script=""; ref=HEAD
while :; do
  case "${1:-}" in
    --cluster) cluster=$2; shift 2;;   # juno (default) | g2
    --script) script=$2; shift 2;;     # slurm/<script>.sbatch instead of slurm/<cluster>.sbatch (juno-cpu: dev partition, no GPU)
    --ref) ref=$2; shift 2;;           # run this commit instead of HEAD (an earlier build for a comparison)
    *) break;;
  esac
done
script=${script:-$cluster}
if [ -n "$(git status --porcelain)" ]; then echo "working tree is dirty; commit first" >&2; exit 1; fi
opts=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do opts+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift
[ $# -gt 0 ] || { echo "usage: $0 [--cluster c] [--script s] [--ref sha] [sbatch options] -- drosophilos.module [args]" >&2; exit 1; }
sha=$(git rev-parse "$ref")
git push -q origin HEAD
git merge-base --is-ancestor "$sha" HEAD || { echo "$ref is not an ancestor of HEAD; push it first" >&2; exit 1; }
ssh "$cluster" "cd ~/drosophilos && git fetch -q origin && git checkout -q --detach $sha && mkdir -p runs && sbatch ${opts[*]} slurm/$script.sbatch $(printf '%q ' "$@")" 2>&1 | grep -v "post-quantum\|store now\|openssh.com"
