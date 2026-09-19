#!/bin/bash
# Submit a drosophilos module run to G2 from this laptop.
#
#   slurm/submit.sh [sbatch options...] -- drosophilos.<module> [args...]
#   slurm/submit.sh --gres=gpu:nvidia_h200_nvl:1 --time=6:00:00 -- drosophilos.bench.perf_campaign --device cuda --out data/perf/g2-h200
#
# Pushes HEAD to origin, fast-forwards ~/drosophilos on G2 to it, submits slurm/g2.sbatch, and
# prints the job id. Refuses a dirty tree: the cluster runs exactly the committed code.
# GPU choices on gpu-preempt (sinfo -o "%N %G"): nvidia_h200_nvl (g-08-01, 4), nvidia_h100_nvl,
# nvidia_a100-sxm4-80gb, nvidia_geforce_rtx_3090 (slow at float64; use --dtype float32).
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -n "$(git status --porcelain)" ]; then echo "working tree is dirty; commit first" >&2; exit 1; fi
opts=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do opts+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift
[ $# -gt 0 ] || { echo "usage: $0 [sbatch options] -- drosophilos.module [args]" >&2; exit 1; }
sha=$(git rev-parse HEAD)
git push -q origin HEAD
ssh g2 "cd ~/drosophilos && git fetch -q origin && git checkout -q --detach $sha && mkdir -p runs && sbatch ${opts[*]} slurm/g2.sbatch $*" 2>&1 | grep -v "post-quantum\|store now\|openssh.com"
