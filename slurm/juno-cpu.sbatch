#!/bin/bash
# Juno CPU job (dev partition, 2 h limit, no GPU): the test suite, compiles, RefSim work.
# Submit through slurm/submit.sh --script juno-cpu -- <module> [args]  (e.g. pytest -q tests).
#
#SBATCH --job-name=droso-cpu
#SBATCH --partition=dev
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=runs/slurm-%j.out
set -euo pipefail
cd ~/drosophilos
# Each job runs its own commit (DROSO_SHA from submit.sh) in its own git worktree, created when
# the job STARTS and removed when it ends, so queued and concurrent jobs never share a checkout
# (2026-09-20: a job's checkout collided with a later submission's and a test job ran the wrong
# commit). data/ and runs/ are symlinks back into ~/drosophilos, so --out data/... lands there;
# an --out under any other directory is lost with the worktree.
if [ -n "${DROSO_SHA:-}" ]; then
  wt=~/drosophilos-jobs/$SLURM_JOB_ID
  mkdir -p ~/drosophilos-jobs
  git worktree add -q --detach "$wt" "$DROSO_SHA"
  trap 'cd ~/drosophilos && git worktree remove --force "$wt"' EXIT
  ln -s ~/drosophilos/data "$wt/data"
  ln -s ~/drosophilos/runs "$wt/runs"
  cd "$wt"
fi
source ~/miniforge3/etc/profile.d/conda.sh
conda activate drosophilos
echo "job $SLURM_JOB_ID on $(hostname), commit $(git rev-parse --short HEAD), $(date -Is)"
exec python -m "$@"
