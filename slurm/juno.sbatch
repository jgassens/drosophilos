#!/bin/bash
# Generic Juno (juno.hpcre.utdallas.edu) job: runs one `python -m drosophilos.<module> ...` in the
# drosophilos conda env from ~/drosophilos. Submit through slurm/submit.sh, which pushes the
# current commit, pulls it on Juno, and passes the module and its arguments here.
#
#   sbatch slurm/juno.sbatch drosophilos.bench.perf_campaign --device cuda ...
#
#SBATCH --job-name=droso
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
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
  # Retry the transient cluster filesystem getcwd failure seen on 2026-09-30.
  for attempt in 1 2 3; do
    if cd ~/drosophilos && git worktree add -q --detach "$wt" "$DROSO_SHA"; then break; fi
    if [ "$attempt" -eq 3 ]; then
      echo "job $SLURM_JOB_ID: could not create worktree for $DROSO_SHA" >&2
      exit 1
    fi
    rm -rf "$wt" || true
    cd ~/drosophilos && git worktree prune || true
    sleep 10
  done
  trap 'cd ~/drosophilos && git worktree remove --force "$wt"' EXIT
  ln -s ~/drosophilos/data "$wt/data"
  ln -s ~/drosophilos/runs "$wt/runs"
  cd "$wt"
fi
source ~/miniforge3/etc/profile.d/conda.sh
conda activate drosophilos
echo "job $SLURM_JOB_ID on $(hostname), commit $(git rev-parse --short HEAD), $(date -Is)"
nvidia-smi -L || true
exec python -m "$@"
