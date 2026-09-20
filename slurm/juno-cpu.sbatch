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
# The commit is checked out when the job STARTS, not when it was submitted: several queued jobs
# can each run their own commit (submit.sh passes it as DROSO_SHA).
if [ -n "${DROSO_SHA:-}" ]; then git checkout -q --detach "$DROSO_SHA"; fi
source ~/miniforge3/etc/profile.d/conda.sh
conda activate drosophilos
echo "job $SLURM_JOB_ID on $(hostname), commit $(git rev-parse --short HEAD), $(date -Is)"
exec python -m "$@"
