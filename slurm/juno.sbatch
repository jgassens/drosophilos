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
# The commit is checked out when the job STARTS, not when it was submitted: several queued jobs
# can each run their own commit (submit.sh passes it as DROSO_SHA).
if [ -n "${DROSO_SHA:-}" ]; then git checkout -q --detach "$DROSO_SHA"; fi
source ~/miniforge3/etc/profile.d/conda.sh
conda activate drosophilos
echo "job $SLURM_JOB_ID on $(hostname), commit $(git rev-parse --short HEAD), $(date -Is)"
nvidia-smi -L || true
exec python -m "$@"
