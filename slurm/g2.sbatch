#!/bin/bash
# Generic G2 (ganymede2.utdallas.edu) job: runs one `python -m drosophilos.<module> ...` in the
# drosophilos conda env from ~/drosophilos. Submit through slurm/submit.sh, which pushes the
# current commit, pulls it on G2, and passes the module and its arguments here.
#
#   sbatch [--gres=gpu:nvidia_h200_nvl:1] slurm/g2.sbatch drosophilos.bench.perf_campaign --device cuda ...
#
#SBATCH --job-name=droso
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=runs/slurm-%j.out
set -euo pipefail
cd ~/drosophilos
source ~/miniforge3/etc/profile.d/conda.sh
conda activate drosophilos
echo "job $SLURM_JOB_ID on $(hostname), commit $(git rev-parse --short HEAD), $(date -Is)"
nvidia-smi -L || true
exec python -m "$@"
