#!/bin/bash
#SBATCH --job-name=droso-campaign
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
# Usage: sbatch campaign.sbatch <block> [a2_campaigns args...]
#   e.g. sbatch campaign.sbatch machine --n 200 --batch 100
# Runs one mix-B perturbation campaign of drosophilos.bench.a2_campaigns on a GPU.
set -euo pipefail
module load miniconda/24.11.1 cuda/12.4
# `source activate` is a no-op in a non-interactive batch shell: use the env's python directly
PY="${DROSO_PY:-$HOME/.conda/envs/drosophilos/bin/python}"
cd "${DROSO_ROOT:-$HOME/drosophilos}"
BLOCK="$1"; shift
mkdir -p data/a2
echo "host $(hostname) gpu $(nvidia-smi -L 2>/dev/null | head -1) block $BLOCK args $*"
"$PY" -u -m drosophilos.bench.a2_campaigns "$BLOCK" --device cuda --out "data/a2/${BLOCK}_4bit_B.jsonl" "$@"
