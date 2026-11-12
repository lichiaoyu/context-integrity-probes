#!/bin/bash
#SBATCH --job-name=ceg-success-matched
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ceg-success-matched-%j.out
#SBATCH --error=logs/ceg-success-matched-%j.err

set -euo pipefail
cd "$HOME/PoisonedRAG"
export PATH="$HOME/envs/camp-b200/bin:$PATH"
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4

python ceg_layerwise_success_matched.py
