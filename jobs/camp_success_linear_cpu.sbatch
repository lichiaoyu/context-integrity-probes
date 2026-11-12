#!/bin/bash
#SBATCH --job-name=camp-success-linear
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/camp-success-linear-%j.out
#SBATCH --error=logs/camp-success-linear-%j.err

set -euo pipefail
cd "$HOME/PoisonedRAG"
export PATH="$HOME/envs/camp-b200/bin:$PATH"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1

python ceg_layerwise_success_linear.py
