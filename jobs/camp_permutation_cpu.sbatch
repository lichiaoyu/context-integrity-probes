#!/bin/bash
#SBATCH --job-name=camp-permutation
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/camp-permutation-%j.out
#SBATCH --error=logs/camp-permutation-%j.err

set -euo pipefail
cd "$HOME/PoisonedRAG"
export PATH="$HOME/envs/camp-b200/bin:$PATH"
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1

python ceg_permutation_check.py
