#!/bin/bash
#SBATCH --job-name=camp-4way
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/camp-4way-%j.out
#SBATCH --error=logs/camp-4way-%j.err

set -euo pipefail

cd ~/PoisonedRAG

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

~/envs/camp-b200/bin/python ceg_crossversion_4way.py \
  --layers all \
  --outdir camp_ceg_crossversion_4way

echo "EXPERIMENT COMPLETED"
