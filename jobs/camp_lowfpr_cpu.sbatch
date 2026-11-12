#!/bin/bash
#SBATCH --job-name=camp-lowfpr
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/camp-lowfpr-%j.out
#SBATCH --error=logs/camp-lowfpr-%j.err

set -euo pipefail
cd ~/PoisonedRAG

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

~/envs/camp-b200/bin/python ceg_lowfpr_eval.py \
  --outdir camp_ceg_lowfpr

echo "LOW-FPR EXPERIMENT COMPLETED"
