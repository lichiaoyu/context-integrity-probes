#!/bin/bash
#SBATCH --job-name=camp-OQ-449
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/camp-OQ-449-%j.out
#SBATCH --error=logs/camp-OQ-449-%j.err

set -euo pipefail

cd "$HOME/PoisonedRAG"
export PATH="$HOME/envs/camp-b200/bin:$PATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

python ceg_crossversion_449.py \
  --original camp_ceg_successful_attention \
  --paraphrased camp_ceg_paraphrase_449/qwen_449_activations \
  --outdir camp_ceg_crossversion_449
