#!/bin/bash
#SBATCH --job-name=camp-layer-T
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/camp-layer-T-%j.out
#SBATCH --error=logs/camp-layer-T-%j.err

set -euo pipefail

cd "$HOME/PoisonedRAG"
export PATH="$HOME/envs/camp-b200/bin:$PATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

python ceg_layerwise_learned_teacher.py \
  --activations-dir camp_ceg_successful_attention \
  --labels camp_ceg_successful_samples/answer_labels_v2.csv \
  --outdir camp_ceg_layerwise_learned_teacher
