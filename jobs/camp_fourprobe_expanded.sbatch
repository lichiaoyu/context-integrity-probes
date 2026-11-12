#!/bin/bash
#SBATCH --job-name=camp-probes
#SBATCH --partition=b200-batch
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/camp-probes-%j.out
#SBATCH --error=logs/camp-probes-%j.err

set -euo pipefail

cd "$HOME/PoisonedRAG"
export PATH="$HOME/envs/camp-b200/bin:$PATH"
export PYTHONUNBUFFERED=1

: "${ACTIVATIONS_DIR:?}"
: "${LABELS_FILE:?}"
: "${OUTPUT_DIR:?}"

python ceg_fourprobe_eval.py \
  --activations-dir "$ACTIVATIONS_DIR" \
  --labels "$LABELS_FILE" \
  --outdir "$OUTPUT_DIR" \
  --seeds 11 22 33 44 55 \
  --epochs 100

echo "Finished: $(date)"
