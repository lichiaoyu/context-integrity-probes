#!/bin/bash
#SBATCH --job-name=camp-ceg-gen
#SBATCH --partition=b200-batch
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/camp-ceg-gen-%j.out
#SBATCH --error=logs/camp-ceg-gen-%j.err

set -euo pipefail

cd "$HOME/PoisonedRAG"
export PATH="$HOME/envs/camp-b200/bin:$PATH"
export PYTHONUNBUFFERED=1

: "${INPUT_FILE:?Missing INPUT_FILE}"
: "${OUT_DIR:?Missing OUT_DIR}"
: "${NUM_SAMPLES:?Missing NUM_SAMPLES}"

echo "Input: $INPUT_FILE"
echo "Output: $OUT_DIR"
echo "Samples per context: $NUM_SAMPLES"
echo "Started: $(date)"

python ceg_sample_responses.py \
  --input "$INPUT_FILE" \
  --outdir "$OUT_DIR" \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --passage-chars 650 \
  --max-tokens 4096 \
  --max-new-tokens 32 \
  --n "$NUM_SAMPLES" \
  --temperature 0.7 \
  --top-p 0.9 \
  --seed 2026

echo "=== Output files ==="
ls -lh "$OUT_DIR"
echo "Finished: $(date)"
