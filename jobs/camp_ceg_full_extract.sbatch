#!/bin/bash
#SBATCH --job-name=camp-ceg-extract
#SBATCH --partition=b200-batch
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/camp-ceg-full-%j.out
#SBATCH --error=logs/camp-ceg-full-%j.err

set -euo pipefail

cd "$HOME/PoisonedRAG"
export PATH="$HOME/envs/camp-b200/bin:$PATH"
export PYTHONUNBUFFERED=1

: "${INPUT_FILE:?Missing INPUT_FILE}"
: "${OUT_DIR:?Missing OUT_DIR}"
: "${PAIR_COUNT:?Missing PAIR_COUNT}"

echo "Dataset: $INPUT_FILE"
echo "Pairs: $PAIR_COUNT"
echo "Output: $OUT_DIR"
echo "Started: $(date)"

python ceg_extract_attention_samepass.py \
  --input "$INPUT_FILE" \
  --outdir "$OUT_DIR" \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --passage-chars 650 \
  --limit-pairs "$PAIR_COUNT"

echo "=== Output files ==="
ls -lh "$OUT_DIR"
echo "Finished: $(date)"
