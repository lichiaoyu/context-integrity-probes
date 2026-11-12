#!/bin/bash
#SBATCH --job-name=camp-case-control
#SBATCH --partition=b200-batch
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/camp-case-control-%j.out
#SBATCH --error=logs/camp-case-control-%j.err

set -euo pipefail

cd "$HOME/PoisonedRAG"
export PATH="$HOME/envs/camp-b200/bin:$PATH"
export PYTHONUNBUFFERED=1

INPUT_FILE="camp_ceg_successful_matched_800_casecontrolled.jsonl"
OUT_DIR="camp_ceg_casecontrolled_attention"
PAIR_COUNT=449

if [ -e "$OUT_DIR" ]; then
    echo "ERROR: Output directory already exists: $OUT_DIR"
    exit 1
fi

echo "Input: $INPUT_FILE"
echo "Output: $OUT_DIR"
echo "Pairs: $PAIR_COUNT"
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
