#!/bin/bash
#SBATCH --job-name=ceg-matched-outcome
#SBATCH --partition=b200-batch
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/ceg-matched-%j.out
#SBATCH --error=logs/ceg-matched-%j.err

set -euo pipefail

cd "$HOME/PoisonedRAG"
export PATH="$HOME/envs/camp-b200/bin:$PATH"
export PYTHONUNBUFFERED=1

INPUT="camp_ceg_successful_matched_800_casecontrolled.jsonl"
OUT="camp_ceg_casecontrolled_samples"

mkdir -p logs "$OUT"

echo "=== Matched-Outcome Validation ==="
echo "Started: $(date)"
echo "Input: $INPUT"
echo "Output: $OUT"

python ceg_sample_responses.py \
  --input "$INPUT" \
  --outdir "$OUT" \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --passage-chars 650 \
  --max-tokens 4096 \
  --max-new-tokens 32 \
  --n 10 \
  --temperature 0.7 \
  --top-p 0.9 \
  --seed 2026

echo "=== Relabeling ==="

python ceg_relabel_v2.py \
  --input "$OUT/answer_labels.csv"

echo "=== Validating outputs ==="

python - <<'PY'
import csv
from pathlib import Path
from collections import Counter

p = Path("camp_ceg_casecontrolled_samples/answer_labels_v2.csv")

with p.open(newline="") as f:
    rows = list(csv.DictReader(f))

assert len(rows) == 8980, f"Expected 8980 responses, got {len(rows)}"

counts = Counter(r["condition"] for r in rows)
assert counts == {"clean": 4490, "poison_1": 4490}, counts

print("VALIDATION PASSED")
print("Responses:", len(rows))
print("Conditions:", dict(counts))
print("Labels:", dict(Counter(r["label"] for r in rows)))
PY

echo "=== Output files ==="
ls -lh "$OUT"
echo "Finished: $(date)"
