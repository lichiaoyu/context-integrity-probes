#!/bin/bash
#SBATCH --job-name=camp-qwen-smoke
#SBATCH --partition=b200-devel
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=logs/camp-qwen-smoke-%j.out
#SBATCH --error=logs/camp-qwen-smoke-%j.err

set -euo pipefail

cd "$HOME/PoisonedRAG"
export PATH="$HOME/envs/camp-b200/bin:$PATH"
export PYTHONUNBUFFERED=1

echo "=== Model smoke test ==="
echo "Python: $(which python)"
echo "Started: $(date)"

python ceg_extract_attention_samepass.py \
  --input camp_ceg_final_matched_250.jsonl \
  --outdir camp_ceg_b200_smoke \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --passage-chars 650 \
  --limit-pairs 2

echo "=== Output files ==="
ls -lh camp_ceg_b200_smoke/

echo "Finished: $(date)"
