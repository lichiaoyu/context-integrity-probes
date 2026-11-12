#!/bin/bash
#SBATCH --job-name=camp-hid-qwen100
#SBATCH --partition=rtx-batch
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/camp-hid-qwen100-%j.out
#SBATCH --error=logs/camp-hid-qwen100-%j.err

set -euo pipefail
cd "$HOME/PoisonedRAG"
export PATH="$HOME/envs/camp-b200/bin:$PATH"
export PYTHONUNBUFFERED=1

OUT="camp_ceg_paraphrase_pilot/hidden_qwen_100"

if [ -e "$OUT" ]; then
    echo "Output already exists: $OUT"
    exit 1
fi

python ceg_extract_tbg.py \
  --input camp_ceg_paraphrase_pilot/qwen_100/paraphrased.jsonl \
  --outdir "$OUT" \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --max-tokens 2048 \
  --passage-chars 650 \
  --limit-pairs 100
