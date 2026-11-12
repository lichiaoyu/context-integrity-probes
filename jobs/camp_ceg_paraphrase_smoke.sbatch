#!/bin/bash
#SBATCH --job-name=camp-paraphrase
#SBATCH --partition=b200-batch
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/camp-paraphrase-%j.out
#SBATCH --error=logs/camp-paraphrase-%j.err

set -euo pipefail

cd "$HOME/PoisonedRAG"
export PATH="$HOME/envs/camp-b200/bin:$PATH"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

python ceg_paraphrase_pilot.py \
  --input camp_ceg_paraphrase_pilot/pilot_100_original.jsonl \
  --outdir camp_ceg_paraphrase_pilot/smoke_10 \
  --pairs 10 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --max-new-tokens 512

echo "Finished: $(date)"
