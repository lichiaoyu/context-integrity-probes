#!/bin/bash
#SBATCH --job-name=camp-para-449
#SBATCH --partition=rtx-batch,b200-batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/camp-para-449-%j.out
#SBATCH --error=logs/camp-para-449-%j.err

set -euo pipefail

cd "$HOME/PoisonedRAG"

export PATH="$HOME/envs/camp-b200/bin:$PATH"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

python ceg_paraphrase_pilot.py \
  --input camp_ceg_paraphrase_449_original.jsonl \
  --outdir camp_ceg_paraphrase_449/qwen_449 \
  --pairs 449 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --max-new-tokens 512

echo "Finished: $(date)"
