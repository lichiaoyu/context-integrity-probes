#!/bin/bash
#SBATCH --job-name=camp-q449-extract
#SBATCH --partition=preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/camp-q449-extract-%j.out
#SBATCH --error=logs/camp-q449-extract-%j.err

set -euo pipefail

cd "$HOME/PoisonedRAG"
export PATH="$HOME/envs/camp-b200/bin:$PATH"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "Started: $(date)"

python ceg_extract_attention_samepass.py \
  --input camp_ceg_paraphrase_449/qwen_449/paraphrased.jsonl \
  --outdir camp_ceg_paraphrase_449/qwen_449_activations \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --passage-chars 650 \
  --max-tokens 2048 \
  --limit-pairs 449

echo "Finished: $(date)"
