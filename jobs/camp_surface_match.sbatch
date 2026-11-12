#!/bin/bash
#SBATCH --job-name=camp-surface
#SBATCH --partition=b200-batch
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/camp-surface-%j.out
#SBATCH --error=logs/camp-surface-%j.err

set -euo pipefail
cd "$HOME/PoisonedRAG"
export PATH="$HOME/envs/camp-b200/bin:$PATH"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

python ceg_surface_match.py \
  --input camp_ceg_paraphrase_pilot/smoke_10/paraphrased.jsonl \
  --outdir camp_ceg_paraphrase_pilot/qwen_controlled_10
