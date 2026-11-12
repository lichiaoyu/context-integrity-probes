#!/bin/bash
#SBATCH --job-name=camp-learned-tx
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/camp-learned-tx-%j.out
#SBATCH --error=logs/camp-learned-tx-%j.err

set -euo pipefail

cd "$HOME/PoisonedRAG"
export PATH="$HOME/envs/camp-b200/bin:$PATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

python ceg_fourprobe_eval.py \
  --activations-dir camp_ceg_successful_attention \
  --labels camp_ceg_successful_samples/answer_labels_v2.csv \
  --outdir camp_ceg_learned_tx_449_overnight \
  --seeds 11 22 33 44 55 \
  --epochs 100

echo "Finished: $(date)"
