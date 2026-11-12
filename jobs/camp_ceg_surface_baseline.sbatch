#!/bin/bash
#SBATCH --job-name=ceg-surface
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/ceg-surface-%j.out
#SBATCH --error=logs/ceg-surface-%j.err

set -euo pipefail
cd "$HOME/PoisonedRAG"
export PATH="$HOME/envs/camp-b200/bin:$PATH"

python ceg_surface_baseline.py
