#!/bin/bash
#SBATCH --job-name=camp-b200-check
#SBATCH --partition=b200-devel
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/camp-b200-check-%j.out
#SBATCH --error=logs/camp-b200-check-%j.err

set -euo pipefail

export PATH="$HOME/envs/camp-b200/bin:$PATH"

python - <<'PY'
import torch
import transformers

print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())

assert torch.cuda.is_available(), "CUDA GPU unavailable"

print("GPU:", torch.cuda.get_device_name(0))

x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
y = x @ x

print("BF16 matrix multiply:", y.shape)
print("Finite result:", torch.isfinite(y).all().item())
PY
