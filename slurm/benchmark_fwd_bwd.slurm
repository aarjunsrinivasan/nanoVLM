#!/bin/bash
#SBATCH --job-name=benchmark_fwd_bwd
#SBATCH --output=logs/benchmark_fwd_bwd/%j.out
#SBATCH --error=logs/benchmark_fwd_bwd/%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=batch
#SBATCH --nodelist=localhost
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Single-GPU benchmark/profile job for eval/benchmark_fwd_bwd.py, per this repo's CLAUDE.md
# GPU rules (one GPU, modest batch size for test runs). Submit with:
#   sbatch slurm/benchmark_fwd_bwd.slurm
# Extra args are forwarded to the script, e.g.:
#   sbatch slurm/benchmark_fwd_bwd.slurm --batch_size 8 --model_id lusxvr/nanoVLM-460M-8k

set -euo pipefail

cd /home/asrinivasan/vlm
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false

echo "Assigned GPU(s): ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv

python -m eval.benchmark_fwd_bwd "$@"
