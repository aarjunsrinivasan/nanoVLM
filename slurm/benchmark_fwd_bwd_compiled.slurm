#!/bin/bash
#SBATCH --job-name=benchmark_fwd_bwd_compiled
#SBATCH --output=logs/benchmark_fwd_bwd_compiled/%j.out
#SBATCH --error=logs/benchmark_fwd_bwd_compiled/%j.err
#SBATCH --time=01:30:00
#SBATCH --partition=batch
#SBATCH --nodelist=localhost
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Single-GPU torch.compile benchmark/profile job for eval/benchmark_fwd_bwd.py --compile, per this
# repo's CLAUDE.md GPU rules (one GPU, modest batch size for test runs). Longer time limit than
# slurm/benchmark_fwd_bwd.slurm (eager) since torch.compile's first-shape compilation adds several
# minutes of wall-clock -- that cost lands inside the script's untimed warmup loop, so it doesn't
# pollute the reported latencies, it just needs the extra wall-clock budget. 90 minutes (not 60):
# the script now calls torch._dynamo.reset() before every batch size to keep each size's compiled
# graph isolated (avoids torch._dynamo's recompile_limit -- see eval/benchmark_fwd_bwd.py), so every
# batch size pays a full real compile cost instead of some sizes silently falling back to eager.
# Separate script (not a flag on the eager one) so eager and compiled runs can't accidentally share
# --results_file/--trace_dir and clobber each other. Submit with:
#   sbatch slurm/benchmark_fwd_bwd_compiled.slurm
# Extra args are forwarded to the script (--compile is always on), e.g.:
#   sbatch slurm/benchmark_fwd_bwd_compiled.slurm --batch_sizes 1 2 4 8

set -euo pipefail

cd /home/asrinivasan/vlm
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false

echo "Assigned GPU(s): ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv

python -m eval.benchmark_fwd_bwd --compile "$@"
