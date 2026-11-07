#!/usr/bin/env bash
# Loss path (lm_loss_impl full vs gather) x (eager vs torch.compile), 460M model, 1 GPU.
# Interleaved (E_full E_gather C_full C_gather, repeat), one PROCESS per arm: dynamo's config
# (train.py sets capture_dynamic_output_shape_ops under --compile), compile cache and recompile_limit
# are process-global, so arms must not share a process.
#
#   CUDA_VISIBLE_DEVICES=<idle gpu> bash eval/profile/loss_compile_ab/run.sh
#
# Overrides (env): OUT REPS STEPS INTERVAL WARMUP PACKING ARMS SKIP_PRIME NUM_WORKERS. Smoke test:
#   CUDA_VISIBLE_DEVICES=0 OUT=/tmp/smoke STEPS=20 INTERVAL=5 WARMUP=0 REPS=1 SKIP_PRIME=1 bash eval/profile/loss_compile_ab/run.sh
set -u
cd "$(dirname "$0")/../../.." || exit 1   # repo root, wherever it is cloned
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?pin exactly one idle GPU, e.g. CUDA_VISIBLE_DEVICES=0}"
export PYTHONUNBUFFERED=1

OUT="${OUT:-eval/profile/loss_compile_ab}"
REPS="${REPS:-2}"
STEPS="${STEPS:-300}"
INTERVAL="${INTERVAL:-25}"
WARMUP="${WARMUP:-4}"           # stats intervals dropped per arm: steady state = steps 100-300 at the defaults
PACKING="${PACKING:-none}"      # held fixed across arms so only the loss path (and compile) varies
ARMS="${ARMS:-E_full E_gather C_full C_gather}"
NUM_WORKERS="${NUM_WORKERS:-4}" # the driver's default of 2 starves compiled arms (see eval/h100/speed_230m/README.md)
mkdir -p "$OUT"

declare -A ARM_FLAGS=(
  [E_full]="--variants full"
  [E_gather]="--variants gather"
  [C_full]="--variants full --compile"
  [C_gather]="--variants gather --compile"
)

# The first process primes the shard cache so no arm pays for downloads; later ones skip it.
prime="--skip_prime_cache"
[ "${SKIP_PRIME:-0}" = "1" ] || prime=""

for rep in $(seq 1 "$REPS"); do
  for arm in $ARMS; do
    echo "=== $arm rep$rep $(date)"
    .venv/bin/python -u -m eval.run_loss_ab ${ARM_FLAGS[$arm]} $prime \
      --attn_packing_impl "$PACKING" --num_workers "$NUM_WORKERS" \
      --max_training_steps "$STEPS" --stats_log_interval "$INTERVAL" --warmup_intervals "$WARMUP" \
      --results_file "$OUT/${arm}_r${rep}.json" > "$OUT/${arm}_r${rep}.log" 2>&1
    echo "    exit=$? $(grep -E 'Fastest:' "$OUT/${arm}_r${rep}.log" | tail -1)"
    prime="--skip_prime_cache"
  done
done
echo "DONE $(date)"
