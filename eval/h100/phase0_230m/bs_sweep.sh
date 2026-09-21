#!/usr/bin/env bash
# Phase 0 batch-size sweep: 3 arms x micro-batch {2,4,8,16}, bs*accum = 16 rows/step, 60 steps each.
set -u
cd "$(dirname "$0")/../../.." || exit 1   # repo root, wherever it is cloned
SECRETS="${SECRETS:-/workspace/.secrets.env}"   # WANDB_API_KEY / HF_TOKEN, kept outside the repo
set -a; [ -f "$SECRETS" ] && . "$SECRETS"; set +a
CACHE="${SHARD_CACHE:-$HOME/.cache/finevision_shards}"   # local parquet shard cache (LRU), any disk with ~30 GiB free
OUT="${OUT:-runs/sweep}"   # per-run logs; override with OUT=... to keep several sweeps apart
mkdir -p "$OUT"
COMMON="--lm_model_type HuggingFaceTB/SmolLM2-135M-Instruct --no_eval --no_lmms_eval --no_hub_push --no_log_wandb \
  --dataset_cache_dir "$CACHE" --max_cache_gb 30 --val_size 5000 \
  --max_training_steps 60 --stats_log_interval 10"
declare -A ARMS=(
  [A]="--loss_impl full --attn_packing_impl none"
  [B]="--loss_impl gather --attn_packing_impl flex_document_causal"
  [C]="--loss_impl gather --attn_packing_impl flex_document_causal --compile"
)
declare -A OOM=()
for bs in 2 4 8 16; do
  accum=$((16 / bs))
  for arm in A B C; do
    [[ -n "${OOM[$arm]:-}" ]] && { echo "skip $arm bs=$bs (OOM earlier)"; continue; }
    log="$OUT/${arm}_bs${bs}.log"
    echo "=== $arm bs=$bs accum=$accum $(date)"
    TORCH_LOGS=recompiles .venv/bin/python -u train.py $COMMON ${ARMS[$arm]} \
      --batch_size $bs --gradient_accumulation_steps $accum 2>&1 \
      | while IFS= read -r l; do printf '%s %s\n' "$(date +%s.%N)" "$l"; done > "$log"
    if grep -q "OutOfMemoryError\|CUDA out of memory" "$log"; then OOM[$arm]=1; echo "  OOM"; fi
    grep "Step:" "$log" | tail -2
  done
done
echo "SWEEP DONE $(date)"
