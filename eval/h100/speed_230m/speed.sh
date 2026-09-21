#!/usr/bin/env bash
# Phase 1: interleaved speed runs (ABC x3), 300 steps each, no eval/checkpoints, logged to wandb group speed-230m.
set -u
cd "$(dirname "$0")/../../.." || exit 1   # repo root, wherever it is cloned
CACHE="${SHARD_CACHE:-$HOME/.cache/finevision_shards}"   # local parquet shard cache (LRU), any disk with ~30 GiB free
S="${S:-runs/speed}"   # per-run logs; override with S=... to keep several sweeps apart
mkdir -p "$S"
SECRETS="${SECRETS:-/workspace/.secrets.env}"   # WANDB_API_KEY / HF_TOKEN, kept outside the repo
set -a; [ -f "$SECRETS" ] && . "$SECRETS"; set +a
COMMON="--lm_model_type HuggingFaceTB/SmolLM2-135M-Instruct --no_eval --no_lmms_eval --no_hub_push \
  --dataset_cache_dir "$CACHE" --max_cache_gb 30 --val_size 5000 --num_workers 4 \
  --batch_size 2 --gradient_accumulation_steps 8 --max_training_steps 300 --stats_log_interval 25 \
  --wandb_group speed-230m --wandb_project nanoVLM"
declare -A ARMS=(
  [A]="--loss_impl full --attn_packing_impl none"
  [B]="--loss_impl gather --attn_packing_impl flex_document_causal"
  [C]="--loss_impl gather --attn_packing_impl flex_document_causal --compile"
)
for rep in 1 2 3; do
  for arm in A B C; do
    echo "=== $arm rep$rep $(date)"
    .venv/bin/python -u train.py $COMMON ${ARMS[$arm]} --wandb_tags speed,arm_$arm --run_name_suffix speed_${arm}_r${rep} \
      2>&1 | while IFS= read -r l; do printf '%s %s\n' "$(date +%s.%N)" "$l"; done > $S/${arm}_r${rep}.log
    grep -E "Step: 275, Loss" $S/${arm}_r${rep}.log | cut -d' ' -f2- | cut -c1-140
  done
done
echo "SPEED DONE $(date)"
