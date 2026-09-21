#!/usr/bin/env bash
# Phase 2: val-loss A/B. Baseline (A) vs fork with compile (C), seed 0 pair first, then seed 1.
# Each run: 10k steps, eval + checkpoint every 500, two-mask eval on 256 val rows, full val pass at the end.
# A run that dies is resumed from its latest checkpoint, up to 3 times.
set -u
cd "$(dirname "$0")/../../.." || exit 1   # repo root, wherever it is cloned
CACHE="${SHARD_CACHE:-$HOME/.cache/finevision_shards}"   # local parquet shard cache (LRU), any disk with ~30 GiB free
S="${S:-runs/ab}"   # per-run logs; override with S=... to keep several A/Bs apart
mkdir -p "$S"
SECRETS="${SECRETS:-/workspace/.secrets.env}"   # WANDB_API_KEY / HF_TOKEN, kept outside the repo
set -a; [ -f "$SECRETS" ] && . "$SECRETS"; set +a
COMMON="--lm_model_type HuggingFaceTB/SmolLM2-135M-Instruct --no_lmms_eval --no_hub_push \
  --dataset_cache_dir "$CACHE" --max_cache_gb 30 --val_size 5000 --num_workers 4 \
  --batch_size 2 --gradient_accumulation_steps 8 --max_training_steps 10000 --eval_interval 500 \
  --stats_log_interval 100 --eval_mask_rows 256 --save_training_state \
  --wandb_group ab-10k-230m --wandb_project nanoVLM"

run_arm() {
  local tag=$1 seed=$2 arm=$3; shift 3
  local args="$*" attempt=0 code run
  while : ; do
    if [ $attempt -eq 0 ]; then
      echo "=== $tag start $(date)"
      .venv/bin/python -u train.py $COMMON $args --seed $seed --run_name_suffix ab_$tag \
        --wandb_tags ab,arm_$arm,seed_$seed > $S/${tag}_try0.log 2>&1
    else
      run=$(ls -dt checkpoints/*_ab_$tag/ 2>/dev/null | head -1)
      if [ -z "$run" ]; then echo "!!! $tag: no checkpoint to resume from, giving up"; return 1; fi
      echo "=== $tag resume attempt $attempt from $run $(date)"
      .venv/bin/python -u train.py $COMMON $args --seed $seed --run_name_suffix ab_$tag \
        --wandb_tags ab,arm_$arm,seed_$seed --resume_from "$run" > $S/${tag}_try${attempt}.log 2>&1
    fi
    code=$?
    if [ $code -eq 0 ]; then echo "=== $tag finished $(date)"; return 0; fi
    attempt=$((attempt+1))
    echo "!!! $tag exited $code, retry $attempt/3 $(date)"
    if [ $attempt -gt 3 ]; then echo "!!! $tag failed after 3 retries"; return 1; fi
    sleep 30
  done
}

run_arm A_s0 0 A "--loss_impl full --attn_packing_impl none"
run_arm C_s0 0 C "--loss_impl gather --attn_packing_impl flex_document_causal --compile"
echo "### seed 0 pair done $(date)"
run_arm A_s1 1 A "--loss_impl full --attn_packing_impl none"
run_arm C_s1 1 C "--loss_impl gather --attn_packing_impl flex_document_causal --compile"
echo "PHASE2 DONE $(date)"
