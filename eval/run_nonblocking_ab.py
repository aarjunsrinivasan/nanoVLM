"""A/B benchmarks VLMConfig.non_blocking_transfers in a real training loop.

Sweeps whether the host->device .to() calls in models/vision_language_model.py's
_process_images and train.py's 4 batch-tensor transfers (input_ids, labels, attention_mask,
doc_id) pass non_blocking=True --

    'blocking'      current default before this change: plain .to(device)
    'non_blocking'  .to(device, non_blocking=True)

-- one real train.py run per arm (same model, same streamed FineVision data, same step count),
sequentially on one GPU, and compares tokens/sec and peak memory. Both DataLoaders already set
pin_memory=True (train.py:253,266), which is a prerequisite for non_blocking=True to actually be
asynchronous rather than a silent no-op -- this sweep isolates the effect of the flag alone.

Caveat specific to this sweep: enabling non_blocking_transfers makes train.py's `data_load_time`
stat (the time up to and including the four input .to(device) calls, train.py:422-433) an
UNDERCOUNT once an arm runs with non_blocking=True -- it then measures time to *launch* the async
copy, not time for it to *complete*, because no torch.cuda.synchronize() is inserted around it.
The stats_history's avg_data_load_time column is therefore NOT a reliable signal of how much this
change saved. The reliable end-to-end signal is avg_tokens_per_second / avg_fw_bw_time_s, both of
which stay correct because loss.item() (train.py:483) forces a full CUDA sync before
batch_end_time is captured, downstream of both the data load and the forward+backward pass. See
eval/profile/nonblocking_transfers_ab.md for how this is handled in the writeup.

Run as a module from the repo root, pinned to a single idle GPU (check `nvidia-smi` first):

    CUDA_VISIBLE_DEVICES=0 python -m eval.run_nonblocking_ab

A full run trains --max_training_steps per arm (minutes, not seconds) and downloads/caches real
dataset shards. Do a short dry run first:

    CUDA_VISIBLE_DEVICES=0 python -m eval.run_nonblocking_ab \
        --max_training_steps 20 --stats_log_interval 5 --warmup_intervals 0

The same --num_workers >= 1 fairness caveat as eval/benchmark_attn_train.py applies: get_dataloaders()
creates a fresh torch.Generator().manual_seed(0) per call, so each arm sees an identical sample
order only when DataLoader workers (not the parent process's carried-over RNG) do the shuffling.
"""
import argparse
import dataclasses
import json
import os
import time

import torch

import train
import models.config as config
from eval.benchmark_attn_train import (
    build_comparison_table,
    json_safe,
    prime_shard_cache,
    run_arm,
    summarize_variant,
)

# 'blocking' first: it is the pre-existing, unoptimized baseline the other arm is measured against.
VARIANTS = ["blocking", "non_blocking"]
BASELINE = "blocking"
VARIANT_TO_FLAG = {"blocking": False, "non_blocking": True}


def build_base_configs(args):
    vlm_cfg = config.VLMConfig(hf_repo_name=None)
    if args.lm_model_type is not None:
        vlm_cfg.lm_model_type = args.lm_model_type
    if args.vit_model_type is not None:
        vlm_cfg.vit_model_type = args.vit_model_type

    train_cfg = config.TrainConfig(
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_training_steps=args.max_training_steps,
        stats_log_interval=args.stats_log_interval,
        val_size=args.val_size,
        num_workers=args.num_workers,
        dataset_cache_dir=args.dataset_cache_dir,
        max_cache_gb=args.max_cache_gb,
        prefetch_shards=args.prefetch_shards,
        compile=args.compile,
        log_wandb=args.log_wandb,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        wandb_group=args.wandb_group,
        use_lmms_eval=False,
        eval_in_epochs=False,  # skips the in-loop val pass and its save_pretrained checkpoint write
    )
    return vlm_cfg, train_cfg


def build_arm_configs(base_vlm_cfg, base_train_cfg, label, checkpoint_root):
    vlm_cfg = dataclasses.replace(
        base_vlm_cfg,
        non_blocking_transfers=VARIANT_TO_FLAG[label],
        vlm_checkpoint_path=f"{checkpoint_root}/{label}",
    )
    train_cfg = dataclasses.replace(base_train_cfg, run_name_suffix=label)
    return label, vlm_cfg, train_cfg


def main():
    parser = argparse.ArgumentParser(
        description="A/B VLMConfig.non_blocking_transfers ('blocking' vs 'non_blocking' host->device "
                    ".to() calls) in a real training loop."
    )
    parser.add_argument("--variants", nargs="*", default=VARIANTS, choices=VARIANTS,
                        help="Arms to run, one train() call each.")
    parser.add_argument("--max_training_steps", type=int, default=200)
    parser.add_argument("--stats_log_interval", type=int, default=25)
    parser.add_argument("--warmup_intervals", type=int, default=1,
                        help="Leading stats intervals dropped before averaging, to exclude startup "
                             "effects (first-step allocator growth, shard cache warmup).")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--val_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Keep >= 1 for a fair cross-arm comparison (see module docstring).")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--lm_model_type", type=str, default=None)
    parser.add_argument("--vit_model_type", type=str, default=None)
    parser.add_argument("--dataset_cache_dir", type=str,
                        default=os.environ.get("NANOVLM_CACHE_DIR")
                        or os.path.expanduser("~/.cache/vlm-dev/finevision_benchmark_shards"))
    parser.add_argument("--max_cache_gb", type=float, default=30.0)
    parser.add_argument("--prefetch_shards", type=int, default=1)
    parser.add_argument("--checkpoint_root", type=str, default="checkpoints/run_nonblocking_ab",
                        help="Required by VLMConfig, but with eval_in_epochs=False nothing is ever written here.")
    parser.add_argument("--prime_cache_batches", type=int, default=None,
                        help="Micro-batches to pull through the loader before any arm runs, so no "
                             "arm pays the shard download cost. Defaults to one full arm's worth.")
    parser.add_argument("--skip_prime_cache", action="store_true")
    parser.add_argument("--log_wandb", action="store_true",
                        help="Off by default so the sweep runs without wandb credentials. When on, "
                             "each arm is a separate run sharing --wandb_group, so they compare "
                             "side by side in the wandb UI.")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="nanoVLM")
    parser.add_argument("--wandb_group", type=str, default="nonblocking_transfers_ab")
    parser.add_argument("--results_file", type=str, default="eval/profile/nonblocking_transfers_ab_results.json",
                        help="Results are grouped under eval/<gpu>/ by the hardware they were measured on.")
    args = parser.parse_args()

    if not args.variants:
        raise ValueError("Nothing to run -- --variants is empty.")
    if not args.dataset_cache_dir:
        raise ValueError("--dataset_cache_dir resolved empty -- pass it explicitly.")
    os.makedirs(args.dataset_cache_dir, exist_ok=True)
    if args.wandb_entity is None:
        args.wandb_entity = config.TrainConfig.wandb_entity

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"GPU: {device_name} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset -- all GPUs visible!')})")
    if device_name == "cpu":
        print("WARNING: no GPU visible -- peak-memory columns will be meaningless, and non_blocking "
              "has no async effect on CPU. Check LD_LIBRARY_PATH includes the CUDA compat libs.")

    base_vlm_cfg, base_train_cfg = build_base_configs(args)

    if not args.skip_prime_cache:
        prime_batches = args.prime_cache_batches
        if prime_batches is None:
            prime_batches = args.max_training_steps * args.gradient_accumulation_steps
        prime_shard_cache(base_train_cfg, base_vlm_cfg, prime_batches)

    per_variant = {}
    rows = []
    for label in args.variants:
        label, vlm_cfg, train_cfg = build_arm_configs(base_vlm_cfg, base_train_cfg, label, args.checkpoint_root)
        summary, wall_clock_s = run_arm(label, train_cfg, vlm_cfg)
        steady_state = summarize_variant(label, summary, wall_clock_s, args.warmup_intervals)
        per_variant[label] = {"summary": summary, "wall_clock_s": wall_clock_s, "steady_state": steady_state}
        rows.append(steady_state)

    print("\n--- Summary ---")
    df = build_comparison_table(rows, baseline=BASELINE)
    print(df.to_string(index=False))

    speedup_col = f"speedup_vs_{BASELINE}"
    best = df.iloc[0]
    speedup_note = f", {best[speedup_col]:.2f}x vs {BASELINE}" if speedup_col in df.columns else ""
    print(f"\nFastest: {best['variant']} ({best['mean_tokens_per_second']:.0f} tokens/s{speedup_note})")

    leanest = df.loc[df["peak_mem_allocated_gib"].idxmin()]
    print(f"Lowest peak memory: {leanest['variant']} ({leanest['peak_mem_allocated_gib']:.2f} GiB allocated)")

    results = {
        "provenance": {**train.get_provenance(), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")},
        "harness_config": vars(args),
        "variants": per_variant,
        "ranking": df.to_dict(orient="records"),
    }
    os.makedirs(os.path.dirname(args.results_file) or ".", exist_ok=True)
    with open(args.results_file, "w") as f:
        json.dump(json_safe(results), f, indent=2, default=str)
    print(f"\nSaved results to {args.results_file}")


if __name__ == "__main__":
    main()
