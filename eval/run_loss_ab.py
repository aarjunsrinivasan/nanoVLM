"""A/B benchmarks VLMConfig.lm_loss_impl in a real training loop.

One real train.py run per loss impl, comparing tokens/sec and peak memory. Results are saved as
JSON under eval/<gpu>/; see eval/h100/loss_gather_ab.md for the measured numbers.

    'full'    materialize logits [B, T, V] over every position, then F.cross_entropy
    'gather'  keep only non-masked positions (targets != -100) before the LM head (default)
    'chunked' gather, then F.linear_cross_entropy, which never materializes logits at all

    CUDA_VISIBLE_DEVICES=0 python -m eval.run_loss_ab

This measures cost, not correctness: numerical equivalence of the three impls is covered by
tests/test_vision_language_model_loss.py. The last_batch_loss column is only a signal that the
arms saw the same data.

Keep --num_workers >= 1, for the reason in eval/benchmark_attn_train.py.
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

# 'full' first: it is the unoptimized baseline the other two are measured against.
VARIANTS = ["full", "gather", "chunked"]
BASELINE = "full"


def build_base_configs(args):
    vlm_cfg = config.VLMConfig(hf_repo_name=None)
    if args.lm_model_type is not None:
        vlm_cfg.lm_model_type = args.lm_model_type
    if args.vit_model_type is not None:
        vlm_cfg.vit_model_type = args.vit_model_type
    vlm_cfg.lm_attn_packing_impl = args.attn_packing_impl

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


def build_arm_configs(base_vlm_cfg, base_train_cfg, loss_impl, checkpoint_root):
    vlm_cfg = dataclasses.replace(
        base_vlm_cfg,
        lm_loss_impl=loss_impl,
        vlm_checkpoint_path=f"{checkpoint_root}/{loss_impl}",
    )
    train_cfg = dataclasses.replace(base_train_cfg, run_name_suffix=loss_impl)
    return loss_impl, vlm_cfg, train_cfg


def main():
    parser = argparse.ArgumentParser(
        description="A/B VLMConfig.lm_loss_impl ('full' vs 'gather' vs 'chunked') in a real "
                    "training loop. Replaces the lost experiments/loss_optimization driver."
    )
    parser.add_argument("--variants", nargs="*", default=VARIANTS, choices=VARIANTS,
                        help="Loss implementations to run, one train() call each.")
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
    parser.add_argument("--compile", action="store_true",
                        help="Apply torch.compile to every arm. Note chunked linear_cross_entropy "
                             "always falls back to eager under compile -- expected, not a regression.")
    parser.add_argument("--attn_packing_impl", type=str, default="none",
                        choices=["none", "dense_block_diagonal", "flex_document_causal"],
                        help="Held constant across arms so this sweep isolates the loss path. "
                             "Defaults to the config default so the numbers describe the loss "
                             "change alone, independent of eval/benchmark_attn_train.py's sweep.")
    parser.add_argument("--lm_model_type", type=str, default=None)
    parser.add_argument("--vit_model_type", type=str, default=None)
    parser.add_argument("--dataset_cache_dir", type=str,
                        default=os.environ.get("NANOVLM_CACHE_DIR")
                        or os.path.expanduser("~/.cache/vlm-dev/finevision_benchmark_shards"))
    parser.add_argument("--max_cache_gb", type=float, default=30.0)
    parser.add_argument("--prefetch_shards", type=int, default=1)
    parser.add_argument("--checkpoint_root", type=str, default="checkpoints/run_loss_ab",
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
    parser.add_argument("--wandb_group", type=str, default="loss_gather_ab")
    parser.add_argument("--results_file", type=str, default="eval/h100/loss_gather_ab_results.json",
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
        print("WARNING: no GPU visible -- peak-memory columns will be meaningless. "
              "Check LD_LIBRARY_PATH includes the CUDA compat libs (see scripts/setup_pod.sh).")

    base_vlm_cfg, base_train_cfg = build_base_configs(args)

    if not args.skip_prime_cache:
        prime_batches = args.prime_cache_batches
        if prime_batches is None:
            prime_batches = args.max_training_steps * args.gradient_accumulation_steps
        prime_shard_cache(base_train_cfg, base_vlm_cfg, prime_batches)

    per_variant = {}
    rows = []
    for loss_impl in args.variants:
        label, vlm_cfg, train_cfg = build_arm_configs(base_vlm_cfg, base_train_cfg, loss_impl, args.checkpoint_root)
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
