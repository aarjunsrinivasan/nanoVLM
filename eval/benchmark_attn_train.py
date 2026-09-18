"""Benchmarks VLMConfig.lm_attn_packing_impl in a real training loop.

Unlike eval/benchmark_attn_packing.py (a synthetic microbenchmark of the isolated attention-core
math on random tensors), this script runs the actual train.py training loop -- real model
(SmolLM2-360M-Instruct + siglip2-base-patch16-512 by default), real streamed FineVision dataset --
once per lm_attn_packing_impl value ('none', 'dense_block_diagonal', 'flex_document_causal', all
eager by default) plus optional compiled arms (--compile_variants, e.g. 'dense_block_diagonal'
with torch.compile -- 'flex_document_causal' can't compile, see train.py's own hard refusal),
sequentially on one GPU, and compares real tokens/sec, peak memory, and loss.

Results are local-only regardless of wandb: printed as a table and saved as JSON under
eval/<gpu>/, grouped by the hardware they were measured on.

Run as a module from the repo root, pinned to a single idle GPU per this repo's CLAUDE.md GPU
rules (check `nvidia-smi` first):
    CUDA_VISIBLE_DEVICES=0 python -m eval.benchmark_attn_train

A full run downloads/caches real dataset shards and trains for --max_training_steps per arm
(minutes, not seconds); run it detached (`nohup`/`tmux`) rather than in an interactive shell you
might lose. Do a short dry run first, e.g.:
    CUDA_VISIBLE_DEVICES=0 python -m eval.benchmark_attn_train \
        --max_training_steps 20 --stats_log_interval 5 --compile_warmup_intervals 1 \
        --variants none dense_block_diagonal --compile_variants dense_block_diagonal

To add only the compiled arm to a run that already has saved eager results (skip re-running the
eager arms, and don't clobber their results file):
    CUDA_VISIBLE_DEVICES=0 python -m eval.benchmark_attn_train --variants \
        --compile_variants dense_block_diagonal \
        --results_file eval/h100/benchmark_attn_train_compile_results.json

Caveat: train.py and data/advanced_datasets.py seed Python's global random/torch RNGs once at
import time, not per train() call. With --num_workers 0, the packing dataset's shuffle state
would carry over between these sequential in-process variant runs, so the variants would not see
an identical shard/sample order. With the default --num_workers 2 (>=1) this isn't a concern:
get_dataloaders() creates a fresh torch.Generator().manual_seed(0) per call, and PyTorch's
DataLoader worker init derives each worker's random/torch seed deterministically from that
generator, reseeding `random` inside each worker process independently of the parent process's
prior state. Keep --num_workers >= 1 for a fair cross-variant comparison.
"""
import argparse
import dataclasses
import gc
import json
import math
import os
import time
from statistics import mean, median

import pandas as pd
import torch

import train
import models.config as config

VARIANTS = ["none", "dense_block_diagonal", "flex_document_causal"]


def build_base_configs(args):
    vlm_cfg = config.VLMConfig(hf_repo_name=None)
    if args.lm_model_type is not None:
        vlm_cfg.lm_model_type = args.lm_model_type
    if args.vit_model_type is not None:
        vlm_cfg.vit_model_type = args.vit_model_type
    if args.attn_flex_block_size is not None:
        vlm_cfg.lm_attn_flex_block_size = args.attn_flex_block_size

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
        log_wandb=False,
        use_lmms_eval=False,
        eval_in_epochs=False,  # skips the in-loop val pass and its save_pretrained checkpoint write
    )
    return vlm_cfg, train_cfg


def arm_label(packing_impl, use_compile):
    """Distinct label per (packing_impl, use_compile) pair -- used as the results dict key, the
    'variant' column value, the checkpoint subdir, and run_name_suffix. Eager arms keep their
    existing bare packing_impl label (e.g. 'dense_block_diagonal') for backward compatibility with
    prior results JSON; compiled arms get a '_compile' suffix (e.g. 'dense_block_diagonal_compile')."""
    return f"{packing_impl}_compile" if use_compile else packing_impl


def build_arm_configs(base_vlm_cfg, base_train_cfg, packing_impl, use_compile, checkpoint_root):
    if use_compile and packing_impl == 'flex_document_causal':
        # Mirrors train.py's own hard refusal (flex_document_causal's internal torch.compile calls
        # produce cross-document-leaking output when nested inside an outer torch.compile(model)) --
        # fail here, before starting any run, rather than deep inside train.train() after other
        # arms may have already completed.
        raise ValueError(
            "flex_document_causal cannot be combined with compile=True (train.py hard-rejects "
            "this combination). Pass 'none' or 'dense_block_diagonal' via --compile_variants instead."
        )
    label = arm_label(packing_impl, use_compile)
    vlm_cfg = dataclasses.replace(
        base_vlm_cfg,
        lm_attn_packing_impl=packing_impl,
        vlm_checkpoint_path=f"{checkpoint_root}/{label}",
    )
    train_cfg = dataclasses.replace(base_train_cfg, run_name_suffix=label, compile=use_compile)
    return label, vlm_cfg, train_cfg


def prime_shard_cache(base_train_cfg, base_vlm_cfg, num_batches):
    if num_batches <= 0:
        return
    print(f"\n--- Priming shard cache: {num_batches} micro-batches (no model, one pass through the loader) ---")
    train_loader, val_loader, iter_train_loader, iter_val_loader = train.get_dataloaders(base_train_cfg, base_vlm_cfg)
    for i in range(num_batches):
        next(iter_train_loader)
        if (i + 1) % 50 == 0:
            print(f"  primed {i + 1}/{num_batches} micro-batches")
    del train_loader, val_loader, iter_train_loader, iter_val_loader
    gc.collect()
    print("--- Cache priming done ---\n")


def run_arm(label, train_cfg, vlm_cfg):
    print(f"\n=== Running arm: {label} ===")
    print(vlm_cfg)
    print(train_cfg)
    start = time.time()
    summary = train.train(train_cfg, vlm_cfg)
    wall_clock_s = time.time() - start
    if summary is None:
        raise RuntimeError(
            f"train.train() returned None for arm {label!r} -- the run did not complete "
            f"max_training_steps (e.g. it was interrupted by SIGTERM). Re-run this arm."
        )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary, wall_clock_s


def summarize_variant(label, summary, wall_clock_s, warmup_intervals):
    history = summary["stats_history"]
    steady = history[warmup_intervals:] if len(history) > warmup_intervals else history
    if not steady:
        raise RuntimeError(
            f"Arm {label!r} logged only {len(history)} stats interval(s), not enough to drop "
            f"{warmup_intervals} warmup interval(s). Lower --warmup_intervals/--compile_warmup_intervals "
            f"or raise --max_training_steps."
        )
    tps = [s["avg_tokens_per_second"] for s in steady]
    return {
        "variant": label,
        "mean_tokens_per_second": mean(tps),
        "median_tokens_per_second": median(tps),
        "peak_mem_allocated_gib": max(s.get("peak_mem_allocated_gib", 0.0) for s in steady),
        "peak_mem_reserved_gib": max(s.get("peak_mem_reserved_gib", 0.0) for s in steady),
        "avg_fw_bw_time_s": mean(s["avg_fw_bw_time"] for s in steady),
        "avg_data_load_time_s": mean(s["avg_data_load_time"] for s in steady),
        "last_batch_loss": history[-1]["batch_loss"],
        "wall_clock_s": wall_clock_s,
    }


def json_safe(obj):
    """Replaces non-finite floats with None so the results file is valid, portable JSON.

    Python's json writes float('inf') as the bare token `Infinity`, which is not in the JSON spec
    and is rejected by strict parsers (jq, Go, most JS tooling). These sweeps hit it every run:
    `eval_in_epochs=False` means no validation ever runs, so train()'s `best_val_loss` stays at its
    float('inf') initial value and lands in the summary dict. None round-trips as null and reads
    correctly as "not measured".
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def build_comparison_table(rows, baseline="none"):
    """Ranked table, fastest first, with a speedup column relative to `baseline`.

    `baseline` is the arm the comparison is against -- 'none' (the leaky status quo) for the
    packing sweep, 'full' for eval/run_loss_ab.py's loss sweep. The column is named after it,
    so the two sweeps' result JSONs stay self-describing.
    """
    df = pd.DataFrame(rows).sort_values("mean_tokens_per_second", ascending=False).reset_index(drop=True)
    if baseline in df["variant"].values:
        baseline_tps = df.loc[df["variant"] == baseline, "mean_tokens_per_second"].iloc[0]
        df[f"speedup_vs_{baseline}"] = df["mean_tokens_per_second"] / baseline_tps
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark VLMConfig.lm_attn_packing_impl in a real training loop (real model, "
                    "real streamed FineVision data), unlike eval/benchmark_attn_packing.py's synthetic "
                    "attention-core microbenchmark."
    )
    parser.add_argument("--variants", nargs="*", default=VARIANTS, choices=VARIANTS,
                         help="Eager arms to run. Pass with no values (--variants) to skip all eager arms, "
                              "e.g. when adding just a compiled arm to an already-benchmarked eager set.")
    parser.add_argument("--compile_variants", nargs="*", default=["dense_block_diagonal"],
                         choices=["none", "dense_block_diagonal"],
                         help="packing_impl values to ALSO run with train_cfg.compile=True, each producing "
                              "a distinct '<impl>_compile' row alongside the eager arms in one comparison. "
                              "'flex_document_causal' is not a valid choice here -- train.py hard-rejects "
                              "it combined with compile=True. Pass with no values to skip compiled arms.")
    parser.add_argument("--max_training_steps", type=int, default=200)
    parser.add_argument("--stats_log_interval", type=int, default=25)
    parser.add_argument("--warmup_intervals", type=int, default=1,
                         help="Leading stats intervals dropped before computing steady-state throughput/memory "
                              "for eager arms (hides flex_document_causal's one-time torch.compile cost).")
    parser.add_argument("--compile_warmup_intervals", type=int, default=2,
                         help="Like --warmup_intervals, but for --compile_variants arms only. Whole-model "
                              "torch.compile(model) has larger, more variable first-call latency (tens of "
                              "seconds to minutes, possibly spread over several early steps as image-count-"
                              "driven shapes vary) than flex_document_causal's ~1.4-1.7s single-op compile "
                              "that --warmup_intervals=1 was sized for. Full stats_history is always saved, "
                              "so steady-state can be resliced later with a different value if this is off.")
    parser.add_argument("--batch_size", type=int, default=2, help="TrainConfig default -- real production value.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="TrainConfig default.")
    parser.add_argument("--val_size", type=int, default=64,
                         help="The val split is built but never evaluated (eval_in_epochs is forced off); kept small only to keep dataloader setup fast.")
    parser.add_argument("--num_workers", type=int, default=2,
                         help="Keep >=1 -- see module docstring for why num_workers=0 breaks the cross-variant identical-data-order assumption.")
    parser.add_argument("--dataset_cache_dir", type=str,
                         default=os.environ.get("NANOVLM_CACHE_DIR") or os.path.expanduser("~/.cache/vlm-dev/finevision_benchmark_shards"),
                         help="Local shard cache shared across all variants, so only the first touches the network.")
    parser.add_argument("--max_cache_gb", type=float, default=30.0)
    parser.add_argument("--prefetch_shards", type=int, default=1)
    parser.add_argument("--lm_model_type", type=str, default=None, help="Override VLMConfig.lm_model_type (default: real SmolLM2-360M-Instruct).")
    parser.add_argument("--vit_model_type", type=str, default=None, help="Override VLMConfig.vit_model_type (default: real siglip2-base-patch16-512).")
    parser.add_argument("--attn_flex_block_size", type=int, default=None)
    parser.add_argument("--checkpoint_root", type=str, default="checkpoints/benchmark_attn_train",
                         help="Required by VLMConfig, but with eval_in_epochs=False nothing is ever written here.")
    parser.add_argument("--results_file", type=str, default="eval/h100/benchmark_attn_train_results.json",
                         help="Results are grouped under eval/<gpu>/ by the hardware they were measured on.")
    parser.add_argument("--prime_cache_batches", type=int, default=None,
                         help="Defaults to max_training_steps * gradient_accumulation_steps (one variant's worth of micro-batches).")
    parser.add_argument("--skip_prime_cache", action="store_true", help="Skip cache priming, e.g. when re-running against an already-warm cache.")
    args = parser.parse_args()

    if not args.dataset_cache_dir:
        raise ValueError("--dataset_cache_dir resolved empty -- pass it explicitly.")
    os.makedirs(args.dataset_cache_dir, exist_ok=True)

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"GPU: {device_name} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset -- all GPUs visible!')})")

    base_vlm_cfg, base_train_cfg = build_base_configs(args)

    if not args.skip_prime_cache:
        prime_batches = args.prime_cache_batches
        if prime_batches is None:
            prime_batches = args.max_training_steps * args.gradient_accumulation_steps
        prime_shard_cache(base_train_cfg, base_vlm_cfg, prime_batches)

    arms = (
        [(v, False, args.warmup_intervals) for v in args.variants]
        + [(v, True, args.compile_warmup_intervals) for v in args.compile_variants]
    )
    if not arms:
        raise ValueError("Nothing to run -- --variants and --compile_variants are both empty.")

    per_variant = {}
    rows = []
    for packing_impl, use_compile, warmup_intervals in arms:
        label, vlm_cfg, train_cfg = build_arm_configs(base_vlm_cfg, base_train_cfg, packing_impl, use_compile, args.checkpoint_root)
        summary, wall_clock_s = run_arm(label, train_cfg, vlm_cfg)
        steady_state = summarize_variant(label, summary, wall_clock_s, warmup_intervals)
        per_variant[label] = {"summary": summary, "wall_clock_s": wall_clock_s, "steady_state": steady_state}
        rows.append(steady_state)

    print("\n--- Summary ---")
    df = build_comparison_table(rows)
    print(df.to_string(index=False))

    best = df.iloc[0]
    speedup_note = f", {best['speedup_vs_none']:.2f}x vs none" if "speedup_vs_none" in df.columns else ""
    print(f"\nFastest: {best['variant']} ({best['mean_tokens_per_second']:.0f} tokens/s{speedup_note})")

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
