"""Quality A/B for the 222M nanoVLM: upstream-style baseline vs the current optimized setup.

Trains ONE arm per invocation (a fresh process per arm, so both arms start from an identical RNG
state -- train.py seeds torch once at import, so in-process sequential arms would not share a
model init) with in-loop validation and checkpoints on, and saves the val-loss history as JSON.

    'baseline'  lm_loss_impl='full', lm_attn_packing_impl='none', non_blocking_transfers=False, eager
                (what the repo did before the loss / attention-masking / transfer changes)
    'current'   lm_loss_impl='gather', lm_attn_packing_impl='flex_document_causal',
                non_blocking_transfers=True (+ torch.compile with --compile)

Model: the README's 222M nanoVLM -- siglip-base-patch16-224 + SmolLM2-135M, mp_pixel_shuffle_factor=2
(49 image tokens per tile). Data: streamed FineVision through the same pipeline as every other
A/B in this repo (the cauldron loader is not in this codebase). max_training_steps also sets the
cosine LR schedule length, so both arms see an identical schedule that anneals to the end.

IMPORTANT -- do not read the in-loop val_loss as the comparison. The two arms score validation
under different attention masks ('none' lets later packed samples attend into earlier unrelated
ones; 'flex_document_causal' does not), and each validation pass re-packs its batches. Re-score
the saved checkpoints on a frozen val set under one common mask with eval/rescore_common_mask.py.

Run from the repo root, pinned to one idle GPU (check nvidia-smi first), one arm after the other:

    CUDA_VISIBLE_DEVICES=0 python -m eval.run_quality_ab --arm baseline
    CUDA_VISIBLE_DEVICES=0 python -m eval.run_quality_ab --arm current

Dry run first:

    CUDA_VISIBLE_DEVICES=0 python -m eval.run_quality_ab --arm current \
        --max_training_steps 10 --eval_interval 5 --stats_log_interval 5 --val_size 64
"""
import argparse
import dataclasses
import json
import os
import time

import torch

import train
import models.config as config

ARMS = {
    "baseline": dict(lm_loss_impl="full", lm_attn_packing_impl="none", non_blocking_transfers=False),
    "current": dict(lm_loss_impl="gather", lm_attn_packing_impl="flex_document_causal", non_blocking_transfers=True),
}

# The README's 222M model (from the initial commit's models/config.py, adapted to the current
# config): SigLIP-B/16-224 (85M) + SmolLM2-135M, 224px tiles, 2x2 pixel shuffle -> 49 tokens/tile.
MODEL_222M = dict(
    vit_hidden_dim=768,
    vit_inter_dim=4 * 768,
    vit_patch_size=16,
    vit_img_size=224,
    vit_n_heads=12,
    vit_n_blocks=12,
    vit_model_type="google/siglip-base-patch16-224",
    lm_hidden_dim=576,
    lm_inter_dim=1536,
    lm_n_heads=9,
    lm_n_kv_heads=3,
    lm_n_blocks=30,
    lm_model_type="HuggingFaceTB/SmolLM2-135M",
    # lm_tokenizer stays SmolLM2-360M-Instruct's: same 49152-token vocab as SmolLM2-135M.
    mp_pixel_shuffle_factor=2,
    mp_image_token_length=49,
)


def json_safe(o):
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))):
        return None
    return o


def build_configs(args):
    vlm_cfg = config.VLMConfig(hf_repo_name=None, **MODEL_222M, **ARMS[args.arm])
    tag = f"{args.arm}_seed{args.seed}"
    vlm_cfg.vlm_checkpoint_path = os.path.join(args.checkpoint_root, tag)
    train_cfg = config.TrainConfig(
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_training_steps=args.max_training_steps,
        eval_in_epochs=True,
        eval_interval=args.eval_interval,
        stats_log_interval=args.stats_log_interval,
        val_size=args.val_size,
        num_workers=args.num_workers,
        dataset_cache_dir=args.dataset_cache_dir,
        max_cache_gb=args.max_cache_gb,
        prefetch_shards=args.prefetch_shards,
        compile=bool(args.compile and args.arm == "current"),
        use_lmms_eval=False,  # needs Slurm and is meaningless at 3k steps
        log_wandb=args.log_wandb,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        wandb_group=args.wandb_group,
        run_name_suffix=tag,
    )
    return vlm_cfg, train_cfg


def main():
    parser = argparse.ArgumentParser(description="Train one arm of the 222M baseline-vs-current quality A/B.")
    parser.add_argument("--arm", required=True, choices=list(ARMS))
    parser.add_argument("--seed", type=int, default=0,
                        help="Model-init / bf16 seed. Data order is fixed by train.py's own seeds, so a "
                             "different --seed measures init noise only, not data-order noise.")
    parser.add_argument("--max_training_steps", type=int, default=3000, help="Optimizer steps; also the LR schedule length.")
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--stats_log_interval", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--val_size", type=int, default=5000,
                        help="Held-out samples taken off the front of the stream (train starts after them, "
                             "identically in both arms).")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--compile", action="store_true", help="torch.compile the 'current' arm (baseline stays eager).")
    parser.add_argument("--dataset_cache_dir", type=str,
                        default=os.environ.get("NANOVLM_CACHE_DIR")
                        or os.path.expanduser("~/.cache/vlm-dev/finevision_benchmark_shards"))
    parser.add_argument("--max_cache_gb", type=float, default=30.0)
    parser.add_argument("--prefetch_shards", type=int, default=1)
    parser.add_argument("--checkpoint_root", type=str, default="checkpoints/quality_ab_222m")
    parser.add_argument("--log_wandb", action="store_true")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="nanoVLM")
    parser.add_argument("--wandb_group", type=str, default="quality_ab_222m")
    parser.add_argument("--results_dir", type=str, default="eval/profile/quality_ab")
    args = parser.parse_args()

    if args.wandb_entity is None:
        args.wandb_entity = config.TrainConfig.wandb_entity
    os.makedirs(args.dataset_cache_dir, exist_ok=True)

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"GPU: {device_name} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset -- all GPUs visible!')})")
    if device_name == "cpu":
        raise RuntimeError("No GPU visible -- check CUDA_VISIBLE_DEVICES and LD_LIBRARY_PATH (CUDA compat libs).")

    vlm_cfg, train_cfg = build_configs(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print(f"\n=== Arm: {args.arm} (seed {args.seed}) ===")
    print(vlm_cfg)
    print(train_cfg)
    start = time.time()
    summary = train.train(train_cfg, vlm_cfg)
    wall_clock_s = time.time() - start
    if summary is None:
        raise RuntimeError("train.train() returned None -- the run did not complete max_training_steps.")

    results = {
        "provenance": {**train.get_provenance(), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")},
        "harness_config": vars(args),
        "vlm_cfg": dataclasses.asdict(vlm_cfg),
        "wall_clock_s": wall_clock_s,
        "summary": summary,
    }
    os.makedirs(args.results_dir, exist_ok=True)
    path = os.path.join(args.results_dir, f"{args.arm}_seed{args.seed}_results.json")
    with open(path, "w") as f:
        json.dump(json_safe(results), f, indent=2, default=str)
    print(f"\nval_history: {summary['val_history']}")
    print(f"Saved results to {path}")


if __name__ == "__main__":
    main()
