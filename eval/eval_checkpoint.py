import argparse
import dataclasses
import json
import math
import os
import subprocess
import time

import torch
from torch.utils.data import DataLoader

import train  # eval_under_masks, seed_worker
import models.config as config
from data.advanced_datasets import ConstantLengthDataset
from data.collators import VQACollator
from data.datasets import VQADataset
from data.processors import get_image_processor, get_tokenizer
from data.shard_cache import get_cached_train_val_datasets, load_or_create_manifest
from models.vision_language_model import VisionLanguageModel

# VLMConfig fields that decide which val tokens a row turns into: every checkpoint scored in one call must agree.
DATA_KEYS = ("lm_max_length", "max_img_size", "vit_img_size", "resize_to_max_side_len", "lm_tokenizer",
             "vlm_extra_tokens", "lm_chat_template", "mp_image_token_length")


def load_vlm_cfg(ckpt):
    with open(os.path.join(ckpt, "config.json")) as f:
        return config.VLMConfig(**json.load(f))


def git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    except Exception:
        return None


def build_val_loader(vlm_cfg, tc, seed):
    """The val DataLoader of train.get_dataloaders, without the train half (which would start train workers)."""
    image_processor = get_image_processor(vlm_cfg.max_img_size, vlm_cfg.vit_img_size, vlm_cfg.resize_to_max_side_len)
    tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer, vlm_cfg.vlm_extra_tokens, vlm_cfg.lm_chat_template)
    _, val_ds = get_cached_train_val_datasets(tc, 1, 0)  # the train dataset is never iterated: no train shard downloads
    val_dataset = VQADataset(val_ds, tokenizer, image_processor, vlm_cfg.mp_image_token_length,
                             tc.relevance_min_rating, tc.image_correspondence_min_rating,
                             tc.visual_dependency_min_rating, tc.formatting_min_rating)
    val_dataset = ConstantLengthDataset(val_dataset, infinite=False, max_sample_length=tc.max_sample_length,
                                        seq_length=vlm_cfg.lm_max_length, num_of_sequences=tc.batch_size * 4, queue_size=8,
                                        max_images_per_example=tc.max_images_per_example,
                                        max_images_per_knapsack=tc.max_images_per_knapsack)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(val_dataset, batch_size=tc.batch_size, collate_fn=VQACollator(tokenizer, vlm_cfg.lm_max_length),
                      num_workers=1, pin_memory=True, persistent_workers=False, drop_last=True,
                      worker_init_fn=train.seed_worker, generator=g)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", nargs="+", help="checkpoint dirs holding config.json + model.safetensors")
    p.add_argument("--dataset_cache_dir", default="~/.cache/finevision_shards")
    p.add_argument("--val_size", type=int, default=5000, help="raw val rows (the 10k A/B used 5000)")
    p.add_argument("--trained_val_size", type=int, default=5000,
                   help="the --val_size the checkpoints were trained with; sets where their train shards start")
    p.add_argument("--allow_train_overlap", action="store_true", help="score on shards the checkpoints trained on")
    p.add_argument("--batch_size", type=int, default=2, help="val batch size; the val packing buffer is 4x this")
    p.add_argument("--max_rows", type=int, default=None, help="smoke test: score only this many packed val rows")
    p.add_argument("--seed", type=int, default=0, help="val DataLoader generator seed (only reorders rows within a buffer)")
    p.add_argument("--max_cache_gb", type=float, default=30.0)
    p.add_argument("--out_dir", default="eval_results/checkpoint_val")
    args = p.parse_args()

    ckpts = [os.path.abspath(os.path.expanduser(c)) for c in args.checkpoints]
    cfgs = [load_vlm_cfg(c) for c in ckpts]
    for c, cfg in zip(ckpts[1:], cfgs[1:]):
        diff = [k for k in DATA_KEYS if getattr(cfg, k) != getattr(cfgs[0], k)]
        if diff:
            raise SystemExit(f"{c} differs from {ckpts[0]} in {diff}: score them in separate calls (different val tokens)")

    tc = config.TrainConfig()
    tc.dataset_cache_dir = os.path.abspath(os.path.expanduser(args.dataset_cache_dir))
    tc.val_size, tc.batch_size, tc.max_cache_gb, tc.num_workers = args.val_size, args.batch_size, args.max_cache_gb, 1

    os.makedirs(tc.dataset_cache_dir, exist_ok=True)
    manifest = load_or_create_manifest(tc.train_dataset_path, tc.dataset_cache_dir)
    n_val = math.ceil(args.val_size / manifest.rows_per_shard)
    n_trained_val = math.ceil(args.trained_val_size / manifest.rows_per_shard)
    print(f"[eval_checkpoint] manifest {manifest.repo_id}@{manifest.revision[:12]}, {len(manifest.files)} shards, "
          f"{manifest.rows_per_shard} rows in the first; val_size {args.val_size} = first {n_val} shards")
    if n_val > n_trained_val and not args.allow_train_overlap:
        raise SystemExit(f"val_size {args.val_size} reads {n_val} shards, but checkpoints trained with val_size "
                         f"{args.trained_val_size} started training at shard {n_trained_val}: shards "
                         f"{n_trained_val}-{n_val - 1} are training data. Lower --val_size or pass --allow_train_overlap")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval_checkpoint] device {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    val_loader = build_val_loader(cfgs[0], tc, args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    for ckpt in ckpts:
        t0 = time.time()
        model = VisionLanguageModel.from_pretrained(ckpt).to(device).eval()
        losses = train.eval_under_masks(model, val_loader, device, max_rows=args.max_rows)
        elapsed = time.time() - t0
        result = {
            "checkpoint": ckpt,
            **losses,
            "max_rows": args.max_rows,
            "partial": args.max_rows is not None,
            "val_size": args.val_size,
            "val_shards": n_val,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "manifest_revision": manifest.revision,
            "manifest_rows_per_shard": manifest.rows_per_shard,
            "train_cfg_used": dataclasses.asdict(tc),
            "git_sha": git_sha(),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "elapsed_s": round(elapsed, 1),
        }
        name = "__".join(ckpt.rstrip("/").split("/")[-2:]) + (f"__rows{args.max_rows}" if args.max_rows else "")
        path = os.path.join(args.out_dir, name + ".json")
        with open(path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"[eval_checkpoint] {ckpt}: doc_masked {losses['doc_masked']:.4f}  unmasked {losses['unmasked']:.4f}  "
              f"({elapsed:.0f}s{', PARTIAL' if args.max_rows else ''}) -> {path}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
