import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from eval.eval_checkpoint import DATA_KEYS, build_val_loader, git_sha, load_vlm_cfg
from data.processors import get_tokenizer
from data.shard_cache import load_or_create_manifest
from models.language_model import packing_impl_override
from models.vision_language_model import VisionLanguageModel
import models.config as config

MASKS = (("doc_masked", "dense_block_diagonal"), ("unmasked", "none"))
DOC_INDEX_BUCKETS = (0, 1, 2)  # last bucket is "2+"
PRECEDING_LEN_EDGES = (0, 1, 512, 1024, 2048, 4097)  # bucket i = [edges[i], edges[i+1])


def doc_index_bucket(d):
    return d if d < DOC_INDEX_BUCKETS[-1] else DOC_INDEX_BUCKETS[-1]


def preceding_len_bucket(n):
    for i in range(len(PRECEDING_LEN_EDGES) - 1):
        if PRECEDING_LEN_EDGES[i] <= n < PRECEDING_LEN_EDGES[i + 1]:
            return i
    return len(PRECEDING_LEN_EDGES) - 2


@torch.no_grad()
def per_doc_stats(model, val_loader, device, image_token_id, mp_image_token_length, max_rows=None):
    """One record per (mask, row, doc). `row` is a running counter *local to this one traversal only*
    -- see the module docstring: packed-row grouping is randomly reshuffled per traversal, so `row`
    (and therefore doc_index/preceding_len for a given raw sample) is not comparable across different
    calls to this function. Aggregate within one (mask) traversal's records; never join across calls
    by (row, doc_index)."""
    loss_impl = model.cfg.lm_loss_impl
    model.cfg.lm_loss_impl = "full"
    out = {}
    try:
        for name, impl in MASKS:
            records, row = [], 0
            with packing_impl_override(model.decoder, impl):
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    labels = batch["labels"].to(device)
                    doc_id = batch["doc_id"].to(device)
                    attn = batch["attention_mask"].to(device)
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type in ("cuda", "cpu") else torch.float16):
                        logits, _ = model(input_ids, batch["images"], attention_mask=attn, targets=labels, doc_id=doc_id)
                    tok_loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), labels.reshape(-1),
                                               ignore_index=-100, reduction="none").reshape(labels.shape)
                    valid = labels != -100
                    B = input_ids.size(0)
                    for b in range(B):
                        row_doc = doc_id[b]
                        uniq = sorted(int(d) for d in torch.unique(row_doc).tolist() if d >= 0)
                        preceding = 0
                        for d in uniq:
                            dm = row_doc == d
                            dv = dm & valid[b]
                            n_valid = int(dv.sum())
                            doc_len = int(dm.sum())
                            if n_valid > 0:
                                n_tiles = int((input_ids[b][dm] == image_token_id).sum()) // mp_image_token_length
                                records.append({"row": row, "doc_index": d, "doc_len": doc_len, "preceding_len": preceding,
                                                "n_valid_tokens": n_valid, "loss_sum": float(tok_loss[b][dv].sum()), "n_tiles": n_tiles})
                            preceding += doc_len
                        row += 1
                    if max_rows is not None and row >= max_rows:
                        break
            out[name] = records
    finally:
        model.cfg.lm_loss_impl = loss_impl
    return out


def check_full_pass_totals(all_stats, max_rows):
    """The one cross-traversal invariant that packing-order randomness (see module docstring)
    cannot break: a full (non-`--max_rows`) traversal exhausts the exact same fixed val set every
    time, so total document count and total valid-token count must be exactly equal across all 8
    (checkpoint, mask) traversals. Skipped for `--max_rows` runs, which legitimately see a random
    partial subset each time (same reason eval_checkpoint.py labels those "PARTIAL")."""
    if max_rows is not None:
        print("[eval_per_doc] --max_rows set: skipping the full-pass total-count cross-check (expected to vary)")
        return
    totals = {(ckpt, mask): (len(recs), sum(r["n_valid_tokens"] for r in recs))
             for ckpt, masks in all_stats.items() for mask, recs in masks.items()}
    ref_key, ref_val = next(iter(totals.items()))
    bad = {k: v for k, v in totals.items() if v != ref_val}
    if bad:
        raise RuntimeError(f"full-pass (doc_count, total_valid_tokens) should be identical across every "
                           f"(checkpoint, mask) traversal (same fixed val set every time) but {ref_key} saw "
                           f"{ref_val} while these differ: {bad}")
    print(f"[eval_per_doc] full-pass total check OK: every traversal saw {ref_val[0]} docs, {ref_val[1]} valid tokens")


def weighted_mean(records, key_loss="loss_sum", key_n="n_valid_tokens"):
    n = sum(r[key_n] for r in records)
    return (sum(r[key_loss] for r in records) / n) if n else float("nan"), n


def write_report(all_stats, ckpt_meta, path):
    lines = ["# Per-document val loss by packing position\n",
            "Token-weighted mean cross-entropy (nats/token). N = valid (non-ignored) token count.\n"]

    lines.append("## Doc-index-0 control (no preceding context either way -- doc_masked and unmasked should")
    lines.append("nearly agree; a large gap here would mean the two masks aren't actually equivalent for doc 0)\n")
    lines.append("Note: doc_masked and unmasked are separate traversals with independently randomized packing, so this")
    lines.append("compares two overlapping-but-not-identical populations of \"whichever documents landed at doc_index 0\",")
    lines.append("not one fixed set scored twice -- still a meaningful check since doc_index 0 is always that row's")
    lines.append("longest document, a fairly stable population either way.\n")
    lines.append("| checkpoint | doc_masked | unmasked | \\|Δ\\| | N |")
    lines.append("|---|--:|--:|--:|--:|")
    for ckpt in all_stats:
        d0 = {m: [r for r in recs if r["doc_index"] == 0] for m, recs in all_stats[ckpt].items()}
        lm, nm = weighted_mean(d0["doc_masked"])
        lu, nu = weighted_mean(d0["unmasked"])
        lines.append(f"| {ckpt_meta[ckpt]['label']} | {lm:.4f} | {lu:.4f} | {abs(lm - lu):.4f} | {min(nm, nu)} |")

    lines.append("\n## Own-mask loss by doc-index bucket (the mask each arm actually trained with:")
    lines.append("unmasked for A, doc_masked for C -- this is the A-vs-C-by-position comparison)\n")
    lines.append("| doc_index | " + " | ".join(ckpt_meta[c]["label"] for c in all_stats) + " |")
    lines.append("|---" * (1 + len(all_stats)) + "|")
    for bucket in DOC_INDEX_BUCKETS:
        cells = []
        for ckpt in all_stats:
            own_mask = ckpt_meta[ckpt]["own_mask"]
            recs = [r for r in all_stats[ckpt][own_mask] if doc_index_bucket(r["doc_index"]) == bucket]
            m, n = weighted_mean(recs)
            cells.append(f"{m:.4f} (N={n})")
        label = str(bucket) if bucket < DOC_INDEX_BUCKETS[-1] else f"{bucket}+"
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines.append("\n## Own-mask loss by preceding-context-length bucket (doc_index >= 1 only, since doc 0 always has 0)\n")
    edges = PRECEDING_LEN_EDGES
    bucket_labels = [f"[{edges[i]},{edges[i+1]})" for i in range(len(edges) - 1)]
    lines.append("| preceding tokens | " + " | ".join(ckpt_meta[c]["label"] for c in all_stats) + " |")
    lines.append("|---" * (1 + len(all_stats)) + "|")
    for bi, blabel in enumerate(bucket_labels):
        if bi == 0:
            continue  # [0,1) is exactly doc_index==0, already shown above
        cells = []
        for ckpt in all_stats:
            own_mask = ckpt_meta[ckpt]["own_mask"]
            recs = [r for r in all_stats[ckpt][own_mask] if r["doc_index"] >= 1 and preceding_len_bucket(r["preceding_len"]) == bi]
            m, n = weighted_mean(recs)
            cells.append(f"{m:.4f} (N={n})" if n else "n/a")
        lines.append(f"| {blabel} | " + " | ".join(cells) + " |")

    lines.append("\n## Counterfactual mask, for reference (A under doc_masked, C under unmasked -- not what either arm trained with)\n")
    lines.append("| doc_index | " + " | ".join(ckpt_meta[c]["label"] for c in all_stats) + " |")
    lines.append("|---" * (1 + len(all_stats)) + "|")
    for bucket in DOC_INDEX_BUCKETS:
        cells = []
        for ckpt in all_stats:
            other_mask = "unmasked" if ckpt_meta[ckpt]["own_mask"] == "doc_masked" else "doc_masked"
            recs = [r for r in all_stats[ckpt][other_mask] if doc_index_bucket(r["doc_index"]) == bucket]
            m, n = weighted_mean(recs)
            cells.append(f"{m:.4f} (N={n})")
        label = str(bucket) if bucket < DOC_INDEX_BUCKETS[-1] else f"{bucket}+"
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines.append("\n## How to read this")
    lines.append("- This is the training-distribution (FineVision) val set, not benchmark data -- benchmark inference never")
    lines.append("  packs documents (`doc_id=None` for both arms), so this cannot explain which benchmark task moved (see plan).")
    lines.append("- Descriptive/correlational: it shows *where* the aggregate doc-masked-vs-unmasked val-loss gap concentrates,")
    lines.append("  not that the mechanism *caused* any specific downstream benchmark result.")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--dataset_cache_dir", default="~/.cache/finevision_shards")
    p.add_argument("--val_size", type=int, default=5000)
    p.add_argument("--trained_val_size", type=int, default=5000)
    p.add_argument("--allow_train_overlap", action="store_true")
    p.add_argument("--batch_size", type=int, default=2, help="matches the 10k A/B's --batch_size 2")
    p.add_argument("--max_rows", type=int, default=None, help="smoke test: score only this many packed val rows per mask")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_cache_gb", type=float, default=30.0)
    p.add_argument("--out_dir", default="eval_results/lmms_ab")
    args = p.parse_args()

    ckpts = [os.path.abspath(os.path.expanduser(c)) for c in args.checkpoints]
    cfgs = [load_vlm_cfg(c) for c in ckpts]
    for c, cfg in zip(ckpts[1:], cfgs[1:]):
        diff = [k for k in DATA_KEYS if getattr(cfg, k) != getattr(cfgs[0], k)]
        if diff:
            raise SystemExit(f"{c} differs from {ckpts[0]} in {diff}")

    tc = config.TrainConfig()
    tc.dataset_cache_dir = os.path.abspath(os.path.expanduser(args.dataset_cache_dir))
    tc.val_size, tc.batch_size, tc.max_cache_gb, tc.num_workers = args.val_size, args.batch_size, args.max_cache_gb, 1
    os.makedirs(tc.dataset_cache_dir, exist_ok=True)
    manifest = load_or_create_manifest(tc.train_dataset_path, tc.dataset_cache_dir)
    n_val = math.ceil(args.val_size / manifest.rows_per_shard)
    n_trained_val = math.ceil(args.trained_val_size / manifest.rows_per_shard)
    if n_val > n_trained_val and not args.allow_train_overlap:
        raise SystemExit(f"val_size {args.val_size} reads {n_val} shards, past the {n_trained_val} the checkpoints "
                         f"trained with -- pass --allow_train_overlap to score on their training shards anyway")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval_per_doc] device {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    val_loader = build_val_loader(cfgs[0], tc, args.seed)
    tokenizer = get_tokenizer(cfgs[0].lm_tokenizer, cfgs[0].vlm_extra_tokens, cfgs[0].lm_chat_template)
    image_token_id = tokenizer.convert_tokens_to_ids(tokenizer.image_token)

    # own_mask: the mask each checkpoint actually trained with (from its own config, not assumed A/C).
    ckpt_meta = {}
    all_stats = {}
    os.makedirs(args.out_dir, exist_ok=True)
    for ckpt, cfg in zip(ckpts, cfgs):
        label = "__".join(ckpt.rstrip("/").split("/")[-2:])
        own_mask = "doc_masked" if cfg.lm_attn_packing_impl != "none" else "unmasked"
        ckpt_meta[label] = {"own_mask": own_mask, "label": label}
        t0 = time.time()
        model = VisionLanguageModel.from_pretrained(ckpt).to(device).eval()
        stats = per_doc_stats(model, val_loader, device, image_token_id, cfg.mp_image_token_length, max_rows=args.max_rows)
        all_stats[label] = stats
        n_docs = sum(len(v) for v in stats.values())
        print(f"[eval_per_doc] {label}: {n_docs} doc records ({time.time() - t0:.0f}s)", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    check_full_pass_totals(all_stats, args.max_rows)

    raw_path = os.path.join(args.out_dir, "per_doc_loss.json")
    with open(raw_path, "w") as f:
        json.dump({"checkpoints": ckpt_meta, "stats": all_stats, "max_rows": args.max_rows,
                   "partial": args.max_rows is not None, "val_size": args.val_size, "batch_size": args.batch_size,
                   "git_sha": git_sha()}, f)
    report_path = os.path.join(args.out_dir, "per_doc_summary.md")
    write_report(all_stats, ckpt_meta, report_path)
    print(f"[eval_per_doc] wrote {raw_path} and {report_path}")


if __name__ == "__main__":
    main()
