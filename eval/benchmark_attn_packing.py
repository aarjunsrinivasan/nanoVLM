"""Benchmarks attention implementations for training-time sequence packing.

`data/advanced_datasets.py`'s `ConstantLengthDataset` packs several unrelated VQA samples into one
row to fill `lm_max_length` (used for both train and val, `train.py:247-251`). But
`models/language_model.py`'s `LanguageModelGroupedQueryAttention.forward` (lines 207-308) only ever
applies one full causal mask over the whole packed row, with no document-boundary awareness -- a
later sample's tokens causally attend into an earlier, unrelated sample's tokens during training.

This script compares, on synthetic batches shaped like this repo's real packed output:
  - `current_packed_dense_sdpa`: today's actual production attention core (verbatim copy of
    language_model.py:262-287) -- fast, but exhibits the cross-sample leak above.
  - `flex_document_causal_{precomputed_mask,mask_rebuilt_per_iter}`: torch.nn.attention.flex_attention
    with a document-causal BlockMask (`doc_id[q] == doc_id[kv]` ANDed with causal) -- correct, no
    cross-sample leakage, and (unlike a dense block-diagonal SDPA mask) block-sparse so the kernel
    skips cross-document blocks instead of computing-then-masking them.
  - `unpacked_padded_sdpa`: no packing at all (today's safe alternative, one sample per row,
    individually padded) -- reference baseline for what packing of either kind actually buys.
  - `sdpa_gqa_dense_causal` (optional, off by default): `current_packed_dense_sdpa` but with SDPA's
    native `enable_gqa=True` instead of `repeat_interleave` -- isolates "native GQA" speedup from
    flex's "document-mask sparsity" speedup.

Does not modify any models/*.py or data/*.py code -- only exercises LanguageModelGroupedQueryAttention's
real submodules (q_proj/k_proj/v_proj/out_proj, RotaryEmbedding, apply_rotary_pos_embd) directly, so
all variants pay identical, non-reimplemented projection/RoPE/out-proj cost and the benchmarked delta
is isolated to the attention core.

By default also runs a cheap correctness self-check (`verify_no_cross_contamination`) proving the
dense variant leaks across a document boundary and the flex variant doesn't, before trusting any
timing numbers.

Run as a module from the repo root (eval/ is a package), pinned to a single idle GPU per this
repo's CLAUDE.md GPU rules (check `nvidia-smi` first):
    CUDA_VISIBLE_DEVICES=0 python -m eval.benchmark_attn_packing --batch_sizes 2 --num_iters 5 --num_warmup 2

A larger sweep (more batch sizes / block sizes) is a separate, heavier job -- submit via
`sbatch slurm/benchmark_attn_packing.slurm <extra args>` rather than running it interactively.

If timing looks off for the flex variants, `TORCH_LOGS=recompiles python -m eval.benchmark_attn_packing ...`
will show whether a shape/guard change is forcing an unexpected recompile mid-sweep.
"""
import argparse
import json
import os
import random

import pandas as pd
import torch
import torch._dynamo
import torch.nn.functional as F
from torch.nn.attention.flex_attention import and_masks, create_block_mask, flex_attention

from eval.benchmark_fwd_bwd import NOOP, run_phase
from models.config import VLMConfig
from models.language_model import LanguageModelGroupedQueryAttention, RotaryEmbedding, apply_rotary_pos_embd

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)


# --------------------------------------------------------------------------------------------- #
# Synthetic packed-batch construction, shaped like ConstantLengthDataset's real output
# --------------------------------------------------------------------------------------------- #

def _sample_lengths(rng, seq_length, avg_len, std, min_len, max_len):
    """Draws sub-sample lengths until they sum to exactly seq_length, mirroring
    ConstantLengthDataset._pack_one_group's invariant that a packed row never exceeds seq_length
    (data/advanced_datasets.py:261-262) -- the final draw is truncated to fit.

    The remaining-budget cap must be applied LAST, after the min_len floor, not before: if
    max(min_len, min(..., remaining)) were used instead, a remaining budget smaller than min_len
    would get floored back up past it, overshooting seq_length on the final segment (surfaced by
    a real crash during testing -- torch.arange(length) not matching the space actually left in
    the row)."""
    lengths, total = [], 0
    while total < seq_length:
        remaining = seq_length - total
        length = min(max(min_len, min(max_len, int(rng.gauss(avg_len, std)))), remaining)
        lengths.append(length)
        total += length
    return lengths


def build_packed_batch(batch_size, seq_length, hidden_dim, avg_sample_length, sample_length_std,
                        min_sample_length, max_sample_length, device, seed):
    """Builds a synthetic batch shaped like ConstantLengthDataset's real output: each row is a
    flush concatenation of several unrelated sub-samples summing to exactly seq_length. Returns
    random hidden states (benchmarking attention compute, not model quality) plus the bookkeeping
    tensors each attention variant needs.

    hidden_states is always float32 -- like the real token_embedding table's output
    (models/vision_language_model.py:64), it's the model's stored (unmodified) dtype; the bf16/fp16
    autocast happens per-op inside the timed region, matching eval/benchmark_fwd_bwd.py's pattern.

    Returns:
        hidden_states: [B, T, D] float32
        doc_id: [B, T] long -- which packed sub-sample owns each position (row-local)
        continuous_position_ids: [B, T] -- today's actual RoPE behavior (no reset at boundaries)
        reset_position_ids: [B, T] -- resets to 0 at each doc_id boundary (the RoPE half of the fix)
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    hidden_states = torch.randn(batch_size, seq_length, hidden_dim, device=device, generator=generator)

    doc_id = torch.zeros(batch_size, seq_length, dtype=torch.long, device=device)
    continuous_position_ids = torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1).clone()
    reset_position_ids = torch.empty(batch_size, seq_length, dtype=torch.long, device=device)

    rng = random.Random(seed)
    docs_per_row = []
    for b in range(batch_size):
        lengths = _sample_lengths(rng, seq_length, avg_sample_length, sample_length_std, min_sample_length, max_sample_length)
        docs_per_row.append(len(lengths))
        pos = 0
        for doc_idx, length in enumerate(lengths):
            doc_id[b, pos:pos + length] = doc_idx
            reset_position_ids[b, pos:pos + length] = torch.arange(length, device=device)
            pos += length

    print(f"Packed batch: seq_length={seq_length}, docs/row min={min(docs_per_row)} "
          f"max={max(docs_per_row)} avg={sum(docs_per_row) / len(docs_per_row):.1f}")

    return hidden_states, doc_id, continuous_position_ids, reset_position_ids


def build_unpacked_batch(doc_id, hidden_states):
    """Explodes a packed [B,T] row into one row per sub-sample, individually right-padded to the
    flattened batch's max sub-sample length -- what VQACollator does without packing. Quantifies
    what packing of either kind actually buys vs. not packing at all."""
    B, T = doc_id.shape
    D = hidden_states.size(-1)
    rows = []
    for b in range(B):
        boundaries = (torch.where(doc_id[b, 1:] != doc_id[b, :-1])[0] + 1).tolist()
        bounds = [0, *boundaries, T]
        rows.extend(hidden_states[b, s:e] for s, e in zip(bounds[:-1], bounds[1:]))

    max_len = max(r.size(0) for r in rows)
    x = torch.zeros(len(rows), max_len, D, device=hidden_states.device, dtype=hidden_states.dtype)
    pad_mask = torch.zeros(len(rows), max_len, dtype=torch.bool, device=hidden_states.device)
    for i, r in enumerate(rows):
        x[i, :r.size(0)] = r
        pad_mask[i, :r.size(0)] = True

    position_ids = torch.arange(max_len, device=x.device).unsqueeze(0).expand(len(rows), -1).clone()
    return x, pad_mask, position_ids


# --------------------------------------------------------------------------------------------- #
# Shared plumbing -- real q_proj/k_proj/v_proj/out_proj + RoPE, so only the attention core differs
# --------------------------------------------------------------------------------------------- #

def build_shared_attn_module(cfg, device):
    """Params stay float32 (real training never hard-casts the model -- see models/config.py's
    lm_loss_impl comment on 'gather' being the H100-fastest path under bf16 *autocast*). The
    bf16/fp16 autocast is applied per-call at each variant's timed region instead, matching how
    train.py/eval/benchmark_fwd_bwd.py actually run this model."""
    return LanguageModelGroupedQueryAttention(cfg).to(device=device).eval()


def project_qkv(attn, x, cos, sin):
    """Replays language_model.py:231-236 verbatim via the real submodules, so q/k/v can't
    silently drift from production."""
    B, T, _ = x.shape
    q = attn.q_proj(x).view(B, T, attn.n_heads, attn.head_dim).transpose(1, 2)
    k = attn.k_proj(x).view(B, T, attn.n_kv_heads, attn.head_dim).transpose(1, 2)
    v = attn.v_proj(x).view(B, T, attn.n_kv_heads, attn.head_dim).transpose(1, 2)
    q, k = apply_rotary_pos_embd(q, k, cos, sin)
    return q, k, v


def run_attention_core(attn, core_fn, x, cos, sin):
    """core_fn(q, k, v) -> [B, n_heads, T, head_dim]. Shares q/k/v/out_proj cost across all
    variants so the benchmarked delta is isolated to the attention-core call."""
    q, k, v = project_qkv(attn, x, cos, sin)
    y = core_fn(q, k, v)
    B, H, T, hd = y.shape
    return attn.out_proj(y.transpose(1, 2).contiguous().view(B, T, H * hd))


# --------------------------------------------------------------------------------------------- #
# Attention cores: today's production path, FlexAttention document masking, GQA-native SDPA
# --------------------------------------------------------------------------------------------- #

def dense_sdpa_core(q, k, v, n_kv_groups, pad_mask, dropout_p):
    """Verbatim copy of language_model.py:262-287's SDPA branch (today's actual production
    attention core) -- k/v repeat_interleave'd to n_heads, one dense causal+padding additive mask
    over the whole packed row, with no document-boundary awareness. This IS the bug being
    benchmarked; if language_model.py's SDPA branch changes, re-sync this copy by line number."""
    k_exp = k.repeat_interleave(n_kv_groups, dim=1)
    v_exp = v.repeat_interleave(n_kv_groups, dim=1)
    T, T_kv = q.size(2), k_exp.size(2)

    additive_mask = (1.0 - pad_mask[:, None, None, :T_kv].float()) * torch.finfo(q.dtype).min
    causal = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool)).view(1, 1, T, T)
    causal_bias = torch.zeros_like(causal, dtype=q.dtype).masked_fill(~causal, torch.finfo(q.dtype).min)

    return F.scaled_dot_product_attention(
        q, k_exp, v_exp, attn_mask=additive_mask + causal_bias, dropout_p=dropout_p, is_causal=False,
    )


def dense_block_diagonal_sdpa_core(q, k, v, n_kv_groups, doc_id, pad_mask, dropout_p):
    """Molmo2-style fix (Ai2's production multimodal LLM, local checkout at
    /home/asrinivasan/vlm_gen/molmo2 -- olmo/models/molmo2/molmo2.py:682-698): AND one extra
    broadcast doc-id-equality compare into the same dense causal+padding mask
    dense_sdpa_core already builds, fed to plain SDPA. No block-sparse kernel, so this pays the
    same full O(T^2) attention compute as today's buggy path (no compute saved), but unlike
    flex_document_causal's create_block_mask, building the mask itself costs almost nothing extra
    -- Molmo2 rebuilds it fresh every step this way (olmo/data/dynamic_packer.py:220-222 repacks
    every batch, same as this repo's ConstantLengthDataset)."""
    k_exp = k.repeat_interleave(n_kv_groups, dim=1)
    v_exp = v.repeat_interleave(n_kv_groups, dim=1)
    T, T_kv = q.size(2), k_exp.size(2)

    causal = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool)).view(1, 1, T, T)
    same_doc = (doc_id.unsqueeze(2) == doc_id.unsqueeze(1)).unsqueeze(1)  # [B,1,T,T]
    allowed = causal & same_doc & pad_mask[:, None, None, :T_kv]
    additive_bias = torch.zeros_like(allowed, dtype=q.dtype).masked_fill(~allowed, torch.finfo(q.dtype).min)

    return F.scaled_dot_product_attention(
        q, k_exp, v_exp, attn_mask=additive_bias, dropout_p=dropout_p, is_causal=False,
    )


def dense_sdpa_gqa_core(q, k, v, pad_mask, dropout_p):
    """Same as dense_sdpa_core but uses SDPA's native enable_gqa=True instead of repeat_interleave --
    isolates "native GQA" speedup from flex_document_causal's "document-mask sparsity" speedup,
    which that variant otherwise bundles into a single number."""
    T, T_kv = q.size(2), k.size(2)
    additive_mask = (1.0 - pad_mask[:, None, None, :T_kv].float()) * torch.finfo(q.dtype).min
    causal = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool)).view(1, 1, T, T)
    causal_bias = torch.zeros_like(causal, dtype=q.dtype).masked_fill(~causal, torch.finfo(q.dtype).min)

    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=additive_mask + causal_bias, dropout_p=dropout_p, is_causal=False, enable_gqa=True,
    )


def _causal_mask_mod(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def _make_document_mask_mod(doc_id):
    def document_mask_mod(b, h, q_idx, kv_idx):
        return doc_id[b, q_idx] == doc_id[b, kv_idx]
    return document_mask_mod


def make_doc_causal_block_mask(doc_id, seq_length, device, block_size, create_block_mask_fn=create_block_mask):
    """create_block_mask is expensive by construction -- determining whether a block is fully
    sparse requires evaluating mask_mod at every point in the block (PyTorch's own docs call this
    out), and with a different doc_id every training step (real ConstantLengthDataset behavior)
    there's nothing to amortize across steps unless create_block_mask_fn itself is compiled
    (pass torch.compile(create_block_mask); see --compile_mask). A fresh mask_mod closure is built
    here each call, but it shares the same underlying code object every time, which is what lets
    torch.compile recognize repeat calls as the same traced graph rather than recompiling."""
    mask_mod = and_masks(_causal_mask_mod, _make_document_mask_mod(doc_id))
    return create_block_mask_fn(mask_mod, B=doc_id.size(0), H=None, Q_LEN=seq_length, KV_LEN=seq_length,
                                 device=device, BLOCK_SIZE=block_size)


def flex_core(compiled_flex_attention, block_mask):
    def core(q, k, v):
        return compiled_flex_attention(q, k, v, block_mask=block_mask, enable_gqa=True)
    return core


# --------------------------------------------------------------------------------------------- #
# Correctness self-check
# --------------------------------------------------------------------------------------------- #

def verify_no_cross_contamination(cfg, device, seq_len=64, atol=1e-4, rtol=1e-3, seed=0):
    """2-document synthetic row (doc 0 = first half, doc 1 = second half). Perturbing doc 0's
    tokens must change dense packed-causal SDPA's output at a doc-1 position (that's the
    contamination bug) but must NOT change flex's document-causal output (the fix). Also checks
    both implementations agree exactly at doc-0 positions, where no earlier doc exists to leak
    from either way -- proof flex agrees with dense where dense is already correct, and diverges
    exactly where dense is wrong."""
    torch.manual_seed(seed)
    half = seq_len // 2

    doc_id = torch.zeros(1, seq_len, dtype=torch.long, device=device)
    doc_id[:, half:] = 1

    attn = build_shared_attn_module(cfg, device)
    rotary = RotaryEmbedding(cfg).to(device)
    reset_pos = torch.cat([torch.arange(half), torch.arange(seq_len - half)]).unsqueeze(0).to(device)
    cos, sin = rotary(reset_pos)
    pad_mask = torch.ones(1, seq_len, dtype=torch.bool, device=device)
    block_mask = make_doc_causal_block_mask(doc_id, seq_len, device, block_size=min(32, seq_len))

    def dense(x):
        core = lambda q, k, v: dense_sdpa_core(q, k, v, attn.n_kv_groups, pad_mask, 0.0)
        return run_attention_core(attn, core, x, cos, sin)

    def flex(x):
        return run_attention_core(attn, flex_core(flex_attention, block_mask), x, cos, sin)

    x = torch.randn(1, seq_len, cfg.lm_hidden_dim, device=device)
    probe = seq_len - 1  # last token: doc 1, causally sees all of doc 0

    dense_before, flex_before = dense(x)[:, probe], flex(x)[:, probe]

    x2 = x.clone()
    x2[:, :half] += 10.0 * torch.randn(1, half, cfg.lm_hidden_dim, device=device)
    dense_after, flex_after = dense(x2)[:, probe], flex(x2)[:, probe]

    dense_changed = not torch.allclose(dense_before, dense_after, atol=atol, rtol=rtol)
    flex_unchanged = torch.allclose(flex_before, flex_after, atol=atol, rtol=rtol)
    doc0_agree = torch.allclose(dense(x)[:, :half], flex(x)[:, :half], atol=atol, rtol=rtol)

    assert dense_changed, "Expected dense packed SDPA to leak across documents -- it didn't; check the synthetic setup."
    assert flex_unchanged, "Expected flex's document mask to block cross-document leakage -- it didn't; check mask_mod/doc_id."
    assert doc0_agree, "dense and flex disagree even where no cross-doc leakage is possible -- something else is wrong."

    print(f"[verify] dense packed SDPA leaks across documents:            {dense_changed} (expected True)")
    print(f"[verify] flex document-causal blocks cross-document leakage:  {flex_unchanged} (expected True)")
    print(f"[verify] dense and flex agree pre-boundary (doc 0, no leak possible either way): {doc0_agree} (expected True)")


# --------------------------------------------------------------------------------------------- #
# Benchmark driver
# --------------------------------------------------------------------------------------------- #

def run_variant(name, core_builder, x, cos, sin, attn, device, autocast_dtype, args, do_profile):
    """core_builder() -> core_fn(q, k, v). Called fresh inside the timed region on every measured
    call (not just once), so variants that rebuild state per-call (e.g. a fresh BlockMask) have
    that cost captured -- and variants that don't just pay a cheap closure allocation.

    Wraps the actual attention call in torch.autocast, not a hard model/tensor dtype cast --
    mirrors eval/benchmark_fwd_bwd.py's forward_setup/full_step_fn (real training never casts
    model params directly to bf16)."""
    results = []

    if "forward" in args.passes:
        def forward_fn():
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=autocast_dtype):
                run_attention_core(attn, core_builder(), x, cos, sin)

        latency_s, peak_mb = run_phase(NOOP, forward_fn, device, f"{name}_forward", args, do_profile)
        print(f"{name:45s} forward:           {latency_s * 1000:8.2f} ms/iter, peak VRAM {peak_mb:8.1f} MB")
        results.append({"variant": name, "pass": "forward", "latency_ms": latency_s * 1000, "peak_vram_mb": peak_mb})

    if "forward_backward" in args.passes:
        def full_step_fn():
            for p in attn.parameters():
                p.grad = None
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                y = run_attention_core(attn, core_builder(), x, cos, sin)
            y.float().pow(2).sum().backward()

        latency_s, peak_mb = run_phase(NOOP, full_step_fn, device, f"{name}_forward_backward", args, do_profile)
        print(f"{name:45s} forward+backward:  {latency_s * 1000:8.2f} ms/iter, peak VRAM {peak_mb:8.1f} MB")
        results.append({"variant": name, "pass": "forward_backward", "latency_ms": latency_s * 1000, "peak_vram_mb": peak_mb})

    return results


ALL_VARIANTS = [
    "current_packed_dense_sdpa",
    "dense_block_diagonal_sdpa",
    "flex_document_causal_precomputed_mask",
    "flex_document_causal_mask_rebuilt_per_iter",
    "unpacked_padded_sdpa",
    "sdpa_gqa_dense_causal",
]
DEFAULT_VARIANTS = [v for v in ALL_VARIANTS if v != "sdpa_gqa_dense_causal"]


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark attention implementations for training-time sequence packing "
                    "(dense packed SDPA vs. FlexAttention document masking vs. no packing)."
    )
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[2], help="2 = TrainConfig.batch_size (real per-GPU value).")
    parser.add_argument("--seq_length", type=int, default=4096, help="= lm_max_length / ConstantLengthDataset.seq_length.")
    parser.add_argument("--avg_sample_length", type=float, default=262.0,
                         help="mp_image_token_length(64) + 198, ConstantLengthDataset._average_length_per_sample's own constant.")
    parser.add_argument("--sample_length_std", type=float, default=96.0)
    parser.add_argument("--min_sample_length", type=int, default=16)
    parser.add_argument("--max_sample_length", type=int, default=4096)
    parser.add_argument("--block_size", type=int, nargs="+", default=[128], help="create_block_mask BLOCK_SIZE; try 64 too (avg sub-sample ~262 tokens is only ~2 blocks at 128).")
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS, choices=ALL_VARIANTS)
    parser.add_argument("--passes", nargs="+", default=["forward", "forward_backward"], choices=["forward", "forward_backward"])
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--compile_flex", dest="compile_flex", action="store_true", default=True)
    parser.add_argument("--no_compile_flex", dest="compile_flex", action="store_false", help="Flex needs compilation for real perf; only disable to debug.")
    parser.add_argument("--compile_mask", dest="compile_mask", action="store_true", default=True,
                         help="Wrap create_block_mask itself in torch.compile for flex_document_causal_mask_rebuilt_per_iter "
                              "(the PyTorch-recommended way to make it cheap enough to call fresh every step).")
    parser.add_argument("--no_compile_mask", dest="compile_mask", action="store_false")
    parser.add_argument("--num_warmup", type=int, default=3)
    parser.add_argument("--num_iters", type=int, default=10)
    parser.add_argument("--row_limit", type=int, default=20, help="Rows to show in each profiler table.")
    parser.add_argument("--trace_dir", default=None, help="If set, export a Chrome trace per phase to this directory.")
    parser.add_argument("--profile", action="store_true", help="Attach torch.profiler op tables to every variant/pass.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verify", dest="verify", action="store_true", default=True)
    parser.add_argument("--skip_verify", dest="verify", action="store_false")
    parser.add_argument("--verify_only", action="store_true", help="Run only the correctness self-check and exit (no GPU sweep).")
    parser.add_argument("--results_file", default="eval/h200/benchmark_attn_packing_results.json",
                         help="Results are grouped under eval/<gpu>/ by the hardware they were measured on.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset -- all GPUs visible!')})")

    cfg = VLMConfig()
    autocast_dtype = getattr(torch, args.dtype)

    if args.verify or args.verify_only:
        print("\n--- Correctness self-check: does packing actually leak, and does the fix actually stop it? ---")
        verify_no_cross_contamination(cfg, device, seed=args.seed)
    if args.verify_only:
        return

    compiled_flex_attention = torch.compile(flex_attention, dynamic=False) if args.compile_flex else flex_attention
    compiled_create_block_mask = torch.compile(create_block_mask, dynamic=False) if args.compile_mask else create_block_mask

    all_results = []
    for batch_size in args.batch_sizes:
        torch._dynamo.reset()

        hidden_states, doc_id, continuous_pos, reset_pos = build_packed_batch(
            batch_size, args.seq_length, cfg.lm_hidden_dim, args.avg_sample_length, args.sample_length_std,
            args.min_sample_length, args.max_sample_length, device, args.seed,
        )
        attn = build_shared_attn_module(cfg, device)
        rotary = RotaryEmbedding(cfg).to(device)
        cos_cont, sin_cont = rotary(continuous_pos)
        cos_reset, sin_reset = rotary(reset_pos)
        pad_mask = torch.ones(batch_size, args.seq_length, dtype=torch.bool, device=device)

        x_unpacked, pad_mask_unpacked, pos_unpacked = build_unpacked_batch(doc_id, hidden_states)
        cos_unpacked, sin_unpacked = rotary(pos_unpacked)

        for block_size in args.block_size:
            print(f"\n=== batch_size={batch_size} seq_length={args.seq_length} block_size={block_size} ===")

            precomputed_block_mask = None
            if "flex_document_causal_precomputed_mask" in args.variants:
                precomputed_block_mask = make_doc_causal_block_mask(doc_id, args.seq_length, device, block_size)

            for name in args.variants:
                if name == "current_packed_dense_sdpa":
                    core_builder = lambda: (lambda q, k, v: dense_sdpa_core(q, k, v, attn.n_kv_groups, pad_mask, 0.0))
                    x, cos, sin = hidden_states, cos_cont, sin_cont
                elif name == "dense_block_diagonal_sdpa":
                    core_builder = lambda: (lambda q, k, v: dense_block_diagonal_sdpa_core(q, k, v, attn.n_kv_groups, doc_id, pad_mask, 0.0))
                    x, cos, sin = hidden_states, cos_reset, sin_reset
                elif name == "sdpa_gqa_dense_causal":
                    core_builder = lambda: (lambda q, k, v: dense_sdpa_gqa_core(q, k, v, pad_mask, 0.0))
                    x, cos, sin = hidden_states, cos_cont, sin_cont
                elif name == "flex_document_causal_precomputed_mask":
                    core_builder = lambda: flex_core(compiled_flex_attention, precomputed_block_mask)
                    x, cos, sin = hidden_states, cos_reset, sin_reset
                elif name == "flex_document_causal_mask_rebuilt_per_iter":
                    core_builder = lambda: flex_core(
                        compiled_flex_attention,
                        make_doc_causal_block_mask(doc_id, args.seq_length, device, block_size, compiled_create_block_mask),
                    )
                    x, cos, sin = hidden_states, cos_reset, sin_reset
                elif name == "unpacked_padded_sdpa":
                    core_builder = lambda: (lambda q, k, v: dense_sdpa_core(q, k, v, attn.n_kv_groups, pad_mask_unpacked, 0.0))
                    x, cos, sin = x_unpacked, cos_unpacked, sin_unpacked
                else:
                    raise ValueError(f"Unknown variant: {name}")

                variant_results = run_variant(name, core_builder, x, cos, sin, attn, device, autocast_dtype, args, args.profile)
                for r in variant_results:
                    r.update({"batch_size": batch_size, "seq_length": args.seq_length, "block_size": block_size})
                all_results.extend(variant_results)

    print("\n--- Summary ---")
    df = pd.DataFrame(all_results)
    print(df.to_string(index=False))

    os.makedirs(os.path.dirname(args.results_file) or ".", exist_ok=True)
    with open(args.results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {args.results_file}")


if __name__ == "__main__":
    main()
