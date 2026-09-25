import argparse
import contextlib
import json
import os

import torch
import torch.nn.functional as F

from eval.eval_checkpoint import DATA_KEYS, build_val_loader, git_sha, load_vlm_cfg
from models.language_model import LanguageModelGroupedQueryAttention, packing_impl_override
from models.vision_language_model import VisionLanguageModel
import models.config as config

_orig_attn_forward = LanguageModelGroupedQueryAttention.forward
_orig_sdpa = torch.nn.functional.scaled_dot_product_attention
_state = None      # None outside `capture(...)`; else the accumulator dict `capture` was given
_layer_idx = None  # None outside the wrapped attention forward of a module in _state["layer_of"]


def _patched_attn_forward(self, *args, **kwargs):
    global _layer_idx
    if _state is None or id(self) not in _state["layer_of"]:
        return _orig_attn_forward(self, *args, **kwargs)
    _layer_idx = _state["layer_of"][id(self)]
    try:
        return _orig_attn_forward(self, *args, **kwargs)
    finally:
        _layer_idx = None


def _patched_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, **kw):
    if _state is None or _layer_idx is None:
        return _orig_sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kw)

    doc_id = _state["doc_id"]
    B, H, Tq, D = q.shape
    Tk = k.shape[2]
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * (D ** -0.5)
    if is_causal:
        causal = torch.tril(torch.ones(Tq, Tk, device=q.device, dtype=torch.bool)).view(1, 1, Tq, Tk)
        scores = scores.masked_fill(~causal, float("-inf"))
    if attn_mask is not None:
        scores = scores + attn_mask.float()
    probs = torch.softmax(scores, dim=-1)  # [B, H, Tq, Tk]

    doc_q = doc_id[:, :Tq]
    doc_k = doc_id[:, :Tk]
    other = doc_k.unsqueeze(1) != doc_q.unsqueeze(2)                     # [B, Tq, Tk]: key's doc differs from query's
    first_real = (doc_k >= 0).float().argmax(dim=1)                     # [B]: first non-pad key position (the sink)
    sink = F.one_hot(first_real, num_classes=Tk).bool().unsqueeze(1)    # [B, 1, Tk]
    other = other & ~sink
    valid_q = (doc_q > 0).float()                                        # [B, Tq]: real tokens not in doc_index 0

    mass = (probs * other.unsqueeze(1).float()).sum(dim=-1)             # [B, H, Tq]: other-doc mass per query
    weighted = mass * valid_q.unsqueeze(1)
    _state["sum_mass"][_layer_idx] += weighted.sum(dim=(0, 2)).detach().to("cpu", torch.float64)
    _state["count"][_layer_idx] += float(valid_q.sum())

    return torch.matmul(probs.to(v.dtype), v)


def new_accumulator(model):
    attn_modules = [m for m in model.decoder.modules() if isinstance(m, LanguageModelGroupedQueryAttention)]
    n_heads = attn_modules[0].n_heads
    return {"sum_mass": {i: torch.zeros(n_heads, dtype=torch.float64) for i in range(len(attn_modules))},
            "count": {i: 0.0 for i in range(len(attn_modules))}}


@contextlib.contextmanager
def capture(model, doc_id, accum):
    """accum: an externally-owned dict from `new_accumulator`, mutated in place -- reuse the same
    one across many calls so stats accumulate over a whole val pass instead of resetting per batch."""
    global _state
    attn_modules = [m for m in model.decoder.modules() if isinstance(m, LanguageModelGroupedQueryAttention)]
    accum["layer_of"] = {id(m): i for i, m in enumerate(attn_modules)}
    accum["doc_id"] = doc_id
    _state = accum
    LanguageModelGroupedQueryAttention.forward = _patched_attn_forward
    torch.nn.functional.scaled_dot_product_attention = _patched_sdpa
    try:
        yield
    finally:
        LanguageModelGroupedQueryAttention.forward = _orig_attn_forward
        torch.nn.functional.scaled_dot_product_attention = _orig_sdpa
        _state = None


@contextlib.contextmanager
def decoupled_packing_override(language_model, lm_impl, attn_impl):
    """Like models.language_model.packing_impl_override, but sets the LanguageModel's own
    packing_impl (governs RoPE position reset, computed once per forward and shared by every block)
    independently from every LanguageModelGroupedQueryAttention's packing_impl (governs that layer's
    masking) -- these are separate attributes on separate objects; packing_impl_override just always
    sets them to the same value, which isn't required by the model's forward logic (see module
    docstring). Use lm_impl='dense_block_diagonal' to keep position reset ON without also requesting
    a flex block mask (that only gets built for lm_impl=='flex_document_causal')."""
    attn_modules = [m for m in language_model.modules() if isinstance(m, LanguageModelGroupedQueryAttention)]
    saved_lm = language_model.packing_impl
    saved_attn = [m.packing_impl for m in attn_modules]
    try:
        language_model.packing_impl = lm_impl
        for m in attn_modules:
            m.packing_impl = attn_impl
        yield
    finally:
        language_model.packing_impl = saved_lm
        for m, v in zip(attn_modules, saved_attn):
            m.packing_impl = v


@torch.no_grad()
def run(model, val_loader, device, max_rows, force_mask="none", keep_position_reset=False):
    """force_mask: the packing_impl to force on every attention module regardless of what it was
    trained with ('none' or 'dense_block_diagonal', see packing_impl_override). For a checkpoint
    trained with 'none' (arm A), force_mask='none' just reproduces its own inference-time behavior.
    For a checkpoint trained with masking (arm C), force_mask='none' is a counterfactual: "what would
    this model's attention do if the mask it was trained under were removed."

    keep_position_reset: only meaningful with force_mask='none'. By default (False), removing the
    mask via packing_impl_override also removes the per-document RoPE position reset (both gated by
    the same flag there), so the measurement conflates "mask removed" with "unfamiliar continuous
    positions." Setting this True uses decoupled_packing_override instead, forcing masking off while
    leaving position behavior exactly as trained -- isolates the mask's own contribution."""
    accum = new_accumulator(model)
    rows = 0
    if keep_position_reset:
        assert force_mask == "none", "keep_position_reset is only meaningful with force_mask='none'"
        ctx = decoupled_packing_override(model.decoder, lm_impl="dense_block_diagonal", attn_impl="none")
    else:
        ctx = packing_impl_override(model.decoder, force_mask)
    with ctx:
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            doc_id = batch["doc_id"].to(device)
            attn = batch["attention_mask"].to(device)
            with capture(model, doc_id, accum):
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type in ("cuda", "cpu") else torch.float16):
                    model(input_ids, batch["images"], attention_mask=attn, targets=None, doc_id=doc_id)
            rows += input_ids.size(0)
            if max_rows is not None and rows >= max_rows:
                break
    return accum, rows


def selftest():
    """CPU-only correctness check, no model/GPU needed: 2 documents (lengths 3 and 2), no padding,
    1 head, head_dim 4. Verifies (a) the patched path returns the same output as real SDPA, and
    (b) the accumulated other-doc mass matches a hand-computed value."""
    torch.manual_seed(0)
    B, H, T, D = 1, 1, 5, 4
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    doc_id = torch.tensor([[0, 0, 0, 1, 1]])
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool)).view(1, 1, T, T)
    attn_mask = torch.zeros(1, 1, T, T).masked_fill(~causal, float("-inf"))  # causal only, no padding

    expected = _orig_sdpa(q, k, v, attn_mask=attn_mask, is_causal=False)

    global _state, _layer_idx
    accum = {"sum_mass": {0: torch.zeros(H, dtype=torch.float64)}, "count": {0: 0.0}, "doc_id": doc_id}
    _state, _layer_idx = accum, 0
    try:
        got = _patched_sdpa(q, k, v, attn_mask=attn_mask, is_causal=False)
    finally:
        _state, _layer_idx = None, None
    assert torch.allclose(got, expected, atol=1e-5), (got - expected).abs().max()

    # Hand check: valid queries are positions 3,4 (doc_id==1, i.e. > 0). Position 0 is the sink (first
    # real key) and is excluded. So query 3 can only place "other-doc" mass on keys {1, 2} (doc 0, not
    # the sink); query 4 the same {1, 2} (position 3 is doc 1 = query's own doc, excluded by `other`).
    probs = torch.softmax((q.float() @ k.float().transpose(-2, -1)) / (D ** 0.5) + attn_mask, dim=-1)[0, 0]
    want = probs[3, 1] + probs[3, 2] + probs[4, 1] + probs[4, 2]
    assert abs(float(accum["sum_mass"][0].sum()) - float(want)) < 1e-5
    assert accum["count"][0] == 2.0

    # Outside `capture`, the patch must be a no-op even if left installed (defensive: it never is,
    # since `capture` always restores originals in `finally`, but assert the gating logic directly).
    assert _patched_sdpa is not _orig_sdpa
    global_state_before = (_state, _layer_idx)
    out = _patched_sdpa(q, k, v, attn_mask=attn_mask, is_causal=False)
    assert torch.allclose(out, expected, atol=1e-5)
    assert (_state, _layer_idx) == global_state_before == (None, None)

    # decoupled_packing_override: real LanguageModelGroupedQueryAttention instances (cheap to build,
    # no VLM/GPU needed) behind a minimal stand-in "language_model" (only needs .packing_impl and
    # .modules()). Checks the LM-level and attention-level flags diverge inside the context and both
    # restore correctly afterward, including when they *started* from different values (the case
    # packing_impl_override could never produce, since it always sets one shared value).
    class _FakeCfg:
        lm_n_heads, lm_n_kv_heads, lm_hidden_dim, lm_dropout = 2, 1, 4, 0.0
        lm_attn_packing_impl, lm_attn_flex_block_size = "none", 128

    class _FakeLM:
        def __init__(self, attn_modules, packing_impl):
            self.packing_impl = packing_impl
            self._attn_modules = attn_modules

        def modules(self):
            return self._attn_modules

    attn1, attn2 = LanguageModelGroupedQueryAttention(_FakeCfg()), LanguageModelGroupedQueryAttention(_FakeCfg())
    attn1.packing_impl, attn2.packing_impl = "dense_block_diagonal", "none"  # deliberately mismatched to start
    lm = _FakeLM([attn1, attn2], packing_impl="flex_document_causal")
    with decoupled_packing_override(lm, lm_impl="dense_block_diagonal", attn_impl="none"):
        assert lm.packing_impl == "dense_block_diagonal"
        assert attn1.packing_impl == "none" and attn2.packing_impl == "none"
    assert lm.packing_impl == "flex_document_causal"
    assert attn1.packing_impl == "dense_block_diagonal" and attn2.packing_impl == "none"

    print("selftest OK")


def write_report(results, path):
    n_layers = max(len(r["mean_by_layer"]) for r in results.values())
    lines = ["# Attention mass on earlier documents\n",
            "Mean softmax probability mass each real, non-first-document query token places on keys",
            "from an earlier, unrelated packed document (excluding the row's attention-sink token),",
            "averaged over heads. N = number of query tokens the mean is over.\n"]
    lines.append("| layer | " + " | ".join(f"{c} mean (N)" for c in results) + " |")
    lines.append("|---" * (1 + len(results)) + "|")
    for layer in range(n_layers):
        cells = []
        for ckpt, r in results.items():
            m, n = r["mean_by_layer"][layer], r["count_by_layer"][layer]
            cells.append(f"{m:.4f} ({int(n)})")
        lines.append(f"| {layer} | " + " | ".join(cells) + " |")
    lines.append("\n## How to read this")
    lines.append("- Descriptive, not causal: confirms the mechanism is real and its rough size, nothing more.")
    lines.append("- This is the training-distribution val set; benchmark inference never packs documents for either arm,")
    lines.append("  so this cannot explain which benchmark task moved (see plan) -- pair qualitatively with")
    lines.append("  eval_per_doc.py's own-mask loss-by-position table for the training-distribution mechanism story.")
    lines.append("- For an A_s* column, this is just how that model already runs (it trained with no masking) --")
    lines.append("  what its own attention actually does. For a plain C_s* column, `--force_mask none` is a")
    lines.append("  counterfactual: C trained *with* masking, so this measures what its attention would do if that")
    lines.append("  mask were removed, not how it behaves under its own training/inference regime (which gives")
    lines.append("  exactly 0 by construction and isn't shown here). Forcing this off also removes C's per-document")
    lines.append("  RoPE position reset at the same time (both are gated by the same packing_impl flag), so a plain")
    lines.append("  C_s* number reflects that combined effect, not masking removal in isolation.")
    lines.append("- A `... (reset kept)` column ran with `--keep_position_reset`: masking forced off exactly as above,")
    lines.append("  but RoPE position reset left ON (decoupled_packing_override) -- isolates the mask's own")
    lines.append("  contribution from the position-scheme change. Compare it to the plain column for the same")
    lines.append("  checkpoint: similar values means the position-scheme change wasn't doing much of the work;")
    lines.append("  a large gap means it was.")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", nargs="*", help="any checkpoint dirs -- see --force_mask for what's measured on each")
    p.add_argument("--force_mask", default="none", choices=("none", "dense_block_diagonal"),
                   help="packing_impl to force on every checkpoint regardless of what it trained with (default: "
                        "'none', i.e. 'how much does this model's attention spread to other docs when unmasked'). "
                        "For an arm-C checkpoint this is a counterfactual -- see the module docstring and plan for "
                        "the RoPE-position-reset confound that comes bundled with it. 'dense_block_diagonal' should "
                        "measure ~0 for every checkpoint (confirmatory, not run by default).")
    p.add_argument("--keep_position_reset", action="store_true",
                   help="only meaningful with --force_mask none (the default): mask forced off as usual, but keep "
                        "RoPE per-document position reset ON (decoupled_packing_override) instead of also switching "
                        "to continuous positions -- isolates the mask's own effect from the position-scheme change. "
                        "Results are labeled '<checkpoint> (reset kept)', a separate column from the plain run.")
    p.add_argument("--dataset_cache_dir", default="~/.cache/finevision_shards")
    p.add_argument("--val_size", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--max_rows", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_cache_gb", type=float, default=30.0)
    p.add_argument("--out_dir", default="eval_results/lmms_ab")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    if args.selftest:
        return selftest()
    if not args.checkpoints:
        p.error("checkpoints required unless --selftest")

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[attn_mass] device {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    val_loader = build_val_loader(cfgs[0], tc, args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    raw_path = os.path.join(args.out_dir, "attn_mass.json")
    results = {}
    if os.path.exists(raw_path):
        try:
            results = json.load(open(raw_path))["results"]
        except Exception as e:  # noqa: BLE001
            print(f"[attn_mass] warning: couldn't load existing {raw_path} to merge into ({e}), starting fresh")
    for ckpt in ckpts:
        label = "__".join(ckpt.rstrip("/").split("/")[-2:])
        if args.keep_position_reset:
            label += " (reset kept)"
        model = VisionLanguageModel.from_pretrained(ckpt).to(device).eval()
        accum, rows = run(model, val_loader, device, args.max_rows, force_mask=args.force_mask,
                          keep_position_reset=args.keep_position_reset)
        n_layers = len(accum["sum_mass"])
        n_heads = accum["sum_mass"][0].numel()
        # mean over heads AND queries (not sum over heads) -- a probability mass, so must stay in [0,1].
        mean_by_layer = [float(accum["sum_mass"][i].sum() / (accum["count"][i] * n_heads)) if accum["count"][i] else float("nan")
                         for i in range(n_layers)]
        results[label] = {"mean_by_layer": mean_by_layer, "count_by_layer": [accum["count"][i] for i in range(n_layers)],
                          "mean_by_layer_head": {i: (accum["sum_mass"][i] / max(accum["count"][i], 1)).tolist()
                                                  for i in range(n_layers)}, "rows_scored": rows}
        overall = sum(accum["sum_mass"][i].sum() for i in range(n_layers)) / (sum(accum["count"].values()) * n_heads)
        print(f"[attn_mass] {label}: {rows} rows, overall mean other-doc mass {overall:.4f}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with open(raw_path, "w") as f:
        json.dump({"results": results, "max_rows": args.max_rows, "batch_size": args.batch_size, "git_sha": git_sha()}, f, indent=1)
    report_path = os.path.join(args.out_dir, "attn_mass_summary.md")
    write_report(results, report_path)
    print(f"[attn_mass] wrote {raw_path} and {report_path}")


if __name__ == "__main__":
    main()
