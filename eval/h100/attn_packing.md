# Attention packing: cross-document masking results (H100)

nanoVLM packs many short samples into one 4096-token row to avoid padding waste. The packing that
shipped does **not** mask across the documents it packs, so every sample attends to the samples
before it in its row. This is a correctness bug, not a tuning knob: `VLMConfig.lm_attn_packing_impl`
defaults to `'none'` purely for backward compatibility
([`models/config.py`](../../models/config.py)).

These are the measurements behind the two fixes — `'dense_block_diagonal'` (Molmo2-style, folds
`doc_id[q] == doc_id[kv]` into the dense SDPA mask) and `'flex_document_causal'` (a FlexAttention
`BlockMask`) — on one H100, both as isolated attention-core math and in the real training loop.

Raw JSON in this directory is **committed**, so the tables below can be checked against the data
that produced them:

- [`benchmark_attn_packing_results.json`](benchmark_attn_packing_results.json) — attention-core microbenchmark
- [`benchmark_attn_train_results.json`](benchmark_attn_train_results.json) — end-to-end training A/B

## Environment

- Single NVIDIA H100 80GB HBM3, driver 570.124.06 with the CUDA 13 forward-compat libraries
  (`cuda-compat-13-0` 580.178.04) — the pinned torch cu130 wheels need driver ≥580, so
  `LD_LIBRARY_PATH` must include `/usr/local/cuda-13.0/compat` (see
  [`scripts/setup_pod.sh`](../../scripts/setup_pod.sh)). Without it torch silently falls back to CPU.
- torch `2.14.0+cu130`, bfloat16.
- Code state: branch `feat/attn-packing`, with `train.py`'s committed conflict markers resolved
  (`97ed280`). Everything except the `flex_document_causal` + compile arm was measured before
  `e733b77` removed `train.py`'s refusal of that combination; see that section below.

## Correctness first: the leak is real

`eval/benchmark_attn_packing.py` runs a self-check before timing anything, and it is the reason the
rest of this document matters. All three assertions held:

| Check | Result |
|---|---|
| Dense packed SDPA (`'none'`) leaks across documents | **True** (expected True) |
| `flex_document_causal` blocks cross-document leakage | **True** (expected True) |
| Dense and flex agree pre-boundary (doc 0, where neither can leak) | **True** (expected True) |

So `'none'` is measurably wrong, and the flex path is measurably correct — the numbers below are
the price of that correctness.

### The fix has two halves: the mask and the RoPE positions

Packing breaks two things, and both fixes (`dense_block_diagonal`, `flex_document_causal`) repair
both:

1. **Attention crosses documents.** Fixed by the document mask (`doc_id[q] == doc_id[kv]`).
2. **RoPE positions run continuously across the row**, so the third document in a row starts at,
   say, position 700 instead of 0. Fixed by resetting position ids to 0 at every document boundary
   (`_compute_reset_position_ids`, called from `LanguageModel.forward`).

The second half is subtler than it looks. RoPE scores depend only on *relative* position
(`q_m · k_n` is a function of `m − n`), so once the mask confines attention to one document, a
constant per-document offset **cancels exactly**. Measured on a packed row of 1,768 tokens: correct
mask with continuous positions matches the unpacked reference to 1.9e-06, which is float noise.

The one thing that breaks the cancellation is RoPE's **dynamic scaling**, which reads the
*absolute* maximum position:

```python
max_seq = position_ids.max() + 1                 # absolute, across the whole packed row
if max_seq > self.original_max_seq_len:          # models/language_model.py:178
    inv_freq = self.inv_freq / (max_seq / self.original_max_seq_len)
```

With continuous positions, a packed row's `max_seq` is its full length, so packing several short
documents can trigger scaling that none of them would trigger alone. That rescales `inv_freq` for
**every** document in the row. Same 1,768-token row with `lm_max_position_embeddings=1024`:

| RoPE positions | max \|diff\| vs unpacked |
|---|---|
| reset per document (what ships) | 9.5e-07 — match |
| continuous | **3.8e-03 — mismatch**, ~40× the test tolerance |

**In the current config this is latent**, not live: packed rows are capped at `lm_max_length=4096`,
below `lm_max_position_embeddings=8192`, so scaling never fires. It becomes live the moment
`lm_max_length` exceeds the backbone's original context, or a backbone with a shorter one is
swapped in.

**It was also untested.** The packed-vs-unpacked tests run in the non-scaling regime, where
continuous positions pass comfortably. Verified by mutation: deleting the reset call from
`LanguageModel.forward` left all four existing packing tests green. Three tests now run in the
scaling regime (`tests/test_vision_language_model_packing.py`): each fix must match the unpacked
reference, and a negative control asserts that continuous positions do **not**. Against the same
mutation, both positive tests fail.

## Attention-core microbenchmark

`batch_size=2`, `seq_length=4096`, `block_size=128`, 17 documents per row, `--num_warmup 3
--num_iters 10`. Isolated attention math on synthetic tensors — no model, no data.

| variant | forward (ms) | forward peak VRAM (MB) | fwd+bwd (ms) | fwd+bwd peak VRAM (MB) |
|---|---|---|---|---|
| `current_packed_dense_sdpa` (`'none'`, **leaky baseline**) | 1.46 | 559 | 3.87 | 638 |
| `dense_block_diagonal_sdpa` | 1.71 | 653 | 4.10 | 688 |
| `flex_document_causal_precomputed_mask` | **0.59** | **309** | **1.92** | **404** |
| `flex_document_causal_mask_rebuilt_per_iter` | 1.19 | 309 | 2.37 | 404 |
| `unpacked_padded_sdpa` (no packing at all) | 1.29 | 506 | 3.16 | 671 |

Reading this:

- **`flex_document_causal` is strictly better than the leaky baseline**: 2.01× faster on fwd+bwd
  (1.92 vs 3.87 ms) at 37% less peak VRAM (404 vs 638 MB). Correctness is free here — it is not a
  trade-off at all.
- **Rebuilding the BlockMask every iteration costs 0.45 ms** on fwd+bwd (2.37 vs 1.92). Worth
  caching the mask when the packing layout is stable across steps.
- **`dense_block_diagonal` is the slowest option** (4.10 ms, 688 MB) — 6% slower and 8% more VRAM
  than the leaky baseline. It materializes a full `[T, T]` boolean mask, so it pays for the
  correctness that flex gets structurally. Its value is that it is `torch.compile`-safe, which flex
  is not (see Caveats).
- Packing itself is worth less than it looks at the attention core: `unpacked_padded_sdpa` beats the
  leaky packed baseline (3.16 vs 3.87 ms). Packing's real win is upstream, in how many real tokens
  a step processes — which is what the end-to-end numbers measure.

## End-to-end training A/B

Real `train.py` loop, one run per arm: SmolLM2-360M-Instruct + siglip2-base-patch16-512
(460,113,984 params), streamed FineVision, `batch_size=2` × 8 gradient accumulation (effective 16),
200 steps, `stats_log_interval=25`. Throughput and memory are **steady state** — the first logging
interval is dropped per arm (two for the compiled arm) to exclude startup and compilation.

The `fw+bw` column in both tables below is **CPU launch time, not device time.** These runs read
`train.py`'s per-interval `avg_fw_bw_time`, which at the time was stopped immediately after
`loss.backward()` returned — before any device sync — so the GPU tail was charged to
`post_process` (see `speed_230m/README.md` †). This matters most here, because two of these arms
are compiled and two are not, and `torch.compile` cuts CPU launch cost specifically: the
`0.351 → 0.203` narrowing therefore **overstates** what compilation does to kernel time. The
`tokens/s`, memory and `speedup` columns are unaffected — they come from `batch_duration`, which
encloses `loss.item()` and so a genuine sync — and no conclusion below rests on `fw+bw`. A
`torch.cuda.synchronize()` now precedes that timer, so future runs measure device time; these are
left as measured. The attention-core microbenchmark above is unaffected: it times through
`eval/benchmark_fwd_bwd.py`, which synchronizes around every measured region.

| arm | tokens/s | peak alloc (GiB) | peak reserved (GiB) | fw+bw (s) | speedup vs `none` | last loss |
|---|---|---|---|---|---|---|
| `dense_block_diagonal` + `torch.compile` | **19,635** | **42.46** | 45.99 | 0.203 | **1.64×** | 0.8970 |
| `flex_document_causal` | 15,315 | 47.47 | 47.62 | 0.272 | 1.28× | 0.8967 |
| `none` (leaky baseline) | 11,990 | 50.26 | 50.51 | 0.351 | 1.00× | 0.8976 |
| `dense_block_diagonal` | 11,942 | 50.26 | 50.47 | 0.351 | 0.996× | 0.8995 |

**The headline: fixing the cross-document leak is free, and in two of three configurations it is
strictly faster than leaving the bug in.**

- **`dense_block_diagonal` eager costs nothing.** 11,942 vs 11,990 tokens/s is a 0.4% difference
  against a leaky baseline — inside run-to-run noise, with identical peak memory (50.26 GiB both).
  The microbenchmark predicted a 6% attention-core penalty; end to end that disappears, because
  attention is a small slice of a step dominated by the ViT, the MLP blocks and the LM head.
- **`flex_document_causal` is 1.28× faster and uses 5.5% less memory** (47.47 vs 50.26 GiB) while
  being the only arm that provably blocks the leak. Correctness is better than free.
- **`dense_block_diagonal` + `torch.compile` is the best configuration overall**: 1.64× throughput
  and 15.5% less peak allocated memory (42.46 vs 50.26 GiB). It is also the only *correct* arm that
  can be compiled at all — see Caveats.
- **The arms agree on loss** (0.8967–0.8995 at step 200), which is the sanity check that they saw
  the same data and the same optimization trajectory. It is not a precision claim; numerical
  equivalence of the masking paths is covered by `tests/test_attn_packing.py` and
  `tests/test_vision_language_model_packing.py`.

### `flex_document_causal` + `torch.compile`: the fastest configuration

`train.py` used to refuse this combination, citing silent cross-document corruption when
`--compile`'s whole-model `torch.compile` nests around flex's already-compiled inner calls. That
refusal has been removed on the evidence below. This arm was measured just before the removal, with
the refusal bypassed in memory and the committed arms' exact `harness_config`; the code path is
otherwise identical to what now ships. Reproduce with
`python -m eval.benchmark_attn_train --variants --compile_variants flex_document_causal`.

| arm | tokens/s | peak alloc (GiB) | fw+bw (s) | speedup vs `none` |
|---|---|---|---|---|
| **`flex_document_causal` + `torch.compile`** | **20,440** | **39.83** | 0.196 | **1.70×** |
| `dense_block_diagonal` + `torch.compile` | 19,635 | 42.46 | 0.203 | 1.64× |

It is the fastest and leanest configuration measured: **+4.1% throughput and −6.2% peak memory**
over compiled dense. Raw data:
[`benchmark_attn_train_flex_compile_results.json`](benchmark_attn_train_flex_compile_results.json).
It hit the same image-count recompile limit as compiled dense (see Caveats) at the same point, so
the comparison between the two is like-for-like.

**The corruption the refusal cited does not reproduce.** Tested against the same oracle as the
committed tests — a packed row must match each document run alone — at the real language model's
dimensions (361,884,480 params: hidden 960, 32 blocks, 15/5 heads), a 1,768-token row whose
document boundaries fall mid-block, forward and all 290 gradient tensors:

| arm | batch | fwd max \|diff\| | grad worst rel err | vs correct-noise band |
|---|---|---|---|---|
| `flex_document_causal` + compile | 1 | 4.77e-06 | 4.50e-06 | 1.0× |
| `flex_document_causal` + compile | 2, different layouts per row | 4.77e-06 | 4.31e-06 | 0.7× fwd / 1.0× grad |
| `none` (negative control) | 1 | 2.94e+00 | 1.64e+00 | **615,872×** |

The "correct-noise band" is the error of the known-correct eager `flex` and `dense` paths; fp32
noise over 32 layers makes a fixed tolerance unreliable, so each arm is judged against them. The
batch-2 case matters because the BlockMask carries a batch dimension, and a per-row mix-up under
compile would be invisible at batch 1.

The committed test that replaced the refusal is
`test_flex_document_causal_under_torch_compile_matches_unpacked_reference` in
`tests/test_vision_language_model_packing.py`: a smaller model, but the same geometry — 1,768-token
rows with mid-block boundaries, two rows with different layouts, forward and every gradient — so a
regression in newer torch fails CI rather than silently training on leaked attention. Pointed at
`'none'`, the same check fails at the first cross-boundary document. The evidence covers torch
`2.14.0+cu130`; the refusal may have been right on the version it was written against.

### Operational recommendation

Use `flex_document_causal` with `--compile` for training runs — the fastest and leanest
configuration measured, and correct. Without `--compile`, `flex_document_causal` is still the
fastest correct option. `dense_block_diagonal` + `--compile` is a close second (−4% throughput) and
has no dependency on FlexAttention. Do not use `'none'`: it is both wrong and, in every
configuration measured here, not actually faster.

### Do not read wall-clock epoch time as throughput

Per-arm `Average time per epoch` was 652s (`none`), 653s (`dense_block_diagonal`), 546s
(`flex_document_causal`) and 727s (`dense_block_diagonal` + compile). The compiled arm has the
**highest** epoch time and the **highest** steady-state throughput at the same time: `torch.compile`
pays its one-off compilation inside the measured epoch, and that cost is amortized over 200 steps
here but would vanish over a real multi-thousand-step run. The tokens/s column excludes it; the
epoch column does not.

## Caveats

- **The main end-to-end table's compiled arm is `dense_block_diagonal` only** because, when it was
  run, `train.py` still refused `flex_document_causal` + `torch.compile`. The flex compiled arm was
  measured separately with identical settings — see "`flex_document_causal` + `torch.compile`"
  above.
- **The flex numbers above are already compiled at the attention level, and so is the eager
  `flex_document_causal` training arm.** `models/language_model.py` lazily wraps both
  `flex_attention` and `create_block_mask` in `torch.compile(..., dynamic=False)`, and the
  microbenchmark's `--compile_flex` / `--compile_mask` both default to on. The
  "flex_attention called without torch.compile() — this will use an unfused implementation" warning
  in the run log comes from the correctness self-check, which deliberately calls the raw op; it does
  not apply to any timed row. So the old "`flex_document_causal` cannot be compiled" referred specifically
  to *whole-model* `torch.compile(model)` wrapped around the already-compiled flex call — never to
  flex running unfused.
- The microbenchmark's synthetic rows use a fixed 17 documents per row at
  `avg_sample_length=262`. Real FineVision rows vary, so treat the core numbers as a mechanism
  explanation and the end-to-end numbers as the operational result.
- `benchmark_attn_packing_results.json` is a flat list of rows with no provenance block (unlike the
  training harness's JSON, which embeds `train.get_provenance()`); its environment is recorded in
  this file instead.
- **When benchmarking compiled arms, watch `torch._dynamo`'s recompile limit.** Alternating
  grad-mode or autocast state against one compiled call site creates a distinct `GLOBAL_STATE`
  guard per flavour, and `dynamic=True` does not relax those — only shape guards. Once the
  per-function cache hits `recompile_limit` (default 8), further variants silently fall back to
  **eager while still being labelled compiled**, which reads as "compilation stopped helping at
  scale" rather than as a measurement artifact. `eval/benchmark_fwd_bwd.py` avoids it by calling
  `torch._dynamo.reset()` once per sweep iteration; check stderr for `recompile_limit` before
  trusting any compiled number.
- **The compiled arm above did hit that limit.** Right after step 0, dynamo logged
  `hit config.recompile_limit (8)` for `VisionLanguageModel.forward`, last reason
  `len(images[0]) == 1` from `_process_images` (`models/vision_language_model.py:53`). Dynamo
  specializes `forward` on the **number of images per sample**; FineVision varies it, so each new
  count recompiles until the cap of 8. Batches with a count outside those 8 then run eager.
  - This does not invalidate the number: steady-state throughput held at 19.3–19.9k tokens/s and
    fw+bw at 0.20–0.21s across every interval after the hit. A mostly-eager arm would regress
    toward the eager arms' 0.35s.
  - It does mean the **1.64× is slightly understated**. With the image count marked dynamic (or
    the cap raised), every batch would run compiled.
  - It is **independent of the packing implementation**: the eager arms logged no recompiles, and
    the cause lives in image handling, not attention. It is a pre-existing issue in upstream code.
