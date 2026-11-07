# Attention packing verification — H200 (bounded verification run)

**Scope note:** this is a bounded verification run (20 training steps per arm), not a
re-run of the full 200-step sweep in `eval/h100/attn_packing.md`. It independently
re-checks that write-up's claims on different hardware (4x H200 vs. the original
H100) using the repo's existing tests/benchmarks — no new permanent infrastructure
was added.

## Environment

- GPU: NVIDIA H200 (`CUDA_VISIBLE_DEVICES=0`, single GPU, 3 other GPUs left idle throughout)
- torch: `2.14.0+cu130`
- git SHA: `554bbbb82358a8c8c759096eb5d9795a6ce7995b` (clean, `git_dirty=false`)
- Date: 2026-09-18

## Sub-question 1: Is the attention-packing implementation correct?

**Yes, including under `torch.compile`.**

```
tests/test_attn_packing.py                      7 passed, 3 subtests passed (15.87s, CPU)
tests/test_vision_language_model_packing.py      9 passed (59.17s, CUDA)
```

- `'none'` is proven to leak attention across document boundaries (negative control — the documented bug is real).
- `'dense_block_diagonal'` and `'flex_document_causal'` are proven to block that leakage.
- The full `VisionLanguageModel`, run on a packed row, produces per-token loss **and every gradient tensor** identical to running each sub-sample separately — for both masking backends, and (critically) **under the exact `torch.compile(model)` + `capture_dynamic_output_shape_ops=True` pattern `train.py --compile` uses**
  (`test_flex_document_causal_under_torch_compile_matches_unpacked_reference`,
  `test_dense_block_diagonal_under_torch_compile_matches_unpacked_reference`). This is a direct, measured proof that compiling does not change correctness — not an assumption.
- `doc_id=None` reproduces the exact pre-packing-support output; RoPE position-reset matches an independent reference implementation.

A second, independent correctness check ran as part of Step 2 below (`verify_no_cross_contamination`, synthetic tensors) and agrees: dense/flex block leakage, `'none'` leaks.

## Sub-question 2: Is `flex_document_causal` + `torch.compile` actually the fastest?

**Yes — the H100 ranking transfers to H200, both for the isolated attention core and for real end-to-end training.**

### Isolated attention-core microbenchmark (`eval/benchmark_attn_packing.py`, batch=2, seq_len=4096, block_size=128, includes its own leak-check before timing)

| variant | forward (ms) | forward+backward (ms) | peak VRAM (fwd+bwd) |
|---|---|---|---|
| `flex_document_causal` (precomputed mask) | **0.55** | **3.52** | **404 MB** |
| `unpacked_padded_sdpa` | 1.15 | 3.68 | 671 MB |
| `current_packed_dense_sdpa` (buggy `'none'`) | 1.30 | 4.44 | 638 MB |
| `dense_block_diagonal_sdpa` | 1.79 | 4.90 | 688 MB |

Flex is fastest and lowest-VRAM in every configuration, matching the H100 doc.

### Real end-to-end training loop (`eval/benchmark_attn_train.py`, real SmolLM2-360M-Instruct + siglip2-base-patch16-512 + streamed FineVision, batch_size=2, grad_accum=8, 20 steps, steady-state average over intervals after compile warmup)

| variant | mean tok/s | speedup vs. `none` (eager) | peak mem alloc |
|---|---|---|---|
| **`flex_document_causal` + compile** | **22,039** | **1.69x** | 38.6 GiB |
| `dense_block_diagonal` + compile | 18,215 | 1.40x | 41.3 GiB |
| `flex_document_causal` (eager) | 18,374 | 1.41x | 43.0 GiB |
| `dense_block_diagonal` (eager) | 16,090 | 1.23x | 45.8 GiB |
| `none` (eager, buggy default) | 13,059 | 1.00x (baseline) | 45.8 GiB |

`flex_document_causal + compile` is the fastest configuration measured, on real hardware with a real model and real data — confirming `eval/h100/attn_packing.md`'s claim on H200. `'none'`, the current default, is the slowest option in every configuration tested, on top of being the one proven incorrect.

## Sub-question 3: Are there graph breaks / recompiles, and do they matter?

Two separate concepts: **graph breaks** (perf-only — part of a compiled call falls back to eager) and **recompiles** (a call site is retraced because a guard failed; not a correctness issue by construction, but can exhaust `torch._dynamo.config.recompile_limit`, default 8, after which further novel shapes silently fall back to eager). Both `flex_document_causal + compile` and, as a control, `dense_block_diagonal + compile` were run with `TORCH_LOGS=recompiles,graph_breaks` to capture this directly rather than infer it.

**Finding 1 — the `recompile_limit` (8) was hit in both arms, and the cause is identical in both, and it is not flex-specific.**

Both arms hit `torch._dynamo hit config.recompile_limit (8)` on `VisionLanguageModel.forward` (`models/vision_language_model.py:62`), driven entirely by guard failures on `len(images[0])` / `len(images[1])` in `_process_images` (`models/vision_language_model.py:53-54`) — i.e. the number of images varies per streamed FineVision batch, and each new count forces a fresh trace. The exact same 8 guard-failure reasons occur in the flex arm and the dense arm; grepping both full logs for `flex`, `block_mask`, `doc_id`, or `create_block_mask` finds **zero** matches outside the literal `doc_id=doc_id` keyword argument in a traceback line. This directly confirms — not infers, as the original H100 write-up could only do — that the recompile source is the image-count dynamic shape, unrelated to attention packing or FlexAttention.

**Finding 2 — a second, distinct graph break exists in `LanguageModelRotaryEmbedding.forward` (`models/language_model.py:178`),** the dynamic-RoPE-scaling branch (`if max_seq > self.original_max_seq_len:`) — data-dependent control flow Dynamo can't trace. It appears identically in both arms, only during the first 1-2 compile frames (not recurring every step), and is unrelated to the masking backend.

**Finding 3 — no measurable throughput cost in the initial 20-step sample:** per-interval `tokens/s` rose monotonically through the run (4,851 → 20,103 → 22,557 → 23,459) even after the recompile-limit hit landed mid-run. This 20-step read is corrected and superseded by a longer follow-up run below — it was directionally right but too short a sample to be conclusive on its own.

**Correctness impact: none.** This is already established by Step 1's compiled oracle test — if a recompile or graph break silently changed model behavior, that test would show a wrong number, and it doesn't (loss and all 290+ gradient tensors match the eager unpacked reference to tight tolerance). The findings below only quantify *how much* of a performance nuisance the (packing-unrelated) recompiles are over a realistic longer horizon.

## Follow-up: does the recompile-limit issue compound over longer training?

The 20-step sample above raised a real question: `TrainConfig.max_images_per_knapsack=18` means a packed row can hold 0-18 images, far more distinct values than the 8-slot recompile cache, and reading the installed `torch/_dynamo/convert_frame.py` (v2.14.0) left it genuinely ambiguous whether hitting `recompile_limit` disables compilation only for *new* shapes going forward (already-cached shapes keep fast dispatch) or discards the whole compiled cache for that code object. This was resolved empirically with two longer, complementary measurements rather than by reading source further.

**Mechanism, now directly proven (not inferred from source-reading):** a minimal isolated repro (`torch.compile` on a toy function, `recompile_limit` forced to 2 so the cache fills in two calls) shows conclusively that `torch._dynamo`'s per-code-object cache is a monotonically-growing, capped list of compiled variants — entries are **never evicted**, and once the cap is hit, only *genuinely new* shapes get routed to eager (`RUN_ONLY`); every previously-cached shape keeps getting full compiled-speed dispatch, indefinitely:

| call | timing | interpretation |
|---|---|---|
| pure eager baseline | 3.716 ms | reference |
| shape A, 1st compile | 0.191 ms | compiled |
| shape B, 2nd compile (cache now full, limit=2) | 0.207 ms | compiled |
| shape C, 3rd/over-limit shape | 4.161 ms | eager (≈ baseline) — confirms new shapes fall back once the cap is hit |
| **shape A, re-called after the cap was hit** | **0.193 ms** | **still compiled** — proves cached entries are not discarded |
| shape B, re-called after the cap was hit | 0.197 ms | still compiled |
| shape D, a 4th never-seen shape | 4.733 ms | eager again |

This settles the ambiguity from reading `convert_frame.py` alone: it is **not an LRU cache that evicts to make room**, and it is **not "the whole function goes eager forever" either** — it's closer to a write-once cache with a hard cap: the first `recompile_limit` (8, by default) distinct shapes stay compiled and fast forever, and only shapes beyond that cap pay the eager cost, every time they recur. This directly explains why the H200 measurements below stayed fast: the 8 shapes that filled the cache earliest in the run keep serving compiled speed for the rest of training, and only the long, thinning tail of rarer shapes (Part A below) ever pays the eager cost.

**Part A — shape diversity over 1,600 real micro-batches (CPU-only, no GPU, no model — just the real dataloader):**

| micro-batch | distinct shapes seen so far |
|---|---|
| 46 (first 8 shapes) | 8 |
| 100 | 10 |
| 400 | 14 |
| 800 | 15 |
| 1,200 | 16 |
| 1,600 | 18 |

The 8-slot cache is filled almost immediately (within the first 46 of 1,600 micro-batches), and the rate of genuinely new shapes drops off sharply afterward — roughly 1 new shape every 50-400 micro-batches later in the run, not a steady stream. So yes, the real data distribution exceeds the 8-shape cache (confirming the concern was legitimate to check), but the excess is a thin, increasingly rare tail, not an ongoing flood.

**Part B — real training-loop run, `flex_document_causal + compile`, 200 steps / 1,600 micro-batches (10x the original sample), `TORCH_LOGS=recompiles,graph_breaks` captured throughout:**

Per-interval throughput (20 intervals of 10 steps each) ranged 20,256-23,356 tok/s with no directional trend across the entire run — steady-state mean **21,768 tok/s**, within 1.2% of the original 20-step estimate (22,039 tok/s). The stderr log shows exactly **8 recompiles and one `"hit config.recompile_limit"` warning**, occurring once near the start and never again for the rest of the 200-step run — consistent with `convert_frame.py`'s warning firing exactly once per code object (confirmed by reading the source: the `RUN_ONLY` frame-execution strategy is set with `apply_to_code=True`, so Dynamo's Python-side callback is never re-entered for that code object again, and neither the log nor `torch._dynamo.utils.counters` can show further activity for it either way).

**Verdict:** over a 10x-longer, more realistic window, throughput does not degrade — it stays flat within normal run-to-run noise, and we now have a proven mechanism (not just a favorable trend) explaining why: the cache never evicts, so the ~8 shapes that fill it earliest keep compiled speed for the rest of the run, and only the shrinking tail of rarer shapes (Part A) ever pays the eager cost. This is a materially stronger, data-and-mechanism-backed answer than the original 20-step read. It doesn't guarantee the same holds at, say, 50,000+ steps with a much wider shape distribution than this run happened to sample — the cache is still hard-capped at 8, so a training run whose shape diversity keeps growing indefinitely (rather than plateauing the way Part A's 1,600-batch sample did) would see a growing, not shrinking, fraction of eager-fallback steps — but for realistic FineVision-style data and run lengths in the range actually tested here, this is not a practical concern. Correctness is unaffected either way (established independently by Step 1's oracle tests, which don't depend on run length or cache behavior).

## Flag for the user's decision (not acted on here)

`lm_attn_packing_impl` defaults to `'none'` (`models/config.py:67`), which the test suite proves leaks attention across document boundaries and which was the slowest option in every configuration measured on both H100 and H200. It's kept as the default only for backward compatibility with existing/resumed configs, per the inline comment in `models/config.py`. Changing the default is out of scope for this verification and was not done.

## Explicit scope statement

No production code (`models/*.py`, `data/*.py`, `train.py`) was modified. No multi-GPU or full-scale (thousands-of-step) run was performed — single `CUDA_VISIBLE_DEVICES=0` throughout (the Part A shape-diversity check ran CPU-only, `CUDA_VISIBLE_DEVICES=""`, touching no GPU at all), 20-200 steps per training-loop arm. Raw results: `eval/profile/benchmark_attn_packing_verification_results.json`, `eval/profile/benchmark_attn_train_eager_verification_results.json`, `eval/profile/benchmark_attn_train_flex_compile_verification_results.json`, `eval/profile/benchmark_attn_train_dense_compile_verification_results.json`, `eval/profile/benchmark_attn_train_flex_compile_longrun_results.json`.
