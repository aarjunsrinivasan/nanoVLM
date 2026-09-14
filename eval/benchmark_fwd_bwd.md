# Forward/backward benchmark & profile results

Satisfies `project_dev.md`'s systems TODO item 1 ("benchmark and profile fwd,backward"). Produced
by [`eval/benchmark_fwd_bwd.py`](benchmark_fwd_bwd.py), run both eager (via
[`slurm/benchmark_fwd_bwd.slurm`](../slurm/benchmark_fwd_bwd.slurm), job 1597; job 1596 was a
first attempt invalidated by a bug found and fixed mid-run, see Caveats) and with `torch.compile`
(via [`slurm/benchmark_fwd_bwd_compiled.slurm`](../slurm/benchmark_fwd_bwd_compiled.slurm), job
1600; job 1599 was a first attempt invalidated by a `torch._dynamo` recompile-limit issue found
and fixed afterward, see Caveats).

## Environment

- Single NVIDIA H200 (143.8 GB), pinned via Slurm `--gres=gpu:1` (`CUDA_VISIBLE_DEVICES=0` inside
  the job — confirmed in the job log, only one GPU visible/used).
- torch `2.14.0+cu130`, bf16 autocast (H200 supports bf16).
- Checkpoint: `lusxvr/nanoVLM-460M-8k` (460,113,984 parameters) — matches this repo's current
  `models/config.py` defaults (SmolLM2-360M-Instruct + siglip2-base-patch16-512).
- `--num_warmup 3 --num_iters 10` (script defaults); batch-size sweep `[1, 2, 4, 8]` for both runs.
- Raw JSON results live under [`eval/h200/`](h200/) (gitignored, not committed) — grouped by the
  GPU they were measured on, since these numbers are hardware-specific and a future rerun on
  different hardware would land in its own sibling folder rather than overwrite these.

## Input / output sizes (directly observed, not inferred)

For one fixed example (`assets/image.png`, a synthetic VQA-style question/answer pair, built
through the real `VQADataset`/`VQACollator` pipeline `train.py` uses):

| Tensor | Shape | Notes |
|---|---|---|
| `input_ids` / `attention_mask` / `labels` | `[B, 1131]` | 1131 = natural tokenized length of this example (no padding/truncation; well under the `lm_max_length=4096` cap) |
| image, per sample | `[17, 3, 512, 512]` | `assets/image.png` is 3000×3999 RGB; `DynamicResize` (`max_img_size=2048`, `resize_to_max_side_len=True`) resizes it to 2048×2048, then `GlobalAndSplitImages` splits it into 4×4=16 patches + 1 global patch = 17 patches of 512×512 |
| forward-only output (`targets=None`) | `[B, 1131, 960]` | **Raw decoder hidden states, not logits** — `models/vision_language_model.py`'s `forward()` only applies the LM head when `targets is not None`, so eval-mode/generation-style calls never see vocab-sized output |
| backward / full-step output (`targets=labels`) | logits `[B, 1131, 49218]` + scalar `loss` | LM head applied (`lm_vocab_size = 49152 base + 66 extra tokens`) |

This forward-vs-backward output asymmetry is real model behavior (confirmed by direct
instrumentation), not a script bug — worth knowing if reusing `forward()`'s return value.

## Eager results (job 1597)

| batch_size | forward (ms) | forward peak VRAM (MB) | backward (ms) | backward peak VRAM (MB) | full step (ms) | full step peak VRAM (MB) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 148.6 | 4,383 | 79.0 | 14,667 | 141.4 | 16,525 |
| **2** (real training's per-GPU batch size) | 274.1 | 6,858 | 109.3 | 22,835 | 223.3 | 24,707 |
| 4 | 510.9 | 8,088 | 144.3 | 38,951 | 294.5 | 41,016 |
| 8 | 1,012.8 | 10,525 | 273.3 | 71,290 | 538.1 | 73,804 |

Full step = forward + backward + optimizer.step() (matches `train.py`'s per-step cost). Peak VRAM
scales roughly linearly with batch size once past the fixed model/optimizer-state floor: full-step
peak ≈ 8.3 GB + 8.2 GB × batch_size at this sequence length. Extrapolating, batch_size=16 would sit
at ≈139 GB (~97% of one H200) — too close to CLAUDE.md's "keep utilization well under 100%" rule
on a shared node, so a batch_size=16 stretch run was deliberately skipped rather than risk maxing
out a shared GPU (this applies to the compiled run below too, for the same reason).

Raw data: [`h200/benchmark_fwd_bwd_results.json`](h200/benchmark_fwd_bwd_results.json) (includes the shapes
above per batch size). Full console logs (profiler op tables included):
`logs/benchmark_fwd_bwd/1597.out` (final/correct run) and `1597.err`. Chrome traces (op-level
breakdown for batch_size=8, the largest swept size) at
`logs/benchmark_fwd_bwd/traces/eager/{forward_only,backward_only,forward_backward_step}.json` —
open at `chrome://tracing` or `https://ui.perfetto.dev`.

## Compiled results, `torch.compile()` (job 1600)

| batch_size | forward (ms) | forward peak VRAM (MB) | backward (ms) | backward peak VRAM (MB) | full step (ms) | full step peak VRAM (MB) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 126.5 | 4,125 | 41.1 | 13,949 | 78.3 | 15,815 |
| **2** (real training's per-GPU batch size) | 255.4 | 6,351 | 62.4 | 21,407 | 138.6 | 23,338 |
| 4 | 482.2 | 7,040 | 109.0 | 36,154 | 225.2 | 38,346 |
| 8 | 937.0 | 8,460 | 207.1 | 65,791 | 410.5 | 68,391 |

Full-step speedup vs. eager: **-44.6% at batch_size=1, -37.9% at batch_size=2, -23.5% at
batch_size=4, -23.7% at batch_size=8** — a real, plausible, roughly-plateauing speedup across the
whole sweep. (An earlier attempt, job 1599, showed the speedup collapsing to ~0% at batch_size=8;
that was a `torch._dynamo` recompile-limit artifact, not a real result — see Caveats. This is the
corrected rerun.) `sample_loss` is ≈3.105 and constant across all four sizes here too, confirming
the model-reset fix (see Caveats) works identically under `torch.compile`.

Raw data: [`h200/benchmark_fwd_bwd_results_compiled.json`](h200/benchmark_fwd_bwd_results_compiled.json).
Full console logs: `logs/benchmark_fwd_bwd_compiled/1600.out` and `.err`. Chrome traces (batch_size=8)
at `logs/benchmark_fwd_bwd_compiled/traces/{forward_only,backward_only,forward_backward_step}.json`.

## Caveats

- **Dummy optimizer**: the full-step phase uses a single-param-group `AdamW(lr=1e-5)` for step-cost
  realism, vs. real training's 3 param groups (MP/vision/language backbones, different LRs per
  `TrainConfig`). Same total parameter count, so step cost should be representative, but not a
  byte-for-byte match.
- **Real training's actual operating point** is `batch_size=2` (`TrainConfig.batch_size`, with 8x
  gradient accumulation for an effective batch of 16) — bolded above.
- **Model state is reset before every batch size** (snapshot of the pretrained weights reloaded via
  `load_state_dict` at the top of each sweep iteration). This was a real fix, not a stylistic one:
  the full-step phase's `optimizer.step()` genuinely updates weights, and an earlier run (job 1596)
  reused the same model instance across the whole sweep without resetting it, so later batch sizes
  were silently benchmarking a model partially overfit to the repeated synthetic example by every
  prior batch size's steps (`sample_loss` collapsed from 3.1 at batch=1 to ~1e-7 at batch=8). Job
  1596's results were discarded; job 1597 (linked above) is the corrected rerun, where
  `sample_loss` is ≈3.105 and constant across all four batch sizes, as expected for a fixed model
  evaluated on the same fixed input.
- **`torch.compile` hit its recompile limit in an earlier attempt (job 1599), fixed for job 1600**
  — job 1599's stderr showed `torch._dynamo hit config.recompile_limit (8)` for
  `models/vision_language_model.py`'s `forward()`. Root cause: the script compiles the model ONCE
  before the batch-size loop, and all 3 phases (forward-only: eval/`no_grad`/no-autocast vs.
  backward-only and full-step: train/autocast) genuinely call the compiled model — a distinct
  grad-mode/autocast "GLOBAL_STATE" guard for dynamo that `dynamic=True` (the original config)
  cannot relax, since it only relaxes shape guards. So each batch size needed ~2 distinct graph
  flavors, and 4 batch sizes × 2 flavors = 8 cache entries against the shared per-function cap of
  8 — landing right at the ceiling, with the last-processed size (8) silently falling back to
  eager instead of a fresh compile once the cache filled. This is why job 1599's batch_size=8
  speedup collapsed to ~0% while smaller sizes showed a real, shrinking-but-present speedup — not
  because compilation stops helping at scale, but because batch_size=8 wasn't actually running
  compiled code for the backward/full-step phases (and its saved Chrome trace was almost certainly
  capturing eager execution mislabeled as compiled).

  Checked the sibling repo `/home/asrinivasan/vlm_gen/nanoVLM`'s own `torch.compile` benchmarking
  (`eval/benchmark_attn_backends.py`) for a pattern to reuse — it avoids this by accident, not by a
  technique worth copying: its "forward" benchmark calls `model.generate(...)`, which
  `torch.compile()`'s `OptimizedModule.__getattr__` resolves straight to the *original* uncompiled
  module (only `forward`/`__call__` route through the compiled graph), so that path never compiles
  at all; its one genuinely-compiled path (backward) never alternates grad-mode against the same
  compiled call site, so it stays under budget by construction. Neither applies here, since all 3
  of this script's phases are deliberately meant to exercise genuine compiled execution.

  **Fix applied**: `main()` now calls `torch._dynamo.reset()` once per batch-size loop iteration
  (right after the existing `model.load_state_dict(initial_state_dict)` reset), and the compile
  call was simplified from `torch.compile(model, dynamic=True)` to plain `torch.compile(model)`
  (matching `train.py`'s real compile call — `dynamic=True`'s cross-batch-size shape sharing no
  longer buys anything once each size gets an isolated fresh compile). `reset()`'s own docstring
  describes exactly the guarantee needed: "as if you had started a fresh process invocation" —
  same isolation as a separate process per batch size, without repeating model-load/startup cost
  or needing to merge multiple results files. Verified: job 1600's stderr contains no
  `recompile_limit` message, and its batch_size=8 full-step speedup is now a real -23.7% (see
  Results above) instead of ~0%.
- **Found and fixed along the way**: this benchmarking effort also surfaced two real, previously
  latent bugs, both caused by newer pinned dependency versions (`uv.lock`) changing defaults that
  older repo code silently relied on:
  1. `models/language_model.py`'s SDPA attention path passed both an explicit `attn_mask` and
     `is_causal=True` to `F.scaled_dot_product_attention`, which `torch==2.14.0` rejects outright.
     This would have broken every real training forward pass, not just this benchmark. Fixed by
     folding the causal mask into the same additive padding mask instead of relying on SDPA's
     built-in causal flag.
  2. `data/datasets.py`'s `_prepare_inputs_and_loss_mask()` computed each conversation turn's token
     length via `len(tokenizer.apply_chat_template(...))`; on `transformers==5.17.0`,
     `apply_chat_template`'s `return_dict` default changed from `False` to `True`, so this returned
     a 2-key dict instead of a token list, and `len(...)` silently returned 2 regardless of message
     length. That corrupted the assistant-turn loss mask so badly that every label ended up
     `-100`, producing `NaN` loss — for real training too, not just this benchmark. Fixed by
     passing `return_dict=True` explicitly and reading `len(segment_ids["input_ids"])`.
