# 10k-step A/B: upstream nanoVLM vs this fork (~230M, 1× H100)

Does the fork's training path change **what the model learns**, or only how fast it trains? Two arms, two seeds,
10,000 steps each, about 25 GPU-hours.

| arm | flags |
|---|---|
| **A, upstream** | `--loss_impl full --attn_packing_impl none` (eager) |
| **C, fork** | `--loss_impl gather --attn_packing_impl flex_document_causal --compile` |

Everything else is identical: `HuggingFaceTB/SmolLM2-135M-Instruct` + siglip2-base-patch16-512 (228,063,936 params),
micro-batch 2 × grad-accum 8, 4 dataloader workers, cosine schedule over the full 10k steps, FineVision.
**Within a seed both arms train on byte-identical batches** (same seed, same data pipeline); the step-100/200/300
losses of a pair agree to 3 decimals. Logs, the driver script and `analyze.py` that produces these tables are here.
wandb group `ab-10k-230m`.

## Two ways to measure val loss, because the arms disagree about what a batch is

Training packs several unrelated samples into one 4096-token row. Upstream lets a sample attend back into earlier,
unrelated samples in its row; the fork masks that off and restarts position ids per document. Scoring each arm under
its own training mask would compare two different metrics, so every checkpoint of both arms is scored **both** ways,
by the same evaluator (`train.py: eval_under_masks`, eager, `loss_impl='full'`, same rows):

- **doc-masked** — each sample attends only to itself. This is the per-sample loss, and it is what inference does.
- **unmasked** — upstream's regime, where a sample sees its packed neighbours.

## Result: final full val pass (whole 5k-sample val set)

| arm | seed | doc-masked | unmasked | train loss | wall clock |
|---|---|---|---|---|---|
| A upstream | 0 | 0.9159 | 0.9144 | 1.1038 | 7.01 h |
| A upstream | 1 | 0.9272 | 0.9257 | 1.1137 | 6.95 h |
| **C fork** | 0 | **0.9061** | 0.9208 | 1.0948 | 5.90 h |
| **C fork** | 1 | **0.8927** | 0.9086 | 1.0701 | 5.03 h |

**Pre-registered rule** (fixed before the runs): the improvement counts only if the gap between arms exceeds the
spread between seeds within an arm.

- **doc-masked: passes.** Mean 0.9216 (A) vs 0.8994 (C), gap **0.0222**, largest within-arm seed spread 0.0134.
  The paired per-seed gaps are 0.0098 (seed 0) and 0.0345 (seed 1) — same direction, very different size.
- **unmasked: does not pass.** Mean 0.9201 vs 0.9147, gap 0.0054, inside the 0.0122 seed spread. Under upstream's own
  contaminated-context metric the two arms are indistinguishable.

That split is the expected signature of the masking fix: it helps where the metric matches inference, and it neither
helps nor hurts under the metric that rewards reading unrelated neighbours.

Per-checkpoint doc-masked val loss (256 rows, every 500 steps) has the fork ahead of its paired baseline at
**39 of 40** checkpoints across both seeds (19/20 at seed 0, the exception being step 0 before training; 20/20 at
seed 1), and the gap widens as training proceeds:

| step | A_s0 | C_s0 | A_s1 | C_s1 |
|---|---|---|---|---|
| 2000 | 1.0325 | 1.0299 | 1.0390 | 1.0330 |
| 4000 | 0.9980 | 0.9908 | 1.0121 | 0.9615 |
| 6000 | 0.9374 | 0.9255 | 0.9527 | 0.9028 |
| 8000 | 0.8953 | 0.8878 | 0.9126 | 0.8750 |

## Which change caused it

Only the attention masking can: `lm_loss_impl='gather'` is mathematically identical to `'full'`
(`tests/test_vision_language_model_loss.py` checks loss and all gradients at 1e-5), and `--compile` does not change
the math either. The quality difference is the packed-row fix; gather and compile are the speed and memory half.

## Speed: a range, not a single number

Wall clock for the full 10k steps, including identical eval overhead in both arms:

| seed | A | C | speedup |
|---|---|---|---|
| 0 | 7.01 h | 5.90 h | 1.19× |
| 1 | 6.95 h | 5.03 h | 1.38× |

Per-interval throughput ratio on identical data:

| steps | seed 0 | seed 1 |
|---|---|---|
| 0–2500 | 1.43× | 1.71× |
| 2500–5000 | 1.23× | 1.61× |
| 5000–7500 | 1.22× | 1.39× |
| 7500–10000 | 1.19× | 1.56× |

The controlled interleaved benchmark (`../speed_230m/`, 300 steps, 2 repetitions) measured 1.54×.

**Why it varies.** The baseline's fw+bw is the same in both seeds (0.210 s / 0.212 s), while the compiled arm's is
0.180 s under seed 0 and 0.124 s under seed 1 — a 45% difference on data with the same images per sample (1.635 vs
1.638). Both runs hit dynamo's `recompile_limit` (8, left at its default) once, early. The plausible cause is which
image-tile shapes happened to be compiled before the limit was reached: batches whose shape missed the cache fall
back to eager for the rest of the run. So the honest claim is **1.2–1.7× depending on which shapes get compiled
first**, not a flat 1.54×. Raising or removing the recompile limit is the obvious follow-up, and was deliberately
not done here so the default behaviour is what is reported.

Memory (peak reserved) is the steadier win: 48.6 GiB (A) vs 38.4 GiB (C), a 10.2 GiB saving, of which 7.6 GiB comes
from the gather loss alone. That is what lets the fork train at micro-batch 4, where upstream is already at 78.4 of
79.2 GiB (`../phase0_230m/summary.md`).

## Caveats

- Two seeds per arm. The paired within-seed comparison is the sensitive one (both arms see identical data); the
  unpaired seed-to-seed spread also carries data-order variance, which is why it is larger than one of the paired gaps.
- Kernel nondeterminism alone puts two identical runs ~0.02–0.25 apart in per-step train loss, so per-step losses are
  not comparable; the val numbers here are averages over 256 rows (curves) and the full val set (finals).
- The 222M model in the README is v0.1 (siglip-224) and cannot be built on this code; this is the ~230M equivalent.
- These runs used the fork's `--save_training_state`; each kept only its latest checkpoint plus its best.

## Reproduce

```bash
python train.py --lm_model_type HuggingFaceTB/SmolLM2-135M-Instruct \
  --loss_impl full --attn_packing_impl none \        # arm C: --loss_impl gather --attn_packing_impl flex_document_causal --compile
  --batch_size 2 --gradient_accumulation_steps 8 --num_workers 4 \
  --max_training_steps 10000 --eval_interval 500 --eval_mask_rows 256 --val_size 5000 \
  --save_training_state --seed 0 --no_lmms_eval --no_hub_push
python eval/h100/ab_10k_230m/analyze.py eval/h100/ab_10k_230m
```
