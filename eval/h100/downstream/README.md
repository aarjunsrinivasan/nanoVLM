# Downstream benchmarks and mechanism, 10k A/B (~230M, step_10000)

Does the cross-document masking fix change anything a benchmark can see, and can the attention leak
it fixes be measured directly? Four checkpoints — the same `A_s0 / A_s1 / C_s0 / C_s1` from the
[10k A/B](../ab_10k_230m/README.md) — scored on seven lmms-eval tasks, plus two analyses of the
mechanism itself on the training-distribution val set.

**Headline: one credible improvement, six tasks where the effect is smaller than the noise floor, and
the noise floor is measured rather than assumed.**

| | |
|---|---|
| credible | **TextVQA +4.52 points** [+3.81, +5.26] pooled |
| inconclusive | DocVQA, InfoVQA, ChartQA, OCRBench, AI2D, ScienceQA |
| trending against | AI2D, −1.10 pooled [−2.16, +0.00] |

`inconclusive` means *undetectable at the stated MDE*, not *no effect*.

## What makes this more than a table of scores

**The decision rule was fixed in advance** (`analyze_lmms_ab.py: verdict`). A task is `credible` only
if all four hold: same sign in both seed pairs, both cluster CIs exclude zero, both Holm-corrected
p-values < 0.05, **and** `|pooled Δ|` exceeds both same-arm cross-seed differences. That last
condition is the important one — `|ctrl A|` and `|ctrl C|` measure what changing the training seed
alone does to a task, so any effect smaller than that is indistinguishable from seed luck.

**ChartQA is the clearest illustration, and it is a result rather than an embarrassment.** It moves
−2.84 under seed 0 and **+4.84** under seed 1 — opposite directions, both individually
"significant" (p = 1.3e-3 and 9.5e-9). Its controls are 4.40 and 3.28. A single-seed experiment would
have reported either a solid win or a solid regression on this task, with a convincing p-value
attached, and both would have been artifacts. This is what two seeds buy you, and it is why the six
inconclusive verdicts are stated as inconclusive instead of mined for a headline.

**TextVQA survives that test.** +3.75 (seed 0) and +5.30 (seed 1), same sign, controls of 0.07 and
1.62 against a pooled effect of 4.52.

## Scores

Points (×100), with lmms-eval's own stderr. Effect = C (doc-masked) − A (no masking); all metrics
higher-is-better. Full tables, CIs, McNemar and paired-t secondaries: [`summary.md`](summary.md).

| task | N | A_s0 | C_s0 | A_s1 | C_s1 | pooled Δ | MDE | verdict |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| textvqa_val | 5000 | 21.61 | 25.36 | 21.68 | 26.98 | **+4.52** | 1.04 | **credible: C > A** |
| docvqa_val | 5349 | 21.57 | 21.48 | 19.66 | 24.01 | +2.13 | 0.90 | inconclusive |
| ocrbench | 1000 | 20.40 | 20.80 | 19.80 | 22.80 | +1.70 | 2.45 | inconclusive |
| chartqa | 2500 | 28.16 | 25.32 | 23.76 | 28.60 | +1.00 | 1.60 | inconclusive |
| scienceqa | 4241 | 38.62 | 39.12 | 39.52 | 39.66 | +0.32 | 1.16 | inconclusive |
| infovqa_val | 2801 | 13.50 | 13.22 | 12.90 | 13.34 | +0.08 | 0.95 | inconclusive |
| ai2d | 3088 | 23.80 | 23.51 | 24.84 | 22.93 | −1.10 | 1.55 | inconclusive |

DocVQA looks like a win on the pooled number but is not one: its seed-0 effect is −0.08 while its
seed-1 effect is +4.34, and its controls are 1.90 and 2.52. Same shape as ChartQA, less extreme.

No arm produced empty responses on any task (`max empty resp` is 0.00% throughout), so nothing here is
a degenerate model scoring by accident.

## The mechanism, measured directly

### Attention mass on earlier documents ([`attn_mass_summary.md`](attn_mass_summary.md))

The mean softmax probability mass a query token places on keys belonging to an earlier, unrelated
packed document, excluding the row's attention-sink token, per layer.

| layer | A_s0 (as it trains) | C_s0, mask removed | C_s0, mask removed but RoPE reset kept |
|---|--:|--:|--:|
| 0 | 0.6126 | 0.6297 | 0.8210 |
| 11 | 0.1655 | 0.2471 | 0.9202 |
| 18 | 0.0152 | 0.0409 | 0.3131 |
| 25 | 0.0028 | 0.0094 | 0.7702 |

Two things fall out of this:

1. **Arm A had to learn to ignore its neighbours, and only partly managed it.** Leakage is heavy in
   early layers (0.61 at layer 0) and decays to near-nothing by layer 25 — capacity spent acquiring a
   property the mask provides for free.
2. **The RoPE position reset and the mask are load-bearing together, not separately.** Removing the
   mask while keeping per-document position reset leaves leakage *far higher at every depth*
   (0.77–0.95). With continuous positions, earlier documents sit far away in position space and RoPE's
   distance decay suppresses them; reset positions make unrelated neighbours look adjacent. This is
   why the two are gated by a single flag, and the `(reset kept)` column is what shows it — it is an
   ablation, not a duplicate measurement.

   **Caveat on that ablation: it is not paired.** The `(reset kept)` runs were a separate invocation,
   so they restarted the seeded val loader and scored the 1st and 2nd traversals — their query-token
   counts (205,760 and 204,952) are A_s0's and A_s1's, not C_s0's (204,852) and C_s1's (204,896).
   Packing is reshuffled per traversal, so each `(reset kept)` column is a different packing of the
   same val set from the plain column beside it. The conclusion survives with room to spare: packing
   variation moves these numbers by well under 1% (A_s0 0.6126 vs A_s1 0.6122 on different packings),
   while the effect being claimed is 0.0094 → 0.7702 at layer 25. But it is an unpaired comparison and
   should be read as one. Scoring both in a single invocation would make it paired.

### Per-document loss by packing position ([`per_doc_summary.md`](per_doc_summary.md))

Loss under each arm's own training mask, bucketed by a document's position in its packed row. The
doc-index-0 control agrees to four decimals between masks (|Δ| = 0.0000), which is the check that the
two masks really are equivalent where no leakage is possible.

## Important limits

- **Benchmark inference never packs documents**, for either arm — `doc_id=None` at eval time. So the
  mechanism analyses explain the *validation-loss* gap and cannot explain which benchmark task moved.
  Both summaries say this; it is repeated here because it is the easiest thing to get wrong.
- **The open question, stated plainly:** if inference never packs, why does TextVQA move at all? The
  plausible reading is that training under leaked attention yields a weaker model and the damage
  concentrates on reading text in images. That is consistent with these numbers and **not proven by
  them.**
- **Cluster CIs carry eval-set noise only.** Training-seed variance cannot be estimated from two seeds
  per arm; the `|ctrl|` columns are the only guide to it, which is exactly why the decision rule uses
  them as a floor.
- The mechanism analyses are descriptive and correlational.
- Seven tasks, fixed before the results were seen. No task was added afterwards.

## Reproduce

All four analysis scripts are in the repository, and both that have selftests pass:

```bash
python -m eval.analyze_lmms_ab --selftest      # statistics: cluster bootstrap, Holm, McNemar
python -m eval.attn_mass_analysis --selftest   # patched-SDPA equivalence + hand-computed mass

# regenerate summary.md and paired_stats.json from the per-sample JSONL
python -m eval.analyze_lmms_ab --root <dir with A_s0/ C_s0/ A_s1/ C_s1/>

# the mechanism passes (need the checkpoints and a FineVision shard cache)
python -m eval.attn_mass_analysis <ckpt>...                        # add --keep_position_reset for the ablation
python -m eval.eval_per_doc <ckpt>...
```

`eval_checkpoint.py` refuses to score checkpoints whose configs would tokenize val rows differently,
and refuses to read val shards that the checkpoints trained on unless `--allow_train_overlap` is
passed. `eval_per_doc.py` asserts that every full traversal sees an identical document and valid-token
count across all eight (checkpoint, mask) passes.

## Files

| path | what |
|---|---|
| [`summary.md`](summary.md) | the full lmms-eval A/B: scores, CIs, Holm p-values, McNemar, MDE |
| [`attn_mass_summary.md`](attn_mass_summary.md) | per-layer cross-document attention mass, with the RoPE ablation |
| [`per_doc_summary.md`](per_doc_summary.md) | val loss by doc-index and preceding-context length |
| `paired_stats.json` | every statistic behind `summary.md` |
| `attn_mass.json`, `per_doc_loss.json.gz` | raw mechanism measurements, per layer / per document |
| `results/*_results.json` | lmms-eval's own results files, one per arm |
| `logs/` | run logs, unmodified |

Per-sample JSONL (39 MB, one row per question per arm) is not in the repo; it is attached to the
Hugging Face checkpoint release. `analyze_lmms_ab.py` needs it to regenerate `summary.md`.

All four arms were scored in one session, at the same commit (`c9d0f1e`), `batch_size 8`, no `--limit`,
and the same wrapper version.
