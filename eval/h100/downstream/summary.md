# lmms-eval: doc masking A/B (step_10000)

Scores in points (x100). Effect = C (doc-masked) - A (no masking); every metric is higher-is-better.
CIs are 95% cluster-bootstrap (clusters = images/documents when known, else questions).
p_holm = cluster-robust t-test p-value, Holm-corrected across tasks within a seed pair.

## Scores (mean, lmms-eval stderr)

| task | N | A_s0 | C_s0 | A_s1 | C_s1 | max empty resp |
|---|--:|--:|--:|--:|--:|--:|
| docvqa_val | 5349 | 21.57 ±0.50 | 21.48 ±0.50 | 19.66 ±0.48 | 24.01 ±0.52 | 0.00% |
| infovqa_val | 2801 | 13.50 ±0.55 | 13.22 ±0.54 | 12.90 ±0.54 | 13.34 ±0.55 | 0.00% |
| chartqa | 2500 | 28.16 ±0.90 | 25.32 ±0.87 | 23.76 ±0.85 | 28.60 ±0.90 | 0.00% |
| textvqa_val | 5000 | 21.61 ±0.57 | 25.36 ±0.60 | 21.68 ±0.57 | 26.98 ±0.61 | 0.00% |
| ocrbench | 1000 | 20.40 ±1.27 | 20.80 ±1.28 | 19.80 ±1.26 | 22.80 ±1.33 | 0.00% |
| ai2d | 3088 | 23.80 ±0.77 | 23.51 ±0.76 | 24.84 ±0.78 | 22.93 ±0.76 | 0.00% |
| scienceqa | 4241 | 38.62 ±0.75 | 39.12 ±0.75 | 39.52 ±0.75 | 39.66 ±0.75 | 0.00% |

## Paired differences

| task | clusters | Δ seed0 [CI] p_holm | Δ seed1 [CI] p_holm | Δ pooled [CI] | \|ctrl A\| | \|ctrl C\| | MDE | verdict |
|---|--:|---|---|---|--:|--:|--:|---|
| docvqa_val | 1286 | -0.08 [-0.93, +0.76] p=1 | +4.34 [+3.40, +5.31] p=5.71e-18 | +2.13 [+1.51, +2.77] | 1.90 | 2.52 | 0.90 | inconclusive |
| infovqa_val | 501 | -0.29 [-1.09, +0.57] p=1 | +0.45 [-0.54, +1.44] p=0.732 | +0.08 [-0.57, +0.75] | 0.61 | 0.13 | 0.95 | inconclusive |
| chartqa | 1509 | -2.84 [-4.35, -1.34] p=0.00132 | +4.84 [+3.25, +6.40] p=9.5e-09 | +1.00 [-0.10, +2.11] | 4.40 | 3.28 | 1.60 | inconclusive |
| textvqa_val | 3166 | +3.75 [+2.80, +4.69] p=1.24e-13 | +5.30 [+4.28, +6.29] p=1.09e-23 | +4.52 [+3.81, +5.26] | 0.07 | 1.62 | 1.04 | credible: C > A |
| ocrbench | 930 | +0.40 [-1.80, +2.58] p=1 | +3.00 [+0.70, +5.27] p=0.0442 | +1.70 [-0.05, +3.40] | 0.60 | 2.00 | 2.45 | inconclusive |
| ai2d | 814 | -0.29 [-1.66, +1.06] p=1 | -1.91 [-3.37, -0.41] p=0.0442 | -1.10 [-2.16, +0.00] | 1.04 | 0.58 | 1.55 | inconclusive |
| scienceqa | 4023 | +0.50 [-0.60, +1.59] p=1 | +0.14 [-0.97, +1.28] p=0.805 | +0.32 [-0.49, +1.15] | 0.90 | 0.54 | 1.16 | inconclusive |

## Secondary tests (question-level, uncorrected)

| task | McNemar p s0 (b/c) | McNemar p s1 (b/c) | paired-t p s0 | paired-t p s1 | CI width question vs cluster (s0, pts) |
|---|---|---|--:|--:|---|
| docvqa_val | n/a | n/a | 0.841 | 0 | 1.65 vs 1.69 |
| infovqa_val | n/a | n/a | 0.487 | 0.337 | 1.59 vs 1.66 |
| chartqa | 0.000211 (144/215) | 2.28e-09 (265/144) | 0.000176 | 1.94e-09 | 3.00 vs 3.01 |
| textvqa_val | n/a | n/a | 2.44e-15 | 0 | 1.80 vs 1.89 |
| ocrbench | 0.789 (65/61) | 0.014 (85/55) | 0.722 | 0.0112 | 4.40 vs 4.38 |
| ai2d | 0.713 (232/241) | 0.0134 (246/305) | 0.679 | 0.0119 | 2.78 vs 2.72 |
| scienceqa | 0.404 (298/277) | 0.835 (290/284) | 0.381 | 0.802 | 2.24 vs 2.19 |

## How to read this

- `credible` needs the same sign in both seed pairs, both cluster CIs excluding 0, both p_holm < 0.05, and
  |pooled Δ| larger than both same-arm cross-seed differences (|ctrl A|, |ctrl C|), which show what a change of
  training seed alone does.
- These tests measure eval-set noise for these 4 trained models. Training-seed noise cannot be estimated from
  2 seeds per arm; the controls are the only guide to it. `inconclusive` means undetectable at this MDE, not no effect.
- MDE = smallest pooled Δ detectable at 80% power, alpha 0.05 (2.8 x cluster SE).
