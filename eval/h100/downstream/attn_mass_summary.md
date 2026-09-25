# Attention mass on earlier documents

Mean softmax probability mass each real, non-first-document query token places on keys
from an earlier, unrelated packed document (excluding the row's attention-sink token),
averaged over heads. N = number of query tokens the mean is over.

| layer | A_s0 mean (N) | A_s1 mean (N) | C_s0 mean (N) | C_s1 mean (N) | C_s0 (reset kept) mean (N) | C_s1 (reset kept) mean (N) |
|---|---|---|---|---|---|---|
| 0 | 0.6126 (205760) | 0.6122 (204952) | 0.6297 (204852) | 0.6355 (204896) | 0.8210 (205760) | 0.8144 (204952) |
| 1 | 0.4161 (205760) | 0.4096 (204952) | 0.4497 (204852) | 0.4532 (204896) | 0.9335 (205760) | 0.9366 (204952) |
| 2 | 0.4947 (205760) | 0.4933 (204952) | 0.5508 (204852) | 0.5443 (204896) | 0.8450 (205760) | 0.8470 (204952) |
| 3 | 0.6638 (205760) | 0.6458 (204952) | 0.7003 (204852) | 0.6783 (204896) | 0.8653 (205760) | 0.8488 (204952) |
| 4 | 0.7153 (205760) | 0.7036 (204952) | 0.7307 (204852) | 0.7299 (204896) | 0.8491 (205760) | 0.8418 (204952) |
| 5 | 0.6746 (205760) | 0.6552 (204952) | 0.6803 (204852) | 0.6908 (204896) | 0.8443 (205760) | 0.8467 (204952) |
| 6 | 0.6679 (205760) | 0.6471 (204952) | 0.6931 (204852) | 0.6848 (204896) | 0.8832 (205760) | 0.8752 (204952) |
| 7 | 0.7059 (205760) | 0.6998 (204952) | 0.7343 (204852) | 0.7321 (204896) | 0.8342 (205760) | 0.8327 (204952) |
| 8 | 0.6640 (205760) | 0.6466 (204952) | 0.6878 (204852) | 0.6865 (204896) | 0.8066 (205760) | 0.8059 (204952) |
| 9 | 0.5741 (205760) | 0.5394 (204952) | 0.5788 (204852) | 0.5552 (204896) | 0.8346 (205760) | 0.8321 (204952) |
| 10 | 0.5101 (205760) | 0.5061 (204952) | 0.5468 (204852) | 0.5155 (204896) | 0.9496 (205760) | 0.9410 (204952) |
| 11 | 0.1655 (205760) | 0.1679 (204952) | 0.2471 (204852) | 0.2434 (204896) | 0.9202 (205760) | 0.9160 (204952) |
| 12 | 0.0692 (205760) | 0.0748 (204952) | 0.1711 (204852) | 0.1689 (204896) | 0.7477 (205760) | 0.7517 (204952) |
| 13 | 0.0239 (205760) | 0.0297 (204952) | 0.0663 (204852) | 0.0609 (204896) | 0.7980 (205760) | 0.8141 (204952) |
| 14 | 0.0307 (205760) | 0.0389 (204952) | 0.1223 (204852) | 0.1246 (204896) | 0.4974 (205760) | 0.5046 (204952) |
| 15 | 0.0416 (205760) | 0.0474 (204952) | 0.1968 (204852) | 0.2026 (204896) | 0.6183 (205760) | 0.6170 (204952) |
| 16 | 0.0204 (205760) | 0.0215 (204952) | 0.0729 (204852) | 0.0650 (204896) | 0.9413 (205760) | 0.9427 (204952) |
| 17 | 0.0262 (205760) | 0.0318 (204952) | 0.0890 (204852) | 0.0921 (204896) | 0.5325 (205760) | 0.5524 (204952) |
| 18 | 0.0152 (205760) | 0.0162 (204952) | 0.0409 (204852) | 0.0462 (204896) | 0.3131 (205760) | 0.3275 (204952) |
| 19 | 0.0153 (205760) | 0.0183 (204952) | 0.0564 (204852) | 0.0611 (204896) | 0.8743 (205760) | 0.8766 (204952) |
| 20 | 0.0119 (205760) | 0.0148 (204952) | 0.0501 (204852) | 0.0619 (204896) | 0.3667 (205760) | 0.3878 (204952) |
| 21 | 0.0124 (205760) | 0.0151 (204952) | 0.0872 (204852) | 0.0846 (204896) | 0.6139 (205760) | 0.6173 (204952) |
| 22 | 0.0083 (205760) | 0.0091 (204952) | 0.0447 (204852) | 0.0422 (204896) | 0.5673 (205760) | 0.5687 (204952) |
| 23 | 0.0114 (205760) | 0.0125 (204952) | 0.0587 (204852) | 0.0615 (204896) | 0.3768 (205760) | 0.3892 (204952) |
| 24 | 0.0121 (205760) | 0.0135 (204952) | 0.0525 (204852) | 0.0657 (204896) | 0.2738 (205760) | 0.3015 (204952) |
| 25 | 0.0028 (205760) | 0.0031 (204952) | 0.0094 (204852) | 0.0103 (204896) | 0.7702 (205760) | 0.7842 (204952) |
| 26 | 0.0048 (205760) | 0.0050 (204952) | 0.0243 (204852) | 0.0297 (204896) | 0.4238 (205760) | 0.4417 (204952) |
| 27 | 0.0036 (205760) | 0.0038 (204952) | 0.0174 (204852) | 0.0218 (204896) | 0.6957 (205760) | 0.7132 (204952) |
| 28 | 0.0151 (205760) | 0.0166 (204952) | 0.0693 (204852) | 0.1009 (204896) | 0.5676 (205760) | 0.5978 (204952) |
| 29 | 0.0205 (205760) | 0.0208 (204952) | 0.0677 (204852) | 0.0812 (204896) | 0.6902 (205760) | 0.7245 (204952) |

## How to read this
- Descriptive, not causal: confirms the mechanism is real and its rough size, nothing more.
- This is the training-distribution val set; benchmark inference never packs documents for either arm,
  so this cannot explain which benchmark task moved (see plan) -- pair qualitatively with
  eval_per_doc.py's own-mask loss-by-position table for the training-distribution mechanism story.
- For an A_s* column, this is just how that model already runs (it trained with no masking) --
  what its own attention actually does. For a plain C_s* column, `--force_mask none` is a
  counterfactual: C trained *with* masking, so this measures what its attention would do if that
  mask were removed, not how it behaves under its own training/inference regime (which gives
  exactly 0 by construction and isn't shown here). Forcing this off also removes C's per-document
  RoPE position reset at the same time (both are gated by the same packing_impl flag), so a plain
  C_s* number reflects that combined effect, not masking removal in isolation.
- A `... (reset kept)` column ran with `--keep_position_reset`: masking forced off exactly as above,
  but RoPE position reset left ON (decoupled_packing_override) -- isolates the mask's own
  contribution from the position-scheme change. Compare it to the plain column for the same
  checkpoint: similar values means the position-scheme change wasn't doing much of the work;
  a large gap means it was.
