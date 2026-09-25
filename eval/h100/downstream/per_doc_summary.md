# Per-document val loss by packing position

Token-weighted mean cross-entropy (nats/token). N = valid (non-ignored) token count.

## Doc-index-0 control (no preceding context either way -- doc_masked and unmasked should
nearly agree; a large gap here would mean the two masks aren't actually equivalent for doc 0)

Note: doc_masked and unmasked are separate traversals with independently randomized packing, so this
compares two overlapping-but-not-identical populations of "whichever documents landed at doc_index 0",
not one fixed set scored twice -- still a meaningful check since doc_index 0 is always that row's
longest document, a fairly stable population either way.

| checkpoint | doc_masked | unmasked | \|Δ\| | N |
|---|--:|--:|--:|--:|
| A_s0 | 0.8974 | 0.8975 | 0.0000 | 999182 |
| C_s0 | 0.8882 | 0.8883 | 0.0000 | 999182 |
| A_s1 | 0.9108 | 0.9109 | 0.0000 | 999182 |
| C_s1 | 0.8726 | 0.8727 | 0.0000 | 999182 |

## Own-mask loss by doc-index bucket (the mask each arm actually trained with:
unmasked for A, doc_masked for C -- this is the A-vs-C-by-position comparison)

| doc_index | A_s0 | C_s0 | A_s1 | C_s1 |
|---|---|---|---|---|
| 0 | 0.8975 (N=999182) | 0.8882 (N=999182) | 0.9109 (N=999182) | 0.8726 (N=999182) |
| 1 | 0.9663 (N=276603) | 0.9588 (N=276603) | 0.9747 (N=276603) | 0.9483 (N=276603) |
| 2+ | 0.9321 (N=145706) | 0.9288 (N=145706) | 0.9343 (N=145706) | 0.9251 (N=145706) |

## Own-mask loss by preceding-context-length bucket (doc_index >= 1 only, since doc 0 always has 0)

| preceding tokens | A_s0 | C_s0 | A_s1 | C_s1 |
|---|---|---|---|---|
| [1,512) | n/a | n/a | n/a | n/a |
| [512,1024) | 1.0879 (N=11752) | 1.0724 (N=11752) | 1.0916 (N=11752) | 1.0530 (N=11752) |
| [1024,2048) | 0.9470 (N=341200) | 0.9405 (N=341200) | 0.9544 (N=341200) | 0.9311 (N=341200) |
| [2048,4097) | 0.9687 (N=69357) | 0.9666 (N=69357) | 0.9700 (N=69357) | 0.9663 (N=69357) |

## Counterfactual mask, for reference (A under doc_masked, C under unmasked -- not what either arm trained with)

| doc_index | A_s0 | C_s0 | A_s1 | C_s1 |
|---|---|---|---|---|
| 0 | 0.8974 (N=999182) | 0.8883 (N=999182) | 0.9108 (N=999182) | 0.8727 (N=999182) |
| 1 | 0.9679 (N=276603) | 0.9966 (N=276603) | 0.9764 (N=276603) | 0.9897 (N=276603) |
| 2+ | 0.9443 (N=145706) | 1.0004 (N=145706) | 0.9470 (N=145706) | 1.0020 (N=145706) |

## How to read this
- This is the training-distribution (FineVision) val set, not benchmark data -- benchmark inference never
  packs documents (`doc_id=None` for both arms), so this cannot explain which benchmark task moved (see plan).
- Descriptive/correlational: it shows *where* the aggregate doc-masked-vs-unmasked val-loss gap concentrates,
  not that the mechanism *caused* any specific downstream benchmark result.
