# Loss path: `full` vs `gather` vs `chunked` (H100)

`VLMConfig.lm_loss_impl` selects how the training loss is computed in
`VisionLanguageModel.forward`:

| impl | what it does |
|---|---|
| `full` | LM head over **every** position, materializing logits `[B, T, V]` (V = 49,218), then `F.cross_entropy` with `ignore_index=-100`. The pre-`68921a5` behaviour. |
| `gather` | Keep only positions with `targets != -100` **before** the LM head, so logits are `[N_valid, V]`. The default. |
| `chunked` | `gather`, then `F.linear_cross_entropy`, which never materializes logits at all. |

In a VLM most positions are masked out of the loss (image tokens, prompt tokens), so `full` spends
most of its LM-head compute and memory on logits it immediately discards. This is the measurement
of how much that costs, reproducible from the repo via [`eval/run_loss_ab.py`](../run_loss_ab.py).

Raw data: [`loss_gather_ab_results.json`](loss_gather_ab_results.json) (committed, strict JSON).
The three arms are also logged to wandb as group `loss_gather_ab` in `arjunsrinivasan/nanoVLM`.

## Environment

Same as [`attn_packing.md`](attn_packing.md): single NVIDIA H100 80GB HBM3, driver 570.124.06 with
`cuda-compat-13-0`, torch `2.14.0+cu130`, bfloat16 autocast. Model SmolLM2-360M-Instruct +
siglip2-base-patch16-512 (460,113,984 params), streamed FineVision, `batch_size=2` × 8 gradient
accumulation (effective 16), 200 steps per arm, eager. `lm_attn_packing_impl` held at its default
(`'none'`) across all three arms so this sweep isolates the loss path alone.

**Provenance caveat.** `loss_gather_ab_results.json` records `git_sha: 79f964f` with
`git_dirty: true`, so these runs came from a working tree that does not correspond exactly to any
commit (the same tree as `attn_packing.md`: `79f964f` with `train.py`'s committed conflict markers
resolved, which landed as `97ed280`). The arms differ only in `--loss_impl`, and all three share
that one tree, so the comparison between them is internally consistent — but it is not
reproducible from a clean checkout of `79f964f`. Re-running from a tagged commit is the fix;
`eval/run_loss_ab.py` reproduces the sweep.

## Results

Steady state, first logging interval dropped per arm. The `fw+bw` column is **CPU launch time, not
device time** — these runs read `train.py`'s per-interval `avg_fw_bw_time`, which at the time was
stopped before any device sync, so the GPU tail was charged to `post_process` (see
`speed_230m/README.md` †). All three arms here are eager, so no arm is flattered relative to
another, and the `tokens/s` and speedup columns come from `batch_duration`, which does enclose a
real sync. The 1.24× and the memory figures stand.

| impl | tokens/s | peak alloc (GiB) | peak reserved (GiB) | fw+bw (s) | speedup vs `full` | last loss |
|---|---|---|---|---|---|---|
| `gather` (default) | **14,865** | 50.26 | 50.47 | 0.268 | **1.24×** | 0.8977 |
| `full` | 11,969 | 53.83 | 55.54 | 0.349 | 1.00× | 0.8981 |
| `chunked` | 11,802 | **50.15** | **50.33** | 0.354 | 0.986× | 0.8953 |

**`gather` is the right default, by a wide margin.** It is 1.24× faster than `full` and cuts peak
allocated memory by 3.57 GiB — 97.2% of everything `chunked` saves — at no throughput cost.

- **The reserved-memory saving is larger than the allocated one.** `full` → `gather` frees 3.57 GiB
  allocated but 5.08 GiB reserved (55.54 → 50.47). The `[B, T, V]` logits tensor is large enough to
  fragment the caching allocator, so dropping it helps twice.
- **`chunked` is almost never worth it on this hardware.** It saves only a further **0.10 GiB** over
  `gather`, and pays **20.6%** of throughput for it — slower even than `full`.
- **Losses agree** (0.8953–0.8981). That is a sanity check that the arms saw the same data, not a
  precision claim: exact equivalence of all three paths — loss *and every parameter gradient* — is
  already proven by `tests/test_vision_language_model_loss.py` against an independent reference
  formula.

### Why `chunked` is slow (hypothesis, not profiled)

`chunked` calls `F.linear_cross_entropy(kept_hidden.float(), ...)`. The `.float()` is required
because the op is not autocast-registered — a bf16 input against the fp32 head weight raises. The
consequence is that the 960 → 49,218 vocabulary projection, one of the largest matmuls in the step,
runs in **fp32** instead of on the bf16 tensor cores. That is consistent with the size of the
regression, but it has not been confirmed with a profiler trace.

## This corrects the previous documentation

`models/config.py` described these trade-offs before any committed measurement existed, citing an
`experiments_h100/loss_gather_ab/` directory that was never checked in. Against the numbers above:

| previous claim | measured | |
|---|---|---|
| `gather` is the fastest of the three | 1.24× vs `full` | confirmed |
| `gather` keeps ~95% of `chunked`'s memory saving | 97.2% | confirmed |
| `chunked` saves a further ~0.2–0.5 GiB | 0.10 GiB | overstated 2–5× |
| `chunked` costs ~3–6% throughput | 20.6% | understated 3–7× |

The comment has been updated to the measured values and now points here.

## Caveats

- Single H100, one 200-step run per arm. Run-to-run noise is not characterized; treat differences
  under ~2% as indistinguishable. Every conclusion above rests on differences far larger than that.
- Eager only. `chunked`'s `linear_cross_entropy` always falls back to eager under `torch.compile`
  (documented upstream behaviour), so a compiled sweep would likely widen the gap.
- Measured with `lm_attn_packing_impl='none'`, the config default. The loss path is independent of
  the attention path, so the ranking should hold under `dense_block_diagonal` or
  `flex_document_causal`, but that combination was not measured here.
