# Loss path (`full` vs `gather`) x eager vs `torch.compile` (H200)

[`loss_gather_ab.md`](../h100/loss_gather_ab.md) compared `lm_loss_impl` in eager mode only. This is the
missing 2x2: `full` vs `gather`, each eager and under `torch.compile`, with everything else fixed.
`chunked` is left out (its `linear_cross_entropy` always falls back to eager under compile).

Raw data: `loss_compile_ab/{E,C}_{full,gather}_r{1,2}.{json,log}` (E = eager, C = compile).
Reproduce with [`loss_compile_ab/run.sh`](loss_compile_ab/run.sh), tabulate with
[`loss_compile_ab/summarize.py`](loss_compile_ab/summarize.py). Not logged to wandb.

## Setup

- 1x NVIDIA H200 (one GPU pinned; all four were idle and no Slurm jobs were queued at launch), torch `2.14.0+cu130`, bf16 autocast, code at `c9d0f1e`.
- SmolLM2-360M-Instruct + siglip2-base-patch16-512 (same model as `loss_gather_ab.md`), streamed FineVision,
  micro-batch 2 x grad-accum 8, 4096-token packed rows, `--num_workers 4`, `lm_attn_packing_impl='none'`.
- 300 steps per run; steady state = steps 100-300 (first 4 of 12 stats intervals dropped, same for every arm).
- **Interleaved, 2 reps** (E_full, E_gather, C_full, C_gather, repeat), **one process per arm**: dynamo's config,
  compile cache and `recompile_limit` are process-global, so arms must not share a process.
- `--num_workers 4`, not the driver's default of 2: two workers starve compiled arms
  (see `eval/h100/speed_230m/README.md`).

## Results

Mean of two reps (range in brackets). Memory is the max over reps.

| arm | tok/s | fw+bw (s) | peak alloc (GiB) | peak reserved (GiB) | last-batch loss |
|---|---|---|---|---|---|
| eager, `full` | 13,347 (13,144-13,549) | 0.315 | 52.54 | 55.56 | 0.7134 |
| eager, `gather` | 13,460 (13,410-13,510) | 0.312 | 48.98 | 50.51 | 0.7143 |
| compile, `full` | **18,134** (17,953-18,315) | 0.228 | 47.96 | 50.88 | 0.7148 |
| compile, `gather` | 17,669 (17,458-17,880) | 0.236 | **44.50** | **46.01** | 0.7144 |

| comparison | tok/s ratio | per rep |
|---|---|---|
| `gather` vs `full`, eager | 1.008x | 0.990x, 1.028x |
| `gather` vs `full`, compile | 0.974x | 0.976x, 0.972x |
| compile vs eager, `full` | 1.359x | 1.352x, 1.366x |
| compile vs eager, `gather` | 1.313x | 1.333x, 1.292x |

Peak memory saved by `gather` vs `full`: eager 3.56 GiB allocated / 5.05 GiB reserved; compile 3.46 / 4.87 GiB.

## What this says

- **`gather` is a memory optimization here, not a speed one.** In eager the two are indistinguishable (the reps
  disagree in sign: 0.990x then 1.028x). Under compile `gather` was ~2.6% *slower* in both reps and the ranges do
  not overlap, but with two reps and within-run interval spread of about +/-3-5% this is "no gain, possibly a
  small loss", not a settled regression.
- **The memory saving is solid and does not depend on compile.** 3.5 GiB allocated and ~5 GiB reserved either way;
  the reserved figure is larger than the allocated one for the reason `loss_gather_ab.md` gives (the `[B, T, V]`
  logits tensor fragments the caching allocator).
- **Compile is worth ~1.3-1.36x for either loss path**, and the two effects stack in memory: `gather` + compile
  peaks at 44.5 GiB vs 52.5 GiB for `full` + eager.
- **All four arms trained equivalently**: last-batch losses end within 0.002. (In the 20-step smoke runs, early
  batch losses differed between arms by up to 0.36 at step 5 and converged by step 15; the earlier speed runs
  show the same early transient between eager and compile. So compare losses late in a run, not early.)
- No graph break comes from the gather. A 20-step `TORCH_LOGS=graph_breaks,recompiles` run of `C_gather` showed
  breaks only in the tokenizer and in the decoder's data-dependent branch at `language_model.py:179` (both arms
  share these). `train.py` already sets `capture_dynamic_output_shape_ops` under `--compile`, which is what lets
  the boolean-mask gather trace in-graph.

## Does not reproduce the H100 result

`loss_gather_ab.md` reports `gather` 1.24x faster than `full` (eager, H100). This run finds 1.008x in eager on an
H200. Two possible reasons, **neither tested here**: (1) that H100 study ran each arm once, not interleaved, and the
repo has seen 24% session-to-session drift on one config (noted in `eval/h100/speed_230m/README.md`); (2) the
H200's higher memory bandwidth makes materializing full logits cheaper. The memory numbers do reproduce almost
exactly (3.57 / 5.08 GiB on H100 vs 3.56 / 5.05 GiB here). Settling the speed question needs an interleaved H100 rerun.

## Caveats

- Two reps per arm, one GPU, one model size, 300 steps, one packing setting (`none`). Differences under ~3% are
  within run-to-run and within-run noise.
- **Every compiled run hit dynamo's `recompile_limit` (8) within the first steps**, from guards on the image-list
  structure (number of images / tiles in the batch), not from the loss path. After that, micro-batches with an
  image shape that has no cached graph run eager, so the compiled arms are a mix of compiled and eager
  micro-batches. It affects both compiled arms alike; the limit was left at its default, as in the 230M speed runs.
- Peak memory is a max over reps, not a mean.
