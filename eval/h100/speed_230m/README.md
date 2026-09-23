# Phase 1: speed, ~230M model, 1× H100 80GB

Upstream's training path vs this fork's, at the settings Phase 0 fixed. Runs are **interleaved** (A B C, A B C) rather
than grouped per arm, because two runs of one config in different sessions have differed by 24% in this repo before
(`loss_gather_ab.md` vs `attn_packing.md`); interleaving spreads any drift over all arms equally.

- Model: `HuggingFaceTB/SmolLM2-135M-Instruct` + siglip2-base-patch16-512, 228,063,936 params
- 300 steps, micro-batch 2 × grad-accum 8, `--num_workers 4`, `--no_eval`, seed 0, warm shard cache
- All arms train on the identical 300 batches (same seed and data pipeline); step-275 losses agree to 4 decimals
- Steady state = steps 100–275, so compile warmup and cache warmup are excluded
- wandb group `speed-230m`; logs and the driver script are in this directory

| arm | loss / packing / compile | tok/s (compute) | vs A | spread | fw+bw † | peak reserved | wall s/step | vs A |
|---|---|---|---|---|---|---|---|---|
| A (upstream) | full / none / eager | 14,674 | 1.00× | 2.4% | 0.286 s | 48.5 GiB | 2.68 | 1.00× |
| B | gather / flex / eager | 15,210 | 1.04× | 0.3% | 0.280 s | 40.9 GiB (−7.6) | 2.59 | 1.04× |
| C | gather / flex / compile | 22,619 | **1.54×** | 4.2% | 0.179 s | 38.4 GiB (**−10.2**) | **1.77** | **1.51×** |

End to end, including compile warmup and the data-cache warmup of the first steps, C finishes the 300 steps 1.36×
faster than A (594 s vs 808 s). The compile cost is one-off, so over a 10k-step run this converges to the 1.51× above.

† **The `fw+bw` column is CPU launch time, not device time.** When these runs were made, `train.py` stopped the
fw+bw timer straight after `loss.backward()` returned, which only *queues* kernels; the first real device sync came a
few lines later at `loss.item()`, so the GPU tail was charged to `post_process` instead. `torch.compile` reduces CPU
launch cost specifically, so this column **overstates C's per-phase advantage** and the split between arms should not
be read as a kernel-time breakdown. Everything else in the table is unaffected: `tok/s`, `wall s/step` and the
**1.54× / 1.51× / 1.36×** headline all derive from `batch_duration`, which encloses `loss.item()` and therefore a
genuine sync. Fixed for future runs by an explicit `torch.cuda.synchronize()` before the timer; these numbers are left
as measured rather than silently restated, and re-running was not judged worth the GPU time since no claim rests on them.

**Caveats**
- `data` wait is 0.001 s everywhere: with 4 workers no arm is data-bound, so these are GPU-side numbers. With the
  upstream default of 2 workers, C is starved and its wall-clock gain drops to 1.17× (Phase 0).
- Both C runs hit dynamo's `recompile_limit` (8) about 2 minutes in, as new image-tile counts appear. The limit was
  left at its default. Throughput stays flat (21.5k–24.3k tok/s) to step 300, so nothing is lost over this horizon;
  whether rarer tile counts later fall back to eager over 10k steps is reported from the Phase 2 runs.
- Two repetitions per arm. The A and C ranges are far apart (14.5k–14.9k vs 22.1k–23.1k); A and B nearly touch
  (14.5k–14.9k vs 15.2k–15.2k), so the 1.04× eager gain is small but consistent in direction.
- `speed.sh` as committed loops `for rep in 1 2 3`, i.e. it was written to do three repetitions per arm, but only
  `r1` and `r2` exist for each arm and every number above is computed over those two. The third repetition did not
  produce a kept log; the driver is left exactly as it was run rather than edited to match the output. So "spread"
  here is the range of two observations — a sanity check on direction, not a dispersion estimate.
