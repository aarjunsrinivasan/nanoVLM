# Phase 0: batch-size sweep and pilot (~230M model, 1× H100 80GB)

These runs set the batch size and give time and data estimates for the A/B experiment. They are sizing runs, not the final speed results. For those, see Phase 1, which uses interleaved, repeated runs on an otherwise idle machine.

## Setup

**Model:** `--lm_model_type HuggingFaceTB/SmolLM2-135M-Instruct`, **228,063,936 params** in total:

| component | params |
|---|---|
| SigLIP2-base-patch16-512 | 86.4M |
| SmolLM2-135M-Instruct | 134.6M |
| projector | rest |

**Common flags:** `--val_size 5000 --no_eval --no_log_wandb`, shard cache under `/workspace/.cache/finevision_shards`. Code at `554bbbb`. torch 2.14.0+cu130.

**Arms:**

| arm | flags |
|---|---|
| A (upstream baseline) | `--loss_impl full --attn_packing_impl none` |
| B | `--loss_impl gather --attn_packing_impl flex_document_causal` |
| C | B + `--compile` (default `recompile_limit` of 8, unchanged) |

**Scripts:** `bs_sweep.sh` and `pilot.sh`, run from anywhere (they cd to the repo root); logs land in `runs/`, override with `OUT=`. `count_data.py` measures data consumption.

## Batch-size sweep (60 steps; micro-batch × accum = 16 rows/step)

Steady tok/s is the mean over steps 20–50.

| arm | bs 2 steady tok/s | bs 2 peak reserved | bs 4 steady tok/s | bs 4 peak reserved | bs 8 |
|---|---|---|---|---|---|
| A | ~14.9k | 43.0 GiB | ~16.9k | **78.4 GiB** | OOM |
| B | ~15.5k | 35.6 GiB | ~18.5k | 65.6 GiB | OOM |
| C | ~23.1k | 32.9 GiB | ~22.0k | 60.4 GiB | OOM |

**Chosen: micro-batch 2 × accum 8, the upstream default.**
- At bs 4, A reaches 78.4 of 79.2 GiB within 60 steps, so one image-heavy batch in a 10k-step run would OOM it.
- C gets no gain from bs 4.
- Note: `ConstantLengthDataset` sizes its packing buffer from `batch_size`, so micro-batch size slightly changes the data. All arms must use the same one.

## Pilot (300 steps, bs 2 × accum 8)

Steady numbers are over steps 100–275.

| arm | tok/s (GPU-side, excl. data wait) | fw_bw per micro-batch | data wait per micro-batch | wall s/step | peak reserved |
|---|---|---|---|---|---|
| A | 14.8k (1.00×) | 0.284 s | 0.011 s | 2.67 (1.00×) | 44.3 GiB |
| B | 15.6k (1.05×) | 0.266 s | 0.000 s | 2.52 (1.06×) | 36.9 GiB |
| C | 22.8k (1.54×) | 0.175 s | 0.047 s, rising to 0.094 | 2.28 (1.17×) | 34.2 GiB |

Notes:
- **`fw_bw` is CPU launch time, not device time.** It comes from `train.py`'s `avg_fw_bw_time`, which at the time was stopped before any device sync, so the GPU tail landed in `post_process` instead (see `../speed_230m/README.md` †). Since C is compiled and A/B are not, and `torch.compile` reduces launch cost specifically, the `0.284 → 0.175` narrowing overstates the kernel-time effect. `tok/s` and `wall s/step` are unaffected, and the sizing decisions on this page rest on memory and wall clock, not on this column.
- **A, steps 0–90:** a CPU-bound data-counting job ran at the same time and slowed A's first ~90 steps (fw_bw 0.5 s). Those intervals are excluded. Nothing in these scripts enforces machine idleness; that contamination was caught by reading the logs, not by the harness.
- **C recompiles:** 8 recompiles, all within the first ~65 s of the loop (about step 5). Each came from a new image-tile count (`images[i][0]` size). C hit `recompile_limit` (8) at about step 5. After that, throughput held steady at 22–24k tok/s through step 300. Whether rarely seen shapes fall back to eager over 10k steps still needs watching.
- **C is data-starved with 2 workers:** data wait rises from 0.017 to 0.094 s per micro-batch. That is why its 1.54× GPU-side speedup shrinks to 1.17× in wall time.

## Data consumption (`count_data.log`)

- 2.51 raw stream samples per packed 4096-token row, and 2.44 documents per row (about 3% filtered).
- For 10k steps: 160k rows → ~402k raw samples → ~166 train shards (~72 GiB), plus 3 val shards (~1.3 GiB).
- All arms read the same shards, so one download serves every run.

## DataLoader workers (arm C, 150 steps, bs 2 × accum 8; logs in `workers/`)

Wall s/step is measured over steps 50–125.

| `--num_workers` | data wait per micro-batch | wall s/step |
|---|---|---|
| 2 (default, pilot) | 0.047 s, rising to 0.094 | 2.28 |
| **4** | 0.001 s | **1.77** |
| 6 | 0.001 s | 1.77 |

**Chosen: `--num_workers 4` for every arm.** 4 already removes the data stall and 6 adds nothing. The worker count changes how shards are assigned to workers, so it changes data order. It must therefore be identical across arms.

## Decision for Phase 2

- Settings: bs 2 × accum 8, 4 workers, 10k steps.
- Estimated time: A ≈ 7.3 h and C ≈ 4.9 h per seed. Two seeds each, plus evals, comes to about 25 h, within budget, so 10k steps stays.
