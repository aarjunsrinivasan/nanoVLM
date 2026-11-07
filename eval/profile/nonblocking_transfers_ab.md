# H2D transfers: `blocking` vs `non_blocking` (H200)

`VLMConfig.non_blocking_transfers` controls whether the four batch tensors moved to device in
`train.py` (`input_ids`, `labels`, `attention_mask`, `doc_id`) and the concatenated image tensor in
`VisionLanguageModel._process_images` use `non_blocking=True` on their `.to(device)` calls. Both
DataLoaders already set `pin_memory=True` (`train.py:253,266`) — pinning was not the missing piece,
only `non_blocking=True` was. Reproducible via [`eval/run_nonblocking_ab.py`](../run_nonblocking_ab.py).

Raw data: [`nonblocking_transfers_ab_results.json`](nonblocking_transfers_ab_results.json)
(committed, strict JSON).

## Environment

Single NVIDIA H200 143GB HBM3e (`CUDA_VISIBLE_DEVICES=1`, 3 other GPUs left idle throughout),
driver 610.43.02, torch `2.14.0+cu130`, bfloat16 autocast. Model SmolLM2-360M-Instruct +
siglip2-base-patch16-512 (460,113,984 params), streamed FineVision, `batch_size=2` × 8 gradient
accumulation (effective 16), 200 steps per arm, eager. `lm_loss_impl`/`lm_attn_packing_impl` held
at their config defaults (`'gather'`/`'none'`) across both arms — this sweep isolates the transfer
path alone. `git_sha=554bbbb`, dirty (uncommitted local changes present during the run — the
`non_blocking_transfers` code change itself). 2026-09-18.

## Results

Steady state, first logging interval dropped per arm.

| arm | tokens/s | peak alloc (GiB) | fw+bw (s) | speedup vs `blocking` | last loss |
|---|---|---|---|---|---|
| `non_blocking` | **16,180** | 50.26 | **0.250** | **1.22×** | 0.8955 |
| `blocking` (pre-change default) | 13,298 | 50.26 | 0.320 | 1.00× | 0.8997 |

**`non_blocking=True` is a real, measurable win: 1.22× faster, no memory or correctness cost.**
Peak allocated memory is identical between arms (50.26 GiB) — `non_blocking` only changes copy
scheduling, not what's materialized. Losses agree closely (0.8955 vs 0.8997, same seed/data), the
usual sanity check that both arms trained on equivalent data, not a precision claim.

A short dry run (20 steps, no warmup dropped) showed the same effect at a similar magnitude
(1.34×: 16,084 vs 12,009 tokens/s), so this isn't a fluke of the particular 200-step window.

### Why (mechanism)

The expected mechanism was that `non_blocking=True` lets the H2D copy overlap with the CPU-side
work of enqueuing the forward pass, since nothing synchronizes between the `.to(device)` calls and
`loss.item()` (`train.py:483`). The measured `avg_data_load_time_s` (0.0006s `blocking` vs 0.0059s
`non_blocking`, both tiny) shows the four *text* tensors (`input_ids`/`labels`/`attention_mask`/
`doc_id`) are small enough that their own transfer time was never the bottleneck in either arm.

The actual saving shows up in `avg_fw_bw_time_s` instead (0.320s → 0.250s, a 22% cut on its own).
That's because the **image** tensor's `.to(device)` call (`_process_images`,
`models/vision_language_model.py:59`) happens *inside* `model.forward()`, inside the timed fw+bw
window, not in `train.py`'s separately-timed data-load phase. Under `blocking=False`'s old
behavior, that call inserted an implicit CUDA sync: the CPU thread stalled until the image copy
completed before it could go on to dispatch the vision encoder's kernels, the language model's 32
transformer blocks, and everything else in the forward+backward graph — a per-step pipeline bubble
where the GPU has nothing queued while the CPU waits. `non_blocking=True` removes that stall: the
CPU keeps dispatching the rest of the graph immediately, and the image copy proceeds on the DMA
engine in parallel, hidden behind that dispatch overhead. This is consistent with the size of the
regression (fw+bw shrinking by almost exactly the CPU dispatch overhead of one step, not by an
amount that scales with the tiny text-tensor size), though it has not been confirmed with a
profiler trace — flagged here as a plausible mechanism, not a proven one.

### Why `avg_data_load_time` is not the headline number

The module docstring in `eval/run_nonblocking_ab.py` and the code change's design plan both warned
that `avg_data_load_time` could become an *undercount* under `non_blocking=True` — measuring time
to launch the copy rather than time for it to complete, since no `torch.cuda.synchronize()` sits
around it. In this run that artifact didn't manifest as a large drop (both arms' data-load times
stayed near zero, since the four text tensors are small regardless), but the caveat still applies
in general and the correct headline metric is `avg_tokens_per_second`/`avg_fw_bw_time_s`, both of
which are trustworthy end-to-end because `loss.item()` forces a full CUDA sync before
`batch_end_time` is captured, downstream of everything.

## Caveats

- Single H200, one 200-step run per arm (plus one 20-step dry run as a consistency check).
  Run-to-run noise across repeats was not characterized; treat differences under ~2% as
  indistinguishable — the 22% difference measured here is far larger than that.
- Eager only; not measured under `torch.compile`.
- Only these 5 existing H2D call sites are covered by `non_blocking_transfers`. No claim is made
  about other potential async-copy opportunities elsewhere in the codebase (e.g. checkpoint D2H).
- The fw+bw mechanism above (implicit sync stalling CPU dispatch) is a hypothesis consistent with
  the measurement, not a profiler-confirmed fact.
