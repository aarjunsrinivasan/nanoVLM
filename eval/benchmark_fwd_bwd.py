"""Benchmark and profile the forward pass, the backward pass, and the full forward+backward
training step of the current VLM, using real pretrained weights downloaded from the Hub.

Does not modify any models/*.py or models/config.py code -- it only exercises the
existing model through its public forward() API.

Run as a module from the repo root (eval/ is a package; a plain `python eval/benchmark_fwd_bwd.py`
would put eval/ itself on sys.path instead of the repo root and fail to import `models`/`data`).
Example (pin to a single idle GPU, per this repo's CLAUDE.md GPU rules):
    CUDA_VISIBLE_DEVICES=0 uv run python -m eval.benchmark_fwd_bwd --batch_sizes 1 2 4 8
"""
import argparse
import json
import os
import time

import pandas as pd
import torch
import torch._dynamo
import torch.optim as optim
from PIL import Image
from torch.profiler import ProfilerActivity, profile

from data.collators import VQACollator
from data.datasets import VQADataset
from data.processors import get_image_processor, get_tokenizer
from models.vision_language_model import VisionLanguageModel

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)


def build_batch(model, image_path, batch_size, max_length):
    """Builds a real training-shaped batch (image + tokenized VQA pair + labels) through
    the actual VQADataset/VQACollator pipeline train.py uses, so the benchmarked forward
    pass exercises the same code path as real training rather than a hand-rolled stand-in."""
    cfg = model.cfg
    tokenizer = get_tokenizer(cfg.lm_tokenizer, cfg.vlm_extra_tokens, cfg.lm_chat_template)
    image_processor = get_image_processor(cfg.max_img_size, cfg.vit_img_size, getattr(cfg, "resize_to_max_side_len", False))

    image = Image.open(image_path).convert("RGB")
    item = {
        "images": [image],
        "texts": [{"user": "What is shown in this image?", "assistant": "A descriptive answer about the image content."}],
    }
    dataset = VQADataset([item] * batch_size, tokenizer, image_processor, cfg.mp_image_token_length)

    natural_len = dataset[0]["input_ids"].size(0)
    if max_length is None:
        max_length = natural_len
    elif max_length < natural_len:
        raise ValueError(
            f"--max_length={max_length} is shorter than the tokenized example ({natural_len} tokens); "
            "samples would be silently discarded by the collator. Increase --max_length."
        )

    collator = VQACollator(tokenizer, max_length)
    samples = [dataset[i] for i in range(batch_size)]
    return collator(samples)


def warmup(setup_fn, measured_fn, num_warmup, device):
    for _ in range(num_warmup):
        setup_fn()
        measured_fn()
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_only(setup_fn, measured_fn, num_iters, device):
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    total_s = 0.0
    for _ in range(num_iters):
        setup_fn()
        if device.type == "cuda":
            torch.cuda.synchronize()  # exclude setup (e.g. the untimed forward before a backward-only measurement)
        start = time.perf_counter()
        measured_fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_s += time.perf_counter() - start

    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0
    return total_s / num_iters, peak_mb


def profile_once(setup_fn, measured_fn, device, label, row_limit, trace_dir):
    """Profiles only `measured_fn` -- `setup_fn` runs first, outside the profiler context, so
    e.g. a backward-only profile isn't polluted by the forward pass that has to precede it."""
    setup_fn()
    if device.type == "cuda":
        torch.cuda.synchronize()

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=True) as prof:
        measured_fn()
        if device.type == "cuda":
            torch.cuda.synchronize()

    print(f"\n--- Profile: {label} ---")
    sort_by = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_by, row_limit=row_limit))

    if trace_dir:
        os.makedirs(trace_dir, exist_ok=True)
        trace_path = os.path.join(trace_dir, f"{label}.json")
        prof.export_chrome_trace(trace_path)
        print(f"  Chrome trace saved to {trace_path} (view at chrome://tracing or https://ui.perfetto.dev)")


def run_phase(setup_fn, measured_fn, device, label, args, do_profile):
    warmup(setup_fn, measured_fn, args.num_warmup, device)
    if do_profile:
        profile_once(setup_fn, measured_fn, device, label, args.row_limit, args.trace_dir)
    return time_only(setup_fn, measured_fn, args.num_iters, device)


NOOP = lambda: None


def benchmark_batch_size(model, image_path, batch_size, max_length, autocast_dtype, device, args, do_profile):
    batch = build_batch(model, image_path, batch_size, max_length)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    images = batch["images"]  # left on CPU; model._process_images() moves it to device internally, same as train.py
    seq_len = input_ids.size(1)
    print(f"\n=== batch_size={batch_size} seq_len={seq_len} ===")

    # --- Observed (not derived) input shapes, printed + recorded once per batch size. This is
    # a pure attribute read on already-materialized tensors, run before any timed/profiled phase
    # below, so it has zero effect on the reported latency/VRAM numbers. ---
    sample_images = images[0]  # per-sample list of image tensors (one entry per image in that sample)
    image_patch_tensor_shape = tuple(sample_images[0].shape) if sample_images else None  # [n_patches, 3, H, W]
    print(f"Input shapes: input_ids={tuple(input_ids.shape)}, attention_mask={tuple(attention_mask.shape)}, labels={tuple(labels.shape)}")
    print(f"Image input: {len(images)} samples x {len(sample_images)} image(s)/sample, each image tensor shape {image_patch_tensor_shape} "
          f"(dim0 = number of patches after DynamicResize+GlobalAndSplitImages)")

    result = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "input_ids_shape": list(input_ids.shape),
        "image_patch_tensor_shape": list(image_patch_tensor_shape) if image_patch_tensor_shape else None,
    }

    # --- Forward only (eval mode, no grad) ---
    model.eval()

    def forward_fn():
        with torch.no_grad():
            model(input_ids, images, attention_mask=attention_mask, targets=None)

    # One untimed, unprofiled call to observe the actual output shape before benchmarking. With
    # targets=None the LM head is never applied (models/vision_language_model.py forward()), so
    # this is the raw decoder hidden state, not vocab logits -- worth surfacing directly rather
    # than assuming forward and backward return the same kind of tensor.
    with torch.no_grad():
        hidden_states, _ = model(input_ids, images, attention_mask=attention_mask, targets=None)
    print(f"Forward-only output (no LM head applied, targets=None): hidden_states.shape={tuple(hidden_states.shape)}")
    result["forward_output_shape"] = list(hidden_states.shape)
    del hidden_states

    fwd_latency_s, fwd_peak_mb = run_phase(NOOP, forward_fn, device, "forward_only", args, do_profile)
    print(f"Forward only:      {fwd_latency_s * 1000:8.2f} ms/iter, peak VRAM {fwd_peak_mb:8.1f} MB")
    result["forward_latency_ms"] = fwd_latency_s * 1000
    result["forward_peak_vram_mb"] = fwd_peak_mb

    # --- Backward only (train mode; the preceding forward is untimed/unprofiled setup) ---
    model.train()
    loss_holder = {}

    # One untimed, unprofiled call to observe a sample loss value before benchmarking. With
    # targets=labels the LM head IS applied, but only to the gathered non-masked positions
    # (VisionLanguageModel.forward's loss branch never materializes full [B,T,vocab_size]
    # logits), so the first return value is always None here -- nothing to log a shape for.
    with torch.no_grad():
        sample_logits, sample_loss = model(input_ids, images, attention_mask=attention_mask, targets=labels)
    print(f"Backward/full-step output (gather-before-head, targets=labels): logits=None (gather-before-head optimization active), loss={sample_loss.item():.4f}")
    result["backward_output_shape"] = None
    result["sample_loss"] = sample_loss.item()
    del sample_logits, sample_loss

    def forward_setup():
        with torch.autocast(device_type=device.type, dtype=autocast_dtype):
            _, loss_holder["loss"] = model(input_ids, images, attention_mask=attention_mask, targets=labels)

    def backward_fn():
        loss_holder["loss"].backward()

    bwd_latency_s, bwd_peak_mb = run_phase(forward_setup, backward_fn, device, "backward_only", args, do_profile)
    print(f"Backward only:     {bwd_latency_s * 1000:8.2f} ms/iter, peak VRAM {bwd_peak_mb:8.1f} MB")
    result["backward_latency_ms"] = bwd_latency_s * 1000
    result["backward_peak_vram_mb"] = bwd_peak_mb

    # --- Full training step (forward + backward + optimizer.step, matches train.py) ---
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)  # dummy optimizer, just to include its step cost

    def full_step_fn():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype):
            _, loss = model(input_ids, images, attention_mask=attention_mask, targets=labels)
        loss.backward()
        optimizer.step()

    step_latency_s, step_peak_mb = run_phase(NOOP, full_step_fn, device, "forward_backward_step", args, do_profile)
    print(f"Full train step:   {step_latency_s * 1000:8.2f} ms/iter, peak VRAM {step_peak_mb:8.1f} MB")
    result["full_step_latency_ms"] = step_latency_s * 1000
    result["full_step_peak_vram_mb"] = step_peak_mb

    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark & profile forward, backward, and full train-step of the current VLM.")
    parser.add_argument("--model_id", default="lusxvr/nanoVLM-460M-8k", help="HF Hub repo id or local checkpoint path.")
    parser.add_argument("--image_path", default="assets/image.png")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 2, 4, 8], help="Batch sizes to sweep.")
    parser.add_argument("--max_length", type=int, default=None, help="Defaults to the natural tokenized length of the example (no padding).")
    parser.add_argument("--num_warmup", type=int, default=3)
    parser.add_argument("--num_iters", type=int, default=10)
    parser.add_argument("--row_limit", type=int, default=20, help="Rows to show in each profiler table.")
    parser.add_argument("--trace_dir", default=None, help="If set, export a Chrome trace per phase (and batch size) to this directory.")
    parser.add_argument("--results_file", default="eval/h200/benchmark_fwd_bwd_results.json", help="Results are grouped under eval/<gpu>/ by the hardware they were measured on.")
    parser.add_argument("--compile", action="store_true", help="Wrap the model with torch.compile() before benchmarking (each batch size gets an isolated fresh compile via torch._dynamo.reset()).")
    parser.add_argument("--profile_batch_size", type=int, default=None,
                         help="Only this batch size gets the detailed torch.profiler op tables (default: the largest in --batch_sizes). "
                              "Every batch size still gets timing + peak VRAM.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset -- all GPUs visible!')})")

    print(f"Loading pretrained model weights: {args.model_id}")
    model = VisionLanguageModel.from_pretrained(args.model_id).to(device)
    if args.compile:
        print("Compiling model with torch.compile() (matches train.py's real compile call; each batch "
              "size gets an isolated fresh compile below rather than sharing one dynamic-shape graph)...")
        model = torch.compile(model)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model loaded with {param_count:,} parameters")

    # The full-step phase calls a real optimizer.step(), which actually updates the weights. Since
    # `model` is reused across the whole batch-size sweep, without resetting it each batch size
    # would benchmark against a model further overfit to the same repeated synthetic example by
    # every previous batch size's steps -- harmless for latency/VRAM, but it silently mutates the
    # model under test and makes `sample_loss` meaningless across sizes. Snapshot once, reload
    # before each batch size so every size benchmarks the same untouched pretrained checkpoint.
    initial_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}

    autocast_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    profile_batch_size = args.profile_batch_size if args.profile_batch_size is not None else max(args.batch_sizes)

    all_results = []
    for batch_size in args.batch_sizes:
        model.load_state_dict(initial_state_dict)
        if args.compile:
            # All 3 phases below (forward-only: eval/no_grad/no-autocast vs. backward/full-step:
            # train/autocast) genuinely call the compiled model, so each batch size needs ~2
            # distinct dynamo graph flavors against torch._dynamo.config.recompile_limit (default
            # 8, shared per compiled function across the WHOLE sweep). Without resetting here, 4
            # batch sizes x 2 flavors hits that cap and later sizes silently fall back to eager
            # instead of a fresh compile. reset() clears dynamo's caches only -- the OptimizedModule
            # wrapper from torch.compile() above stays valid and just retraces+recompiles fresh on
            # its next call, isolating each batch size as if it were a fresh process invocation.
            torch._dynamo.reset()
        do_profile = (batch_size == profile_batch_size)
        result = benchmark_batch_size(model, args.image_path, batch_size, args.max_length, autocast_dtype, device, args, do_profile)
        result.update({"model_id": args.model_id, "param_count": param_count, "compiled": args.compile})
        all_results.append(result)

    print("\n--- Summary ---")
    df = pd.DataFrame(all_results)
    print(df.to_string(index=False))

    with open(args.results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {args.results_file}")


if __name__ == "__main__":
    main()
