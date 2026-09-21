import os
import re
import json
import math
import shutil
import hashlib
import time
import torch
import torch._dynamo
import wandb
import numpy
import random
import argparse
import contextlib
import subprocess
import torch.optim as optim
from statistics import mean
from dataclasses import asdict
from datetime import timedelta
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_dataset, concatenate_datasets, get_dataset_config_names, load_from_disk

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

PG_CPU = None

from data.datasets import VQADataset
from data.collators import VQACollator
from data.data_utils import synchronized_dataloader_step
from data.advanced_datasets import ConstantLengthDataset
from data.data_progress import DataProgress
from data.shard_cache import get_cached_train_val_datasets
from data.processors import get_image_processor, get_tokenizer

import models.config as config
from models.vision_language_model import VisionLanguageModel
from models.language_model import packing_impl_override

#Otherwise, the tokenizer will throw a warning
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import warnings
warnings.filterwarnings("ignore", message=".*Length of IterableDataset.*")

# Fix for "Decompressed data too large" error with certain PNGs
import PIL.PngImagePlugin
PIL.PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

def init_dist():
    dist.init_process_group(backend='nccl', timeout=timedelta(minutes=30))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    # torch.cuda.manual_seed(0)           # seed *this* GPU only

def destroy_dist():
    dist.destroy_process_group()

def is_dist():
    return dist.is_available() and dist.is_initialized()

def is_master():
    return dist.get_rank() == 0 if is_dist() else True

def get_world_size():
    return dist.get_world_size() if is_dist() else 1

def get_rank():
    return dist.get_rank() if is_dist() else 0

def dist_gather(obj):
    """
    Gather *any* picklable object from every rank without allocating
    temporary CUDA buffers.  Returns a list [rank0_obj, rank1_obj, …].

    Falls back to a single-rank list when torch.distributed is not initialised.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return [obj]

    result = [None] * dist.get_world_size()
    dist.all_gather_object(result, obj, group=PG_CPU)  # CPU path
    return result

def dist_mean_scalar(x: float | int) -> float:
    if not (dist.is_available() and dist.is_initialized()):
        return float(x)

    t = torch.tensor(x, device=torch.cuda.current_device(), dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)           # in‑place, returns None
    t /= dist.get_world_size()
    return t.item()

def wrap_model(model):
    local_rank = int(os.environ["LOCAL_RANK"])
    return DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

def get_run_name(train_cfg, vlm_cfg):
    batch_size = f"bs{int(train_cfg.batch_size*get_world_size()*train_cfg.gradient_accumulation_steps)}"
    max_training_steps = f"{train_cfg.max_training_steps}"
    learning_rate = f"lr_vision_{train_cfg.lr_vision_backbone}-language_{train_cfg.lr_language_backbone}-{train_cfg.lr_mp}"
    num_gpus = f"{get_world_size()}xGPU"
    date = time.strftime("%m%d-%H%M%S")
    vit = f"{vlm_cfg.vit_model_type.split('/')[-1]}" + f"_{vlm_cfg.max_img_size}"
    mp = f"mp{vlm_cfg.mp_pixel_shuffle_factor}"
    llm = f"{vlm_cfg.lm_model_type.split('/')[-1]}"

    run_name = f"nanoVLM_{vit}_{mp}_{llm}_{num_gpus}_{batch_size}_{max_training_steps}_{learning_rate}_{date}"
    if train_cfg.run_name_suffix:
        run_name = f"{run_name}_{train_cfg.run_name_suffix}"
    return run_name

def get_provenance():
    """Git commit, torch version and GPU behind a run, so logged numbers can be traced back."""
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=repo_dir, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    status = git("status", "--porcelain", "--", "models", "data", "train.py")
    return {
        "git_sha": git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "torch_version": torch.__version__,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

def get_hub_train_val_datasets(train_cfg):
    dataset_names_to_load = train_cfg.train_dataset_name
    if "shards" in train_cfg.train_dataset_name:
        print("Loading shards")
        total_shards = 56
        dataset_names_to_load = [train_cfg.train_dataset_path + f"/shard_{i}" for i in range(total_shards)]

    if "all" in dataset_names_to_load:
        dataset_names_to_load = get_dataset_config_names(train_cfg.train_dataset_path)

    # Load and combine all training datasets
    combined_train_data = []

    for dataset_name in dataset_names_to_load:
        print(f"Loading dataset: {dataset_name}")
        if "shard_" in dataset_name:
            try:
                train_ds = load_from_disk(dataset_name)
                combined_train_data.append(train_ds)
                continue
            except Exception as e:
                print(f"Warning: Failed to load dataset shard '{dataset_name}' from '{train_cfg.train_dataset_path}'. Error: {e}")
                continue
        try:
            train_ds = load_dataset(train_cfg.train_dataset_path, dataset_name, streaming=train_cfg.stream_dataset, on_bad_files='warn')['train']
            if train_cfg.stream_dataset:
                next(iter(train_ds)) # Check if the dataset is loaded correctly
            else:
                train_ds[0] # Check if the dataset is loaded correctly
            combined_train_data.append(train_ds)
        except Exception as e:
            if is_master():
                print(f"Warning: Failed to load dataset config '{dataset_name}' from '{train_cfg.train_dataset_path}'. Error: {e}")
            continue

    if not combined_train_data:
        raise ValueError("No valid datasets were loaded. Please check your dataset path and configurations.")
    
    train_ds = concatenate_datasets(combined_train_data)

    if not train_cfg.stream_dataset:
        train_ds = train_ds.shuffle(seed=0) # Shuffle the training dataset, so train and val get equal contributions from all concatenated datasets  


    if is_dist():  # We need to shard the dataset in DDP since we are using an iterable dataset instead of the distributed sampler
        train_ds = train_ds.shard(num_shards=get_world_size(), index=get_rank())

    # train_ds = train_ds.shuffle(buffer_size=10000, seed=0) # Shuffle the training dataset, so train and val get equal contributions from all concatenated datasets  

    val_size = int(train_cfg.val_size/get_world_size())
    print(f"Val size per GPU: {val_size}")

    if train_cfg.stream_dataset:
        val_ds = train_ds.take(val_size)
        train_ds = train_ds.skip(val_size)
    else:
        val_ds = train_ds.select(range(val_size))
        train_ds = train_ds.select(range(val_size, len(train_ds)))

    return train_ds, val_ds

def get_dataloaders(train_cfg, vlm_cfg, data_state=None):
    """data_state (from a save_training_state checkpoint) resumes the train stream exactly where that run stopped."""
    print(f"Getting dataloaders from {train_cfg.train_dataset_path}")
    # Create datasets
    image_processor = get_image_processor(vlm_cfg.max_img_size, vlm_cfg.vit_img_size, vlm_cfg.resize_to_max_side_len)
    tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer, vlm_cfg.vlm_extra_tokens, vlm_cfg.lm_chat_template)

    if train_cfg.dataset_cache_dir is not None:
        if not train_cfg.stream_dataset:
            raise ValueError("dataset_cache_dir requires stream_dataset=True")
        if tuple(train_cfg.train_dataset_name) != ("default",):
            raise ValueError("dataset_cache_dir reads all parquet shards of the repo, so train_dataset_name must be ('default',)")
        # Don't iterate the datasets here: that would start download threads before the DataLoader forks its workers
        train_ds, val_ds = get_cached_train_val_datasets(train_cfg, get_world_size(), get_rank(), data_state)
        print(f"Val size per GPU: {int(train_cfg.val_size/get_world_size())}")
    else:
        if data_state is not None:
            raise ValueError("Resuming the data position needs dataset_cache_dir (the shard reader skips consumed rows)")
        train_ds, val_ds = get_hub_train_val_datasets(train_cfg)

    train_dataset = VQADataset(
        train_ds,
        tokenizer,
        image_processor,
        vlm_cfg.mp_image_token_length,
        train_cfg.relevance_min_rating,
        train_cfg.image_correspondence_min_rating,
        train_cfg.visual_dependency_min_rating,
        train_cfg.formatting_min_rating,
    )
    val_dataset = VQADataset(
        val_ds,
        tokenizer,
        image_processor,
        vlm_cfg.mp_image_token_length,
        train_cfg.relevance_min_rating,
        train_cfg.image_correspondence_min_rating,
        train_cfg.visual_dependency_min_rating,
        train_cfg.formatting_min_rating,
    )

    train_dataset = ConstantLengthDataset(train_dataset, infinite=False, max_sample_length=train_cfg.max_sample_length, seq_length=vlm_cfg.lm_max_length, num_of_sequences=train_cfg.batch_size*4, queue_size=8,
                                        max_images_per_example=train_cfg.max_images_per_example, max_images_per_knapsack=train_cfg.max_images_per_knapsack,
                                        resume_state=data_state)

    val_dataset = ConstantLengthDataset(val_dataset, infinite=False, max_sample_length=train_cfg.max_sample_length, seq_length=vlm_cfg.lm_max_length, num_of_sequences=train_cfg.batch_size*4, queue_size=8,
                                        max_images_per_example=train_cfg.max_images_per_example, max_images_per_knapsack=train_cfg.max_images_per_knapsack)

    # Create collators
    vqa_collator = VQACollator(tokenizer, vlm_cfg.lm_max_length)

    g = torch.Generator()
    g.manual_seed(train_cfg.seed)

    # Create dataloaders

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,    # =per device BS in DDP
        collate_fn=vqa_collator,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
        persistent_workers=False,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        collate_fn=vqa_collator,
        num_workers=1,
        pin_memory=True,
        persistent_workers=False,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    # Warmup dataloaders to kickstart worker processes
    print("Warming up dataloaders...")   
    iter_train_loader = iter(train_loader)
    iter_val_loader = iter(val_loader)
    # The warmup batches are consumed but never trained on (upstream behavior); DataProgress counts the train one as
    # consumed. A resumed run skips both: its train stream already starts right after the last batch the saved run
    # consumed, and its first eval should see the same val batches as every eval after the first in a straight run.
    train_loader.warmup_stream_state = []
    if data_state is None:
        train_loader.warmup_stream_state = next(iter_train_loader).get("stream_state", [])
        next(iter_val_loader)
    print("Warmup complete.")

    return train_loader, val_loader, iter_train_loader, iter_val_loader

# Cosine learning rate schedule with warmup (from Karpathy)
# https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py#L353
def get_lr(it, max_lr, max_steps):
    min_lr = max_lr * 0.1
    warmup_steps = max_steps * 0.03
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

TRAINING_STATE = "training_state.pt"
DATA_TRACE = os.environ.get("NANOVLM_DATA_TRACE") == "1"  # print a hash of every train micro-batch, to check data order across runs/resumes

def resolve_resume_dir(path):
    """A run dir resumes its `latest` checkpoint; a step dir resumes itself."""
    latest = os.path.join(path, "latest")
    if os.path.exists(latest):
        with open(latest) as f:
            path = os.path.join(path, f.read().strip())
    if not os.path.exists(os.path.join(path, TRAINING_STATE)):
        raise FileNotFoundError(f"No {TRAINING_STATE} in {path} (was the run saved with --save_training_state?)")
    return path

def save_training_state(step_dir, state, keep_dirs=()):
    """Write training_state.pt next to the weights already saved in step_dir, point `latest` at it, and prune.

    Both files are written to a temp name and renamed, so `latest` only ever names a complete checkpoint. Older step
    dirs of the run are deleted, except those in keep_dirs (the best checkpoint keeps its weights, not its state).
    """
    tmp = os.path.join(step_dir, TRAINING_STATE + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, os.path.join(step_dir, TRAINING_STATE))
    run_dir = os.path.dirname(step_dir)
    with open(os.path.join(run_dir, "latest.tmp"), "w") as f:
        f.write(os.path.basename(step_dir))
    os.replace(os.path.join(run_dir, "latest.tmp"), os.path.join(run_dir, "latest"))
    keep = {os.path.abspath(d) for d in keep_dirs if d}
    for name in os.listdir(run_dir):
        d = os.path.join(run_dir, name)
        if not name.startswith("step_") or not os.path.isdir(d) or os.path.abspath(d) == os.path.abspath(step_dir):
            continue
        if os.path.abspath(d) in keep:
            if os.path.exists(os.path.join(d, TRAINING_STATE)):
                os.remove(os.path.join(d, TRAINING_STATE))
        else:
            shutil.rmtree(d)

@torch.no_grad()
def eval_under_masks(model, val_loader, device, max_rows=None):
    """Token-weighted val loss of the uncompiled model, run eagerly, under the per-document and the unmasked mask.

    Every arm is scored the same way whatever it trained with: the same val rows (a fresh val iterator per mask), the
    'full' loss path, and SDPA attention ('dense_block_diagonal' computes what 'flex_document_causal' does). The
    compiled graph is never called, so this adds no recompiles. max_rows=None scores the whole val set.
    """
    m = model.module if is_dist() else model
    m = getattr(m, "_orig_mod", m)
    loss_impl = m.cfg.lm_loss_impl
    m.cfg.lm_loss_impl = "full"
    out = {}
    try:
        for name, impl in (("doc_masked", "dense_block_diagonal"), ("unmasked", "none")):
            loss_sum, n_tokens, rows = 0.0, 0, 0
            with packing_impl_override(m.decoder, impl):
                for batch in val_loader:
                    labels = batch["labels"].to(device)
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type in ['cuda', 'cpu'] else torch.float16):
                        _, loss = m(batch["input_ids"].to(device), batch["images"], attention_mask=batch["attention_mask"].to(device),
                                    targets=labels, doc_id=batch["doc_id"].to(device))
                    n = (labels != -100).sum().item()
                    loss_sum += loss.item() * n
                    n_tokens += n
                    rows += labels.size(0)
                    if max_rows is not None and rows >= max_rows:
                        break
            out[name] = loss_sum / max(n_tokens, 1)
    finally:
        m.cfg.lm_loss_impl = loss_impl
    return out

def train(train_cfg, vlm_cfg):
    resume_dir, resume_state = None, None
    if train_cfg.resume_from:
        resume_dir = resolve_resume_dir(train_cfg.resume_from)
        resume_state = torch.load(os.path.join(resume_dir, TRAINING_STATE), map_location="cpu", weights_only=False)
        print(f"Resuming from {resume_dir} ({resume_state['global_step']} optimizer steps done)")
    train_loader, val_loader, iter_train_loader, iter_val_loader = get_dataloaders(train_cfg, vlm_cfg, resume_state["data"] if resume_state else None)

    if is_dist():
        print("Rank", get_rank(), "Waiting for all workers to get dataloaders...")
        if is_master():
            print("Waiting for all workers to get dataloaders...")
        dist.barrier(device_ids=int(os.environ["LOCAL_RANK"]))
        if is_master():
            print("All workers have gotten dataloaders.")

    run_name = resume_state["run_name"] if resume_state else get_run_name(train_cfg, vlm_cfg)
    if train_cfg.log_wandb and is_master():
        wandb_kwargs = {}
        if resume_state and resume_state.get("wandb_run_id"):
            # Continue the same run. wandb drops (with a warning) the replayed steps between the checkpoint and where the
            # run died, keeping the values first logged there; the resume is exact, so they are the same steps anyway.
            # (Rewinding the run instead needs wandb's private-preview rewind feature.)
            wandb_kwargs.update(id=resume_state["wandb_run_id"], resume="must")
        run = wandb.init(
            entity=train_cfg.wandb_entity,
            project=train_cfg.wandb_project,
            group=train_cfg.wandb_group,
            tags=list(train_cfg.wandb_tags),
            config={
                "VLMConfig": asdict(vlm_cfg),
                "TrainConfig": asdict(train_cfg),
                "provenance": get_provenance(),
            },
            name=run_name,
            **wandb_kwargs,
        )
        # Define a custom x-axis for lmms-eval metrics
        lmms_eval_step = "<lmms-eval-step>"
        run.define_metric(name="lmms_eval/*", step_metric=lmms_eval_step)

    # Initialize model
    if resume_state is not None:
        model = VisionLanguageModel.from_pretrained(resume_dir)
    elif train_cfg.resume_from_vlm_checkpoint:
        print(f"Resuming from VLM checkpoint: {vlm_cfg.vlm_checkpoint_path}")
        model = VisionLanguageModel.from_pretrained(vlm_cfg.vlm_checkpoint_path)
    else:
        model = VisionLanguageModel(vlm_cfg, load_backbone=vlm_cfg.vlm_load_backbone_weights)
    
    if is_master():
        print(f"nanoVLM initialized with {sum(p.numel() for p in model.parameters()):,} parameters") 
        print(f"Training summary{' (global)' if is_dist() else ''}: {-1*get_world_size()} samples, batch size {int(train_cfg.batch_size*get_world_size()*train_cfg.gradient_accumulation_steps)}{', training on ' + str(get_world_size()) + ' GPUs' if is_dist() else ''}")
        if is_dist():
            print(f"Training summary per GPU: batch size {train_loader.batch_size}")
        print(f"Validation summary{' (global)' if is_dist() else ''}: {-1*get_world_size()} samples, batch size {int(train_cfg.batch_size*get_world_size()*train_cfg.gradient_accumulation_steps)}{', training on ' + str(get_world_size()) + ' GPUs' if is_dist() else ''}")
        if is_dist():
            print(f"Validation summary per GPU: batch size {val_loader.batch_size}")

    # Define optimizer groups
    # Since we have pretrained vision and language backbones, but a newly initialized modality projection layer, it doesn't make sense to train them with the same learning rate
    # You could opt to fully freeze the backbones and only train the MP layer, but finetuning them with a lower learning rate makes the training as a whole easier
    param_groups = []
    if train_cfg.lr_mp > 0:
        param_groups.append({'params': list(model.MP.parameters()), 'lr': train_cfg.lr_mp})
    else:
        for p in list(model.MP.parameters()):
            p.requires_grad = False
    if train_cfg.lr_vision_backbone > 0:
        param_groups.append({'params': list(model.vision_encoder.parameters()), 'lr': train_cfg.lr_vision_backbone})
    else:
        for p in list(model.vision_encoder.parameters()):
            p.requires_grad = False
    if train_cfg.lr_language_backbone > 0:
        param_groups.append({'params': list(model.decoder.parameters()), 'lr': train_cfg.lr_language_backbone})
    else:
        for p in list(model.decoder.parameters()):
            p.requires_grad = False

    optimizer = optim.AdamW(param_groups)
    all_params = [p for group in optimizer.param_groups for p in group['params']]

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    if device.type == "mps":
        torch.backends.mps.enable_fallback_to_cpu = True
        torch.mps.empty_cache()
    
    print(f"Using device: {device}")
    model.to(device)
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"])  # after .to(device): AdamW state follows its params' device on load
    
    if train_cfg.compile:
        # VisionLanguageModel.forward's loss branch gathers hidden states with a boolean mask
        # (targets != -100), which has a data-dependent output shape (aten.nonzero) -- without
        # this, Dynamo graph-breaks and recompiles on it every time the kept-token count changes
        # shape bucket. With it, that gather traces as part of one dynamic-shape graph instead.
        # Separately, and unaffected by this flag: the chunked linear_cross_entropy call in the
        # same branch always falls back to eager under torch.compile (documented upstream
        # behavior) -- expected, not a regression to chase in a --compile profiler trace.
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        model = torch.compile(model)
    if is_dist():
        print("Wrapping model for DDP")
        model = wrap_model(model)
        print("Model wrapped for DDP")

    epoch_times = []
    best_val_loss = float('inf')
    best_model_path = None
    logged_eval_steps = set()
    global_step = 0
    epoch = 0
    data_progress = DataProgress(max(train_cfg.num_workers, 1), resume_state["data"] if resume_state else None)
    data_progress.update(train_loader.warmup_stream_state)
    if resume_state is not None:
        global_step = resume_state["global_step"]
        epoch = resume_state["epoch"] - 1  # the epoch loop increments it first
        best_val_loss, best_model_path = resume_state["best_val_loss"], resume_state["best_model_path"]
        rng = resume_state["rng"]
        torch.set_rng_state(rng["torch"])
        if rng["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        random.setstate(rng["python"])
        numpy.random.set_state(rng["numpy"])

    def training_state(optimizer_steps_done):
        return {
            "global_step": optimizer_steps_done,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "best_model_path": best_model_path,
            "optimizer": optimizer.state_dict(),
            "data": data_progress.state_dict(),
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "python": random.getstate(),
                "numpy": numpy.random.get_state(),
            },
            "run_name": run_name,
            "wandb_run_id": run.id if train_cfg.log_wandb else None,
            "train_cfg": asdict(train_cfg),
            "vlm_cfg": asdict(vlm_cfg),
        }
    overall_peak_mem_allocated_gib = 0.0
    logged_tokens_per_second = []
    stats_history = []  # one entry per stats_log_interval, for callers that want warmup-trimmed stats
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    
    # Training stats accumulators
    accumulated_stats = {
        'tokens_per_second': [],
        'data_load_time': [],
        'fw_bw_time': [],
        'post_process_time': [],
        'images_per_sample': [],
    }
    
    while global_step < train_cfg.max_training_steps:
        epoch += 1
        epoch_start_time = time.time()
        model.train()
        total_train_loss = 0
        total_tokens_processed = 0
        optimizer.zero_grad()
        data_load_start = time.time()

        print("Starting training loop")
        for i, batch in enumerate(synchronized_dataloader_step(iter_train_loader, is_dist())):
            is_update_step = (i + 1) % train_cfg.gradient_accumulation_steps == 0
            batch_start_time = time.time()
            images = batch["images"]
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            doc_id = batch["doc_id"].to(device)
            data_progress.update(batch.get("stream_state", []))
            if DATA_TRACE:
                print(f"data_trace step {global_step} micro {i} {hashlib.md5(batch['input_ids'].numpy().tobytes()).hexdigest()[:12]}")
            data_load_time = time.time() - data_load_start

            # When using DDP with gradient accumulation,
            # skip gradient synchronization on intermediate steps to save time.
            # Gradients only need to be synced at the end of each accumulation cycle.
            if (is_dist()
                and train_cfg.gradient_accumulation_steps > 1
                and not is_update_step):
                context = model.no_sync()
            else:
                context = contextlib.nullcontext()

            fw_bw_start = time.time()
            autocast_context = torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16 if device.type in ['cuda', 'cpu'] else torch.float16
            )
            with autocast_context:
                with context:
                    _, loss = model(input_ids, images, attention_mask=attention_mask, targets=labels, doc_id=doc_id)

            if train_cfg.gradient_accumulation_steps > 1:
                loss = loss / train_cfg.gradient_accumulation_steps

            loss.backward()

            fw_bw_time = time.time() - fw_bw_start
            post_process_start = time.time()
            if is_update_step:
                if train_cfg.max_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=train_cfg.max_grad_norm)

                param_group_idx = 0
                if train_cfg.lr_mp > 0:
                    adj_lr_mp = get_lr(global_step, train_cfg.lr_mp, train_cfg.max_training_steps)
                    optimizer.param_groups[param_group_idx]['lr'] = adj_lr_mp
                    param_group_idx += 1

                if train_cfg.lr_vision_backbone > 0:
                    adj_lr_vision_backbone = get_lr(global_step, train_cfg.lr_vision_backbone, train_cfg.max_training_steps)
                    optimizer.param_groups[param_group_idx]['lr'] = adj_lr_vision_backbone
                    param_group_idx += 1

                if train_cfg.lr_language_backbone > 0:
                    adj_lr_language_backbone = get_lr(global_step, train_cfg.lr_language_backbone, train_cfg.max_training_steps)
                    optimizer.param_groups[param_group_idx]['lr'] = adj_lr_language_backbone
              
                optimizer.step()
                optimizer.zero_grad()

            batch_loss = loss.item()
            if train_cfg.gradient_accumulation_steps > 1:
                batch_loss = batch_loss * train_cfg.gradient_accumulation_steps
            total_train_loss += batch_loss

            num_tokens = torch.sum(attention_mask).item() # Sum of attention mask gives number of tokens
            total_tokens_processed += num_tokens
            post_process_time = time.time() - post_process_start

            images_per_sample = [len(image_pack) for image_pack in images]

            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time
            tokens_per_second = get_world_size() * num_tokens / batch_duration  # Multiply by world size to get global tokens/s

            # Accumulate training stats
            accumulated_stats['tokens_per_second'].append(tokens_per_second)
            accumulated_stats['data_load_time'].append(data_load_time)
            accumulated_stats['fw_bw_time'].append(fw_bw_time)
            accumulated_stats['post_process_time'].append(post_process_time)
            accumulated_stats['images_per_sample'].extend(images_per_sample)
            
            if train_cfg.eval_in_epochs and global_step % train_cfg.eval_interval == 0 and is_update_step:
                print("Starting evaluation")
                model.eval()
                if device == "cuda":
                    torch.cuda.empty_cache()
                with torch.no_grad():
                    total_val_loss = 0
                    val_batches = 0
                    for batch in synchronized_dataloader_step(iter_val_loader, is_dist()):
                        if val_batches > 64:
                            print(f"Evaluated {val_batches} batches")
                            break
                        images = batch["images"]
                        input_ids = batch["input_ids"].to(device)
                        labels = batch["labels"].to(device)
                        attention_mask = batch["attention_mask"].to(device)
                        doc_id = batch["doc_id"].to(device)

                        with autocast_context:
                            _, loss = model(input_ids, images, attention_mask=attention_mask, targets=labels, doc_id=doc_id)

                        total_val_loss += loss.item()
                        val_batches += 1
                    
                    iter_val_loader = iter(val_loader)
                    avg_val_loss = total_val_loss / val_batches if val_batches > 0 else 0
                    avg_val_loss = mean(dist_gather(avg_val_loss)) if is_dist() else avg_val_loss

                    mask_losses = {}
                    if train_cfg.eval_mask_rows > 0:
                        mask_losses = eval_under_masks(model, val_loader, device, train_cfg.eval_mask_rows)
                        if is_dist():
                            mask_losses = {k: mean(dist_gather(v)) for k, v in mask_losses.items()}

                    checkpoint_path_step = ""
                    if is_master():
                        # Save a checkpoint for this evaluation step
                        checkpoint_path_step = os.path.join(vlm_cfg.vlm_checkpoint_path, run_name, f"step_{global_step}")
                        save_model = model.module if is_dist() else model # unwrap the model for saving if DDP
                        save_model.save_pretrained(save_directory=checkpoint_path_step)

                        if train_cfg.use_lmms_eval and global_step % (train_cfg.eval_interval*2) == 0:
                            # Submit evaluation job
                            cmd = f"sbatch eval.slurm {checkpoint_path_step} {global_step} {run_name} {train_cfg.lmms_eval_limit} {train_cfg.lmms_eval_tasks} {train_cfg.lmms_eval_batch_size}"
                            print(f"Submitting evaluation job: {cmd}")
                            subprocess.run(cmd, shell=True)

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        if is_master():
                            best_model_path = checkpoint_path_step
                    
                    if train_cfg.save_training_state and is_master():
                        # The eval runs before this step's global_step += 1, so global_step + 1 optimizer steps are done
                        save_training_state(checkpoint_path_step, training_state(global_step + 1), keep_dirs=[best_model_path])

                    if is_master():
                        mask_msg = "".join(f", {k} val loss: {v:.4f}" for k, v in mask_losses.items())
                        print(f"Step: {global_step}, Val Loss: {avg_val_loss:.4f}{mask_msg}, Tokens/s: {tokens_per_second:.2f}")
                        if train_cfg.log_wandb:
                            run.log({"val_loss": avg_val_loss, **{f"val/{k}_loss": v for k, v in mask_losses.items()}}, step=global_step)

                model.train()

            # Log training stats every N steps (ALL RANKS must participate in collective ops)
            if global_step % train_cfg.stats_log_interval == 0 and len(accumulated_stats['tokens_per_second']) > 0 and is_update_step:
                # ALL RANKS: Perform collective operations for training stats
                stats = {}
                for key in ['tokens_per_second', 'data_load_time', 'fw_bw_time', 'post_process_time', 'images_per_sample']:
                    if is_dist():
                        all_values = dist_gather(accumulated_stats[key])
                        all_values_flat = [item for sublist in all_values for item in sublist]  # Flatten list of lists
                        stats[f'avg_{key}'] = mean(all_values_flat)
                    else:
                        stats[f'avg_{key}'] = mean(accumulated_stats[key])
                
                for key in ['data_load_time', 'fw_bw_time', 'post_process_time', 'images_per_sample']:
                    if is_dist():
                        all_values = dist_gather(accumulated_stats[key])
                        all_values_flat = [item for sublist in all_values for item in sublist]
                        stats[f'max_{key}'] = max(all_values_flat)
                    else:
                        stats[f'max_{key}'] = max(accumulated_stats[key])

                if is_dist():
                    all_images_values = dist_gather(accumulated_stats['images_per_sample'])
                    all_images_flat = [item for sublist in all_images_values for item in sublist]
                    stats['min_images_per_sample'] = min(all_images_flat)
                else:
                    stats['min_images_per_sample'] = min(accumulated_stats['images_per_sample'])

                # Peak memory since the last stats log (this rank only)
                if device.type == "cuda":
                    stats['peak_mem_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30
                    stats['peak_mem_reserved_gib'] = torch.cuda.max_memory_reserved() / 2**30
                    overall_peak_mem_allocated_gib = max(overall_peak_mem_allocated_gib, stats['peak_mem_allocated_gib'])
                    torch.cuda.reset_peak_memory_stats()
                logged_tokens_per_second.append(stats['avg_tokens_per_second'])
                stats_history.append({**stats, 'batch_loss': batch_loss, 'global_step': global_step})

                if is_master():
                    print(f"Step: {global_step}, Loss: {batch_loss:.4f}, Tokens/s: {stats['avg_tokens_per_second']:.0f}, "
                          f"fw_bw: {stats['avg_fw_bw_time']:.3f}s, data: {stats['avg_data_load_time']:.3f}s, "
                          f"peak alloc: {stats.get('peak_mem_allocated_gib', 0.0):.2f} GiB, peak reserved: {stats.get('peak_mem_reserved_gib', 0.0):.2f} GiB")

                # MASTER ONLY: Log to wandb
                if train_cfg.log_wandb and is_master():
                    run.log({
                        **{f"training_stats/{key}": value for key, value in stats.items()},
                    }, step=global_step)

                    # Check for and log new lmms-eval results
                    eval_results_dir = os.path.join('eval_results', run_name)
                    if os.path.exists(eval_results_dir):
                        logged_results_count = 0
                        for result_file in os.listdir(eval_results_dir):
                            # Match only files like "step_1234.json" (no extra text)
                            match = re.fullmatch(r"step_(\d+)\.json", result_file)
                            if not match:
                                continue  # skip if the filename has extra text like taskname

                            try:
                                step = int(match.group(1))
                                if step not in logged_eval_steps:
                                    with open(os.path.join(eval_results_dir, result_file), 'r') as f:
                                        eval_data = json.load(f)

                                    lmms_results = eval_data.get('results', {})
                                    if lmms_results:
                                        metrics = {f"lmms_eval/{key}": value for key, value in lmms_results.items()}
                                        metrics[lmms_eval_step] = eval_data['global_step']
                                        if logged_results_count > 0:
                                            print(f"Logging more than one lmms-eval result for step {global_step}, try to avoid this.")
                                        run.log(metrics, step=global_step + logged_results_count)
                                        logged_results_count += 1
                                        print(f"Logged lmms-eval results from step {eval_data['global_step']}")

                                    logged_eval_steps.add(step)
                            except (ValueError, KeyError, json.JSONDecodeError) as e:
                                print(f"Warning: Could not process eval result file {result_file}. Error: {e}")
                                continue
                
                # ALL RANKS: Reset accumulators
                for key in accumulated_stats:
                    accumulated_stats[key] = []

            # Log batch loss  
            if is_update_step:
                # ALL RANKS: gather loss from all ranks if DDP
                if is_dist():
                    batch_loss_gathered = dist_mean_scalar(batch_loss)
                else:
                    batch_loss_gathered = batch_loss
                    
                # MASTER ONLY: Log to wandb
                if train_cfg.log_wandb and is_master():
                    run.log({
                        "batch_loss": batch_loss_gathered,
                        **({"grad_norm": grad_norm} if train_cfg.max_grad_norm is not None else {})
                    }, step=global_step)
                
            if is_update_step:
                global_step += 1
                if global_step >= train_cfg.max_training_steps:
                    break
            data_load_start = time.time()

        if resume_state is not None and global_step < train_cfg.max_training_steps:
            # A new epoch would re-apply the resume position to a fresh pass over the data
            raise NotImplementedError("The train stream ran out in a resumed run; restarting an epoch after a resume is not supported")
        iter_train_loader = iter(train_loader)
        avg_train_loss = total_train_loss / i
        # gather average batch loss from all ranks if DDP
        avg_train_loss = mean(dist_gather(avg_train_loss)) if is_dist() else avg_train_loss  

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)

        # gather and sum total_tokens_processed across all ranks if DDP
        total_tokens_processed = sum(dist_gather(total_tokens_processed)) if is_dist() else total_tokens_processed  
        epoch_tokens_per_second = total_tokens_processed / epoch_duration

        if is_master():
            if train_cfg.log_wandb:
                run.log({"epoch_loss": avg_train_loss,
                         "epoch_duration": epoch_duration,
                         "epoch_tokens_per_second": epoch_tokens_per_second})

            print(f"Epoch: {epoch}, Step: {global_step}/{train_cfg.max_training_steps}, Train Loss: {avg_train_loss:.4f} | Time: {epoch_duration:.2f}s | T/s: {epoch_tokens_per_second:.2f}")

    if train_cfg.save_training_state and is_master():
        # Final checkpoint (the last in-loop eval is at most eval_interval steps earlier) and a full val pass under both masks
        model.eval()
        final_losses = eval_under_masks(model, val_loader, device, max_rows=None)
        print("Final full val pass: " + ", ".join(f"{k} val loss: {v:.4f}" for k, v in final_losses.items()))
        if train_cfg.log_wandb:
            run.log({f"val_final/{k}_loss": v for k, v in final_losses.items()}, step=global_step)
            for k, v in final_losses.items():
                run.summary[f"val_final/{k}_loss"] = v
        final_dir = os.path.join(vlm_cfg.vlm_checkpoint_path, run_name, f"step_{global_step}")
        (model.module if is_dist() else model).save_pretrained(save_directory=final_dir)
        save_training_state(final_dir, training_state(global_step), keep_dirs=[best_model_path])

    # Summary Statistics
    if is_master():
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        total_training_time = sum(epoch_times)
        batch_size = int(train_cfg.batch_size*get_world_size()*train_cfg.gradient_accumulation_steps)
        total_samples_processed = batch_size * global_step
        avg_time_per_sample = total_training_time / total_samples_processed
        print(f"Average time per epoch: {avg_epoch_time:.2f}s")
        print(f"Average time per sample: {avg_time_per_sample:.4f}s")

        # Push the best model to the hub (Please set your user name in the config!)
        if vlm_cfg.hf_repo_name is not None and best_model_path:
            print(f"Training complete. Pushing best model from {best_model_path} to Hugging Face Hub...")
            hf_model = VisionLanguageModel.from_pretrained(best_model_path)
            hf_model.push_to_hub(vlm_cfg.hf_repo_name)

        if train_cfg.log_wandb:
            run.summary["avg_epoch_time"] = avg_epoch_time
            run.summary["avg_time_per_sample"] = avg_time_per_sample
            run.summary["overall_peak_mem_allocated_gib"] = overall_peak_mem_allocated_gib
            run.summary["best_val_loss"] = best_val_loss
            if logged_tokens_per_second:
                run.summary["mean_tokens_per_second"] = mean(logged_tokens_per_second)
            run.finish()

        return {
            "run_name": run_name,
            "global_step": global_step,
            "avg_epoch_time": avg_epoch_time,
            "total_training_time": total_training_time,
            "avg_time_per_sample": avg_time_per_sample,
            "overall_peak_mem_allocated_gib": overall_peak_mem_allocated_gib,
            "best_val_loss": best_val_loss,
            "mean_tokens_per_second": mean(logged_tokens_per_second) if logged_tokens_per_second else None,
            "stats_history": stats_history,
        }

def main():
    global PG_CPU
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr_mp', type=float, help='Learning rate for the mapping network')
    parser.add_argument('--lr_vision_backbone', type=float, help='Learning rate for the vision backbone')
    parser.add_argument('--lr_language_backbone', type=float, help='Learning rate for the language backbone')
    parser.add_argument('--vlm_checkpoint_path', type=str, help='Path to the VLM checkpoint for loading or saving')
    parser.add_argument('--compile', action='store_true', default=None, help='Use torch.compile to optimize the model')
    parser.add_argument('--log_wandb', type=bool, help='Log to wandb')
    parser.add_argument('--resume_from_vlm_checkpoint', type=bool, default=False, help='Resume training from VLM checkpoint specified by vlm_checkpoint_path (or default if not provided)')
    parser.add_argument('--no_log_wandb', action='store_true', help='Do not log to wandb')
    parser.add_argument('--train_dataset_path', type=str, help='Train dataset path')
    parser.add_argument('--dataset_cache_dir', type=str, help='Download whole dataset shards on demand into this local cache dir and read them from disk')
    parser.add_argument('--max_cache_gb', type=float, help='LRU size cap of the local shard cache in GiB')
    parser.add_argument('--prefetch_shards', type=int, help='Upcoming shards per DataLoader worker to download in the background')
    parser.add_argument('--num_workers', type=int, help='Train DataLoader workers')
    parser.add_argument('--relevance_min_rating', type=int, help='Minimum relevance rating of images per sample')
    parser.add_argument('--image_correspondence_min_rating', type=int, help='Minimum image correspondence rating of images per sample')
    parser.add_argument('--visual_dependency_min_rating', type=int, help='Minimum visual dependency rating of images per sample')
    parser.add_argument('--formatting_min_rating', type=int, help='Minimum formatting rating of images per sample')
    parser.add_argument('--lm_model_type', type=str, help='Language backbone, e.g. HuggingFaceTB/SmolLM2-135M for the ~230M VLM')
    parser.add_argument('--loss_impl', type=str, choices=['full', 'gather', 'chunked'], help='Training loss path (see VLMConfig.lm_loss_impl)')
    parser.add_argument('--attn_packing_impl', type=str, choices=['none', 'dense_block_diagonal', 'flex_document_causal'], help='Cross-sample attention masking for packed training rows (see VLMConfig.lm_attn_packing_impl)')
    parser.add_argument('--attn_flex_block_size', type=int, help='flex_attention create_block_mask BLOCK_SIZE (see VLMConfig.lm_attn_flex_block_size)')
    parser.add_argument('--batch_size', type=int, help='Micro-batch size per GPU')
    parser.add_argument('--gradient_accumulation_steps', type=int, help='Micro-batches per optimizer step')
    parser.add_argument('--max_training_steps', type=int, help='Optimizer steps (also sets the LR schedule length)')
    parser.add_argument('--eval_interval', type=int, help='Validate and checkpoint every N optimizer steps')
    parser.add_argument('--no_eval', action='store_true', help='Skip in-loop validation and checkpointing')
    parser.add_argument('--val_size', type=int, help='Number of validation samples')
    parser.add_argument('--stats_log_interval', type=int, help='Log training stats every N optimizer steps')
    parser.add_argument('--wandb_entity', type=str, help='wandb entity (user or team)')
    parser.add_argument('--wandb_project', type=str, help='wandb project')
    parser.add_argument('--wandb_group', type=str, help='wandb group, one per experiment')
    parser.add_argument('--wandb_tags', type=str, help='Comma-separated wandb tags')
    parser.add_argument('--run_name_suffix', type=str, help='Appended to the generated run name')
    parser.add_argument('--no_lmms_eval', action='store_true', help='Do not submit lmms-eval jobs (they need Slurm)')
    parser.add_argument('--no_hub_push', action='store_true', help='Do not push the best model to the Hugging Face Hub')
    parser.add_argument('--seed', type=int, help='Seed for model init and data packing order (default 0 reproduces earlier runs)')
    parser.add_argument('--save_training_state', action='store_true', help='At each eval also save optimizer/step/RNG/data position so the run can be resumed exactly (keeps only the latest)')
    parser.add_argument('--resume_from', type=str, help='Resume exactly from a run dir (its latest checkpoint) or a step dir saved with --save_training_state')
    parser.add_argument('--eval_mask_rows', type=int, help='At each eval, also score the uncompiled model under the per-document and the unmasked mask on this many val rows')

    args = parser.parse_args()

    vlm_cfg = config.VLMConfig()
    train_cfg = config.TrainConfig()

    if args.lr_mp is not None:
        train_cfg.lr_mp = args.lr_mp
    if args.lr_vision_backbone is not None:
        train_cfg.lr_vision_backbone = args.lr_vision_backbone
    if args.lr_language_backbone is not None:
        train_cfg.lr_language_backbone = args.lr_language_backbone
    if args.vlm_checkpoint_path is not None:
        vlm_cfg.vlm_checkpoint_path = args.vlm_checkpoint_path
    if args.compile is not None:
        train_cfg.compile = args.compile
    if args.no_log_wandb is True:
        train_cfg.log_wandb = False
    if args.train_dataset_path is not None:
        train_cfg.train_dataset_path = args.train_dataset_path
    if args.dataset_cache_dir is not None:
        train_cfg.dataset_cache_dir = args.dataset_cache_dir
    if args.max_cache_gb is not None:
        train_cfg.max_cache_gb = args.max_cache_gb
    if args.prefetch_shards is not None:
        train_cfg.prefetch_shards = args.prefetch_shards
    if args.num_workers is not None:
        train_cfg.num_workers = args.num_workers
    if args.relevance_min_rating is not None:
        train_cfg.relevance_min_rating = args.relevance_min_rating
    if args.image_correspondence_min_rating is not None:
        train_cfg.image_correspondence_min_rating = args.image_correspondence_min_rating
    if args.visual_dependency_min_rating is not None:
        train_cfg.visual_dependency_min_rating = args.visual_dependency_min_rating
    if args.formatting_min_rating is not None:
        train_cfg.formatting_min_rating = args.formatting_min_rating
    if args.lm_model_type is not None:
        vlm_cfg.lm_model_type = args.lm_model_type
    if args.loss_impl is not None:
        vlm_cfg.lm_loss_impl = args.loss_impl
    if args.attn_packing_impl is not None:
        vlm_cfg.lm_attn_packing_impl = args.attn_packing_impl
    if args.attn_flex_block_size is not None:
        vlm_cfg.lm_attn_flex_block_size = args.attn_flex_block_size
    if args.batch_size is not None:
        train_cfg.batch_size = args.batch_size
    if args.gradient_accumulation_steps is not None:
        train_cfg.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.max_training_steps is not None:
        train_cfg.max_training_steps = args.max_training_steps
    if args.eval_interval is not None:
        train_cfg.eval_interval = args.eval_interval
    if args.no_eval:
        train_cfg.eval_in_epochs = False
    if args.seed is not None:
        train_cfg.seed = args.seed
    if args.save_training_state:
        train_cfg.save_training_state = True
    if args.resume_from is not None:
        train_cfg.resume_from = args.resume_from
    if args.eval_mask_rows is not None:
        train_cfg.eval_mask_rows = args.eval_mask_rows
    if args.val_size is not None:
        train_cfg.val_size = args.val_size
    if args.stats_log_interval is not None:
        train_cfg.stats_log_interval = args.stats_log_interval
    if args.wandb_entity is not None:
        train_cfg.wandb_entity = args.wandb_entity
    if args.wandb_project is not None:
        train_cfg.wandb_project = args.wandb_project
    if args.wandb_group is not None:
        train_cfg.wandb_group = args.wandb_group
    if args.wandb_tags is not None:
        train_cfg.wandb_tags = tuple(t for t in args.wandb_tags.split(',') if t)
    if args.run_name_suffix is not None:
        train_cfg.run_name_suffix = args.run_name_suffix
    if args.no_lmms_eval:
        train_cfg.use_lmms_eval = False
    if args.no_hub_push:
        vlm_cfg.hf_repo_name = None

    if args.resume_from_vlm_checkpoint and args.vlm_checkpoint_path is not None:
        train_cfg.resume_from_vlm_checkpoint = True
        # When resuming a full VLM, we don't need to load individual backbone weights from original sources
        vlm_cfg.vlm_load_backbone_weights = False

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        init_dist()
        PG_CPU = dist.new_group(backend="gloo")   # host‑RAM, zero GPU allocations

    if is_dist() and (train_cfg.save_training_state or train_cfg.resume_from):
        raise NotImplementedError("--save_training_state / --resume_from track one rank's data position; single-GPU only")

    if train_cfg.seed != 0:
        # Seed 0 keeps the import-time seeding above untouched, so runs from before this flag reproduce exactly
        torch.manual_seed(train_cfg.seed)
        torch.cuda.manual_seed_all(train_cfg.seed)
        random.seed(42 + train_cfg.seed)  # main-process knapsack shuffles (num_workers=0); workers are seeded via the DataLoader generator

    if is_master():
        print("--- VLM Config ---")
        print(vlm_cfg)
        print("--- Train Config ---")
        print(train_cfg)

    train(train_cfg, vlm_cfg)

    if is_dist():
        destroy_dist()

if __name__ == "__main__":
    main()
