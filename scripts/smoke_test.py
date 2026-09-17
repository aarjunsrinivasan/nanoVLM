import argparse
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train
import models.config as config

parser = argparse.ArgumentParser(description="Small real-training sanity check (few steps, small val), per repo convention.")
parser.add_argument('--attn_packing_impl', type=str, default='none', choices=['none', 'dense_block_diagonal', 'flex_document_causal'],
                     help='See VLMConfig.lm_attn_packing_impl. Run once per value to sanity-check all three before trusting them.')
parser.add_argument('--compile', action='store_true', help='Wrap the model with torch.compile(), like a real --compile training run.')
args = parser.parse_args()

# checkpoints/smoketest_<impl>[_compiled] -- distinct per combination so back-to-back runs
# (e.g. one per lm_attn_packing_impl value) don't clobber each other's checkpoint dir.
checkpoint_suffix = args.attn_packing_impl + ("_compiled" if args.compile else "")

vlm_cfg = config.VLMConfig(
    hf_repo_name=None,                      # never push to HF Hub during a smoke test
    vlm_checkpoint_path=f"checkpoints/smoketest_{checkpoint_suffix}",
    lm_attn_packing_impl=args.attn_packing_impl,
)
train_cfg = config.TrainConfig(
    max_training_steps=6,       # "few iters"
    stats_log_interval=2,       # see training_stats/* fire at least twice
    val_size=16,                # keep the val skip()/take() on the streamed dataset fast
    log_wandb=False,            # no wandb login configured on this machine
    use_lmms_eval=False,        # avoid submitting a real `sbatch eval.slurm` job
    num_workers=2,              # keep CPU usage modest on the shared node
    dataset_cache_dir=os.environ.get("NANOVLM_CACHE_DIR"),  # set to read through the local shard cache
    compile=args.compile,
    run_name_suffix=checkpoint_suffix,
)

if train.is_master():
    print("--- VLM Config ---"); print(vlm_cfg)
    print("--- Train Config ---"); print(train_cfg)

train.train(train_cfg, vlm_cfg)
