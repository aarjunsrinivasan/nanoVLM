import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train
import models.config as config

vlm_cfg = config.VLMConfig(
    hf_repo_name=None,                      # never push to HF Hub during a smoke test
    vlm_checkpoint_path="checkpoints/smoketest",
)
train_cfg = config.TrainConfig(
    max_training_steps=6,       # "few iters"
    stats_log_interval=2,       # see training_stats/* fire at least twice
    val_size=16,                # keep the val skip()/take() on the streamed dataset fast
    log_wandb=False,            # no wandb login configured on this machine
    use_lmms_eval=False,        # avoid submitting a real `sbatch eval.slurm` job
    num_workers=2,              # keep CPU usage modest on the shared node
    dataset_cache_dir=os.environ.get("NANOVLM_CACHE_DIR"),  # set to read through the local shard cache
)

if train.is_master():
    print("--- VLM Config ---"); print(vlm_cfg)
    print("--- Train Config ---"); print(train_cfg)

train.train(train_cfg, vlm_cfg)
