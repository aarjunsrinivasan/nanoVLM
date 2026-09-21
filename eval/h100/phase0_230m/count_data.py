# Counts raw stream samples consumed and documents per packed row (bs=2, same pipeline as train.py).
# Shard cache dir: $SHARD_CACHE, default ~/.cache/finevision_shards.
import os, pathlib, sys, collections, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))  # repo root, wherever it is cloned
import models.config as config
import data.datasets as dsmod
import train
counter = collections.Counter()
orig = dsmod.VQADataset.iter_for_worker
def counting(self):
    for d in self.dataset:
        counter["raw"] += 1
        yield self._process_data(d)
dsmod.VQADataset.iter_for_worker = counting
vlm_cfg, tc = config.VLMConfig(), config.TrainConfig()
vlm_cfg.lm_model_type = "HuggingFaceTB/SmolLM2-135M-Instruct"
tc.batch_size, tc.num_workers, tc.val_size = 2, 0, 5000
tc.dataset_cache_dir = os.environ.get("SHARD_CACHE", os.path.expanduser("~/.cache/finevision_shards"))
tc.max_cache_gb = 30
train_loader, _, it, _ = train.get_dataloaders(tc, vlm_cfg)
counter["raw"] = 0  # drop val warmup reads; train warmup batch already consumed
rows = docs = 0; snaps = {}
for i, b in enumerate(it, 1):
    for r in b["doc_id"]:
        rows += 1; docs += int((torch.unique(r[r >= 0])).numel())
    if i in (200, 600):
        snaps[i] = (counter["raw"], rows, docs); print(i, snaps[i], flush=True)
    if i == 600: break
(r0, w0, d0), (r1, w1, d1) = snaps[200], snaps[600]
print(f"raw samples per packed row: {(r1-r0)/(w1-w0):.3f}  docs per row: {(d1-d0)/(w1-w0):.3f}")
