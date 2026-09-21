import copy
import json
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import Features, Value
from torch.utils.data import DataLoader

from data.advanced_datasets import ConstantLengthDataset
from data.collators import VQACollator
from data.data_progress import DataProgress
from data.shard_cache import get_cached_train_val_datasets
from train import seed_worker

ROWS_PER_SHARD, N_SHARDS, SEQ_LEN, BATCH_SIZE = 40, 7, 64, 2


class _Tokenizer:
    pad_token_id = 0


class _FakeVQA:
    """Stands in for VQADataset: turns a raw row into a sample whose tokens all equal the row id (so packed rows
    identify exactly which raw rows they contain), with deterministic lengths and some rows filtered out."""

    tokenizer = _Tokenizer()
    mp_image_token_length = 0

    def __init__(self, dataset):
        self.dataset = dataset

    def iter_for_worker(self):
        for row in self.dataset:
            rid = row["id"]
            if rid % 7 == 3:  # like a ratings-filtered sample
                yield None
                continue
            n = 5 + (rid * 37) % 36
            ids = torch.full((n,), rid + 1, dtype=torch.long)
            yield {"input_ids": ids, "attention_mask": torch.ones(n, dtype=torch.long), "labels": ids.clone(), "images": []}


class TestResumeDataPosition(unittest.TestCase):
    """A stream resumed from DataProgress.state_dict() must yield exactly the batches the uninterrupted stream yields
    after the same point, through the real shard reader (row-group and cross-shard skipping), the packing producer
    (buffer repacking with the restored RNG, dropping consumed groups) and the DataLoader's worker rotation."""

    def setUp(self):
        self.cache = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.cache, "full"))
        files, sizes = [], []
        for k in range(N_SHARDS):
            name = f"full/data-{k:05d}.parquet"
            ids = list(range(k * ROWS_PER_SHARD, (k + 1) * ROWS_PER_SHARD))
            pq.write_table(pa.table({"id": pa.array(ids, pa.int64())}), os.path.join(self.cache, name), row_group_size=8)
            files.append(name)
            sizes.append(os.path.getsize(os.path.join(self.cache, name)))
        manifest = {"repo_id": "local/test", "revision": "0", "files": files, "sizes": sizes,
                    "rows_per_shard": ROWS_PER_SHARD, "features": Features({"id": Value("int64")}).to_dict()}
        with open(os.path.join(self.cache, "manifest.json"), "w") as f:
            json.dump(manifest, f)

    def tearDown(self):
        shutil.rmtree(self.cache)

    def _loader(self, num_workers, data_state=None):
        cfg = SimpleNamespace(dataset_cache_dir=self.cache, train_dataset_path="local/test", val_size=ROWS_PER_SHARD,
                              max_cache_gb=None, num_workers=num_workers, prefetch_shards=1, cache_evict_grace_min=2.0)
        train_ds, _ = get_cached_train_val_datasets(cfg, world_size=1, rank=0, data_state=data_state)
        cld = ConstantLengthDataset(_FakeVQA(train_ds), seq_length=SEQ_LEN, max_sample_length=SEQ_LEN,
                                    num_of_sequences=BATCH_SIZE * 4, queue_size=2, resume_state=data_state)
        g = torch.Generator()
        g.manual_seed(0)
        return DataLoader(cld, batch_size=BATCH_SIZE, collate_fn=VQACollator(_Tokenizer(), SEQ_LEN), num_workers=num_workers,
                          drop_last=True, worker_init_fn=seed_worker, generator=g)

    def _check(self, num_workers, cut, total=24):
        straight, progress, snapshot = [], DataProgress(max(num_workers, 1)), None
        for i, batch in enumerate(self._loader(num_workers)):
            progress.update(batch["stream_state"])
            straight.append(batch["input_ids"])
            if i + 1 == cut:
                snapshot = copy.deepcopy(progress.state_dict())
            if len(straight) == total:
                break
        self.assertEqual(len(straight), total, "test data too small for this many batches")

        resumed = []
        for batch in self._loader(num_workers, data_state=snapshot):
            resumed.append(batch["input_ids"])
            if len(resumed) == total - cut:
                break
        self.assertEqual(len(resumed), total - cut)
        for k, (a, b) in enumerate(zip(straight[cut:], resumed)):
            self.assertTrue(torch.equal(a, b), f"num_workers={num_workers} cut={cut}: batch {cut + k} differs after resume")

    def test_resume_matches_straight_run(self):
        for num_workers in (1, 2, 3):
            for cut in (1, 4, 7, 11):
                with self.subTest(num_workers=num_workers, cut=cut):
                    self._check(num_workers, cut)

    def test_stream_actually_crosses_shards(self):
        # Guard that the scenario above exercises cross-shard skipping: 24 batches must reach past each stream's first shard
        seen = set()
        for i, batch in enumerate(self._loader(2)):
            seen.update(int(t) - 1 for t in batch["input_ids"].unique() if t != 0)
            if i == 23:
                break
        shards_touched = {rid // ROWS_PER_SHARD for rid in seen}
        self.assertGreaterEqual(len(shards_touched), 4)


if __name__ == "__main__":
    unittest.main()
