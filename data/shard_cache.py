"""Local shard cache for streaming a parquet dataset repo from the Hugging Face Hub.

Instead of streaming parquet over HTTP range requests, whole shards are downloaded on
demand into `cache_dir` (each DataLoader worker also fetches its next shard(s) in the
background), read from local disk, and evicted least-recently-used once the cache
exceeds `max_cache_gb`. Enabled via `TrainConfig.dataset_cache_dir`.

Concurrency model (DataLoader workers x DDP ranks, no coordinator):
- Downloads are serialized per shard with a FileLock + re-check, since
  `hf_hub_download(local_dir=...)` alone can download the same file twice.
- A shard only appears at its final path via atomic rename, so an existing file
  with the expected size is complete.
- Eviction only removes shards whose mtime is older than a grace period. Each process
  heartbeats the shards it holds (current + lookahead), val shards are pinned, and on
  Linux a shard that is already open survives unlink anyway.
"""
import os
import json
import math
import time
import threading
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pyarrow.parquet as pq
from filelock import FileLock, Timeout
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.hf_api import RepoFile
from huggingface_hub.utils import disable_progress_bars
from datasets import Features, IterableDataset, Image
from torch.utils.data import get_worker_info

MANIFEST_NAME = "manifest.json"
GiB = 1024 ** 3
_WARN_INTERVAL_S = 300


@dataclass(frozen=True)
class ShardManifest:
    repo_id: str
    revision: str        # commit sha, pinned so every process reads the same files
    files: tuple         # sorted parquet paths within the repo
    sizes: tuple         # bytes, aligned with files
    rows_per_shard: int  # rows in the first shard
    features: dict       # datasets.Features.to_dict()


@dataclass(frozen=True, eq=False)
class CacheContext:
    repo_id: str
    revision: str
    cache_dir: str
    files: tuple           # this rank's shards, in read order
    sizes: dict            # filename -> bytes
    pinned: frozenset      # never evicted (val shards of all ranks)
    prefetch: int          # upcoming shards of the same worker to download in the background
    max_cache_bytes: int   # None disables eviction
    grace_s: float         # never evict shards touched more recently than this


# Per-process state (each DataLoader worker is its own process).
_state_lock = threading.Lock()
_held = {}  # filename -> number of readers in this process holding it
_executor = None
_executor_pid = None
_heartbeat_pid = None
_last_warn = 0.0


def _lock(cache_dir, name, timeout=-1):
    path = os.path.join(cache_dir, ".locks", name + ".lock")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return FileLock(path, timeout=timeout)


def _is_complete(path, expected_size):
    try:
        size = os.path.getsize(path)
    except FileNotFoundError:
        return False
    return expected_size is None or size == expected_size


def _touch(path):
    try:
        os.utime(path)
    except FileNotFoundError:
        pass


def _remove(path):
    try:
        os.remove(path)
        return True
    except FileNotFoundError:
        return False


def _download(repo_id, revision, cache_dir, filename, expected_size):
    """Return (local path, downloaded?) for `filename`, downloading it at most once across processes."""
    path = os.path.join(cache_dir, filename)
    if _is_complete(path, expected_size):
        return path, False
    with _lock(cache_dir, filename):
        if _is_complete(path, expected_size):
            return path, False
        # hf_hub_download trusts an existing file whose etag metadata matches, so drop a bad one first
        _remove(path)
        hf_hub_download(repo_id, filename, repo_type="dataset", revision=revision, local_dir=cache_dir)
    return path, True


def ensure_local(filename, ctx):
    path, downloaded = _download(ctx.repo_id, ctx.revision, ctx.cache_dir, filename, ctx.sizes.get(filename))
    _touch(path)
    if downloaded:
        evict_if_needed(ctx)
    return path


def _hold(names):
    with _state_lock:
        for name in names:
            _held[name] = _held.get(name, 0) + 1


def _unhold(names):
    with _state_lock:
        for name in names:
            _held[name] -= 1
            if _held[name] == 0:
                del _held[name]


def _held_snapshot():
    with _state_lock:
        return set(_held)


def _start_heartbeat(ctx):
    """Keep the mtime of shards held by this process fresh, independent of how fast they are consumed."""
    global _heartbeat_pid
    if _heartbeat_pid == os.getpid():
        return
    _heartbeat_pid = os.getpid()
    interval = max(1.0, min(30.0, ctx.grace_s / 4))

    def beat():
        while True:
            for name in _held_snapshot():
                _touch(os.path.join(ctx.cache_dir, name))
            time.sleep(interval)

    threading.Thread(target=beat, name="shard-cache-heartbeat", daemon=True).start()


def evict_if_needed(ctx):
    """Delete least-recently-used shards until the cache is under 0.9 * cap. Returns bytes freed."""
    global _last_warn
    if ctx.max_cache_bytes is None:
        return 0
    lock = _lock(ctx.cache_dir, "evict", timeout=0)
    try:
        lock.acquire()
    except Timeout:
        return 0  # another process is already evicting
    try:
        now = time.time()
        held = _held_snapshot()
        total, candidates = 0, []
        for dirpath, dirnames, filenames in os.walk(ctx.cache_dir):
            dirnames[:] = [d for d in dirnames if d != ".locks"]
            for name in filenames:
                full = os.path.join(dirpath, name)
                rel = os.path.relpath(full, ctx.cache_dir)
                is_shard = name.endswith(".parquet") and not rel.startswith(".cache")
                if not (is_shard or name.endswith(".incomplete")):  # in-flight downloads count towards the cap
                    continue
                try:
                    st = os.stat(full)
                except FileNotFoundError:
                    continue
                total += st.st_size
                if is_shard and rel not in ctx.pinned and rel not in held and now - st.st_mtime > ctx.grace_s:
                    candidates.append((st.st_mtime, st.st_size, rel, full))

        if total <= ctx.max_cache_bytes:
            return 0
        target = int(0.9 * ctx.max_cache_bytes)
        freed = 0
        for _, size, rel, full in sorted(candidates):
            if total - freed <= target:
                break
            if _remove(full):
                freed += size
                _remove(os.path.join(ctx.cache_dir, ".cache", "huggingface", "download", rel + ".metadata"))
        if total - freed > ctx.max_cache_bytes and now - _last_warn > _WARN_INTERVAL_S:
            _last_warn = now
            print(f"[shard_cache] Warning: cache is {(total - freed) / GiB:.1f} GiB, over max_cache_gb={ctx.max_cache_bytes / GiB:.1f}, "
                  f"but no shard is evictable (in use, pinned, or touched within {ctx.grace_s:.0f}s)")
        return freed
    finally:
        lock.release()


def sweep_stale_incomplete(cache_dir, older_than_s=30 * 60):
    """Remove partial downloads left behind by killed processes (live downloads keep their mtime fresh)."""
    now = time.time()
    for dirpath, _, filenames in os.walk(os.path.join(cache_dir, ".cache", "huggingface", "download")):
        for name in filenames:
            if not name.endswith(".incomplete"):
                continue
            full = os.path.join(dirpath, name)
            try:
                if now - os.path.getmtime(full) > older_than_s:
                    _remove(full)
            except FileNotFoundError:
                pass


def _check_features(features):
    images = features.get("images")
    if not isinstance(getattr(images, "feature", None), Image):
        raise ValueError(f"Expected an 'images' column holding a list of Image features, got {images}")


def load_or_create_manifest(repo_id, cache_dir):
    """Load `cache_dir/manifest.json`, or list the repo once and create it (one process at a time)."""
    path = os.path.join(cache_dir, MANIFEST_NAME)
    with _lock(cache_dir, "manifest"):
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            if data["repo_id"] != repo_id:
                raise ValueError(f"{cache_dir} caches {data['repo_id']}, not {repo_id}; use a different dataset_cache_dir")
            return ShardManifest(**{**data, "files": tuple(data["files"]), "sizes": tuple(data["sizes"])})

        print(f"[shard_cache] Listing parquet shards of {repo_id} (one-time, saved to {path})")
        api = HfApi()
        revision = api.dataset_info(repo_id).sha
        entries = sorted(
            (e for e in api.list_repo_tree(repo_id, repo_type="dataset", revision=revision, recursive=True)
             if isinstance(e, RepoFile) and e.path.endswith(".parquet")),
            key=lambda e: e.path,
        )
        if not entries:
            raise ValueError(f"No parquet files found in dataset repo {repo_id}")

        first, _ = _download(repo_id, revision, cache_dir, entries[0].path, entries[0].size)
        pf = pq.ParquetFile(first)
        features = Features.from_arrow_schema(pf.schema_arrow)
        _check_features(features)
        manifest = ShardManifest(
            repo_id=repo_id,
            revision=revision,
            files=tuple(e.path for e in entries),
            sizes=tuple(e.size for e in entries),
            rows_per_shard=pf.metadata.num_rows,
            features=features.to_dict(),
        )
        pf.close()

        tmp = f"{path}.tmp{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(asdict(manifest), f)
        os.replace(tmp, path)
        return manifest


def _prefetch_executor():
    global _executor, _executor_pid
    if _executor_pid != os.getpid():
        _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="shard-cache-prefetch")
        _executor_pid = os.getpid()
    return _executor


def _prefetch(filename, ctx):
    try:
        ensure_local(filename, ctx)
    except Exception as e:  # the reader retries when it reaches this shard
        print(f"[shard_cache] Warning: background download of {filename} failed: {e}")


def _read_shard(filename, ctx):
    pf = None
    for _ in range(3):
        path = ensure_local(filename, ctx)
        try:
            pf = pq.ParquetFile(path)
            break
        except FileNotFoundError:  # evicted between download and open
            continue
        except (pa.ArrowInvalid, OSError) as e:
            print(f"[shard_cache] Warning: could not open {filename} ({e}), re-downloading")
            _remove(path)
    if pf is None:
        print(f"[shard_cache] Warning: skipping unreadable shard {filename}")
        return
    try:
        for i in range(pf.num_row_groups):
            try:
                rows = pf.read_row_group(i).to_pylist()
            except (pa.ArrowInvalid, OSError) as e:
                print(f"[shard_cache] Warning: skipping rest of {filename} after read error: {e}")
                return
            yield from rows
    finally:
        pf.close()


def _iter_shard_rows(pos, ctx):
    """Yield the rows of `ctx.files[p]` for p in `pos` (datasets calls this with one position at a time)."""
    worker = get_worker_info()
    # DataLoader workers get every num_workers-th shard, so this worker's next shards are p + k * stride
    stride = worker.num_workers if worker is not None else 1
    if worker is not None:
        disable_progress_bars()  # a tqdm bar per worker download is just noise
    _start_heartbeat(ctx)
    for p in pos:
        filename = ctx.files[p]
        lookahead = [ctx.files[p + k * stride] for k in range(1, ctx.prefetch + 1) if p + k * stride < len(ctx.files)]
        held = [filename] + lookahead
        _hold(held)
        try:
            for name in lookahead:
                if not _is_complete(os.path.join(ctx.cache_dir, name), ctx.sizes.get(name)):
                    _prefetch_executor().submit(_prefetch, name, ctx)
            yield from _read_shard(filename, ctx)
        finally:
            _unhold(held)


def _rank_block(files, world_size, rank):
    per, extra = divmod(len(files), world_size)
    start = rank * per + min(rank, extra)
    return files[start:start + per + (1 if rank < extra else 0)]


def get_cached_train_val_datasets(train_cfg, world_size, rank):
    """Train/val IterableDatasets for this rank that read shards through the local cache. Val gets the first shards."""
    cache_dir = os.path.abspath(os.path.expanduser(train_cfg.dataset_cache_dir))
    os.makedirs(cache_dir, exist_ok=True)
    sweep_stale_incomplete(cache_dir)
    manifest = load_or_create_manifest(train_cfg.train_dataset_path, cache_dir)
    features = Features.from_dict(manifest.features)

    n_val = max(math.ceil(train_cfg.val_size / manifest.rows_per_shard), world_size)  # every rank needs >= 1 val shard
    val_files, train_files = manifest.files[:n_val], manifest.files[n_val:]
    max_cache_bytes = int(train_cfg.max_cache_gb * GiB) if train_cfg.max_cache_gb is not None else None

    if rank == 0:
        working_set = world_size * (max(train_cfg.num_workers, 1) * (train_cfg.prefetch_shards + 1) + 1) * max(manifest.sizes)
        print(f"[shard_cache] {len(train_files)} train / {n_val} val shards in {cache_dir}, "
              f"working set ~{working_set / GiB:.1f} GiB, max_cache_gb={train_cfg.max_cache_gb}")
        if max_cache_bytes is not None and working_set > 0.8 * max_cache_bytes:
            print("[shard_cache] Warning: max_cache_gb is close to the working set; expect shards to be evicted and re-downloaded")

    sizes = dict(zip(manifest.files, manifest.sizes))

    def build(files, prefetch):
        ctx = CacheContext(
            repo_id=manifest.repo_id,
            revision=manifest.revision,
            cache_dir=cache_dir,
            files=tuple(files),
            sizes={f: sizes[f] for f in files},
            pinned=frozenset(val_files),
            prefetch=prefetch,
            max_cache_bytes=max_cache_bytes,
            grace_s=train_cfg.cache_evict_grace_min * 60,
        )
        return IterableDataset.from_generator(_iter_shard_rows, features=features, gen_kwargs={"pos": list(range(len(files))), "ctx": ctx})

    train_ds = build(_rank_block(train_files, world_size, rank), train_cfg.prefetch_shards)
    val_ds = build(_rank_block(val_files, world_size, rank), 0).take(int(train_cfg.val_size / world_size))
    return train_ds, val_ds
