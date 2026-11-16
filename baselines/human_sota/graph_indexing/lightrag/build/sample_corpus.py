"""Shared MuSiQue sample used by the timing probe and the shard jobs."""
import json
import random
from pathlib import Path


def load_documents(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            title = (row.get("title") or "").strip()
            text = row.get("text") or ""
            content = f"{title}\n\n{text}" if title else text
            if content.strip():
                rows.append((row["doc_id"], content))
    return rows


def load_sample(path: Path, n: int, seed: int):
    rows = load_documents(path)
    rng = random.Random(seed)
    if n < len(rows):
        rows = rng.sample(rows, n)
    return rows


def take_shard(rows, shard_index: int, num_shards: int):
    if num_shards <= 1:
        return rows
    if shard_index < 0 or shard_index >= num_shards:
        raise SystemExit(f"shard-index {shard_index} out of range for {num_shards}")
    return rows[shard_index::num_shards]
