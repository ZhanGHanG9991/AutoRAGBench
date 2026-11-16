"""Write embeddings in batches to a float32 memmap.

The search algorithm is unchanged. Callers that would otherwise hold every
vector as a Python list can read the memmap instead. This module does not
open LanceDB and does not build an index.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def batched_ranges(n_items: int, batch_size: int):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    start = 0
    while start < n_items:
        end = min(start + batch_size, n_items)
        yield start, end
        start = end


def iter_lance_record_batches(chunks, embeddings, chunk_doc_ids, char_ranges, batch_size):
    """Yield Lance rows in batches. Stored text is the chunk, with no title prepended.

    This helper does not open LanceDB. The index builder calls it only when
    EXP3_LANCE_BATCH is set, so a large corpus is not held as one Python list.
    """
    n_items = len(chunks)
    if len(embeddings) != n_items or len(chunk_doc_ids) != n_items or len(char_ranges) != n_items:
        raise ValueError("record columns differ in length")
    for start, end in batched_ranges(n_items, batch_size):
        batch = []
        for i in range(start, end):
            vector = embeddings[i]
            values = vector.tolist() if hasattr(vector, "tolist") else list(vector)
            batch.append(
                {
                    "id": f"chunk_{i}",
                    "text": chunks[i],
                    "vector": values,
                    "doc_id": chunk_doc_ids[i],
                    "chunk_index": i,
                    "char_start": int(char_ranges[i][0]),
                    "char_end": int(char_ranges[i][1]),
                }
            )
        yield batch


def encode_to_memmap(embedder, texts, path, batch_size=64):
    """embedder.encode(list[str]) -> 2d array. One batch is in memory at a time."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not texts:
        array = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(0, 0))
        array.flush()
        return array
    first_end = min(batch_size, len(texts))
    first = np.asarray(embedder.encode(list(texts[:first_end])), dtype=np.float32)
    if first.ndim != 2:
        raise ValueError("encoder must return a 2d array")
    array = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(texts), first.shape[1]))
    array[:first_end] = first
    for start, end in batched_ranges(len(texts), batch_size):
        if start == 0:
            continue
        batch = np.asarray(embedder.encode(list(texts[start:end])), dtype=np.float32)
        if batch.shape[1] != array.shape[1]:
            raise ValueError("encoder width changed between batches")
        array[start:end] = batch
    array.flush()
    return array
