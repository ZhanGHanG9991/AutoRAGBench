"""exp1109-night hotfix: exact (flat) vector search kept resident on the GPU.

Implementation-only change. The frozen serve-index has no vector index (only the FTS index), so the
original LanceDBStore.search_vector is an exact brute-force scan over every row with LanceDB's default
squared-L2 distance, re-reading ~21.5 GB per query. This class loads the same vector column once into
GPU memory and computes the same exact squared-L2 distance against every row, then returns the same
top_k rows in ascending distance with the same columns. No approximation, no change of top_k, metric,
fusion, reranker or query expansion. BM25 (LanceDB FTS) and everything else are inherited unchanged.

"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pyarrow as pa
import torch

from agentic_autorag.engine.vector_store import LanceDBStore

_META_COLUMNS = ["id", "text", "doc_id", "chunk_index", "char_start", "char_end"]


class ExactGPUVectorStore(LanceDBStore):
    def __init__(self, db_path, table_name="documents", device="cuda", chunk_rows=262144):
        super().__init__(db_path)
        self.open_table(table_name)
        assert not torch.backends.cuda.matmul.allow_tf32  # no TF32 anywhere in the distance path (elementwise anyway)
        t0 = time.time()
        table = self.table
        n = int(table.count_rows())
        dim = table.schema.field("vector").type.list_size
        self.device = torch.device(device)
        self.n, self.dim, self.chunk_rows = n, dim, int(chunk_rows)
        self.vectors = torch.empty((n, dim), dtype=torch.float32, device=self.device)
        metas, pos = [], 0
        reader = table.search().select(_META_COLUMNS + ["vector"]).limit(n).to_batches(131072)
        for batch in reader:
            k = batch.num_rows
            if k == 0:
                continue
            vec = batch.column(batch.schema.get_field_index("vector"))
            flat = vec.flatten().to_numpy(zero_copy_only=False)
            assert flat.dtype == np.float32 and flat.size == k * dim
            self.vectors[pos : pos + k].copy_(torch.from_numpy(flat.reshape(k, dim)), non_blocking=False)
            metas.append(pa.Table.from_batches([batch.select(_META_COLUMNS)]))
            pos += k
        assert pos == n, (pos, n)
        self.meta = pa.concat_tables(metas).combine_chunks()
        self._buf = torch.empty((min(self.chunk_rows, n), dim), dtype=torch.float32, device=self.device)
        self._dist = torch.empty(n, dtype=torch.float32, device=self.device)
        torch.cuda.synchronize(self.device)
        print(f"EXACT_GPU_VECTOR_READY rows={n} dim={dim} seconds={time.time() - t0:.1f}", file=sys.stderr, flush=True)

    @torch.no_grad()
    def _exact_l2(self, q: torch.Tensor) -> torch.Tensor:
        # squared L2 = sum_d (x_d - q_d)^2, fp32, every row (same metric as LanceDB's flat search; no expansion trick).
        for s in range(0, self.n, self.chunk_rows):
            e = min(s + self.chunk_rows, self.n)
            buf = self._buf[: e - s]
            torch.sub(self.vectors[s:e], q, out=buf)
            buf.square_()
            torch.sum(buf, dim=1, out=self._dist[s:e])
        return self._dist

    def search_vector(self, query_embedding, top_k=5, doc_id=None):
        if doc_id is not None:  # document-scoped tasks keep the original filtered LanceDB path (unused for hotpotqa)
            return super().search_vector(query_embedding, top_k=top_k, doc_id=doc_id)
        # Same float64 list -> float32 conversion LanceDB applies to the query vector.
        qv = np.asarray(self._normalize_vector(query_embedding), dtype=np.float32)
        q = torch.from_numpy(qv).to(self.device)
        dist = self._exact_l2(q)
        k = min(int(top_k), self.n)
        vals, idx = torch.topk(dist, k, largest=False, sorted=True)
        idx_np = idx.cpu().numpy()
        rows = self.meta.take(pa.array(idx_np)).to_pylist()
        for row, v in zip(rows, vals.cpu().tolist()):
            row["_distance"] = float(v)
        return rows
