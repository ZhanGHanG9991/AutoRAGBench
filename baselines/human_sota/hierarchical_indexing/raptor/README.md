# RAPTOR

Paper: P. Sarthi, S. Abdullah, A. Tuli, S. Khanna, A. Goldie, C. D. Manning. *RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval*. ICLR 2024.
Upstream: https://github.com/parthsarthi03/raptor.git
Commit: `7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767`. License: `UPSTREAM_LICENSE`.

The AutoRAGBench adaptation builds recursive UMAP/GMM summary trees using
`sentence-transformers/multi-qa-mpnet-base-cos-v1` and Qwen3.5-4B summaries.
Retrieval ranks nodes in the collapsed tree, expands their original document IDs,
deduplicates in first-selected order, and returns corpus text. Generated summaries
are index content only. Evidence is prefix-capped at 256 passages / 500,000
title-plus-text characters. No generator call is needed by `/retrieve`.

Evaluated: CWQ, FEVER, HotpotQA, MultiHop-RAG, MuSiQue, PubMedQA, QASPER, WebQSP.
NQ is outside this batch. Shared configuration: 100-token leaves and summaries,
five layers, reduction dimension 10, GMM threshold 0.1, clustering input cap 3500,
retrieval context 2000 tokens, traversal top-k 5, UMAP seed 224 and GMM seed 0.
QASPER builds a tree per paper and uses `meta.doc_id`. Other tasks use a corpus-order
forest with 1024 leaves/tree and a SQLite/FAISS collection; IVF search uses nprobe 64.
KG tasks preserve whole relation lines and append supplied topic entities to queries.
Build adapters handle tiny/singleton clusters and propagate clustering positions.
Earlier equivalent-corpus indexes were reused, with document-ID remapping where
needed. Fresh builds use current corpus IDs directly. Legacy `/answer` is retained.

## Rebuild

Follow [workspace assembly](../../_common/fixed_evidence_adapter/README.md).
Install the pinned upstream and `build/requirements.txt` into Python 3.12 at
`$W/venv`; `build/environment.lock.txt` records the original dependency versions.
No upstream patch is needed. Download `sentence-transformers/multi-qa-mpnet-base-cos-v1`
revision `d51b22a1dfa8184e9258074e56e2875e50612dca` to
`$W/models/multi-qa-mpnet-base-cos-v1`. Populate the `cl100k_base` tiktoken cache
at `$W/tiktoken_cache` while online, then use offline mode for serving.
Provide the public `data_manifest.json` next to the corpus directory (used by the
collection builder), `CORPUS_DIR`, `LLM_BASE_URL` and `SCRATCH_DIR`, then run:

```bash
bash "$M/build/rebuild.sh" "$W" "$TASK" "$W/index"
bash "$W/submission/start.sh"
```

The wrapper uses the archived per-paper builder for QASPER, or the archived
sharded forest builder followed by assembly for other tasks. It defaults to one
shard; larger corpora can use disjoint `--shard-id` values with a common
`--shard-count`, then one `--phase assemble` call. This changes scheduling only.
Models, trees, generated summaries, FAISS files, environments and corpus data are
not included. No learned model is fine-tuned. See `build/layout.json` for source aliases.

## VI-G starting package

QASPER has a distinct native `/retrieve` wrapper. It descends selected nodes to
leaves, resolves each leaf to a contiguous span of the scoped corpus document,
deduplicates, and stops at 12,000 UTF-8 bytes, 32,000 characters or 256 passages.
This differs from the Human-SOTA service, which returns whole source documents.
Use `--starting-package` during assembly. Its archived `runtime_env.sh` expects a
relocatable CPython 3.12 installation in `$W/python` and dependencies in
`$W/venv/lib/python3.12/site-packages`. Its optional `build.sh` retains a Slurm
allocation guard: set `BUILD_PARTITION` to your partition. The portable rebuild
wrapper above does not require Slurm. Starting indexes are not included.
