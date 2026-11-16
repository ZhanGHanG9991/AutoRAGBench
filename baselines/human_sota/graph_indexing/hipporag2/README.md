# HippoRAG 2

Paper: B. J. Gutiérrez, Y. Shu, W. Qi, S. Zhou, Y. Su. *From RAG to Memory: Non-Parametric Continual Learning for Large Language Models* (HippoRAG 2). ICML 2025.
Upstream: https://github.com/OSU-NLP-Group/HippoRAG.git
Commit: `d5c8329422e0a0b834a15874545cb6a74b4f9b26`. License: `UPSTREAM_LICENSE`.

The AutoRAGBench adapter builds passage/entity/fact graphs using corpus-only
OpenIE and `BAAI/bge-base-en-v1.5` embeddings. Qwen3.5-4B supplies OpenIE at build
time and Recognition Memory via `$LLM_BASE_URL` at retrieval time. Upstream graph
retrieval, Recognition Memory, PPR and dense fallback select five original chunks;
the adapter validates each chunk against the cited corpus document before returning it.
Generated text is never returned as evidence. Legacy `/answer` remains unused by
the fixed-evidence protocol.

Evaluated: CWQ, MultiHop-RAG, MuSiQue, PubMedQA, QASPER, WebQSP.
FEVER, HotpotQA and NQ are outside this batch. Shared parameters: 220-word chunks,
512-token BGE input, normalized CLS embeddings, linking/retrieval top-k 5,
PPR damping 0.5, passage-node weight 0.05, NER/triple caps 1024/2048 tokens.
KG chunks preserve relation lines; supplied topic entities augment KG queries.
QASPER has independent per-paper graphs and requires `meta.doc_id`.
The online deadline is 57 seconds, with no online retry per logical call.
CWQ and PubMedQA share one runtime revision; the other four tasks share another.
`build/layout.json` preserves this distinction. Build-time JSON/schema validation,
bounded retries, Unicode repair and read-only content-addressed cache reuse are retained.
Upstream Recognition Memory examples are omitted; its system instruction is retained.

## Rebuild

Follow [workspace assembly](../../_common/fixed_evidence_adapter/README.md).
Install the pinned upstream and `build/environment.lock.txt` into Python 3.12 at
`$W/venv`. No upstream patch is needed; bundled upstream modules matched the commit.
Download the public `BAAI/bge-base-en-v1.5` checkpoint to
`$W/models/bge-base-en-v1.5` before offline serving. Provide a public corpus,
`LLM_BASE_URL` for Qwen3.5-4B and `SCRATCH_DIR`, then run:

```bash
bash "$M/build/rebuild.sh" "$W" "$TASK" "$W/index"
bash "$W/submission/start.sh"
```

CWQ uses the newer corpus builder; other tasks retain the earlier builder whose
indexes were reused after corpus-equivalence checks. Both are included, and fresh
builds use current corpus IDs. The earlier build runtime allows 16 offline parse
attempts; the newer one allows eight. Online calls remain single-attempt.
Optional `HIPPO_LEGACY_CALL_CACHE` points to compatible build-only OpenIE responses;
it has no default and is not required for a fresh build. Models, graph indexes,
embedding stores, OpenIE responses and environments are omitted. No fine-tuning
is required. `rageng` cache identities are preserved for index compatibility.

## VI-G starting package

MuSiQue uses the same retrieval engine but a separate native `/retrieve` wrapper
and exact corpus-chunk mapping. It deduplicates and prefix-bounds evidence at
12,000 UTF-8 bytes, 32,000 characters and 256 passages; it also checks four calls
and 32,000 tokens explicitly. Assemble with `--starting-package`.
Its `runtime_env.sh` expects relocatable CPython 3.12 in `$W/python` and dependencies
in `$W/venv/lib/python3.12/site-packages`. Its supplied `build.sh` uses the newer
builder with the starting runtime and requires a Slurm allocation plus
`BUILD_PARTITION`. The portable wrapper above rebuilds the Human-SOTA index path.
No starting-package models or indexes are shipped.
