# Clue-RAG

Paper: Y. Su, Y. Fang, Y. Zhou, C. Yang. *Clue-RAG: Towards Accurate and Cost-Efficient Graph-Based RAG via Multi-Partite Graph-Based Index*. ICDE 2026.
Upstream: https://github.com/Feesuu/ClueRAG.git
Pinned commit: `cb44eaab580752f4cab626507f205c4eed0d2c64`.
Evaluated tasks: **MuSiQue, PubMedQA and QASPER** (only these tasks among the nine benchmark tasks).

## Adaptation

Clue-Index extracts knowledge units and entities into a multipartite graph. Local vector
tables replace upstream OceanBase and hosted embedding/reranking services. BGE-M3 supplies
embeddings, BGE-reranker-v2-M3 supplies scores, and spaCy en_core_web_trf performs NER.
The upstream chunker uses 1200 cl100k tokens with 100 overlap; alpha=1.0 and indexing
concurrency=256. All tasks use a 1024-token retrieval context.
The service returns original corpus chunks selected by final Q-Iter reranking, preserving
upstream order and stripping Title/Content transport wrappers. QASPER is document-scoped.
MuSiQue/PubMedQA use the frozen vector-table warmup variant; QASPER retains the earlier
server. MuSiQue also sets PYTHONHASHSEED=20261022 in its launcher.

## Rebuild and serve

Use Linux with CUDA and the [shared build instructions](../../_common/db5_build/README.md).
Clone upstream into `$AUTORAGBENCH_ROOT/humansota/ClueRAG` and check out the pinned commit.
Stage method `clue` and the desired task with `stage.py`; install `build/setup_env.sh`
from the staged upstream `rageng/` directory. Run shared `build.sh` with public corpus,
train/dev paths and the fixed generator gateway, then `assemble.py` into a new workspace.
`service/layout.json` maps each task to shared files and its exact task overlays.
The assembler reconstructs `submission/`, auxiliary modules, models and index layout.
Set `PORT`, `CORPUS_DIR`, `SCRATCH_DIR`, and `LLM_BASE_URL`, then run
`bash "$WORKSPACE/submission/start.sh"`. `GET /health` checks readiness;
`POST /retrieve` returns `{"passages": [{"doc_id": ..., "title": ..., "text": ...}]}`.
All generated test-time text uses the Qwen3.5-4B `generator` at `LLM_BASE_URL`.

## Assets and dependencies

Omitted assets: BGE model weights, spaCy model, NLTK punkt_tab, tiktoken cache,
`index/store`, `index/graph_cache.json`, `index/chunks.json`, and `index/index_stats.json`.
Use public Hugging Face BAAI model snapshots; install the public spaCy 3.8.0 model wheel
and NLTK punkt_tab archive as described in `build/setup_env.sh`. Rebuild indexes with
`index_task.py`. Populate `submission/cluerag_upstream` from the pinned upstream clone.
The upstream repository has no LICENSE file at the pinned commit.

Models, corpus data and indexes are not included. The original model weights are not all
revision-pinned in the available build records; newly downloaded snapshots may differ.
No full GPU rebuild or end-to-end inference was performed for this source-only release.
