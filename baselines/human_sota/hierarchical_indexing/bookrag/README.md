# BookRAG

Paper: S. Wang, Y. Zhou, Y. Fang. *BookRAG: A Hierarchical Structure-Aware Index-based Approach for Retrieval-Augmented Generation on Complex Documents*. Proc. VLDB Endow., 2026.
Upstream: https://github.com/sam234990/BookRAG.git
Pinned commit: `5ddf9eb4b4806cddf3c2a98e195f72866b1ab694`.
Evaluated tasks: **MuSiQue** (only these tasks among the nine benchmark tasks).

## Adaptation

Corpus documents become a section/leaf tree and an extracted entity graph. Document-local
entity resolution uses BGE-base-en-v1.5 and BGE-reranker-base. The service returns original
leaf/chapter text chosen by section selection, graph/PPR and Skyline, in first-hit order.
It deduplicates exact repeats and skips map/reduce answer generation.
MuSiQue uses up to 3 retrieval subquestions, 24 candidate sections, at most 5 supplementary
sections per step, and at most 256 evidence passages. Query sampling uses seed 20261022.
The build command keeps the campaign's 126000-second cap and 128 extraction workers.
Compatibility patches repair entity-ranking length comparison, shallow-section Skyline
execution, reranker input allowance, and eager imports in the strategy dispatcher.

## Rebuild and serve

Use Linux with CUDA and the [shared build instructions](../../_common/db5_build/README.md).
Clone upstream into `$AUTORAGBENCH_ROOT/humansota/BookRAG` and check out the pinned commit.
Apply `patches/upstream.diff` with `git apply` from the clone before staging.
Stage method `bookrag` and the desired task with `stage.py`; install `build/setup_env.sh`
from the staged upstream `rageng/` directory. Run shared `build.sh` with public corpus,
train/dev paths and the fixed generator gateway, then `assemble.py` into a new workspace.
`service/layout.json` maps each task to shared files and its exact task overlays.
The assembler reconstructs `submission/`, auxiliary modules, models and index layout.
Set `PORT`, `CORPUS_DIR`, `SCRATCH_DIR`, and `LLM_BASE_URL`, then run
`bash "$WORKSPACE/submission/start.sh"`. `GET /health` checks readiness;
`POST /retrieve` returns `{"passages": [{"doc_id": ..., "title": ..., "text": ...}]}`.
All generated test-time text uses the Qwen3.5-4B `generator` at `LLM_BASE_URL`.

## Assets and dependencies

Required assets: `models/bge-base-en-v1.5`, `models/bge-reranker-base`,
`adapter/vendor/Core`, `adapter/tiktoken_cache`, `venv`, and
`index/{bookindex.pkl,entities.npy,index_stats.json}`. Indexes are rebuilt from the corpus;
models are public BAAI snapshots, and the tokenizer cache is generated with tiktoken.
`build/runtime.py` is the index-time adapter; `service/adapter/runtime.py` is the frozen
query-time variant. Keep both. The upstream license scope and third-party notices accompany
the full license texts under `LICENSES/`.

Models, corpus data and indexes are not included. The original model weights are not all
revision-pinned in the available build records; newly downloaded snapshots may differ.
No full GPU rebuild or end-to-end inference was performed for this source-only release.
