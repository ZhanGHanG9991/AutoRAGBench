# HAMMER

Paper: Y. Zhou, Z. Wang, Y. Fang. *HAMMER: An Automatic RAG Tuning System via Hierarchical Memory-Guided Monte Carlo Tree Search*. Proc. ACM Manag. Data (SIGMOD), 2026.
Upstream: https://github.com/Cyan9061/HAMMER.git
Pinned commit: `679514dc40779ecb4c56d175bef660bd616c62e6`.
Evaluated tasks: **MuSiQue and QASPER** (only these tasks among the nine benchmark tasks).

## Adaptation

Public-train query selection retains 30% using BGE-large-en-v1.5, threshold 0.249,
150 samples per node and top-50 exclusion. Memory-guided MCTS uses 20 iterations,
seed 20261022 and a 120000-second build cap. Local Qwen3.5-4B replaces the original
memory/simulation generator and all symbolic pipeline LLM names.
Evidence consists of original chunks after decomposition, retrieval, fusion and reranking,
in processed-node order with duplicate removal. QASPER requires `meta.doc_id`.
MuSiQue: token chunks 1024, overlap .45, sparse top-9, 4 decompositions, distance-score
fusion, InRanker top-21, 2 additional nodes. QASPER: sentence chunks 1024, overlap .4,
hybrid top-6/BM25 weight .1, 3 decompositions, reciprocal-rank fusion, flashrank top-4,
6 additional nodes. Full selected parameters are in each task's `artifact.json`.
The QASPER service retains the earlier model-path/startup implementation; this release
does not repair that evaluated variant's isolated-startup limitation.

## Rebuild and serve

Use Linux with CUDA and the [shared build instructions](../../_common/db5_build/README.md).
Clone upstream into `$AUTORAGBENCH_ROOT/humansota/HAMMER` and check out the pinned commit.
Stage method `hammer` and the desired task with `stage.py`; install `build/setup_env.sh`
from the staged upstream `rageng/` directory. Run shared `build.sh` with public corpus,
train/dev paths and the fixed generator gateway, then `assemble.py` into a new workspace.
`service/layout.json` maps each task to shared files and its exact task overlays.
The assembler reconstructs `submission/`, auxiliary modules, models and index layout.
Set `PORT`, `CORPUS_DIR`, `SCRATCH_DIR`, and `LLM_BASE_URL`, then run
`bash "$WORKSPACE/submission/start.sh"`. `GET /health` checks readiness;
`POST /retrieve` returns `{"passages": [{"doc_id": ..., "title": ..., "text": ...}]}`.
All generated test-time text uses the Qwen3.5-4B `generator` at `LLM_BASE_URL`.

## Assets and dependencies

Omitted assets: search caches/indexes, converted public `data/corpus.json` and
`data/qa.jsonl`, embedding/reranker weights and HF/tokenizer caches. Rebuild them with
`optimize.py`; package `search/best`, the converted public data and selected artifact.
`download_models.sh` populates the nine-model search-space cache and the Qwen2.5-7B
splitter tokenizer (tokenizer only). The shared helper obtains BGE-M3 and query-selection
BGE-large. The service uses the fixed gateway for generated text; symbolic Qwen2 names
in the selected configuration do not select an external generator.
The upstream repository has no LICENSE file at the pinned commit.

Models, corpus data and indexes are not included. The original model weights are not all
revision-pinned in the available build records; newly downloaded snapshots may differ.
No full GPU rebuild or end-to-end inference was performed for this source-only release.
