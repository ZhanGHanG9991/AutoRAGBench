# CARROT

Paper: Z. Wang, H. Yuan, W. Dong, G. Cong, F. Li. *CARROT: A Learned Cost-Constrained Retrieval Optimization System for RAG*. ICDE 2026.
Upstream: https://github.com/wang0702/CARROT
Pinned commit: `476d333462a7afe6651c908bdf58c8bdc50eb0fc`.
Evaluated tasks: **PubMedQA and QASPER** (only these tasks among the nine benchmark tasks).

## Adaptation

A BGE-M3 index retrieves the top 60 candidates. A trained Siamese predictor selects one of
six rerankers and MCTS parameters; native MCTS orders chunks under a 1024-token budget.
The service returns original corpus chunks in the final MCTS state order. Title-prefixed
model input is kept separate from evidence text. QASPER restricts candidates to `meta.doc_id`.
Both tasks use 100 public training questions, seed 20261022 and 60 training epochs.
Rerankers: Jina v1 turbo English, Jina v2 multilingual, BGE v2-M3, BGE large,
BGE base, and GTE multilingual base. Predicted C is clipped to [2,4], iterations to
[10,30], and lambda to [0,1]. The adapter enforces the upstream token-budget constraint.
The GTE package uses local custom code and resolved files for offline startup.

## Rebuild and serve

Use Linux with CUDA and the [shared build instructions](../../_common/db5_build/README.md).
Clone upstream into `$AUTORAGBENCH_ROOT/humansota/CARROT` and check out the pinned commit.
Stage method `carrot` and the desired task with `stage.py`; install `build/setup_env.sh`
from the staged upstream `rageng/` directory. Run shared `build.sh` with public corpus,
train/dev paths and the fixed generator gateway, then `assemble.py` into a new workspace.
`service/layout.json` maps each task to shared files and its exact task overlays.
The assembler reconstructs `submission/`, auxiliary modules, models and index layout.
Set `PORT`, `CORPUS_DIR`, `SCRATCH_DIR`, and `LLM_BASE_URL`, then run
`bash "$WORKSPACE/submission/start.sh"`. `GET /health` checks readiness;
`POST /retrieve` returns `{"passages": [{"doc_id": ..., "title": ..., "text": ...}]}`.
All generated test-time text uses the Qwen3.5-4B `generator` at `LLM_BASE_URL`.

## Assets and dependencies

Weights, indexes, `predictor.pt`, HF custom-code caches and tokenizer caches are omitted.
The shared model helper downloads the seven public models listed in `profile_native.py`.
GTE custom code is pinned to `40ced75c3017eb27626c9d4ea981bde21a2662f4`;
its `auto_map` is rewritten to local modules, with weights unchanged.
Training creates the predictor from public train examples; `carrot_build_index.py` creates
`index/chunks.json` and `index/embeddings.npy`. Keep `artifact.json` and all six rerankers.
The profile and optimizer modules remain in `service/submission/` because the service
imports their model-loading and budget helpers. `build/profile_native.py` preserves the
original training version; the frozen service import also contains later resume/retry code.

Models, corpus data and indexes are not included. The original model weights are not all
revision-pinned in the available build records; newly downloaded snapshots may differ.
No full GPU rebuild or end-to-end inference was performed for this source-only release.
