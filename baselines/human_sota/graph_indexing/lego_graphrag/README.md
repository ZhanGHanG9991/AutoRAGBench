# LEGO-GraphRAG

Paper: Y. Cao, Z. Gao, Z. Li, X. Xie, S. K. Zhou, J. Xu. *LEGO-GraphRAG: Modularizing Graph-based Retrieval-Augmented Generation for Design Space Exploration*. Proc. VLDB Endow., 2024.
Upstream: https://github.com/gzy02/LEGO-GraphRAG.git
Pinned commit: `52da02d1aed48e6fbb3d40e90797d9bea2dd8c5a`.
Evaluated tasks: **MuSiQue, PubMedQA and QASPER** (only these tasks among the nine benchmark tasks).

## Adaptation

The text adaptation extracts triples with the fixed generator from 200-word corpus
windows, then constructs a graph with all-MiniLM-L6-v2 embeddings and synonym links.
Native LEGO instances select reasoning paths; the service resolves their triples back to
original corpus passages in path/first-occurrence order. It never returns generated paths.
All tasks return up to 8 passages. MuSiQue: SBE-PPR+OSAR-LLMs, paths+passages;
PubMedQA: SBE-PPR+OSAR-LLMs, paths; QASPER: SBE-PPR+SBR-SPR, paths+passages.
MuSiQue/QASPER select configurations on public dev; PubMedQA uses the fixed default.
Retrieval obeys the four-call gateway budget. Synonym matrix blocks use 1024 rows to
bound memory without changing similarity thresholds. The upstream patch normalizes
invalid non-breaking-space indentation in `Instance/utils/Tools.py`.

## Rebuild and serve

Use Linux with CUDA and the [shared build instructions](../../_common/db5_build/README.md).
Clone upstream into `$AUTORAGBENCH_ROOT/humansota/LEGO-GraphRAG` and check out the pinned commit.
Apply `patches/upstream.diff` with `git apply` from the clone before staging.
Stage method `lego` and the desired task with `stage.py`; install `build/setup_env.sh`
from the staged upstream `rageng/` directory. Run shared `build.sh` with public corpus,
train/dev paths and the fixed generator gateway, then `assemble.py` into a new workspace.
`service/layout.json` maps each task to shared files and its exact task overlays.
The assembler reconstructs `submission/`, auxiliary modules, models and index layout.
Set `PORT`, `CORPUS_DIR`, `SCRATCH_DIR`, and `LLM_BASE_URL`, then run
`bash "$WORKSPACE/submission/start.sh"`. `GET /health` checks readiness;
`POST /retrieve` returns `{"passages": [{"doc_id": ..., "title": ..., "text": ...}]}`.
All generated test-time text uses the Qwen3.5-4B `generator` at `LLM_BASE_URL`.

## Assets and dependencies

Omitted assets: OpenIE outputs, graph/index files, all-MiniLM-L6-v2 and
BGE-reranker-v2-M3 model weights, and Python dependencies. Models come from the public
sentence-transformers/BAAI repositories. `openie.py`, `build_graph.py`, `devgrid.py`
and `select_lego.py` rebuild the corpus graph and selected configuration.
The workspace needs `index/{graph.pkl,node_emb.npy,triple_pids.pkl,doc_nodes.pkl,
passages.jsonl,graph_stats.json}`, `models`, and patched upstream `Instance/` copied
as `submission/lego_instance/`.

Models, corpus data and indexes are not included. The original model weights are not all
revision-pinned in the available build records; newly downloaded snapshots may differ.
No full GPU rebuild or end-to-end inference was performed for this source-only release.
