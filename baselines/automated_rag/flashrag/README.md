# FlashRAG

Paper: J. Jin, Y. Zhu, Z. Dou, G. Dong, X. Yang, C. Zhang, T. Zhao, Z. Yang, J.-R. Wen. *FlashRAG: A Modular Toolkit for Efficient Retrieval-Augmented Generation Research*. Companion Proceedings of the ACM Web Conference (WWW), 2025.
Upstream: https://github.com/RUC-NLPIR/FlashRAG
Commit: `7926ac3c3c5ce0781f53d60ebac48fc511abed12`. MIT license: `UPSTREAM_LICENSE`.

This benchmark adaptation implements a fixed menu corresponding to FlashRAG's sequential, iterative, IRCOT, and fixed single-/multi-step adaptive pipelines in `exp3lib/flashrag_flow.py`. The served adapter does not import upstream pipeline classes. No upstream files were changed or copied into the evaluated service, so no upstream patch is needed. It excludes no-retrieval, compression, Self-RAG, separate generative models, and learned adaptive routing.

BM25 indexes full corpus documents with bm25s. Dense indexes use source-exact 256-token windows with overlap 32, normalized embeddings, batch size 64, and FAISS IndexFlatIP. The embedding choices are e5-large-v2 and bge-large-en-v1.5; reranking uses bge-reranker-v2-m3. Prefixes are applied only during encoding. `/retrieve` returns ordered original corpus excerpts and matching IDs/titles, filtered by `meta.doc_id` when provided. Retrieval generation uses only Qwen3.5-4B at `LLM_BASE_URL`, with thinking disabled.

The ten registered candidates are in `service/submission/flashrag/candidates.json`: BM25/e5/BGE with and without reranking, iterative BM25 (three retrieval rounds, two generation calls), IRCOT BM25 (up to four generation calls), and fixed single-/multi-step adaptive BM25. Standard retrieval takes 50 hits; reranking fetches 100 and retains 50. The multi-step adaptive loop retains its second generation call, although that call contributes no additional evidence.

Selection uses 200 public train-tune examples with split seed 20261027 and the benchmark's fixed-evidence Qwen reader/metric. Ties favor fewer registered generation calls, then earlier candidate order. Most tasks stop the entire search at the first retrieval-token overflow, discard that candidate's partial evaluation, and select from completed candidates only. MultiHop-RAG instead skips token/context-overflow candidates and continues in registered order. WebQSP includes a pre-send token check and converts context-length HTTP errors into token-budget errors. See `build/SELECTION_RULE.md`.

| Task | Selected candidate / outcome |
| --- | --- |
| MuSiQue | std_e5_rerank |
| MultiHop-RAG | std_bge_rerank |
| PubMedQA | std_bm25 |
| QASPER | std_bge |
| WebQSP | std_e5_rerank |
| CWQ | std_bge_rerank |
| HotpotQA | Build failed: two dense indexes exceeded six hours; no evaluated service |

HotpotQA's approximately 5.25 million chunks required about three hours for e5; BGE remained incomplete at the deadline. Its build configuration is included in `build/configs/hotpotqa/config.json`. NQ and FEVER are outside this release's evaluated set.
MultiHop-RAG alone uses a 131,072-token generator context and per-question retrieval-token budget, still capped at four calls and 60 seconds. Its training-reader budget remains 32,000 tokens, one call, and 512 output tokens. Other tasks use 32,000 retrieval tokens; the overall build limit is always 21,600 seconds. Configure the benchmark gateway/context accordingly; the task overlay preserves the service-side cap.

Rebuild on Linux with Python 3.12 and one suitable GPU allocation:

```bash
git clone https://github.com/RUC-NLPIR/FlashRAG upstream
git -C upstream checkout 7926ac3c3c5ce0781f53d60ebac48fc511abed12
python build/prepare.py --task qasper --workspace "$WORKSPACE" --upstream upstream --models "$MODEL_DIR" --tune "$TUNE_JSON"
"$WORKSPACE/runtime/env/bin/python" -m pip install -r build/requirements.txt
export AUTORAGBENCH_ROOT=/path/to/AutoRAGBench CORPUS_DIR=/path/to/task/corpus
# Also set WORKSPACE and LLM_BASE_URL; no external optimizer credential is needed.
timeout --signal=TERM --kill-after=60 21600 bash build/run.sh qasper
PORT=8000 bash "$WORKSPACE/submission/start.sh"
```

The adaptation uses the recorded runtime dependency versions rather than upstream's older torch/bm25s requirement pins. `TUNE_JSON` supplies the original 200-example public train-tune split in custom-exam format (`id`, `question`, `canonical_answer`, `answer_variants`, `supporting_doc_ids`, and optional `meta`). No split membership or question data is bundled; exact historical reproduction requires that split. `AUTORAGBENCH_ROOT` supplies the benchmark scoring/packing modules.

Download `intfloat/e5-large-v2`, `BAAI/bge-large-en-v1.5`, and `BAAI/bge-reranker-v2-m3` into `MODEL_DIR/<basename>`, plus the `Qwen/Qwen3.5-4B` tokenizer under `Qwen3.5-4B`. `EXP3_QWEN_TOKENIZER` overrides its build-time path. There is no auxiliary training. Weights, indexes, environments, and search outputs are omitted; the original scripts did not pin model-weight revisions.

`prepare.py` stages the original builders and task-specific search modules under `workspace/prep/`, copies supplied models to `workspace/runtime/models/`, and creates `runtime/env/`. The package writer assembles `workspace/submission/` after selection. Evaluation needs that submission, runtime, and `indexes/`. To reuse historical selected candidates with matching rebuilt indexes, overlay `service/submission/` and `service/configs/<task>/submission/` into `workspace/submission/`. Retained `EXP3_*` and `ARB_DEADLINE` identifiers preserve compatibility.
