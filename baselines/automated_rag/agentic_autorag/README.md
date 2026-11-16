# Agentic AutoRAG

Paper: L. B. Strand, R. Jakob, K. O'Sullivan, M. Kreft. *Agentic AutoRAG: RAG Pipeline Optimization through Reasoning-Driven Agents*. arXiv:2610.08452.
Upstream: https://github.com/Agentic-Systems-Lab/Agentic-AutoRAG
Commit: `e1c5590e79dc68c86bba849519a7cc4e762f135a`. MIT license: `UPSTREAM_LICENSE`.

The adaptation reads the task JSONL corpus directly, builds LanceDB vector/hybrid indexes, preserves document IDs and character spans, and evaluates proposals with the benchmark's fixed-evidence training reader/metric. It moves the winning trial's actual index into `serve-index/` without re-embedding. `/retrieve` resolves selected spans to original corpus text and titles; `meta.doc_id` restricts evidence to the requested document.

GPT-6.1 Sol (`openai/gpt-6.1-sol`) proposes configurations; DeepSeek V4.1 Flash (`deepseek/deepseek-flash`) remains the examiner/judge. Credentials come only from `OPENAI_API_KEY` and `DEEPSEEK_API_KEY`. `OPENAI_BASE_URL` defaults to `https://api.openai.com/v1`; `DEEPSEEK_BASE_URL` defaults to `https://api.deepseek.com`. Retrieval and training-reader generation use Qwen3.5-4B at `LLM_BASE_URL`, with thinking disabled. These provider model identifiers must be available on the configured accounts.

Search uses seed 20261027, 200 training-tune examples, concurrency 1, at most 200 trials, optimizer output cap 16,384, and a 19,800-second inner optimization cap within the 21,600-second build budget. Timeout exports the best completed trial. The large menu fixes recursive 256-token chunks with 32-token overlap and two embedding choices; the loose menu searches recursive/fixed chunks of 128/256/384/512 tokens and overlaps 0/32/64. Compression, reordering, and duplicate detection are disabled. PubMedQA actually used the loose menu without MedCPT; the unused MedCPT menu is omitted.

Evaluated tasks and frozen runtime settings (all use temperature 0):

| Task | Embedding | Retrieval / expansion | top-k → rerank top-n |
| --- | --- | --- | --- |
| HotpotQA | bge-large-en-v1.5 | hybrid RRF / multi-query | 50 → 15 |
| MuSiQue | bge-large-en-v1.5 | hybrid alpha=0.2 / HyDE | 50 → 50 |
| MultiHop-RAG | bge-m3 | hybrid RRF / HyDE | 40 → 12 |
| PubMedQA | e5-large-v2 | vector / none | 50 → 3 |
| QASPER | bge-large-en-v1.5 | vector / none | 50 → 18 |
| WebQSP | bge-large-en-v1.5 | hybrid RRF / multi-query | 50 → 30 |
| CWQ | all-MiniLM-L6-v2 | hybrid RRF / multi-query | 50 → 50 |

Reranking uses bge-reranker-v2-m3 except PubMedQA, which uses ms-marco-MiniLM-L-6-v2. NQ and FEVER are outside this release's evaluated set. HotpotQA's task overlay replaces LanceDB flat vector scanning with exact GPU vector search; its index and selected configuration are unchanged. Complete runtime parameters are in `service/configs/<task>/winning-trial.json`.

Rebuild on Linux with Python 3.12 and one suitable GPU allocation:

```bash
git clone https://github.com/Agentic-Systems-Lab/Agentic-AutoRAG upstream
git -C upstream checkout e1c5590e79dc68c86bba849519a7cc4e762f135a
git -C upstream apply "$PWD/patches/upstream.diff"
python build/prepare.py --task qasper --workspace "$WORKSPACE" --upstream upstream --models "$MODEL_DIR" --tune "$TUNE_JSON"
"$WORKSPACE/runtime/env/bin/python" -m pip install -r build/requirements.txt ./upstream
export AUTORAGBENCH_ROOT=/path/to/AutoRAGBench CORPUS_DIR=/path/to/task/corpus
# Also set WORKSPACE, LLM_BASE_URL, OPENAI_API_KEY and DEEPSEEK_API_KEY.
timeout --signal=TERM --kill-after=60 21600 bash build/run.sh qasper
PORT=8000 bash "$WORKSPACE/submission/start.sh"
```

`TUNE_JSON` must contain the original 200-example public train-tune split (seed 20261027), in upstream custom-exam JSON format: `id`, `question`, `canonical_answer`, `answer_variants`, `supporting_doc_ids`, and `meta` where present. No question data or split membership is included here; exact historical reproduction requires that split. Do not construct it from test data. `AUTORAGBENCH_ROOT` supplies `tasks` and `multireader` modules.

Download weights into `MODEL_DIR/<basename>` from `sentence-transformers/all-MiniLM-L6-v2`, `BAAI/bge-large-en-v1.5`, `BAAI/bge-m3`, `intfloat/e5-large-v2`, `BAAI/bge-reranker-v2-m3`, and `cross-encoder/ms-marco-MiniLM-L-6-v2`; also provide the `Qwen/Qwen3.5-4B` tokenizer under `Qwen3.5-4B`. `EXP3_QWEN_TOKENIZER` overrides its build-time path. No auxiliary model is trained. Weights, environments, indexes, and generated training-search artifacts are omitted. Weight revisions were not pinned by the original scripts.

`prepare.py` assembles common service files plus the selected task overlay, stages the original drivers, and copies supplied models and patched upstream code into `workspace/runtime/`. The original package writer creates `workspace/submission/`; evaluation then needs only this submission, `runtime/`, `serve-index/`, and `winning-trial.json`. To use the historical selected configuration with separately rebuilt matching assets, overlay `service/submission/` and `service/configs/<task>/submission/` into `workspace/submission/`, and copy the task's `winning-trial.json` to the workspace root. Retained `EXP3_*`, `G61_*`, and `ARB_DEADLINE` identifiers are compatibility settings.
