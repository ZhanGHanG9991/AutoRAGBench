# Adaptive-RAG

Paper: S. Jeong, J. Baek, S. Cho, S. J. Hwang, J. C. Park. *Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity*. NAACL 2024.
Upstream: https://github.com/starsuzi/Adaptive-RAG
Commit: `0c88670af8707667eb5c1163151bb5ce61b14acb`.
Upstream license: [UPSTREAM_LICENSE](UPSTREAM_LICENSE). No upstream patch is required.

The AutoRAGBench adaptation trains a T5-large A/B/C query-complexity classifier.
It routes queries to no retrieval, one BM25 retrieval, or iterative BM25 retrieval with
Qwen3.5-4B at `LLM_BASE_URL`. The classifier emits labels only at test time.
`POST /retrieve` returns documents encountered along the chosen route, deduplicated in
first-appearance order; generated reasoning is excluded. Route A returns no evidence.
Evaluated tasks: nq_open, hotpotqa, musique, multihop_rag, pubmedqa, qasper and fever.

All tasks use BM25 top 5; iterative retrieval allows up to 3 reasoning steps and 8 paragraphs,
with a soft 24,000-token guard and hard limits of 4 generator calls, 32,000 tokens and 60 seconds.
QASPER respects the supplied document restriction. Per-task launchers set task identity and
answer format through `tasks/<task>/info.json`; all tasks share the classifier.
Training uses balanced silver labels from public training examples: 1,800 per task,
except PubMedQA's 400. An internal training holdout selects the checkpoint.

## Rebuild and serve

1. Clone upstream at the commit above and install `build/requirements.txt` alongside its
   dependencies in Python 3.11. Download `google-t5/t5-large` to the clone's `models/t5-large`.
2. Follow [build/README.md](build/README.md): recreate the public-training split,
   build the task BM25 index with `build_index.py`, generate route labels using
   `label_train.py --task TASK --src formal`, and train the shared classifier using
   `train_classifier.py --tasks ... --labels-dir LABELS --out MODEL --skip-dev`.
   The training script searches learning rates 3e-5 and 1e-4 for up to eight epochs.
3. Use `../../_common/formal_v2_packaging/assemble.py adaptive_rag TASK WORKSPACE`.
   Put the task's frozen BM25 files in `WORKSPACE/index/`, the trained classifier in
   `WORKSPACE/models/classifier/`, and a Python environment in `WORKSPACE/.venv/`.
4. Set `PORT`, `CORPUS_DIR`, `SCRATCH_DIR`, `LLM_BASE_URL`; run `bash WORKSPACE/submission/start.sh`.
   The service supports `/health` and `/retrieve`; the final answer is evaluator-owned.

No upstream source, weights, indexes, silver labels, runtime libraries or data are included.
See the shared packaging README for portable build-root variables and historical data layout.
`ARAG_*`, `RAGENG_ROOT` and `rageng` identifiers are retained for compatibility.
