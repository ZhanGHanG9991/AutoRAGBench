# Build recipe

Use the exact upstream commit in the method README. Set `AUTORAGBENCH_ROOT` to a writable
benchmark/build root. Place that checkout at `AUTORAGBENCH_ROOT/humansota/AdaptiveRAG`.
Copy `service/code/*.py` and then this `build/` tree into the checkout's `rageng/` directory.
Run the commands below there, using the environment installed from `requirements.txt` and
upstream dependencies. `PYTHONPATH` must include `rageng/` (and upstream modules where needed).
Set `MODEL_DIR` to the model cache containing `Qwen3.5-4B/tokenizer.json`; set `BUILD_TMP`
to the filesystem holding build outputs so the original free-space guards remain meaningful.
The scripts preserve historical directory guards and public split names. Arrange the
matching public corpora/training files under `data_formal-v1/TASK/public/` and, for CWQ /
MultiHop-RAG revisions, `data_formal-v2/TASK/public/`. No hidden split is used.
The source release does not contain those historical data files, trained weights or indexes.

Download `google-t5/t5-large` into the checkout's `models/t5-large` directory.
Set `RAGENG_ROOT=$AUTORAGBENCH_ROOT`, `ARAG_ROOT=$AUTORAGBENCH_ROOT/humansota/AdaptiveRAG`
and point `LLM_BASE_URL` at the benchmark Qwen3.5-4B build endpoint.

1. Run the common `make_split.py` to create `humansota/formal-v1-split/TASK/train_fit.jsonl`.
2. For each task, expose the corresponding corpus at `data/TASK/public/corpus/`, then run
   `python build_index.py --task TASK` without `--dense`. It writes `rageng/index/TASK`.
   For silver-label construction, copy this BM25 index to
   `humansota/shared-indexes/formal-v1/TASK-naive_bm25` under the benchmark root.
3. Run `python label_train.py --task TASK --src formal --workers 4 --out LABELS/TASK_train.jsonl`
   for all seven evaluated tasks. The source is the public-training split only; expected
   row counts are 1,800 per task and 400 for PubMedQA. Labels select the least complex
   successful route, with the original task-family bias when no route succeeds.
4. Run `python train_classifier.py --tasks musique hotpotqa multihop_rag nq_open fever pubmedqa qasper
   --labels-dir LABELS --out ../models/classifier-formal-v1 --skip-dev` as one command.
   The balanced training sampler and A/B/C-logit decoding are preserved. Checkpoint selection
   uses an internal training holdout and the two default learning rates.
5. Rebuild BM25 from the evaluated task corpus when it differs from the training corpus.
   Assemble the source and install that index at `WORKSPACE/index/`, the shared classifier at
   `WORKSPACE/models/classifier/`, and the Python runtime at `WORKSPACE/.venv/`.

The service code contains the same `ragcore`, `guard`, `strategies`, `llm`, `retry_http`
and classifier modules used during training. Copying them into `rageng/` supplies all local imports.
