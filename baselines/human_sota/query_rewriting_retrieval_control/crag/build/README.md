# Build recipe

Use the exact upstream commit in the method README. Set `AUTORAGBENCH_ROOT` to a writable
benchmark/build root. Place that checkout at `AUTORAGBENCH_ROOT/humansota/CorrectiveRAG`.
Copy `service/code/*.py` and then this `build/` tree into the checkout's `rageng/` directory.
Run the commands below there, using the environment installed from `requirements.txt` and
upstream dependencies. `PYTHONPATH` must include `rageng/` (and upstream modules where needed).
Set `MODEL_DIR` to the model cache containing `Qwen3.5-4B/tokenizer.json`; set `BUILD_TMP`
to the filesystem holding build outputs so the original free-space guards remain meaningful.
The scripts preserve historical directory guards and public split names. Arrange the
matching public corpora/training files under `data_formal-v1/TASK/public/` and, for CWQ /
MultiHop-RAG revisions, `data_formal-v2/TASK/public/`. No hidden split is used.
The source release does not contain those historical data files, trained weights or indexes.

Download Contriever to `models/contriever-msmarco` in the checkout. The trainer's shared
T5 initialization is read from `AUTORAGBENCH_ROOT/humansota/AdaptiveRAG/models/t5-large`.
Put upstream `scripts/` on `PYTHONPATH`, or set `CRAG_SCRIPTS` to that directory.

1. Run the common `make_split.py`, then `python formal-v1/formal_split.py`.
   The latter creates evaluator/tuning QID partitions with seed 20261019:
   1,600/200 examples for six tasks and 300/100 for PubMedQA.
2. For each task, run `python formal-v1/formal_index.py prepare --task TASK`.
   Read the resulting `index/<task>-contriever[-w200]/plan.json` to obtain the shard count;
   run `formal_index.py shard --task TASK --shard N` for each zero-based shard, then
   `formal_index.py merge --task TASK`. This creates float16 embeddings and corpus offsets.
3. Run `formal_pairs.py retrieve --task TASK` from `formal-v1/` for musique, hotpotqa,
   nq_open, fever, pubmedqa and multihop_rag; run `formal_pairs.py pool` once.
   Pair filtering, negative sampling and class balancing are implemented in this script.
   QASPER is held out from evaluator training. Auxiliary pair files produced by the
   original pooling function are not additional experiments and are not shipped.
4. From `rageng/`, run `CRAG_FORMAL=1 python train_evaluator_rageng.py --part 1`, then
   `CRAG_FORMAL=1 python train_evaluator_rageng.py --part 2`. Each part is four epochs;
   use `models/evaluator-rageng-formal-v1/ep7` as the delivered evaluator.
   `formal-v1/run_train.sbatch` preserves the original resource/training entrypoint;
   replace its partition placeholder and create its logs directory before use.
5. Run `python formal-v1/formal_fit.py --all`. It uses only the separate threshold-tune
   partition and records `threshold_rule`, branch counts and the selected top-N.
   The minimum branch count is 10 and the paper-threshold retention branch is enabled.
   Use the released per-task configs for the reported thresholds. The selection variant
   retains the same thresholds and changes `decompose_mode` to `selection`, top-N to 3;
   excerption uses the published task-specific top-N values.
6. Assemble source with the common CLI, copy the matching task index and epoch-7 evaluator
   into the workspace paths specified in the method README, then install the service runtime.

`fit_contriever_musique.py`, `fit_contriever_task.py` and `build_evaluator_pairs.py` supply
helpers imported by the formal pipeline. Their older standalone entrypoints are not part of
this recipe. `rageng` and the `formal-v1` naming are retained to keep path guards unchanged.
