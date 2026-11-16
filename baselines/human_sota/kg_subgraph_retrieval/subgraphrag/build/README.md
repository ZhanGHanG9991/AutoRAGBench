# Build recipe

Use the exact upstream commit in the method README. Set `AUTORAGBENCH_ROOT` to a writable
benchmark/build root. Place that checkout at `AUTORAGBENCH_ROOT/humansota/SubgraphRAG`.
Copy `service/code/*.py` and then this `build/` tree into the checkout's `rageng/` directory.
Run the commands below there, using the environment installed from `requirements.txt` and
upstream dependencies. `PYTHONPATH` must include `rageng/` (and upstream modules where needed).
Set `MODEL_DIR` to the model cache containing `Qwen3.5-4B/tokenizer.json`; set `BUILD_TMP`
to the filesystem holding build outputs so the original free-space guards remain meaningful.
The scripts preserve historical directory guards and public split names. Arrange the
matching public corpora/training files under `data_formal-v1/TASK/public/` and, for CWQ /
MultiHop-RAG revisions, `data_formal-v2/TASK/public/`. No hidden split is used.
The source release does not contain those historical data files, trained weights or indexes.

Download the GTE model into the checkout's `models/gte-large-en-v1.5` directory.
Set `RAGENG_ROOT=$AUTORAGBENCH_ROOT`; model loading needs its accompanying remote-code modules
available locally for offline serving. Graph construction reads only the task corpus triples.

For WebQSP:

1. Run common `make_split.py`. Encode with `python encode.py --task webqsp --encoder ENCODER
   --train ROOT/humansota/formal-v1-split/webqsp/train_fit.jsonl
   --triples ROOT/data_formal-v1/webqsp/public/corpus/triples.tsv
   --out ROOT/humansota/SubgraphRAG/rageng/artifacts-formalv1/webqsp --scheme formalv1-train1800 --n-train 1800`
   (join the wrapped arguments; replace ROOT with the absolute build root).
2. Run `train.py --task webqsp --artifacts ARTIFACTS --train TRAIN --triples TRIPLES
   --scheme formalv1-train1800 --n-train 1800 --holdout 200 --minutes 25 --epochs 15 --patience 5`.
3. `sweep_holdout.py --phase rank` and then `--phase generate` evaluate candidate K values
   on the internal 200-question training holdout; `select_k.py` selects from those summaries.
   The released service fixes WebQSP K to 800.

For CWQ:

1. Run common `make_sgrag_train_v2.py`; it creates
   `humansota/formal-v2-split/cwq-sgrag-train2000/{train_fit,holdout}.jsonl` (1,800/200).
2. Run `encode_v2_kg.py --task cwq --encoder ENCODER --triples TRIPLES --out EMBEDDINGS`
   using `data_formal-v2/cwq/public/corpus/triples.tsv`.
3. Run `encode_v2b_q.py --encoder ENCODER --train TRAIN --holdout HOLDOUT --out ARTIFACTS`.
   Place the entity/relation arrays and KG `meta.json` from step 2 in ARTIFACTS.
4. Run `train_v2b.py --artifacts ARTIFACTS --persist OUTPUT --train TRAIN --holdout HOLDOUT
   --triples TRIPLES --minutes 90 --epochs 15 --patience 5 --workers 6`.
   `sweep_holdout_v2b.py` supports rank/generate selection using the same public-training holdout.
   The released service fixes CWQ K to 400 and uses the formalv2-train1800 metadata scheme.
5. Assemble the workspace with the learned checkpoint, entity/relation arrays and matching
   metadata under `artifacts/`. The encoder path is `models/encoder`; the budget tokenizer
   is `models/qwen/tokenizer.json`. Preserve corpus entity ordering and hashes in the metadata.

The two tasks use the adapter's native tensor implementation; the separate upstream-reproduction
checkout is not used by the served code or by these artifact builders.
Starting-package rebuilding regenerates corpus embeddings only, retaining the trained checkpoint.
