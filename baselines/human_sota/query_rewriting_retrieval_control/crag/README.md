# CRAG (Corrective Retrieval Augmented Generation)

Paper: S.-Q. Yan, J.-C. Gu, Y. Zhu, Z.-H. Ling. *Corrective Retrieval Augmented Generation*. Findings of IJCNLP-AACL 2026.
Upstream: https://github.com/HuskyInSalt/CRAG
Commit: `de7c2961ae624a1483a138c5798e1f6d0c4fb0e0`.
The upstream commit contains no LICENSE file. No upstream source is bundled;
its nine packaged Python modules were unchanged, so no patch is required.

The AutoRAGBench adaptation replaces web retrieval with closed-corpus Contriever retrieval.
A T5-large evaluator assigns relevance scores; threshold decisions retain, refine or
re-retrieve corpus evidence. Qwen3.5-4B at `LLM_BASE_URL` performs any generated rewrite.
`POST /retrieve` returns ordered, refined contiguous corpus excerpts before answering.
Both `crag_exc` (excerption) and `crag_sel` (selection) were evaluated on
nq_open, hotpotqa, musique, multihop_rag, pubmedqa, qasper and fever.

Both variants retrieve 10 initial documents and 25 rewrite candidates.
Selection retains top 3 units; excerption retains top 16, except PubMedQA top 6.
MultiHop-RAG and QASPER use 200-word Contriever chunks; other tasks use whole corpus entries.
QASPER restricts retrieval to the supplied document and is excluded from evaluator training.
Exact thresholds, tuning rules and other settings are in `service/configs/<task>/<variant>/config.json`.

| Task | Upper threshold | Lower threshold |
|---|---:|---:|
| nq_open | 0.592 | -0.995 |
| hotpotqa | 0.59 | -0.99 |
| musique | 1.004642367362976 | -0.9316878318786621 |
| multihop_rag | 1.0134928226470947 | -0.9970312118530273 |
| pubmedqa | 0.59 | -0.99 |
| qasper | 0.592 | -0.995 |
| fever | 1.01560378074646 | -0.995 |

## Rebuild and serve

1. Clone upstream at the commit above; install its dependencies and `build/requirements.txt`
   in a Python 3.11 environment. Download `facebook/contriever-msmarco` and
   `google-t5/t5-large`. The evaluator is trained locally, not downloaded as a final checkpoint.
2. Follow [build/README.md](build/README.md): recreate the public-training split,
   build Contriever indexes with `formal_index.py`, construct pairs with `formal_pairs.py`,
   train with `CRAG_FORMAL=1 train_evaluator_rageng.py --part 1` then `--part 2`,
   and tune with `formal_fit.py`. The delivered evaluator is epoch 7.
   Threshold tuning uses 200 examples per task (100 for PubMedQA), separate from evaluator
   training, minimum branch count 10, and the recorded paper-threshold retention rule.
3. Use `../../_common/formal_v2_packaging/assemble.py crag TASK WORKSPACE --variant crag_exc`
   (or `crag_sel`). This combines common code with the selected task/variant files.
4. Put upstream `scripts/` at `WORKSPACE/upstream/scripts/`, the task index at `index/`,
   Contriever at `models/contriever-msmarco/`, and epoch 7 at `models/evaluator/`.
   Install the service environment at `WORKSPACE/.venv/`.
5. Set `PORT`, `CORPUS_DIR`, `SCRATCH_DIR`, `LLM_BASE_URL`; run `bash WORKSPACE/submission/start.sh`.
   The service supports `/health` and `/retrieve`; answering remains evaluator-owned.

Models, indexes, training pairs, runtime libraries and data are not included.
See the shared packaging README for portable build-root variables and historical data layout.
`CRAG_*`, `RAGENG_*` and `rageng` identifiers are retained for compatibility.
