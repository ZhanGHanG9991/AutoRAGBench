# LinearRAG

Paper: L. Zhuang, S. Chen, Y. Xiao, H. Zhou, Y. Zhang, H. Chen, Q. Zhang, X. Huang. *LinearRAG: Linear Graph Retrieval Augmented Generation on Large-scale Corpora*. arXiv, 2025.
Upstream: https://github.com/DEEP-PolyU/LinearRAG.git
Commit: `bcc94e66c221f798801255efba09311d6fbcd8d6`. License: `UPSTREAM_LICENSE`.

The AutoRAGBench adapter builds a relation-free entity/sentence/passage graph
with `sentence-transformers/all-mpnet-base-v2` and spaCy `en_core_web_trf`.
It uses 240-word windows at stride 220 and bounded NER batches; retrieval uses entity
activation and graph/PPR scoring, returning the top five original corpus chunks.
The corpus adapter removes only the known title/ordinal metadata wrapper and
validates the remaining body against the source document. Indexing and `/retrieve`
make no generator calls. The unused legacy `/answer` route uses Qwen3.5-4B.

Evaluated: CWQ, FEVER, HotpotQA, MultiHop-RAG, MuSiQue, PubMedQA, QASPER, WebQSP.
NQ is outside this batch. Shared parameters: three iterations, activation threshold
0.4, passage ratio 2, sentence top-k 3, retrieval top-k 5, PPR damping 0.5,
passage-node weight 0.05, embedding batch 128, eight workers; hybrid attribute
fallback and vectorized retrieval are disabled. QASPER builds per-paper graphs
and requires `meta.doc_id`.

FEVER is a documented budget adaptation: the evaluated service uses LinearRAG's
own exact dense top-five component because graph/PPR warmup exceeded the question
budget. It uses the same passage embeddings/index. HotpotQA warms graph retrieval
before health; PubMedQA and the larger tasks retain their distinct frozen servers
and corpus validators. `build/layout.json` selects these exact variants.

## Rebuild

Follow [workspace assembly](../../_common/fixed_evidence_adapter/README.md).
Create Python 3.12 at `$W/.venv` and install the pinned upstream dependencies.
`build/environment.lock.txt` records dependency versions; set `SPACY_MODEL_WHEEL`
to the absolute path of the downloaded `en_core_web_trf` wheel before installing it.
No upstream patch is needed. Obtain the models from their public model repositories:

| Model | Revision / placement |
|---|---|
| sentence-transformers/all-mpnet-base-v2 | `e8c3b32edf5434bc2275fc9bab85f82640a19130`; `$W/models/all-mpnet-base-v2` |
| spacy/en_core_web_trf | `58223163ebdcfdff182d526cc8deb1c73cd8d5a3`; install the wheel into `$W/.venv` with spaCy 3.7.3 |

Use the public corpus at `data_formal-v2/$TASK/public/corpus/corpus.jsonl`.
The archived builder deliberately checks this path suffix; a local directory
alias can provide that layout when your dataset root uses another name.
With `CORPUS_DIR` set to that corpus directory, run on a GPU:

```bash
bash "$M/build/rebuild.sh" "$W" "$TASK" "$W/index"
bash "$W/submission/start.sh"
```

The wrapper calls the original corpus builder with the frozen runtime. The
earlier `build_formal.py` is retained because FEVER, HotpotQA and PubMedQA reused
byte-identical earlier corpus indexes; its direct CLI checks `data_formal-v1/`.
No model training is required. Model weights, graph/index files, optional SQLite
metadata, corpus data and Python environments are excluded and must be rebuilt.
This batch includes no LinearRAG VI-G starting package.
