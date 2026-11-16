# SubgraphRAG

Paper: M. Li, S. Miao, P. Li. *Simple is Effective: The Roles of Graphs and Large Language Models in Knowledge-Graph-Based Retrieval-Augmented Generation* (SubgraphRAG). ICLR 2025.
Upstream: https://github.com/Graph-COM/SubgraphRAG
Commit: `48eb05e56d11293fa780849512310462c45b956b`.
Upstream license: [UPSTREAM_LICENSE](UPSTREAM_LICENSE). No upstream source is bundled.
The served adaptation implements the upstream directional distance encoding and two-layer
retriever MLP with native tensor aggregation; it does not require a patched upstream checkout.

The AutoRAGBench adapter builds its graph from corpus `triples.tsv`, embeds entity/relation
names with `Alibaba-NLP/gte-large-en-v1.5`, and trains the MLP on public training questions.
It localizes a two-hop subgraph with a 100,000-triple ceiling and ranks triples.
`POST /retrieve` returns original corpus triples in ranked order after the Qwen tokenizer
budget guard, capped at 256 passages. This platform cap differs from the configured K.
No generated answer or graph summary is returned as evidence.

| Evaluated task | Ranked K | Training scheme | Training examples |
|---|---:|---|---:|
| webqsp | 800 | formalv1-train1800 | 1800 |
| cwq | 400 | formalv2-train1800 | 1800 |

## Rebuild and serve

1. Clone upstream at the commit above; install `build/requirements.txt` in Python 3.11.
   Download `Alibaba-NLP/gte-large-en-v1.5` including its model code, and the
   `Qwen/Qwen3.5-4B` tokenizer. Only the tokenizer is loaded by the evidence budget guard.
2. Follow [build/README.md](build/README.md): encode WebQSP with `encode.py` and train with
   `train.py`; encode CWQ with `encode_v2_kg.py` and `encode_v2b_q.py`, then train with
   `train_v2b.py`. The 200-question public-training holdout and `sweep_holdout*.py` /
   `select_k.py` provide model/K selection; no hidden data is required.
3. Use `../../_common/formal_v2_packaging/assemble.py subgraphrag TASK WORKSPACE`.
   Put `ckpt.pt`, `ent_emb.f16.npy`, `rel_emb.f16.npy`, `meta.json` in `WORKSPACE/artifacts/`,
   the encoder in `models/encoder/`, `tokenizer.json` in `models/qwen/`, and the runtime in `.venv/`.
4. Set `PORT`, `CORPUS_DIR`, `SCRATCH_DIR`, `LLM_BASE_URL`; run `bash WORKSPACE/submission/start.sh`.
   The corpus must include matching `triples.tsv` and `corpus.jsonl` with stable document IDs.

## Starting packages (Section VI-G)

`starting_package/` contains the supplied WebQSP and CWQ code, with task launchers under
`configs/<task>/`. It retains the Human-SOTA retriever and K, adds relocatable runtime setup,
and adds `build.sh` / `code/rebuild_index.py` to rebuild corpus embeddings while preserving
the learned MLP. Assemble with `--starting-package`; follow the shared packaging README for
standalone Python-prefix layout. Inside an allocated build environment, set `BUILD_PARTITION`
and run `bash WORKSPACE/build.sh NEW_OUTPUT`; select that output when assembling new artifacts.

Weights, learned MLPs, vectors, tokenizer, runtimes and data are not included.
See the shared packaging README for build-root variables and historical split layout.
`SGRAG_*`, `RAGENG_ROOT` and `rageng` identifiers are retained for compatibility.
