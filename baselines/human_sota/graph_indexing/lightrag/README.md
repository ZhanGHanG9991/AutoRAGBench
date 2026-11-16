# LightRAG

Paper: Z. Guo, L. Xia, Y. Yu, T. Ao, C. Huang. *LightRAG: Simple and Fast Retrieval-Augmented Generation*. arXiv, 2024.
Upstream: https://github.com/HKUDS/LightRAG
Commit: `28ff1b05f2ac3f3e6fa14dd2cd33656579bd0c9c`.
Upstream license: [UPSTREAM_LICENSE](UPSTREAM_LICENSE).
The packaged upstream library matches this commit; no patch is required.

The AutoRAGBench adaptation extracts corpus entities/relations with Qwen3.5-4B, merges
sharded graphs, and builds `BAAI/bge-m3` vectors. A local embedding worker serves the vectors.
Hybrid graph retrieval uses Qwen3.5-4B at `LLM_BASE_URL` for generated query text.
`POST /retrieve` returns original corpus chunks after library token limits and the final
sending guard; generated entity/relation descriptions are never returned as evidence.
Evaluated tasks: multihop_rag, musique and pubmedqa.

All three use hybrid mode, reranking disabled and a 30,000-token library context budget;
the final guard reserves the 512-token answer allowance within 32,000 tokens.
Graph construction uses 1,200-token chunks with 100-token overlap, extraction/merge
completion caps of 2,048 tokens, and BGE-M3 embeddings. Task-specific launchers select
answer-format metadata; corpus and storage paths are supplied at assembly/runtime.

## Rebuild and serve

1. Clone upstream at the commit above. Install it and `build/requirements.txt` in a
   Python 3.11 environment. Create a separate Python 3.12 embedding environment with
   torch and transformers; download `BAAI/bge-m3` and the `Qwen/Qwen3.5-4B` tokenizer.
2. Follow [build/README.md](build/README.md): create corpus shards with `formal_split.py`,
   extract graphs with `build_graph.py`, merge using `merge_shards.py`, then create real
   BGE vectors with `embed_rebuild.py`. Initial extraction uses zero embeddings;
   the embedding rebuild is required before serving.
3. Use `../../_common/formal_v2_packaging/assemble.py lightrag TASK WORKSPACE`.
   Copy upstream `lightrag/` to `WORKSPACE/upstream/lightrag/`, merged storage to
   `index/storage/`, BGE-M3 to `models/bge-m3/`, and `tokenizer.json` to `models/qwen/`.
   Install the service runtime at `.venv/` and embedding runtime at `bge-venv/`.
4. Set `PORT`, `CORPUS_DIR`, `SCRATCH_DIR`, `LLM_BASE_URL`; run `bash WORKSPACE/submission/start.sh`.
   The launcher copies graph storage into scratch before library writes and starts the
   local embedding worker. Use a short scratch path for the Unix-domain socket.

## Starting package (Section VI-G)

`starting_package/` contains the supplied MultiHop-RAG service plus `runtime_env.sh`,
`build.sh` and `code/rebuild_index.py`. Retrieval is inherited from Human-SOTA;
the new build entrypoint reconstructs the whole corpus graph and vectors through the
local generator and embedding worker. Assemble with `--starting-package` and use the
standalone Python prefixes described in the shared packaging README. Set `BUILD_PARTITION`
inside an allocated build environment, then run `bash WORKSPACE/build.sh NEW_OUTPUT`.
This direct rebuild entrypoint differs from the original sharded construction pipeline.

Models, indexes, tokenizer, runtimes, generated graph content and data are not included.
See the shared packaging README for portable build-root variables.
`rageng` identifiers are retained for compatibility.
