# Build recipe

Use the exact upstream commit in the method README. Set `AUTORAGBENCH_ROOT` to a writable
benchmark/build root. Place that checkout at `AUTORAGBENCH_ROOT/humansota/LightRAG`.
Copy `service/code/*.py` and then this `build/` tree into the checkout's `rageng/` directory.
Run the commands below there, using the environment installed from `requirements.txt` and
upstream dependencies. `PYTHONPATH` must include `rageng/` (and upstream modules where needed).
Set `MODEL_DIR` to the model cache containing `Qwen3.5-4B/tokenizer.json`; set `BUILD_TMP`
to the filesystem holding build outputs so the original free-space guards remain meaningful.
The scripts preserve historical directory guards and public split names. Arrange the
matching public corpora/training files under `data_formal-v1/TASK/public/` and, for CWQ /
MultiHop-RAG revisions, `data_formal-v2/TASK/public/`. No hidden split is used.
The source release does not contain those historical data files, trained weights or indexes.

Install the pinned upstream library in the graph/service environment. Install torch and
transformers separately for `bge_server.py`; download `BAAI/bge-m3` locally.
Set `LLM_BASE_URL` to the Qwen3.5-4B build endpoint and use corpus-local document IDs.

1. `formal_split.py --corpus CORPUS --pilot-storage PILOT_STORAGE --out SHARDS
   --nshards N --expect-formal COUNT --expect-pilot PILOT_COUNT` creates round-robin ID files.
   This helper verifies that reused pilot chunks match the corpus. For a clean rebuild without
   prior storage, `build_graph.py --shard-index I --num-shards N` partitions the full corpus;
   use this instead of supplying `--id-file` from a reused-storage split.
2. Run `build_graph.py --corpus CORPUS --out SHARD --llm "$LLM_BASE_URL" --group corpus
   --route gateway --n 20000000 --seed 20261018 --shard-index I --num-shards N
   --async-llm 64 --parallel-insert 64 --embedding-async 8 --max-tokens 2048
   --max-wall-sec 13200 --flush-every 200 --disk-guard` for each shard.
   When using step 1's ID files, add `--id-file IDS` and set `--shard-index 0 --num-shards 1`.
   Resume the same output directory until all intended documents are processed.
3. Run `merge_shards.py --shards SHARD/storage ... --out MERGED --llm "$LLM_BASE_URL"
   --max-tokens 2048 --concurrency 48 --skip-vdb --expect-docs COUNT --disk-guard`.
   COUNT is the number of unique intended corpus documents; the merged storage must cover them all.
4. Start `bge_server.py --model MODEL --sock SOCKET --device cuda` in the embedding runtime.
   Run `embed_rebuild.py --storage MERGED/storage --bge-sock SOCKET --batch-size 16 --disk-guard`.
   This replaces zero vectors from extraction with BGE-M3 vectors; it is mandatory for serving.
5. Assemble source with the common CLI and copy the finished storage to `WORKSPACE/index/storage`.
   Install the pinned upstream library at `WORKSPACE/upstream/lightrag`, BGE model and Qwen
   tokenizer as described in the method README, plus the two Python environments.

`build_graph.py` is the actual historical graph-builder script under a descriptive name;
its chunking, batching, resumability, token caps and extraction behavior are unchanged.
`sample_corpus.py` is its corpus-selection dependency, and `disk_guard.py` provides its
free-space guard. No diagnostic or comparison runs are required by this recipe.
The starting-package `rebuild_index.py` offers a separate full-corpus insertion route.
