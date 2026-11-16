# Formal-v2 workspace assembly

Shared assembly helpers for CRAG, Adaptive-RAG, SubgraphRAG and LightRAG.
`package_lib.py` retains the original copy/hash/runtime functions; campaign evaluation,
submission, archive mutation, and freeze orchestration are omitted.
`assemble.py` overlays the released common service code, task configuration and optional CRAG variant.
`catalog.json` lists only the evaluated task/method combinations in this release batch.

```bash
python assemble.py crag nq_open /path/to/new-workspace --variant crag_exc
python assemble.py subgraphrag webqsp /path/to/new-start --starting-package
```

The destination must not exist. Assembly copies source/configuration only, resolves
`@WORKSPACE@` in JSON, and makes shell launchers executable. Install the pinned upstream
checkout and the built indexes/models as described in each method README.
Runtime environments are not included. For services, create `.venv` inside the workspace;
LightRAG also needs `bge-venv`. Starting packages additionally require relocatable
`python/bin/python3.11` and `.venv/lib/python3.11/site-packages`; LightRAG's embedding
worker uses `bge-python/bin/python3.12` and `bge-venv/lib/python3.12/site-packages`.
The original `copy_runtime(source_venv, workspace)` and `copy_bge_runtime(workspace)`
helpers can copy those prefixes. The latter reads `AUTORAGBENCH_ROOT/envs/rag-base`.

Build-time path variables: `AUTORAGBENCH_ROOT` is the benchmark/build root;
`MODEL_DIR` is the downloaded model root (default: workspace `models`);
`WORKSPACE` defaults to the parent of the executing code directory;
`BUILD_TMP` is the scratch filesystem checked by disk guards (default `/tmp`).
Set these explicitly for historical build scripts. Their `humansota/<Upstream>/rageng`
subdirectory layout is preserved. Set `BUILD_PARTITION` when using a starting-package
build wrapper inside a Slurm allocation; replace `<partition>` in scheduler templates.
`RAGENG_ROOT`, `ARAG_*`, `CRAG_*`, `SGRAG_*` and `rageng` names are retained for compatibility.

`make_split.py` recreates the stratified public-training split with seed 20261019.
`make_sgrag_train_v2.py` recreates the CWQ training/holdout split. These operate only on
public training data under `data_formal-v1` / `data_formal-v2` and write under `humansota`.
The historical layout is required by the build guards; supply the matching public data.
No corpus, labelled records, model weights, indexes, runtime libraries or results are bundled.
