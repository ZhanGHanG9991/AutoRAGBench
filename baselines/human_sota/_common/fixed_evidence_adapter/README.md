# Fixed-evidence adapter support

Shared AutoRAGBench corpus validators, preserved from the evaluated services:
`fe_corpus.py` is the RAPTOR/HippoRAG version; `linear_title_wrapper/` removes
LinearRAG's ordinal/title prefix; `linear_empty_title/` additionally accepts the
ordinal-only prefix for empty titles. All three validate whitespace-normalized
complete bodies. `starting_provenance.py`
is the distinct VI-G leaf/chunk mapper, with its original evidence caps.
These versions are intentionally separate. Assembly installs the right module as
`adapter/fe_corpus.py` or `adapter/provenance.py`, preserving imports.

## Workspace assembly

Set `M` to a method directory, `U` to its upstream git checkout, `TASK` to a task
key from its README, and `W` to a new output directory. Check out the exact commit
documented by that method; install its dependencies into the indicated environment.
Then, from the method directory or with absolute paths:

```bash
python "$M/../../_common/fixed_evidence_adapter/assemble.py" \
  --method "$M" --task "$TASK" --workspace "$W" --upstream "$U"
```

Add `--starting-package` for an available VI-G package. `build/layout.json` is the
authoritative map from deduplicated release files to workspace-relative filenames.
Assembly checks upstream HEAD, copies only source modules, and stages archived
builders under `build_sources/`; it refuses an existing workspace. Install upstream
at the recorded commit; no upstream patches are required for these four methods.
Assembly does not create environments, download models or build indexes.

Complete the method's model/index steps. For every service set `PORT`,
`CORPUS_DIR` (directory containing `corpus.jsonl`), `SCRATCH_DIR` (writable),
and `LLM_BASE_URL` (the local Qwen3.5-4B gateway, including `/v1`). Start
`bash "$W/submission/start.sh"`; readiness is `GET /health`, retrieval is
`POST /retrieve` with `{"question":"...","meta":{}}`.
Use a fresh scratch directory per corpus. Models/indexes remain read-only while
serving. `FE_CORPUS_CACHE_DIR`, when set, overrides the validator's scratch cache.
Loopback addresses are service bind addresses/protocol checks, not remote endpoints.

The portable assembly/rebuild wrappers are release additions; served retrieval
code is copied from the frozen workspaces. Syntax and source mapping can be checked
without model downloads or evaluation. Rebuilding indexes, especially summaries
and graph extraction, requires the corpus, models and substantial compute.
