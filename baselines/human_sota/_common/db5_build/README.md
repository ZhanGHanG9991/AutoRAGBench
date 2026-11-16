# Shared build and workspace assembly

These utilities rebuild the five DB5 Human-SOTA adaptations using public corpus/train/dev
inputs and an already running Qwen3.5-4B gateway. They do not schedule jobs or evaluate tests.
Use a Linux CUDA environment; provision compute separately. `AUTORAGBENCH_ROOT` is the
absolute AutoRAGBench checkout. No default external installation path is assumed.

1. Clone each required upstream at the commit in its method README into
   `$AUTORAGBENCH_ROOT/humansota/<UpstreamName>`. Apply that method's patch with `git apply`.
   Keep upstream datasets out of all builds; only the benchmark's public inputs are used.
2. Set the variables below and stage the released code. `stage.py` copies build scripts
   into upstream `rageng/` and the shared working directory; it does not run them.
3. Install the staged `rageng/setup_env.sh` dependencies and obtain auxiliary model assets
   before enabling offline mode. Use the same Python/library versions for build and service.
4. Run the build command with explicit inputs, then assemble a fresh workspace.

```bash
export AUTORAGBENCH_ROOT=/path/to/AutoRAGBench
export RAGENG_ROOT="$AUTORAGBENCH_ROOT"
B="$AUTORAGBENCH_ROOT/baselines/human_sota/_common/db5_build"
METHOD=carrot                    # bookrag | carrot | clue | hammer | lego
TASK=pubmedqa                   # use only a task listed in that method's README
python "$B/stage.py" --root "$AUTORAGBENCH_ROOT" --method "$METHOD" --task "$TASK"
# Install the selected upstream's staged rageng/setup_env.sh here.
# Run with an interpreter providing huggingface_hub and tiktoken:
python "$B/prepare_models.py" --root "$AUTORAGBENCH_ROOT" --method "$METHOD"
export PUBLIC_DIR="$AUTORAGBENCH_ROOT/data/formal-v2/$TASK/public"
export TRAIN_FILE="$PUBLIC_DIR/train.jsonl"
export DEV_FILE=""              # explicit public dev file for MuSiQue/QASPER
export LLM_BASE_URL=http://127.0.0.1:8000/v1
BUILD=/path/to/new-build
WORKSPACE=/path/to/new-workspace
bash "$B/build.sh" "$METHOD" "$TASK" "$BUILD"
python "$B/assemble.py" --root "$AUTORAGBENCH_ROOT" --method "$METHOD" \
  --task "$TASK" --build "$BUILD" --workspace "$WORKSPACE"
export PORT=8080 CORPUS_DIR="$PUBLIC_DIR/corpus" SCRATCH_DIR=/path/to/scratch
mkdir -p "$SCRATCH_DIR"
bash "$WORKSPACE/submission/start.sh"
```

The builds used 9,500 train / 500 dev examples for MuSiQue, 3,419 train / 179 dev
for QASPER, and all 500 public training examples for PubMedQA, with no PubMedQA dev split.
Supply the public split files separately; no examples or split outputs are bundled.
The exact train/dev file hashes are recorded outside this public release.

CARROT requires an eval-mode metered build gateway, `CARROT_GATEWAY_URL` (its root URL)
and `CARROT_ADMIN_TOKEN_FILE` (a local credential file); the credential is not bundled.
Its native label search resets the gateway per public training question. Configure the
upstream benchmark gateway yourself before running `build.sh`.
The shared model helper preserves the pinned GTE custom-code revision and creates a
self-contained `gte-multilingual-reranker-base-offline` package with local `auto_map`.
Load the Jina/GTE models once during online preparation to populate their custom-code
cache under `humansota/db5-formal-v2-20261105/model-cache/huggingface`.

HAMMER additionally needs its staged `rageng/download_models.sh` to populate
`rageng/.hf-home`; `prepare_models.py` supplies the query-selection model and BGE-M3.
Run its build in the benchmark's Singularity environment. The preserved QASPER adapter
still expects the build-time model directory; its isolated-startup failure is not repaired.

BookRAG's model helper must run before `setup_env.sh` unless the two BAAI models are already
present. The upstream patch must be applied before `stage_source.py` copies `Core/`.
For Clue-RAG, download the public spaCy en_core_web_trf 3.8.0 wheel and NLTK punkt_tab
archive into upstream `vendor/` as expected by `setup_env.sh`; these are dependencies,
not benchmark datasets. LEGO/Clue environments are copied into workspace `.venv/`;
NLTK data are also copied into `pylib/nltk_data` for the unchanged launcher setting.
All packaged virtual environments must be relocatable on the destination Linux installation.

`package_helpers.py` retains the original index/model packing operations; `assemble.py`
overlays the exact frozen service variant using `service/layout.json`. CARROT's frozen
artifact stores training provenance plus the six-class catalog; HAMMER's artifact stores
its selected parameters. BookRAG uses `venv/`, while the other methods use `.venv/`.
All services need corpus-derived indexes and local models before startup; none are bundled.
The helper's hard links require build, model cache and workspace to share a filesystem.

Build defaults retain the campaign's 126000-second BookRAG cap and 120000-second HAMMER
cap; the original allocation envelopes were 24 hours for BookRAG/LEGO/Clue and 36 hours
for HAMMER/CARROT. There is no automatic scheduler submission here.

The source was syntax-checked and patches were checked against pinned upstream files.
A full rebuild and GPU service smoke test remain unverified. Upstream model revisions
were not uniformly recorded, so newly downloaded weights may differ from the originals.

Assembly refuses a rebuilt HAMMER/LEGO configuration that differs from the evaluated
artifact, preventing a frozen configuration from being paired with incompatible new indexes.
