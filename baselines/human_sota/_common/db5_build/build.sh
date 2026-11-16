#!/bin/bash
# Build one method using public inputs and an already running benchmark generator.
set -euo pipefail
METHOD=${1:?method}; TASK=${2:?task}; OUT=${3:?new build directory}
R=${AUTORAGBENCH_ROOT:?benchmark checkout required}
CAMP=$R/humansota/db5-formal-v2-20261105
PUBLIC=${PUBLIC_DIR:?public task directory required}
CORPUS=$PUBLIC/corpus/corpus.jsonl
TRAIN=${TRAIN_FILE:-$PUBLIC/train.jsonl}
DEV=${DEV_FILE:-}
: "${LLM_BASE_URL:?fixed generator gateway required}"
[ ! -e "$OUT" ] || { echo "refusing to reuse build output"; exit 2; }
mkdir -p "$OUT"
export RAGENG_DATA_VERSION=formal-v2
export HF_HOME="$CAMP/model-cache/huggingface"
export CARROT_GTE_OFFLINE_DIR="$CAMP/model-cache/gte-multilingual-reranker-base-offline"
export HAMMER_QUERY_EMBED_MODEL="$CAMP/model-cache/bge-large-en-v1.5"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 MKL_NUM_THREADS=12
# CARROT label search resets the metered gateway between training questions.
if [ "$METHOD" = carrot ]; then
  : "${CARROT_GATEWAY_URL:?gateway admin endpoint required}"
  : "${CARROT_ADMIN_TOKEN_FILE:?gateway admin credential file required}"
fi
case "$METHOD" in
  bookrag)
    B=$R/humansota/BookRAG; PY=$B/.venv/bin/python; IDX=$OUT/index
    export TIKTOKEN_CACHE_DIR=$B/rageng/tiktoken_cache PYTHONPATH=$B/rageng:$B/rageng/vendor
    "$PY" "$B/rageng/build.py" --task "$TASK" --corpus "$CORPUS" --out "$IDX" --models "$B/models" \
      --llm "$LLM_BASE_URL" --seconds 126000 --workers 128
    CONFIG=("$IDX/index_stats.json")
    ;;
  lego)
    L=$R/humansota/LEGO-GraphRAG; PY=$L/.venv/bin/python; IDX=$OUT/index
    mkdir -p "$IDX"
    "$PY" "$L/rageng/openie.py" --corpus "$PUBLIC/corpus" --out "$IDX" --llm "$LLM_BASE_URL" --key EMPTY
    "$PY" "$L/rageng/build_graph.py" --index "$IDX" --emb "$L/models/all-MiniLM-L6-v2"
    if [ "$TASK" = pubmedqa ]; then
      "$PY" "$CAMP/campaign/build/select_lego.py" --task "$TASK" --out "$OUT/selection.json"
    else
      "$PY" "$CAMP/source_overrides/lego/devgrid.py" --task "$TASK" --index "$IDX" --llm "$LLM_BASE_URL" \
        --out "$OUT/devgrid" --questions "$DEV" --workers 8
      "$PY" "$CAMP/campaign/build/select_lego.py" --task "$TASK" --summary "$OUT/devgrid/summary.json" --out "$OUT/selection.json"
    fi
    CONFIG=("$IDX/graph_stats.json" "$OUT/selection.json")
    ;;
  clue)
    CL=$R/humansota/ClueRAG; PY=$CL/.venv/bin/python; IDX=$OUT/index
    export NLTK_DATA=$CL/nltk_data TIKTOKEN_CACHE_DIR=$CL/tiktoken_cache
    "$PY" "$CL/rageng/index_task.py" --task "$TASK" --corpus "$PUBLIC/corpus" --out "$IDX" \
      --llm "$LLM_BASE_URL" --key EMPTY --alpha 1.0 --concurrency 256
    CONFIG=("$IDX/index_stats.json")
    ;;
  carrot)
    CR=$R/humansota/CARROT; PY=$CR/.venv/bin/python; IDX=$OUT/index; TRAINING=$OUT/training
    MODELS=$CR/rageng/profile6/models
    "$PY" "$CAMP/campaign/build/carrot_build_index.py" --upstream "$CR" --models "$MODELS" \
      --corpus "$PUBLIC/corpus" --output "$IDX"
    export CARROT_UPSTREAM=$CR CARROT_MODEL_ROOT=$MODELS CARROT_RAGENG_ROOT=$R
    "$PY" "$CAMP/source_overrides/carrot/profile_native.py" --task "$TASK" --output "$TRAINING" \
      --public "$PUBLIC" --train "$TRAIN" --index "$IDX" --count 100 --top-k 60 --seed 20261022
    CONFIG=("$IDX/index_stats.json" "$TRAINING/provenance.json" "$TRAINING/summary.json")
    ;;
  hammer)
    H=$R/humansota/HAMMER; PY=$H/.venv/bin/python; SEARCH=$OUT/search; CACHE=$OUT/cache
    mkdir -p "$CACHE"
    export HAMMER_BUILD_SANDBOX=singularity HAMMER_LLM_BASE_URL=$LLM_BASE_URL
    unset VLLM_API_KEY
    "$PY" "$CAMP/source_overrides/hammer/optimize.py" --task "$TASK" --output "$SEARCH" \
      --public "$PUBLIC" --train "$TRAIN" --cache "$CACHE" --iterations 20 --seconds 120000 --seed 20261022
    CONFIG=("$SEARCH/recommended.json" "$SEARCH/completed.json" "$SEARCH/search_budget.json")
    ;;
esac


ARGS=(--method "$METHOD" --task "$TASK" --corpus "$CORPUS" --train "$TRAIN" --output "$OUT")
[ -z "$DEV" ] || ARGS+=(--dev "$DEV")
for FILE in "${CONFIG[@]}"; do ARGS+=(--config "$FILE"); done
"$PY" "$CAMP/campaign/build/write_manifest.py" "${ARGS[@]}"
