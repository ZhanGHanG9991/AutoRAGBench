#!/bin/bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUN=$(dirname -- "$HERE")
export EXP3_PREP="$HERE"
export EXP3_TASK="webqsp"
export EXP3_CORPUS="${CORPUS_DIR:-$RUN/../task/corpus}/corpus.jsonl"
export EXP3_DEVICE=cuda HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 EXP3_DISABLE_THINKING=1 LITELLM_LOCAL_MODEL_COST_MAP=True
export EXP3_MODELS_DIR="$RUN/runtime/models"
export PYTHONPATH="$HERE/lib:$HERE/runtime:$RUN/runtime/vendor"
export EXP3_LANCE_DIR="$RUN/serve-index" EXP3_TRIAL_JSON="$RUN/winning-trial.json"
exec "$RUN/runtime/env/bin/python" "$HERE/runtime/exp3_autorag_server.py"
