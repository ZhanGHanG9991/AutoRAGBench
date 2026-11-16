#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside an allocated GPU job}"
: "${AUTORAGBENCH_ROOT:?Set to the benchmark repository}"
: "${WORKSPACE:?Set to the prepared workspace}"
: "${CORPUS_DIR:?Set to the task corpus directory}"
: "${LLM_BASE_URL:?Set to the Qwen3.5-4B endpoint}"
BENCH=${1:?task}
export EXP3_RUN="$WORKSPACE" EXP3_TASK="$BENCH" EXP3_PREP="$WORKSPACE/prep"
export EXP3_CORPUS="$CORPUS_DIR/corpus.jsonl" EXP3_DEVICE=cuda EXP3_SELF_CHECK=1 EXP3_DISABLE_THINKING=1
export EXP3_MODELS_DIR="$WORKSPACE/runtime/models"
export EXP3_QWEN_TOKENIZER="${EXP3_QWEN_TOKENIZER:-$EXP3_MODELS_DIR/Qwen3.5-4B}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 LITELLM_LOCAL_MODEL_COST_MAP=True
export PYTHONPATH="$EXP3_PREP/lib:$WORKSPACE/runtime/vendor:$AUTORAGBENCH_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export NO_PROXY="${NO_PROXY:-},127.0.0.1,localhost" no_proxy="${no_proxy:-},127.0.0.1,localhost"
export ARB_DEADLINE="${ARB_DEADLINE:-$(($(date +%s)+21600))}"
PY="$WORKSPACE/runtime/env/bin/python"
"$PY" "$EXP3_PREP/jobs/build_flashrag_indexes.py" --bench "$BENCH" --out "$WORKSPACE/indexes"
"$PY" "$EXP3_PREP/jobs/flashrag_tune.py" --bench "$BENCH" --run "$WORKSPACE"
"$PY" "$EXP3_PREP/jobs/flashrag_select.py" --bench "$BENCH" --run "$WORKSPACE"
"$PY" "$EXP3_PREP/jobs/package_submission.py" --framework flashrag --bench "$BENCH" --run "$WORKSPACE" --mode formal
