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
: "${OPENAI_API_KEY:?Set the optimizer provider credential}"
: "${DEEPSEEK_API_KEY:?Set the examiner/judge provider credential}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
export DEEPSEEK_BASE_URL="${DEEPSEEK_BASE_URL:-https://api.deepseek.com}"
export EXP3_LANCE_ROOT="$WORKSPACE/lance-trials" EXP3_EMBED_MEMMAP=1 EXP3_LANCE_BATCH=4096
export EXP3_EMBED_MEMMAP_PATH="$WORKSPACE/embed.memmap"
"$PY" "$EXP3_PREP/jobs/autorag_optimize.py" --bench "$BENCH" --run "$WORKSPACE"
if [[ ! -e "$WORKSPACE/submission/start.sh" ]]; then
  "$PY" "$EXP3_PREP/jobs/package_submission.py" --framework autorag --bench "$BENCH" --run "$WORKSPACE" --mode formal
fi
