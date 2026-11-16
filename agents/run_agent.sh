#!/bin/bash
# Run one agent (agents/registry.json) in its run directory, inside the sandbox, until task/.deadline.
# Usage (in build.sbatch, after preflight): agents/run_agent.sh <run_id> <agent>   — needs LLM_BASE_URL.
# The harness's stdout (JSON events) is the trace, captured outside the sandbox.
set -uo pipefail
source "${RAGENG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/configs/pilot.env"
source $RAGENG_ROOT/scripts/sandbox.sh
RUN_ID=$1; AGENT=$2
RUN=$RUNS_DIR/$RUN_ID; REC=$RECORDS_DIR/$RUN_ID
reg() { $RAG_BASE_PY -c "import json,sys; r=json.load(open('$RAGENG_ROOT/agents/registry.json')); print(eval(sys.argv[1]))" "$1"; }
HARNESS=$(reg "r['$AGENT']['harness']"); PROVIDER=$(reg "r['$AGENT']['provider']")
REPROMPT=$(reg "int(r['$AGENT']['reprompt'])"); KEYS=$(reg "' '.join(r['_providers']['$PROVIDER']['keys'])")
HARNESS_PATH=$(reg "':'.join('$RAGENG_ROOT/'+h['bin'] for h in r['_harnesses'].values())")
VERSION_CMD=$(reg "r['_harnesses']['$HARNESS']['version']")

source $RAGENG_ROOT/llm/$PROVIDER/env.sh || exit 3   # provider endpoints + keys (host side only)
mkdir -p $RUN/home/tmp
if [ "$HARNESS" = codex ]; then   # fresh per-run CODEX_HOME: provider config only, no sessions / memories
  mkdir -p $RUN/home/.codex
  cp $RAGENG_ROOT/llm/$PROVIDER/codex_home/models.json $RUN/home/.codex/
  sed "s#^model_catalog_json = .*#model_catalog_json = \"$RUN/home/.codex/models.json\"#" \
    $RAGENG_ROOT/llm/$PROVIDER/codex_home/config.toml > $RUN/home/.codex/config.toml
elif [ "$HARNESS" = deepseek_harness ]; then   # fresh per-run DSH_HOME: home-level profile patch only
  mkdir -p $RUN/home/.dsh
  cp $RAGENG_ROOT/llm/$PROVIDER/dsh_home/cordis.patch.yml $RUN/home/.dsh/
elif [ "$HARNESS" = opencode ]; then   # fresh per-run HOME: global opencode config only
  mkdir -p $RUN/home/.config/opencode
  cp $RAGENG_ROOT/llm/$PROVIDER/opencode/opencode.json $RUN/home/.config/opencode/
fi

SB_HOME=$RUN/home; SB_PWD=$RUN/workspace; SB_GPU=1; SB_NET=host
SB_RW=($RUN/home $RUN/workspace $RUN/selfcheck)
mkdir -p $RUN/selfcheck
# read-only: what the agent and task/self_check need, nothing of the harness configuration (configs/pilot.env,
# scripts/, agents/run_agent.sh, llm/, judges/, records/, hidden/ stay invisible; preflight checks it)
SB_RO=($RUN/task $RAGENG_ROOT/envs/rag-base $RAGENG_ROOT/envs/python $RAGENG_ROOT/envs/vllm
       $RAGENG_ROOT/harness $SOFTWARE_DIR $RAGENG_ROOT/agents/harness $RAGENG_ROOT/gateway $RAGENG_ROOT/evaluator.py
       $RAGENG_ROOT/tasks $RAGENG_ROOT/multireader/contract.py $RAGENG_ROOT/multireader/__init__.py $GENERATOR_PATH/config.json $GENERATOR_PATH/tokenizer.json $GENERATOR_PATH/tokenizer_config.json $GENERATOR_PATH/chat_template.jinja $RAGENG_ROOT/configs/sandbox $RAGENG_ROOT/selfcheck)
SB_ENV=(PATH=$HARNESS_PATH:$SOFTWARE_DIR/node/bin:/usr/local/bin:/usr/bin:/bin
        HOME=$RUN/home TMPDIR=$RUN/home/tmp SHELL=/bin/bash LANG=C.UTF-8 TERM=dumb
        LLM_BASE_URL=$LLM_BASE_URL CORPUS_DIR=$RUN/task/corpus TASK_DIR=$RUN/task WORKSPACE=$RUN/workspace
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} SLURM_JOB_ID=$SLURM_JOB_ID RAGENG_PORT_BASE=$RAGENG_PORT_BASE RAGENG_PORT_JOB=$RAGENG_PORT_JOB
        HF_ENDPOINT=${HF_ENDPOINT_AGENT:-$HF_MIRROR} HF_HUB_DISABLE_XET=1 PIP_INDEX_URL=$PIP_MIRROR UV_INDEX_URL=$PIP_MIRROR
        AGENT_HARNESS=$HARNESS AGENT_REPROMPT=$REPROMPT DEADLINE=$(cat $RUN/task/.deadline)
        PROMPT_FILE=$RUN/home/.prompt.txt)
for k in $KEYS; do SB_ENV+=("$k=${!k}"); done
# optional provider-outage handling (agents/harness/common.sh): set PROVIDER_HEALTH_URL in llm/<provider>/env.sh
[ -n "${PROVIDER_HEALTH_URL:-}" ] && SB_ENV+=(PROVIDER_HEALTH_URL=$PROVIDER_HEALTH_URL PROVIDER_KEY_VAR=${KEYS%% *})
if [ "$HARNESS" = codex ]; then
  SB_ENV+=(CODEX_HOME=$RUN/home/.codex)
elif [ "$HARNESS" = deepseek_harness ]; then
  SB_ENV+=(DSH_HOME=$RUN/home/.dsh)
elif [ "$HARNESS" = opencode ]; then
  SB_ENV+=(OPENCODE_DISABLE_AUTOUPDATE=1)
else
  for k in $(reg "' '.join(r['_claude_env'])"); do [ -n "${!k:-}" ] && SB_ENV+=("$k=${!k}"); done
fi
cp $REC/prompt.txt $RUN/home/.prompt.txt
[ "$HARNESS" != codex ] && printf '\nYou are running in a non-interactive mode. So make sure every process you are running finishes before you write your last message.\n' >> $RUN/home/.prompt.txt

# record the harness version and configuration before the run
VERSION=$(sandbox_exec bash -c "$VERSION_CMD" 2>/dev/null | tail -1)
case $HARNESS in
  codex) MODEL=$(grep -m1 '^model =' $RUN/home/.codex/config.toml | cut -d'"' -f2)
         EFFORT=$(grep -m1 '^model_reasoning_effort' $RUN/home/.codex/config.toml | cut -d'"' -f2) ;;
  deepseek_harness) MODEL=$(grep -m1 '^    model:' $RUN/home/.dsh/cordis.patch.yml | awk '{print $2}')
         EFFORT=$(awk '$1=="reasoningEffort:" || $1=="reasoning:" {print $2; exit}' $RUN/home/.dsh/cordis.patch.yml) ;;
  opencode) MODEL=$($RAG_BASE_PY -c "import json; print(json.load(open('$RUN/home/.config/opencode/opencode.json'))['model'])")
         EFFORT=$(grep -m1 '^VARIANT=' $RAGENG_ROOT/agents/harness/opencode.sh | cut -d= -f2 | cut -d' ' -f1) ;;
  *) MODEL=${ANTHROPIC_MODEL:-}; EFFORT=${CLAUDE_CODE_EFFORT_LEVEL:-} ;;
esac
T0=$(date -Is)
echo "[$T0] start agent=$AGENT harness=$HARNESS version=$VERSION sandbox=$SANDBOX" >> $REC/agent_launcher.log
sandbox_exec bash $RAGENG_ROOT/agents/harness/$HARNESS.sh > $REC/agent_trace.jsonl 2>> $REC/agent_stderr.log
CODE=$?
cat > $REC/agent_meta.json <<EOF
{"agent": "$AGENT", "harness": "$HARNESS", "harness_version": "$VERSION", "provider": "$PROVIDER",
 "model": "$MODEL", "reasoning_effort": "$EFFORT", "reprompt": $REPROMPT, "sandbox": "$SANDBOX",
 "start": "$T0", "end": "$(date -Is)", "exit_code": $CODE, "deadline": "$(date -d @$(cat $RUN/task/.deadline) -Is)"}
EOF
# session segments, reprompts, crashes, usage-limit errors -> agent_meta.json "session_stats" (judge evidence)
$RAG_BASE_PY $RAGENG_ROOT/traces/agent_segments.py $REC/agent_stderr.log $REC/agent_trace.jsonl $REC/agent_meta.json \
  >> $REC/agent_launcher.log 2>&1 || echo "[$(date -Is)] agent_segments.py failed" >> $REC/agent_launcher.log
rm -f $RUN/home/.prompt.txt
# flush first: `cp` of files just written on Lustre can take unflushed data for holes (all-zero copies)
sync -f $RUN/home 2>/dev/null || sync
[ "$HARNESS" = codex ] && cp -r $RUN/home/.codex/sessions $REC/agent_sessions 2>/dev/null
[ "$HARNESS" = claude_code ] && cp -r $RUN/home/.claude/projects $REC/agent_sessions 2>/dev/null
[ "$HARNESS" = deepseek_harness ] && cp -r $RUN/home/.dsh/sessions $REC/agent_sessions 2>/dev/null
[ "$HARNESS" = opencode ] && cp -r $RUN/home/.local/share/opencode $REC/agent_sessions 2>/dev/null
exit "$CODE"
