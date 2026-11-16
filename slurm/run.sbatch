#!/bin/bash
#SBATCH --job-name=rageng-run
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --no-requeue
# One job per run: build (agent in the sandbox until the deadline) -> freeze -> judge gate (agents: the LLM judges
# read the build evidence; any flag ends the run without a test evaluation) -> fresh generator -> evaluate the frozen
# system on the hidden test -> score. Jobs of any number of runs and experiments run side by side: each has its
# own vLLM, gateways, ports (RAGENG_PORT_BASE) and directories; the hidden test store is shared read-only.
#   run.sbatch <run_id> <system> [--eval-only]
# Every completed run ends with records/<run_id>/run_result.json ({"outcome": "evaluated_multireader" | "judge_flagged" |
# "workspace_over_limit"}). The test lives in $HIDDEN_DIR/<version>/<task>/, which no sandbox binds; evidence, test
# questions and reader predictions go to $HIDDEN_DIR/multireader/<run_id>/, records/<run_id>/ only gets text-free scores.
# Exit 3 = platform failure before the agent started, twice (resubmit the whole run); exit 4 = platform failure in the
# judge gate or the evaluation (resubmit with --eval-only: the workspace is frozen; judges without a verdict run
# again); exit 2 = refused (configuration); exit 5 = agent failure, requires a fresh run.
# RAGENG_SANDBOXED_BUILD: the agent runs only through sandbox_exec (agents/run_agent.sh); preflight fails closed if
# hidden paths (hidden test store included) are visible. See scripts/unsafe_builds.sh.
set -uo pipefail
source "${RAGENG_ROOT:?export RAGENG_ROOT=<repository root> before sbatch}/configs/pilot.env"
source $RAGENG_ROOT/scripts/services.sh
RUN_ID=$1; SYSTEM=$2; MODE=${3:-full}
RUN=$RUNS_DIR/$RUN_ID; REC=$RECORDS_DIR/$RUN_ID
exec > >(tee -a $REC/run_job.log) 2>&1
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
TASK=$($RAG_BASE_PY -c "import json;print(json.load(open('$REC/run.json'))['task'])")
PROTOCOL=$($RAG_BASE_PY -c "import json; print(json.load(open('$REC/run.json')).get('evaluation_protocol','answer-v1'))")
KIND=$($RAG_BASE_PY -c "import json;print('agent' if '$SYSTEM' in json.load(open('$RAGENG_ROOT/agents/registry.json')) else 'baseline')")
HID=$HIDDEN_DIR/$DATA_VERSION/$TASK
OUT=$HIDDEN_DIR/results/$DATA_VERSION/$RUN_ID
echo "JOB_START $(date -Is) job=$SLURM_JOB_ID host=$(hostname) run=$RUN_ID system=$SYSTEM kind=$KIND mode=$MODE data=$DATA_VERSION sandbox=$SANDBOX"
[ "$DATA_VERSION" != pilot ] || { echo "REFUSED: pilot data has no hidden store (use build.sbatch + eval.sbatch)"; exit 2; }
[ -r $HID/test_questions.jsonl ] && [ -r $HID/test_gold.jsonl ] || { echo "MISSING hidden test $HID"; exit 3; }
[ "$KIND" = baseline ] || [ "$SANDBOX" != none ] || { echo "REFUSED: agents need a sandbox when the test is on the cluster"; exit 2; }
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv -l 30 > $REC/run_gpu.csv 2>&1 &
GPU_MON=$!
trap 'kill $GPU_MON $HFF 2>/dev/null; declare -F svc_stop >/dev/null && svc_stop' EXIT
fail() { echo "PLATFORM_FAILED $1"; exit 4; }
valid_json() { "$RAG_BASE_PY" -c 'import json,sys; assert isinstance(json.load(open(sys.argv[1])),dict)' "$1" 2>/dev/null; }
scan_json() {
  local target=$1; shift
  "$@" > "$target.tmp" && valid_json "$target.tmp" && mv "$target.tmp" "$target" || fail "audit:$target"
}
HFF=
export GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)

# ---------------------------------------------------------------- build
if [ "$MODE" != --eval-only ] && [ "$KIND" = agent ]; then
  build_services() {  # generator, build gateway, HF filter, preflight; returns 1 on a platform failure
    svc_init $REC/build_services
    local t0=$(date +%s); start_vllm
    wait_http "http://127.0.0.1:$VLLM_PORT/health" 1200 || { echo "VLLM_FAILED"; return 1; }
    echo "VLLM_READY $(( $(date +%s)-t0 ))s"
    start_gateway build $GATEWAY_PORT $REC/gateway_build.jsonl
    wait_http "http://127.0.0.1:$GATEWAY_PORT/health" 60 || { echo "GATEWAY_FAILED"; return 1; }
    export LLM_BASE_URL=http://127.0.0.1:$GATEWAY_PORT/v1
    # Hugging Face for the agent: models yes, datasets no (scripts/hf_filter.py)
    # shared cache of immutable (commit-addressed) model files for all runs: faster, fewer mirror errors at start
    $RAG_BASE_PY $RAGENG_ROOT/scripts/hf_filter.py --port $HF_FILTER_PORT --upstream $HF_MIRROR --log $REC/hf_requests.jsonl \
      --cache $RAGENG_ROOT/cache/hf_filter > $REC/build_services/hf_filter.log 2>&1 &
    HFF=$!
    wait_http "http://127.0.0.1:$HF_FILTER_PORT/api/models/BAAI/bge-small-en-v1.5" 60 || echo "HF_FILTER_UPSTREAM_UNREACHABLE (agent HF downloads may fail)"
    export HF_ENDPOINT_AGENT=http://127.0.0.1:$HF_FILTER_PORT
    bash $RAGENG_ROOT/scripts/preflight.sh build $RUN_ID $SYSTEM || { echo "PREFLIGHT_FAILED"; return 1; }
  }
  # a platform failure before the agent starts is retried once in this job (the agent has not seen anything yet)
  for attempt in 1 2; do
    build_services && break
    svc_stop; kill $HFF 2>/dev/null; HFF=
    [ $attempt = 2 ] && { echo "BUILD_SERVICES_FAILED twice"; exit 3; }
    echo "BUILD_SERVICES_RETRY $(date -Is)"; sleep 60
  done
  echo "{\"event\": \"agent_start\", \"ts\": $(date +%s)}" >> $REC/gateway_build.jsonl
  chmod u+w $RUN/task
  echo $(( $(date +%s) + BUILD_SECONDS )) > $RUN/task/.deadline
  chmod a-w $RUN/task/.deadline $RUN/task
  echo "AGENT_START $(date -Is) deadline=$(date -d @$(cat $RUN/task/.deadline) -Is)"
  bash $RAGENG_ROOT/agents/run_agent.sh $RUN_ID $SYSTEM
  AGENT_CODE=$?
  echo "$AGENT_CODE" > "$REC/agent_exit_code"
  echo "AGENT_END $(date -Is)"
  kill $HFF 2>/dev/null; HFF=

  $VLLM_PY $RAGENG_ROOT/scripts/freeze.py freeze $RUN $REC || fail "build evidence"
  grep -c '"event": "call"' $REC/gateway_build.jsonl | sed 's/^/BUILD_GENERATOR_CALLS /'
  svc_stop
  # build analysis before anything of the test exists in this job
  $RAG_BASE_PY $RAGENG_ROOT/traces/sanitize.py $REC --extra-env VLLM_API_KEY || fail "build evidence"
  HARNESS=$($RAG_BASE_PY -c "import json;print(json.load(open('$RAGENG_ROOT/agents/registry.json'))['$SYSTEM']['harness'])")
  $RAG_BASE_PY $RAGENG_ROOT/traces/parse_trace.py $HARNESS $REC/agent_trace.jsonl -o $REC/solve_parsed.txt --usage $REC/agent_usage.json || fail "build evidence"
  $RAG_BASE_PY $RAGENG_ROOT/audit/provenance.py $RUN/workspace $RUN/task/corpus/corpus.jsonl $REC/gateway_build.jsonl $REC/provenance.json || fail "build evidence"
  $RAG_BASE_PY $RAGENG_ROOT/audit/scan_workspace.py $RUN/workspace $REC/workspace_scan.json $RUN/home || fail "build evidence"
  echo "BUILD_DONE $(date -Is)"
elif [ ! -f $REC/workspace_manifest.json ]; then
  [ "$KIND" = baseline ] || { echo "NOT_FROZEN: --eval-only needs a frozen workspace"; exit 2; }
  # baselines with an index (naive_bm25, reference_rag) build it now, before freezing, so that million-passage corpora
  # meet the 20-minute service start limit (baselines/ragcore.py)
  if [ -x $RUN/workspace/submission/build.sh ]; then
    echo "BASELINE_BUILD_START $(date -Is)"
    env -i PATH=/usr/local/bin:/usr/bin:/bin HOME=$RUN/home TMPDIR=$RUN/home CORPUS_DIR=$RUN/task/corpus \
      CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-} HF_HUB_OFFLINE=1 \
      bash $RUN/workspace/submission/build.sh > $REC/baseline_build.log 2>&1 || { echo "BASELINE_BUILD_FAILED"; tail -5 $REC/baseline_build.log; exit 3; }
    echo "BASELINE_BUILD_END $(date -Is) $(cat $RUN/workspace/index/built.json)"
  fi
  $VLLM_PY $RAGENG_ROOT/scripts/freeze.py freeze $RUN $REC || fail "build evidence"
fi

# Recheck frozen evidence on eval-only recovery; a partial build audit cannot be reused.
if [ "$KIND" = agent ]; then
  for evidence in freeze workspace_manifest provenance workspace_scan agent_meta; do
    valid_json "$REC/$evidence.json" || fail "missing/invalid $evidence"
  done
  # Timeout at the declared budget is normal; an API/CLI crash invalidates this run.
  AGENT_CODE=$($RAG_BASE_PY -c "import json;print(json.load(open('$REC/agent_meta.json'))['exit_code'])") || fail agent_meta
  if [ "$PROTOCOL" = fixed-evidence-v1 ]; then
    "$RAG_BASE_PY" "$RAGENG_ROOT/multireader/build_status.py" "$REC/agent_meta.json" || {
      echo "AGENT_FAILED exit=$AGENT_CODE; abnormal final segment; fresh run required"; exit 5;
    }
  else
    case $AGENT_CODE in 0|124) ;; *) echo "AGENT_FAILED exit=$AGENT_CODE; fresh run required"; exit 5 ;; esac
  fi
fi

# ---------------------------------------------------------------- workspace limit
# The frozen workspace (index, models, .venv, code: every file, apparent size, scripts/freeze.py) may not exceed
# WORKSPACE_LIMIT_GB; a larger deliverable is not judged or evaluated and scores the naive baseline.
WS_BYTES=$($RAG_BASE_PY -c "import json;print(json.load(open('$REC/freeze.json'))['workspace_bytes'])" 2>/dev/null) || fail freeze_metadata
if [ "$KIND" = agent ] && [ -n "$WS_BYTES" ] && [ "$WS_BYTES" -gt $(( WORKSPACE_LIMIT_GB * 1000000000 )) ]; then
  echo "{\"outcome\": \"workspace_over_limit\", \"workspace_bytes\": $WS_BYTES, \"limit_gb\": $WORKSPACE_LIMIT_GB, \"job\": \"$SLURM_JOB_ID\", \"time\": \"$(date -Is)\"}" > $REC/run_result.json.tmp && valid_json "$REC/run_result.json.tmp" && mv "$REC/run_result.json.tmp" "$REC/run_result.json" || fail result_write
  echo "NOT_EVALUATED: workspace $WS_BYTES bytes > ${WORKSPACE_LIMIT_GB} GB"; echo "JOB_END $(date -Is)"; exit 0
fi

# ---------------------------------------------------------------- judge gate (agents)
# The judges read only build evidence (judges/prepare_sandbox.py), never the test. Every judge must return a clean
# verdict before the test is evaluated; a flag ends the run here (a human can overrule a judge with
# records/<run_id>/judge_override.json and evaluate with --eval-only). Judges without a verdict = platform failure.
run_result() {  # run_result <outcome> [json fields]: the marker the GPU pool waits for
  echo "{\"outcome\": \"$1\"${2:+, $2}, \"job\": \"$SLURM_JOB_ID\", \"time\": \"$(date -Is)\"}" > $REC/run_result.json.tmp && valid_json "$REC/run_result.json.tmp" && mv "$REC/run_result.json.tmp" "$REC/run_result.json" || fail result_write
}
if [ "$KIND" = agent ]; then
  # judge evidence: does test question text appear in what the agent wrote? (outside the sandbox; paths only)
  valid_json "$REC/leak_scan_build.json" || scan_json "$REC/leak_scan_build.json" $RAG_BASE_PY $RAGENG_ROOT/audit/leak_scan.py $HID/test_questions.jsonl \
    $RUN/task/train.jsonl $RUN/workspace $RUN/home $RUN/selfcheck
  echo "LEAK_SCAN_BUILD $(cut -c1-300 $REC/leak_scan_build.json)"
  GATE=$($RAG_BASE_PY -c "import json;print(json.load(open('$REC/judge_gate.json'))['outcome'])" 2>/dev/null)
  if [ "$GATE" != passed ]; then
    MISSING=$($RAG_BASE_PY $RAGENG_ROOT/judges/gate.py $REC --missing)
    echo "JUDGE_START $(date -Is) judges=${MISSING:-none}"
    [ -n "$MISSING" ] && bash $RAGENG_ROOT/judges/run_judges.sh $RUN_ID $MISSING > $REC/judges_job.log 2>&1
    $RAG_BASE_PY $RAGENG_ROOT/judges/gate.py $REC | sed 's/^/JUDGE_GATE /'
    GATE=$($RAG_BASE_PY -c "import json;print(json.load(open('$REC/judge_gate.json'))['outcome'])" 2>/dev/null)
  fi
  echo "JUDGE_GATE_RESULT $(date -Is) $GATE"
  case $GATE in
    passed) ;;
    flagged) run_result judge_flagged "\"flagged\": $($RAG_BASE_PY -c "import json;print(json.dumps(json.load(open('$REC/judge_gate.json'))['flagged']))")"
             echo "NOT_EVALUATED: judge gate flagged the run"; echo "JOB_END $(date -Is)"; exit 0 ;;
    *) echo "JUDGE_INCOMPLETE"; exit 4 ;;
  esac
fi

# ---------------------------------------------------------------- clean slate
# The agent's sandbox (PID namespace) is gone with its main process; nothing but our own processes may hold the GPU.
for i in $(seq 60); do
  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
  [ "$n" = 0 ] && break; sleep 3
done
[ "$n" = 0 ] || { echo "GPU_NOT_FREE: $(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)"; exit 4; }
bash $RAGENG_ROOT/scripts/unsafe_builds.sh > $REC/unsafe_jobs_at_eval.txt 2>&1 || true

# Fixed-evidence evaluation (the paper's protocol): extract evidence once on the hidden test, seal it, five readers.
if [ "$PROTOCOL" = fixed-evidence-v1 ]; then
  # reference systems have no judge gate (multireader/run.py reads "agent" from adapter.json; absent = agent)
  [ "$KIND" = baseline ] && [ ! -f "$REC/adapter.json" ] && echo "{\"agent\": false, \"baseline\": \"$SYSTEM\"}" > "$REC/adapter.json"
  (cd "$RAGENG_ROOT" && "$VLLM_PY" -m multireader.run --run-id "$RUN_ID") || fail multireader_pipeline
  echo "JOB_END $(date -Is)"
  exit 0
fi
echo "REFUSED: evaluation protocol $PROTOCOL is not part of this release (fixed-evidence-v1 only)"; exit 2
