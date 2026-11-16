#!/bin/bash
# Evaluate one frozen system on one question file, inside an eval job (vLLM already running).
#   eval_system.sh <name> <workspace> <corpus_dir> <questions.jsonl> <out_dir>
# Starts its own metered gateway (unix socket when EVAL_NET=none), runs evaluator.py with the service
# launched through scripts/run_service.sh, writes <out_dir>/{predictions.jsonl, gateway_eval.jsonl, service.log}.
# EVAL_WALL_SECONDS (set by slurm/run.sbatch for formal runs; default 0 = none): wall-clock limit of the whole
# evaluation, startup included; questions not answered by then score 0 (evaluator.py --eval-seconds).
set -uo pipefail
source "${RAGENG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/configs/pilot.env"
NAME=$1; WS=$2; CORPUS=$3; Q=$4; OUT=$5
mkdir -p $OUT
SOCK=/tmp/rg-${SLURM_JOB_ID:-0}-$NAME; rm -rf $SOCK; mkdir -p $SOCK
SCRATCH=/tmp/rg-${SLURM_JOB_ID:-0}-$NAME-scratch; rm -rf $SCRATCH
TOKEN=$(head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n')
umask 077; echo $TOKEN > $SOCK/.gw_token; echo $TOKEN > $OUT/.ev_token; umask 022
GW_ARGS=(--mode eval --upstream http://127.0.0.1:$VLLM_PORT --log $OUT/gateway_eval.jsonl
         --calls $BUDGET_CALLS --tokens $BUDGET_TOKENS --seconds $BUDGET_SECONDS --admin-token-file $SOCK/.gw_token)
if [ "$EVAL_NET" = none ]; then
  GW_ARGS+=(--uds $SOCK/gateway.sock); GW_CLIENT=(--gateway-uds $SOCK/gateway.sock)
  SVC_CLIENT=(--service-uds $SOCK/service.sock); GW_FOR_SERVICE=$SOCK
else
  GWP=$((SERVICE_PORT + 10)); GW_ARGS+=(--port $GWP); GW_CLIENT=(--gateway http://127.0.0.1:$GWP)
  SVC_CLIENT=(); GW_FOR_SERVICE=http://127.0.0.1:$GWP
fi
$VLLM_PY $RAGENG_ROOT/gateway/gateway.py "${GW_ARGS[@]}" > $OUT/gateway.log 2>&1 &
GW=$!
trap 'kill $GW 2>/dev/null; rm -rf $SOCK $SCRATCH' EXIT
for i in $(seq 60); do
  if [ "$EVAL_NET" = none ]; then curl -sf --unix-socket $SOCK/gateway.sock http://local/health >/dev/null && break
  else curl -sf http://127.0.0.1:$GWP/health >/dev/null && break; fi
  sleep 1
done
$VLLM_PY $RAGENG_ROOT/evaluator.py --response-mode "${RAGENG_RESPONSE_MODE:-answer}" --workspace $WS --questions $Q --out $OUT/predictions.jsonl \
  "${GW_CLIENT[@]}" "${SVC_CLIENT[@]}" --admin-token-file $OUT/.ev_token \
  --service-cmd "bash $RAGENG_ROOT/scripts/run_service.sh $WS $CORPUS $SCRATCH $SERVICE_PORT $EVAL_NET $GW_FOR_SERVICE" \
  --corpus-dir $CORPUS --scratch-dir $SCRATCH --service-log $OUT/service.log --port $SERVICE_PORT \
  --answer-timeout $BUDGET_SECONDS --eval-seconds ${EVAL_WALL_SECONDS:-0} \
  --startup-timeout ${LINEARRAG_STARTUP_TIMEOUT:-1200}
