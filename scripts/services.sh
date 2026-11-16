# Sourced inside Slurm jobs (bash). Starts the evaluator-owned services:
#   vLLM (fixed generator) and the metering gateway, each under a restart loop.
# Requires configs/pilot.env to be sourced first. Logs go to $SVC_LOG_DIR.

svc_init() {
  SVC_LOG_DIR=$1; mkdir -p "$SVC_LOG_DIR"
  export VLLM_API_KEY=$(head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n')
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  export HF_HOME=$RAGENG_ROOT/envs/.hf-home VLLM_CACHE_ROOT=$RAGENG_ROOT/envs/.vllm-cache
  export VLLM_USE_FLASHINFER_SAMPLER=0      # avoid FlashInfer JIT (needs ninja/nvcc) at startup
  SVC_PIDS=()
}

# supervise <name> <cmd...>: rerun the command if it dies, log every restart.
supervise() {
  local name=$1; shift
  (
    trap 'kill $child 2>/dev/null; exit 0' TERM INT
    n=0
    while true; do
      echo "[$(date -Is)] start $name (#$n)" >> "$SVC_LOG_DIR/supervisor.log"
      "$@" >> "$SVC_LOG_DIR/$name.log" 2>&1 &
      child=$!
      wait $child
      echo "[$(date -Is)] $name exited code=$? (#$n)" >> "$SVC_LOG_DIR/supervisor.log"
      n=$((n+1)); sleep 5
    done
  ) &
  SVC_PIDS+=($!)
}

start_vllm() {
  supervise vllm env PATH="$RAGENG_ROOT/envs/vllm/bin:$PATH" "$RAGENG_ROOT/envs/vllm/bin/vllm" serve "$GENERATOR_PATH" \
    --served-model-name "$GENERATOR_NAME" --api-key "$VLLM_API_KEY" \
    --host 127.0.0.1 --port "$VLLM_PORT" \
    --gpu-memory-utilization "$VLLM_GPU_UTIL" --max-model-len "$VLLM_MAX_LEN" --max-num-seqs "$VLLM_MAX_SEQS" \
    --limit-mm-per-prompt '{"image":0,"video":0}' ${VLLM_EXTRA_ARGS:-}
}

wait_http() {  # wait_http <url> <timeout_s>
  local t0=$(date +%s)
  until curl -sf -o /dev/null "$1"; do
    (( $(date +%s) - t0 > $2 )) && { echo "timeout waiting for $1"; return 1; }
    sleep 3
  done
}

# start_gateway <mode> <port> <logfile>
start_gateway() {
  supervise "gateway-$1-$2" "$VLLM_PY" "$RAGENG_ROOT/gateway/gateway.py" --mode "$1" --port "$2" \
    --upstream "http://127.0.0.1:$VLLM_PORT" --log "$3" \
    --calls "$BUDGET_CALLS" --tokens "$BUDGET_TOKENS" --seconds "$BUDGET_SECONDS"
}

svc_stop() {
  for p in "${SVC_PIDS[@]}"; do kill -TERM "$p" 2>/dev/null; done
  sleep 3   # anything left over is reaped by Slurm's cgroup cleanup at job end
}
