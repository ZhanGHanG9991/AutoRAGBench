#!/bin/bash
# Start a frozen submission for evaluation, isolated by scripts/sandbox.sh.
#   run_service.sh <workspace> <corpus_dir> <scratch_dir> <port> <net:host|none> <gateway: url or sock dir>
# net=host: the service reaches the gateway at <gateway>/v1 over TCP.
# net=none: no network; scripts/relay.py bridges 127.0.0.1 ports to unix sockets in <gateway> (a dir).
# The service sees only PORT, LLM_BASE_URL, CORPUS_DIR, SCRATCH_DIR, HOME, TMPDIR, PATH, CUDA_VISIBLE_DEVICES.
# TMPDIR points into SCRATCH_DIR: under singularity --containall /tmp is a 64 MB tmpfs.
set -uo pipefail
source "${RAGENG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/configs/pilot.env"
source $RAGENG_ROOT/scripts/sandbox.sh
WS=$1; CORPUS=$2; SCRATCH=$3; PORT=$4; NET=$5; GW=$6
mkdir -p $SCRATCH/tmp
SB_HOME=$SCRATCH; SB_PWD=$WS; SB_GPU=1; SB_NET=$NET
SB_RO=($WS $CORPUS $RAGENG_ROOT/envs/rag-base $RAGENG_ROOT/envs/python $RAGENG_ROOT/scripts/relay.py)
SB_RW=($SCRATCH)
LLM_PORT=$((PORT + 50))
if [ "$NET" = none ]; then LLM=http://127.0.0.1:$LLM_PORT/v1; SB_RW+=($GW); else LLM=$GW/v1; fi
SB_ENV=(PORT=$PORT LLM_BASE_URL=$LLM CORPUS_DIR=$CORPUS SCRATCH_DIR=$SCRATCH HOME=$SCRATCH
        TMPDIR=$SCRATCH/tmp PATH=/usr/local/bin:/usr/bin:/bin LANG=C.UTF-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-})
if [ "$NET" = none ]; then
  exec_cmd=($RAG_BASE_PY $RAGENG_ROOT/scripts/relay.py --sock-dir $GW --port $PORT --llm-port $LLM_PORT -- bash submission/start.sh)
else
  exec_cmd=(bash submission/start.sh)
fi
sandbox_exec "${exec_cmd[@]}"
