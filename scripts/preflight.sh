#!/bin/bash
# Preflight checks inside a Slurm job, after vLLM and the gateway are up. Any failure -> exit 3
# (the orchestrator records a platform failure; no agent time is spent).
#   preflight.sh build <run_id> <agent>     before the agent starts (LLM_BASE_URL = build gateway)
# Writes <records>/preflight.json: {"stage", "ok", "checks": {name: {"ok", "detail"}}}.
set -uo pipefail
source "${RAGENG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/configs/pilot.env"
source $RAGENG_ROOT/scripts/sandbox.sh
STAGE=$1
PF_LINES=$(mktemp /tmp/rageng-preflight.XXXXXX)
check() {  # check <name> <cmd...>: records ok/fail and the last lines of output
  local name=$1; shift
  local out; out=$("$@" 2>&1); local rc=$?
  $RAG_BASE_PY -c "import json,sys; print(json.dumps({'name': sys.argv[1], 'ok': sys.argv[2] == '0', 'detail': sys.argv[3][-400:]}))" \
    "$name" "$rc" "$(echo "$out" | tail -3)" >> $PF_LINES
  echo "PREFLIGHT $name rc=$rc $(echo "$out" | tail -1 | cut -c1-200)"
}
write_json() {  # write_json <path>: summary file; returns 0 iff every check passed
  $RAG_BASE_PY - "$1" "$STAGE" "$PF_LINES" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[3])]
ok = all(r["ok"] for r in rows)
json.dump({"stage": sys.argv[2], "ok": ok, "checks": {r["name"]: {"ok": r["ok"], "detail": r["detail"]} for r in rows}},
          open(sys.argv[1], "w"), indent=1)
sys.exit(0 if ok else 1)
PY
}

chat() {  # chat <base_url>: one generator call through a gateway
  curl -sf --max-time 60 "$1/chat/completions" -H 'Content-Type: application/json' -d '{"model":"generator","max_tokens":8,
    "messages":[{"role":"user","content":"Reply with the word OK."}],"chat_template_kwargs":{"enable_thinking":false}}' \
    | $RAG_BASE_PY -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
}

if [ "$STAGE" = build ]; then
  RUN_ID=$2; AGENT=$3; RUN=$RUNS_DIR/$RUN_ID; REC=$RECORDS_DIR/$RUN_ID
  PROVIDER=$($RAG_BASE_PY -c "import json;print(json.load(open('$RAGENG_ROOT/agents/registry.json'))['$AGENT']['provider'])")
  HARNESS=$($RAG_BASE_PY -c "import json;print(json.load(open('$RAGENG_ROOT/agents/registry.json'))['$AGENT']['harness'])")
  HARNESS_PATH=$($RAG_BASE_PY -c "import json;print(':'.join('$RAGENG_ROOT/'+h['bin'] for h in json.load(open('$RAGENG_ROOT/agents/registry.json'))['_harnesses'].values()))")
  VERSION_CMD=$($RAG_BASE_PY -c "import json;print(json.load(open('$RAGENG_ROOT/agents/registry.json'))['_harnesses']['$HARNESS']['version'])")
  check disk_free_gb bash -c "g=\$(df -BG --output=avail $RAGENG_ROOT | tail -1 | tr -dc 0-9); echo \${g}G; [ \$g -gt 50 ]"
  check generator_via_gateway chat "$LLM_BASE_URL"
  PROVIDER_CHECK=(bash -c "source $RAGENG_ROOT/llm/$PROVIDER/env.sh >/dev/null
    url=\$($RAG_BASE_PY -c \"import json;print(json.load(open('$RAGENG_ROOT/agents/registry.json'))['_providers']['$PROVIDER']['models_url'])\")
    key=\$($RAG_BASE_PY -c \"import json;print(json.load(open('$RAGENG_ROOT/agents/registry.json'))['_providers']['$PROVIDER']['keys'][0])\")
    code=\$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 -H \"Authorization: Bearer \${!key}\" \$url); echo http=\$code; [ \$code = 200 ]")
  check provider_api "${PROVIDER_CHECK[@]}"
  # sandbox as the agent sees it: GPU, CLI, hidden paths, hidden host processes
  SB_HOME=$RUN/home; SB_PWD=$RUN/workspace; SB_GPU=1; SB_NET=host; mkdir -p $RUN/home/tmp $RUN/selfcheck
  SB_RW=($RUN/home $RUN/workspace $RUN/selfcheck)
  SB_RO=($RUN/task $RAGENG_ROOT/envs/rag-base $RAGENG_ROOT/envs/python $RAGENG_ROOT/envs/vllm $RAGENG_ROOT/harness
         $SOFTWARE_DIR $RAGENG_ROOT/agents/harness $RAGENG_ROOT/gateway $RAGENG_ROOT/evaluator.py $RAGENG_ROOT/tasks
         $RAGENG_ROOT/multireader/contract.py $RAGENG_ROOT/multireader/__init__.py $GENERATOR_PATH/config.json $GENERATOR_PATH/tokenizer.json $GENERATOR_PATH/tokenizer_config.json $GENERATOR_PATH/chat_template.jinja $RAGENG_ROOT/configs/sandbox $RAGENG_ROOT/selfcheck)   # = agents/run_agent.sh
  SB_ENV=(PATH=$HARNESS_PATH:$SOFTWARE_DIR/node/bin:/usr/local/bin:/usr/bin:/bin
          HOME=$RUN/home DSH_HOME=$RUN/home/tmp/dsh-preflight LANG=C.UTF-8 SLURM_JOB_ID=$SLURM_JOB_ID RAGENG_PORT_BASE=$RAGENG_PORT_BASE RAGENG_PORT_JOB=$RAGENG_PORT_JOB CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-})
  check sandbox_gpu sandbox_exec $RAG_BASE_PY -c "import torch; assert torch.cuda.is_available(); x=torch.ones(1<<20,device='cuda'); print(torch.cuda.get_device_name(0), float(x.sum()))"
  check sandbox_cli sandbox_exec bash -c "$VERSION_CMD"
  check sandbox_hides_paths sandbox_exec bash -c "for p in $RAGENG_ROOT/benchmarks $RAGENG_ROOT/records $RAGENG_ROOT/llm $RAGENG_ROOT/.env $DATA_DIR $HIDDEN_DIR $RAGENG_ROOT/eval_inputs $RAGENG_ROOT/configs/pilot.env $RAGENG_ROOT/scripts $RAGENG_ROOT/agents/run_agent.sh $RAGENG_ROOT/agents/registry.json $RAGENG_ROOT/judges $RAGENG_ROOT/slurm $RAGENG_ROOT/gpu_pool $(ls -d $RUNS_DIR/* | grep -v "^$RUN\$" | head -1); do [ -e \$p ] && { echo VISIBLE \$p; exit 1; }; done; env | grep -F $HIDDEN_DIR && { echo HIDDEN_PATH_IN_ENV; exit 1; }; grep -rlF $HIDDEN_DIR $RAGENG_ROOT/configs/sandbox && { echo HIDDEN_PATH_IN_CONFIG; exit 1; }; echo hidden"
  check sandbox_hides_host_pids sandbox_exec bash -c "n=\$(ls /proc | grep -c '^[0-9]'); echo procs=\$n; for p in /proc/[0-9]*; do tr '\\0' ' ' < \$p/cmdline 2>/dev/null; echo; done | grep -q '[v]llm serve' && { echo vllm_visible; exit 1; }; exit 0"
  # a service started the way self_check starts it (env -i + the GPU variables) must see the GPU
  check selfcheck_service_gpu sandbox_exec bash -c "env -i PATH=/usr/local/bin:/usr/bin:/bin LD_LIBRARY_PATH=\"\$LD_LIBRARY_PATH\" CUDA_VISIBLE_DEVICES=\"\$CUDA_VISIBLE_DEVICES\" $RAG_BASE_PY -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'"
  # the whole self_check chain on 3 labelled questions with the protocol example, inside the sandbox
  FAKE=$RUN/home/tmp/preflight
  # The protocol example inherits read-only directory modes from task/example.
  # Only this private copy needs writable directories for cleanup; never chmod its corpus symlink.
  [ ! -d "$FAKE/workspace" ] || chmod -R u+w "$FAKE/workspace"
  rm -rf $FAKE; mkdir -p $FAKE/task $FAKE/selfcheck
  LQ=$RUN/task/dev.jsonl; [ -f $LQ ] || LQ=$RUN/task/train.jsonl
  cp -r $RUN/task/example $FAKE/workspace; cp $RUN/task/{self_check,task.json} $FAKE/task/; head -3 $LQ > $FAKE/task/dev.jsonl
  ln -s $RUN/task/corpus $FAKE/task/corpus
  SB_ENV+=(TMPDIR=$RUN/home/tmp)
  check selfcheck_chain sandbox_exec bash -c "$FAKE/task/self_check $FAKE/task/dev.jsonl 3 && python3 -c \"
import json,glob; f=sorted(glob.glob('$FAKE/selfcheck/*/predictions.jsonl'))[-1]; p=[json.loads(l) for l in open(f)]
assert len(p)==3 and all(not x['error'] and (x['gateway']['calls']>0 or x.get('reader_gateway',{}).get('calls',0)>0) for x in p), p; print('answered', len(p))\""
  tail -n 1 $PF_LINES | grep -q '"ok": false' && cp -r $FAKE/selfcheck $REC/preflight_selfcheck 2>/dev/null
  chmod -R u+w "$FAKE/workspace"
  rm -rf $FAKE
  write_json $REC/preflight.json && exit 0 || exit 3

fi
