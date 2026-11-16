#!/bin/bash
# Create runs/<run_id>/{task,workspace,home} and records/<run_id>/ (login node).
# Usage: scripts/new_run.sh <run_id> <task> [workspace_template_dir]
#   workspace_template_dir: pre-fill the workspace (baselines); omitted for agent runs.
set -euo pipefail
source "${RAGENG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/configs/pilot.env"
RUN_ID=$1; TASK_NAME=$2; TEMPLATE=${3:-}
RUN=$RUNS_DIR/$RUN_ID; REC=$RECORDS_DIR/$RUN_ID; PUB=$DATA_DIR/$TASK_NAME/public
[ -e "$RUN" ] && { echo "run $RUN exists"; exit 1; }
[ -f "$PUB/train.jsonl" ] || { echo "missing $PUB (run tasks.prepare)"; exit 1; }
mkdir -p $RUN/task $RUN/workspace $RUN/home $REC
cp -al $PUB/corpus $RUN/task/corpus   # hard links: same files as the data version (read-only), no copy
cp $PUB/train.jsonl $RUN/task/
[ -f $PUB/dev.jsonl ] && cp $PUB/dev.jsonl $RUN/task/
# The paper's protocol (fixed-evidence-v1): the system returns evidence (POST /retrieve), the evaluator answers.
PROTOCOL=${RAGENG_EVAL_PROTOCOL:-fixed-evidence-v1}
[ "$PROTOCOL" = fixed-evidence-v1 ] || { echo "Unsupported protocol $PROTOCOL (this release ships fixed-evidence-v1 only)"; exit 2; }
EXAMPLE=example_fixed; README_TEMPLATE=README_fixed.md; PROMPT_TEMPLATE=prompt_fixed.md
cp -r $RAGENG_ROOT/materials/$EXAMPLE $RUN/task/example
cp $RAGENG_ROOT/materials/timer.sh $RAGENG_ROOT/selfcheck/self_check $RUN/task/
# absolute paths of this installation (the sandbox sees no configs/pilot.env)
sed -i "s#@RAGENG_ROOT@#$RAGENG_ROOT#g" $RUN/task/self_check
sed -i "s#@RAG_BASE_PY@#$RAG_BASE_PY#g" $RUN/task/example/submission/start.sh
cp $RAGENG_ROOT/tasks/$TASK_NAME/info.json $RUN/task/task.json
$RAG_BASE_PY - "$RUN/task/task.json" "$PROTOCOL" <<'PY'
import json,sys
p=sys.argv[1];d=json.load(open(p));d['evaluation_protocol']=sys.argv[2];json.dump(d,open(p,'w'),indent=2)
PY
chmod +x $RUN/task/timer.sh $RUN/task/self_check
if [ -n "$TEMPLATE" ]; then
  cp -r $TEMPLATE/. $RUN/workspace/
  cp $RUN/task/task.json $RUN/workspace/submission/task.json
  [ -f $TEMPLATE/../ragcore.py ] && cp $TEMPLATE/../ragcore.py $RUN/workspace/submission/
  sed -i "s#@RAG_BASE_PY@#$RAG_BASE_PY#g" $RUN/workspace/submission/*.sh
  rm -f $RUN/workspace/models.txt
  for m in $(cat $TEMPLATE/models.txt 2>/dev/null); do mkdir -p $RUN/workspace/models; cp -r $RAGENG_ROOT/envs/models/$m $RUN/workspace/models/; done
fi

set -a; source $RAGENG_ROOT/configs/pilot.env; set +a
(cd $RAGENG_ROOT && $RAG_BASE_PY scripts/render.py materials/$README_TEMPLATE $RUN $TASK_NAME) > $RUN/task/README.md
(cd $RAGENG_ROOT && $RAG_BASE_PY scripts/render.py materials/$PROMPT_TEMPLATE $RUN $TASK_NAME) > $REC/prompt.txt
# .deadline is written by build.sbatch when the agent starts (ignored by the manifest).
$VLLM_PY $RAGENG_ROOT/scripts/freeze.py hash $RUN/task $REC/task_manifest.json
chmod -R a-w $RUN/task/corpus $RUN/task/*.jsonl $RUN/task/*.json $RUN/task/example $RUN/task/README.md \
  $RUN/task/self_check $RUN/task/timer.sh
cat > $REC/run.json <<EOF
{"run_id": "$RUN_ID", "task": "$TASK_NAME", "template": "$TEMPLATE", "created": "$(date -Is)",
 "build_seconds": $BUILD_SECONDS, "generator": "$GENERATOR_PATH", "evaluation_protocol": "$PROTOCOL", "data_version": "$DATA_VERSION"}
EOF
echo "created $RUN"
