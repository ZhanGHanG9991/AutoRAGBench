#!/bin/bash
# Run the judges (judges/judges.json, or the ones named) on one run, in parallel, each session in its own sandboxed
# copy of the evidence; a session that writes no complete verdict is run once more. A judge with "votes": N runs N
# independent sessions (<judge>@1..N) and judges/votes.py writes the per-field majority. Called by slurm/run.sbatch
# (judge gate, before the test evaluation) and by slurm/judge.sbatch (manual re-judging).
#   run_judges.sh <run_id> [judge ...]
# Outputs in records/<run_id>/: judgement_<judge>.json (+ judgement_<judge>.vote<k>.json),
# judges/<session>.trace.jsonl, judges/<session>.stderr.
set -uo pipefail
source "${RAGENG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/configs/pilot.env"
source $RAGENG_ROOT/scripts/sandbox.sh
RUN_ID=$1; shift
RUN=$RUNS_DIR/$RUN_ID; REC=$RECORDS_DIR/$RUN_ID
J=$RAGENG_ROOT/judges
JUDGES=${*:-$($RAG_BASE_PY -c "import json;print(' '.join(k for k in json.load(open('$J/judges.json')) if not k.startswith('_')))")}
PROVIDER=$($RAG_BASE_PY -c "import json;print(json.load(open('$J/judges.json'))['_provider'])")
TIMEOUT=$($RAG_BASE_PY -c "import json;print(json.load(open('$J/judges.json'))['_timeout_seconds'])")
source $RAGENG_ROOT/llm/$PROVIDER/env.sh >/dev/null
KEY=$($RAG_BASE_PY -c "import json;print(json.load(open('$RAGENG_ROOT/agents/registry.json'))['_providers']['$PROVIDER']['keys'][0])")
mkdir -p $REC/judges
BASE=$REC/judges/evidence
$RAG_BASE_PY $J/prepare_sandbox.py $RUN $REC $BASE
sync -f $BASE 2>/dev/null || sync
judge_one() {  # judge_one <session>: "<judge>" or "<judge>@<k>" (one vote of a best-of-N judge)
  local u=$1 j=${1%@*} out=judgement_${1%@*}.json D=$REC/judges/${1/@/.vote}
  [ "$u" != "$j" ] && out=judgement_$j.vote${u#*@}.json
  # copy by reading the data (shutil, sendfile): `cp -r` of files just written on Lustre can take unflushed data
  # for holes and produce full-size all-zero copies (seen with 10 parallel copies of a fresh evidence dir)
  rm -rf $D; $RAG_BASE_PY -c "import shutil, sys; shutil.copytree(sys.argv[1], sys.argv[2], symlinks=True)" $BASE $D
  mkdir -p $D/.home/.codex $D/.home/tmp
  cp $RAGENG_ROOT/llm/$PROVIDER/codex_home/models.json $D/.home/.codex/
  sed "s#^model_catalog_json = .*#model_catalog_json = \"$D/.home/.codex/models.json\"#" \
    $RAGENG_ROOT/llm/$PROVIDER/codex_home/config.toml > $D/.home/.codex/config.toml
  (cd $RAGENG_ROOT && $RAG_BASE_PY judges/render_prompt.py $j $REC) > $D/.home/prompt.txt
  local SB_HOME=$D/.home SB_PWD=$D SB_GPU=0 SB_NET=host
  local SB_RW=($D) SB_RO=($RAGENG_ROOT/harness $SOFTWARE_DIR)
  local SB_ENV=(PATH=$RAGENG_ROOT/harness/codex/bin:$SOFTWARE_DIR/node/bin:/usr/local/bin:/usr/bin:/bin HOME=$D/.home
          TMPDIR=$D/.home/tmp CODEX_HOME=$D/.home/.codex LANG=C.UTF-8 TERM=dumb "$KEY=${!KEY}")
  echo "[$(date -Is)] judge $u start" >> $REC/judges/run.log
  sandbox_exec bash -c "timeout -s TERM -k 30 $TIMEOUT codex exec --skip-git-repo-check \
      --dangerously-bypass-approvals-and-sandbox --json -C $D \"\$(cat $D/.home/prompt.txt)\" < /dev/null" \
    > $REC/judges/${u/@/.vote}.trace.jsonl 2> $REC/judges/${u/@/.vote}.stderr
  echo "[$(date -Is)] judge $u exit=$?" >> $REC/judges/run.log
  if [ -f $D/judgement.json ]; then cp $D/judgement.json $REC/$out
  else echo '{"error": "judge wrote no judgement.json"}' > $REC/$out; fi
  $RAG_BASE_PY $RAGENG_ROOT/traces/parse_trace.py codex $REC/judges/${u/@/.vote}.trace.jsonl \
    -o $REC/judges/${u/@/.vote}.parsed.txt >/dev/null
  rm -rf $D
}
UNITS=$($RAG_BASE_PY $J/votes.py $REC sessions $JUDGES)   # first round: every session of the judges asked for
for attempt in 1 2; do
  for u in $UNITS; do judge_one $u & done
  wait
  $RAG_BASE_PY $J/votes.py $REC resolve $JUDGES
  # judges (of those asked for) still without a verdict -> their incomplete sessions run once more
  MISSING=$($RAG_BASE_PY $J/gate.py $REC --missing | tr ' ' '\n' | grep -xF -f <(echo $JUDGES | tr ' ' '\n') | tr '\n' ' ')
  [ -z "${MISSING// /}" ] && break
  UNITS=$($RAG_BASE_PY $J/votes.py $REC units $MISSING)
  echo "[$(date -Is)] no verdict from: $MISSING (attempt $attempt); rerun: $UNITS" >> $REC/judges/run.log
done
$RAG_BASE_PY $RAGENG_ROOT/traces/sanitize.py $REC/judges
echo "JUDGES_DONE $RUN_ID $(date -Is)"
