#!/bin/bash
# Runs inside the sandbox. stdout = DeepSeek Harness `dsh --profile headless --json` events (the trace);
# stderr = launcher log. Env: PROMPT_FILE, DEADLINE (epoch), AGENT_REPROMPT (0|1), TASK_DIR, WORKSPACE,
# DSH_HOME (prepared by run_agent.sh: cordis.patch.yml only), DEEPSEEK_API_KEY.
set -uo pipefail
export DSH_TELEMETRY_DISABLED=1 DSH_PERMISSION_MODE=danger-full-access   # no approvals: the sandbox is the boundary
FLAGS=(--profile headless --json)
LAST=$HOME/tmp/dsh_last.jsonl
SID=
LAST_EXIT=0
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"   # left, reprompt_loop (reprompt policy)
run() {
  provider_refresh
  local l; l=$(left); (( l > 0 )) || return 0
  echo "[$(date -Is)] dsh $1 (left=${l}s)" >&2
  if [ -n "${PROVIDER_HEALTH_URL:-}${REPROMPT_PROVIDER_PROBE:-}" ]; then
    segment_timeout dsh "${FLAGS[@]}" "${@:2}" < /dev/null | tee "$LAST"
    LAST_EXIT=${PIPESTATUS[0]}
  else
    timeout -s TERM -k 60 $l dsh "${FLAGS[@]}" "${@:2}" < /dev/null | tee "$LAST"
    LAST_EXIT=${PIPESTATUS[0]}
  fi
  echo "[$(date -Is)] dsh $1 exit=$LAST_EXIT" >&2
  # the opening `session` event names the Session that --session-id resumes
  [ -n "$SID" ] || SID=$(head -1 $LAST | sed -n 's/.*"sessionId":"\([^"]*\)".*/\1/p')
  return "$LAST_EXIT"
}
cd "$WORKSPACE"
# a launch that never opens a Session (seen once in the sandbox: `dsh --version` lost its arguments) is retried
i=0
while (( i < 3 )); do
  run initial "$(cat "$PROMPT_FILE")"
  [ -n "$SID" ] && break
  if [ "$LAST_EXIT" != 0 ] && provider_down; then
    wait_provider || break
    continue
  fi
  i=$((i + 1))
  sleep 5
done
# SID comes from the initial session; without one the precheck ends the loop before any resume
REPROMPT_PRECHECK='[ -n "$SID" ]'
if [ "${AGENT_REPROMPT:-0}" = 1 ]; then reprompt_loop --session-id "$SID"; fi

# Keep the child status, including normal budget timeout124, for run.sbatch.
[ -n "$SID" ] || { echo "DSH_NO_SESSION after launch attempts" >&2; exit 3; }
exit "$LAST_EXIT"
