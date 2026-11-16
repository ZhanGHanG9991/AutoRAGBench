#!/bin/bash
# Runs inside the sandbox. stdout = Claude Code stream-json events (the trace); stderr = launcher log.
# Env: PROMPT_FILE, DEADLINE (epoch), AGENT_REPROMPT (0|1), TASK_DIR, WORKSPACE, ANTHROPIC_* (provider).
set -uo pipefail
# no built-in web tools (same for every harness: the agent reaches the network through its shell)
FLAGS=(--dangerously-skip-permissions --output-format stream-json --verbose --disallowedTools=WebSearch,WebFetch)
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"   # left, reprompt_loop (reprompt policy)
run() {
  local l; l=$(left); (( l > 0 )) || return 0
  echo "[$(date -Is)] claude $1 (left=${l}s)" >&2
  timeout -s TERM -k 60 $l claude "${@:2}" < /dev/null
  echo "[$(date -Is)] claude $1 exit=$?" >&2
}
cd "$WORKSPACE"
run initial -p "$(cat "$PROMPT_FILE")" "${FLAGS[@]}"
# -p is a flag (print mode); the prompt is the positional argument, so the message can come last
if [ "${AGENT_REPROMPT:-0}" = 1 ]; then reprompt_loop -c "${FLAGS[@]}" -p; fi
