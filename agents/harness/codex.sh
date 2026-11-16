#!/bin/bash
# Runs inside the sandbox. stdout = Codex JSON events (the trace); stderr = launcher log.
# Env: PROMPT_FILE, DEADLINE (epoch), AGENT_REPROMPT (0|1), TASK_DIR, WORKSPACE.
# Optional: PROVIDER_HEALTH_URL (provider outages: see common.sh).
set -uo pipefail
FLAGS=(--skip-git-repo-check --dangerously-bypass-approvals-and-sandbox --json)
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"   # left, segment_timeout, run_initial, reprompt_loop
# a session exists once Codex has written its rollout file (resume --last needs one)
HAS_SESSION='find "${CODEX_HOME:-/nonexistent}/sessions" -name "*.jsonl" -print -quit 2>/dev/null | grep -q .'
LAST_EXIT=0
# Explicit request-level 429 is not a network outage: health can remain green.
# Cool down before the existing same-session reprompt, with a finite total wait.
# Waiting consumes the original build budget; no deadline extension or request replay.
CODEX_RATE_LIMIT_WAIT=${CODEX_RATE_LIMIT_WAIT:-120}
CODEX_RATE_LIMIT_MAX_WAIT=${CODEX_RATE_LIMIT_MAX_WAIT:-900}
RATE_LIMIT_WAITED=0
RATE_LIMIT_COOLDOWN=0
run() {
  local l; l=$(left); (( l > 0 )) || return 0
  echo "[$(date -Is)] codex $1 (left=${l}s)" >&2
  local events delay remaining
  RATE_LIMIT_COOLDOWN=0
  events=$(mktemp "${TMPDIR:-/tmp}/codex-segment.XXXXXX") || { LAST_EXIT=1; return 1; }
  segment_timeout codex exec "${@:2}" < /dev/null | tee "$events"
  LAST_EXIT=${PIPESTATUS[0]}
  echo "[$(date -Is)] codex $1 exit=$LAST_EXIT" >&2
  if [ "$LAST_EXIT" != 0 ] && [ -n "${PROVIDER_HEALTH_URL:-}" ] &&
      grep -Eq '^\{[[:space:]]*"type":[[:space:]]*"(error|turn.failed)".*(429 Too Many Requests|rate_limit_exceeded)' "$events"; then
    delay=$CODEX_RATE_LIMIT_WAIT
    remaining=$(( CODEX_RATE_LIMIT_MAX_WAIT - RATE_LIMIT_WAITED ))
    (( delay > remaining )) && delay=$remaining
    remaining=$(( $(left) - REPROMPT_MIN_LEFT ))
    (( delay > remaining )) && delay=$remaining
    if (( delay > 0 )); then
      echo "[$(date -Is)] PROVIDER_RATE_LIMIT: confirmed 429; cooldown=${delay}s (total=$((RATE_LIMIT_WAITED + delay))/${CODEX_RATE_LIMIT_MAX_WAIT}s); build clock continues" >&2
      sleep "$delay"
      RATE_LIMIT_WAITED=$(( RATE_LIMIT_WAITED + delay ))
      RATE_LIMIT_COOLDOWN=1
    else
      echo "[$(date -Is)] PROVIDER_RATE_LIMIT_WAIT_EXHAUSTED: existing rapid-exit guard remains active" >&2
    fi
  fi
  rm -f -- "$events"
  return "$LAST_EXIT"
}
cd "$WORKSPACE"
while :; do
  run_initial "${FLAGS[@]}" -C "$WORKSPACE" "$(cat "$PROMPT_FILE")"
  # A rejected first request may not have created a resumable session yet.
  if [ "$LAST_EXIT" != 0 ] && [ "$RATE_LIMIT_COOLDOWN" = 1 ] && ! eval "$HAS_SESSION"; then
    continue
  fi
  break
done
if [ "${AGENT_REPROMPT:-0}" = 1 ]; then reprompt_loop resume --last "${FLAGS[@]}"; fi
exit "$LAST_EXIT"
