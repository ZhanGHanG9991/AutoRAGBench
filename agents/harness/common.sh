# Sourced by the harness adapters (inside the sandbox). Needs DEADLINE (epoch) and a run() function:
#   run <segment name> <harness arguments...>   starts one session segment, bounded by the deadline.
# Reprompt policy: when the agent ends its session, resume the same session as long as at least
# REPROMPT_MIN_LEFT seconds remain, with a fixed continuation message; no limit on the number of
# resumes. Guard: REPROMPT_MAX_RAPID segments in a row that end within REPROMPT_RAPID seconds stop the loop
# (the harness cannot work; the run is then a platform failure, see traces/agent_segments.py).
#
# Provider outages (optional). When the provider's env.sh sets PROVIDER_HEALTH_URL, run_agent.sh passes it in:
# - a failed segment while the provider probe fails is not counted towards the guard; the loop waits until the
#   provider is back and then resumes the same session (an initial segment that opened no session starts again);
# - left() and the segment timeout follow task/.deadline (re-read every time) instead of the DEADLINE fixed at
#   sandbox start, so an outside process may pause the build clock during an outage by moving task/.deadline later.
# Without a probe the behaviour is the plain reprompt loop.
REPROMPT_MIN_LEFT=${REPROMPT_MIN_LEFT:-900}   # env override for tests only; the sandbox never sets it
REPROMPT_RAPID=60
REPROMPT_MAX_RAPID=5
REPROMPT_PROVIDER_POLL=${REPROMPT_PROVIDER_POLL:-30}
REPROMPT_PROVIDER_RECHECK=${REPROMPT_PROVIDER_RECHECK:-10}
SEGMENT_POLL=${SEGMENT_POLL:-2}
if [ -z "${REPROMPT_PROVIDER_PROBE:-}" ] && [ -n "${PROVIDER_HEALTH_URL:-}" ]; then
  # the provider's health URL answers 200 (the key is sent on stdin, not in argv)
  REPROMPT_PROVIDER_PROBE='printf "Authorization: Bearer %s\n" "${!PROVIDER_KEY_VAR:-}" | curl -sf -m 10 --noproxy "*" -H @- -o /dev/null "$PROVIDER_HEALTH_URL"'
fi

# Seconds left in the build: task/.deadline when readable (it moves later while the provider is down).
left() {
  local d
  d=$(cat "${TASK_DIR:-/nonexistent}/.deadline" 2>/dev/null)
  [[ "$d" =~ ^[0-9]+$ ]] || d=$DEADLINE
  echo $(( d - $(date +%s) ))
}

reprompt_message() {  # fixed continuation message
  local l=$1
  echo "You still have $((l / 3600))h $((l % 3600 / 60))m remaining. Please continue improving your result and maximize performance."
}

# provider_refresh: hook called before every segment (e.g. to re-read a rotating key); no-op by default.
provider_refresh() {
  return 0
}

# provider_down: a probe is configured and fails twice, REPROMPT_PROVIDER_RECHECK seconds apart.
provider_down() {
  [ -n "${REPROMPT_PROVIDER_PROBE:-}" ] || return 1
  provider_refresh
  eval "$REPROMPT_PROVIDER_PROBE" && return 1
  sleep "$REPROMPT_PROVIDER_RECHECK"
  provider_refresh
  ! eval "$REPROMPT_PROVIDER_PROBE"
}

# wait_provider: poll until the probe succeeds (0) or fewer than REPROMPT_MIN_LEFT seconds remain (1). While an
# outside process pauses the clock, left() stays constant, so the wait ends only when the provider is back or the pause cap
# (the Slurm step limit minus the evaluation reserve) is used up.
wait_provider() {
  local t0; t0=$(date +%s)
  echo "[$(date -Is)] PROVIDER_DOWN: model endpoint unreachable; waiting, not counted as a rapid exit (left=$(left)s)" >&2
  while (( $(left) >= REPROMPT_MIN_LEFT )); do
    sleep "$REPROMPT_PROVIDER_POLL"
    provider_refresh
    if eval "$REPROMPT_PROVIDER_PROBE"; then
      echo "[$(date -Is)] PROVIDER_BACK after $(( $(date +%s) - t0 ))s (left=$(left)s)" >&2; return 0
    fi
  done
  echo "[$(date -Is)] PROVIDER_DOWN until less than ${REPROMPT_MIN_LEFT}s were left" >&2; return 1
}

# segment_timeout <command...>: `timeout -s TERM -k 60 <left> command`, but the end follows task/.deadline when it
# moves later during a run. Exit 124 when the deadline ended the command, as timeout(1) reports it.
# timeout(1) runs the command in its own process group and forwards our TERM to that group, then KILLs after 60 s.
segment_timeout() {
  local l pid rc fired=0
  l=$(left); (( l > 0 )) || return 124
  timeout -s TERM -k 60 $(( l + 172800 )) "$@" &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    if (( fired == 0 && $(left) <= 0 )); then
      kill -TERM "$pid" 2>/dev/null; fired=1
    fi
    sleep "$SEGMENT_POLL"
  done
  wait "$pid"; rc=$?
  (( fired )) && rc=124
  return "$rc"
}

# run_initial <harness arguments...>: the first segment. If it fails while the provider is down and opened no session
# (HAS_SESSION: optional command that succeeds once a session exists), wait for the provider and start it again.
run_initial() {
  while :; do
    provider_refresh
    run initial "$@"
    if [ "${LAST_EXIT:-0}" != 0 ] && ! { [ -n "${HAS_SESSION:-}" ] && eval "$HAS_SESSION"; } && provider_down; then
      wait_provider || return
      continue
    fi
    return
  done
}

# reprompt_loop <harness arguments before the message...>: the message is appended as the last argument.
# REPROMPT_PRECHECK (optional): a command that must succeed before each resume (e.g. a session id is known).
reprompt_loop() {
  local i=0 rapid=0 l t0
  while :; do
    l=$(left); (( l < REPROMPT_MIN_LEFT )) && break
    if [ -n "${REPROMPT_PRECHECK:-}" ] && ! eval "$REPROMPT_PRECHECK"; then
      echo "[$(date -Is)] reprompt precheck failed ($REPROMPT_PRECHECK), cannot resume" >&2; break
    fi
    i=$((i + 1)); t0=$(date +%s)
    provider_refresh
    run "resume-$i" "$@" "$(reprompt_message $l)"
    if [ "${LAST_EXIT:-0}" != 0 ] && provider_down; then
      wait_provider || break
      rapid=0; continue
    fi
    if (( $(date +%s) - t0 < REPROMPT_RAPID )); then rapid=$((rapid + 1)); else rapid=0; fi
    if (( rapid >= REPROMPT_MAX_RAPID )); then
      echo "[$(date -Is)] REPROMPT_STOPPED: $rapid sessions in a row ended within ${REPROMPT_RAPID}s" >&2; break
    fi
  done
}
