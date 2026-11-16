#!/bin/bash
# Runs inside the sandbox. stdout = opencode `run --format json` events (the trace); stderr = launcher log.
# Env: PROMPT_FILE, DEADLINE (epoch), AGENT_REPROMPT (0|1), TASK_DIR, WORKSPACE, DEEPSEEK_API_KEY;
# $HOME/.config/opencode/opencode.json prepared by run_agent.sh (model, permissions, no web tools).
set -uo pipefail
VARIANT=high   # reasoning effort, same as the Codex and DeepSeek Harness agents
FLAGS=(--format json --variant $VARIANT --thinking --auto)   # --auto: no approvals, the sandbox is the boundary
LAST=$HOME/tmp/opencode_last.jsonl
SID=
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"   # left, reprompt_loop (reprompt policy)
# OpenCode fetches ripgrep from GitHub on its first glob/grep; GitHub may be unreachable from compute nodes,
# so put the release it would fetch (harness/opencode/vendor/rg, 15.1.0 musl) where it looks for it.
# On a shared filesystem a copy made right after the vendor file appeared can be empty, and cp (copy_file_range)
# can fail on Lustre, so copy with plain read/write (dd) and retry; the last try links to the vendor file.
RG=$HOME/.cache/opencode/bin/rg; mkdir -p "$(dirname "$RG")"
RG_SRC="$(dirname "${BASH_SOURCE[0]}")/../../harness/opencode/vendor/rg"
for i in 1 2 3 4 5 6; do
  "$RG" --version 2>/dev/null | grep -q "^ripgrep 15\.1\.0" && break   # an empty executable also exits 0
  [ $i -gt 1 ] && sleep 10
  rm -f "$RG"
  if [ $i -lt 6 ]; then dd if="$RG_SRC" of="$RG" bs=4M status=none; chmod 755 "$RG"; else ln -s "$RG_SRC" "$RG"; fi
done
"$RG" --version 2>/dev/null | grep -q "^ripgrep 15\.1\.0" && echo "[$(date -Is)] rg ready: $("$RG" --version | head -1)" >&2 \
  || echo "[$(date -Is)] RG_SEED_FAILED (OpenCode glob/grep will try to download ripgrep)" >&2
run() {
  local l; l=$(left); (( l > 0 )) || return 0
  echo "[$(date -Is)] opencode $1 (left=${l}s)" >&2
  # stdin must be closed: `opencode run` waits for piped input otherwise
  timeout -s TERM -k 60 $l opencode run "${FLAGS[@]}" "${@:2}" < /dev/null | tee $LAST
  echo "[$(date -Is)] opencode $1 exit=${PIPESTATUS[0]}" >&2
  # every event carries the sessionID that --session resumes
  [ -n "$SID" ] || SID=$(head -1 $LAST | sed -n 's/.*"sessionID":"\([^"]*\)".*/\1/p')
}
cd "$WORKSPACE"
for i in 1 2 3; do   # a launch that never opens a session (e.g. a failed first-start plugin install) is retried
  run initial "$(cat "$PROMPT_FILE")"
  [ -n "$SID" ] && break
  sleep 5
done
# SID comes from the initial session; without one the precheck ends the loop before any resume
REPROMPT_PRECHECK='[ -n "$SID" ]'
if [ "${AGENT_REPROMPT:-0}" = 1 ]; then reprompt_loop --session "$SID"; fi
