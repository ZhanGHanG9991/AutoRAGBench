#!/bin/bash
# Login node. Print the running or pending agentic jobs (configs/agentic_jobs.txt: by job name or by the content of
# the submitted batch script) that are not sandboxed builds: no RAGENG_SANDBOXED_BUILD marker or no sandbox_exec.
# Empty output = the hidden test may be on the cluster. Unreadable scripts count as unsafe (fail closed).
# One line per job: <jobid> <name> <reason>.
source "${RAGENG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/configs/pilot.env"
CFG=$RAGENG_ROOT/configs/agentic_jobs.txt
NAMES=$(awk '$1 == "name" {print $2}' $CFG | paste -sd'|')
SCRIPTS=$(awk '$1 == "script" {print $2}' $CFG | paste -sd'|')
squeue -h -u $USER -t R,PD,CF,S -o '%i %j' | while read -r id name; do
  script=$(scontrol write batch_script "$id" - 2>/dev/null)
  if [ -z "$script" ]; then
    grep -qE "$NAMES" <<<"$name" && echo "$id $name batch-script-unreadable"
    continue
  fi
  grep -qE "$NAMES" <<<"$name" || grep -qE "$SCRIPTS" <<<"$script" || continue
  grep -q 'RAGENG_SANDBOXED_BUILD' <<<"$script" || { echo "$id $name no-sandbox-marker"; continue; }
  grep -q 'sandbox_exec' <<<"$script" || echo "$id $name no-sandbox_exec"
done
exit 0
