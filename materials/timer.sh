#!/bin/bash
# Prints the remaining build time as H:MM, or "expired".
d=$(cat "$(dirname "$0")/.deadline" 2>/dev/null) || { echo "timer not started"; exit 0; }
left=$(( d - $(date +%s) ))
if (( left <= 0 )); then echo "expired"; else printf '%d:%02d remaining\n' $((left/3600)) $((left%3600/60)); fi
