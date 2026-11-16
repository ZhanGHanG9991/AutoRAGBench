# Sourced (bash). One entry point for running something isolated, whatever the node supports.
# SANDBOX (configs/pilot.env): singularity | bwrap | none   — chosen from the M0' probe.
#
#   SB_RO=(dir ...)        read-only binds (kept at the same absolute path)
#   SB_RW=(dir ...)        read-write binds
#   SB_ENV=(K=V ...)       the only environment variables the process gets
#   SB_HOME=dir            HOME inside the sandbox (must be in SB_RW)
#   SB_PWD=dir             working directory
#   SB_GPU=1|0             expose the GPU
#   SB_NET=host|none       network namespace
#   sandbox_exec cmd args...
#
# Everything under the project tree that is not bound is invisible; PIDs outside the sandbox are invisible
# (singularity --containall / bwrap --unshare-pid). Paths stay identical inside and outside, so frozen
# venvs and indexes keep working. The paper's runs used singularity; bwrap hides SANDBOX_HIDE_DIR
# (default: the parent of the repository) behind a tmpfs.

REAL_WORK=$(readlink -f "${SANDBOX_HIDE_DIR:-$RAGENG_ROOT/..}")

_real() { readlink -f "$1"; }

sandbox_exec() {
  local mode=${SANDBOX:-none}
  case $mode in
    singularity) _sb_singularity "$@" ;;
    bwrap)       _sb_bwrap "$@" ;;
    none)        _sb_none "$@" ;;
    *) echo "unknown SANDBOX=$mode" >&2; return 2 ;;
  esac
}

_sb_envfile() {  # writes K=V pairs as a sourced env file (handles commas/quotes/brackets), prints its path
  # values in single quotes: singularity's --env-file parser keeps the backslashes of printf %q escapes (e.g.
  # ANTHROPIC_MODEL=deepseek-flash[1m] arrived as deepseek-flash\[1m\]; commas and spaces likewise). Verified
  # 2026-10-19: brackets, commas, spaces, '=' pass unchanged; '$' is still expanded and "'" keeps a backslash.
  local f kv v; f=$(mktemp /tmp/rageng-env.XXXXXX); chmod 600 "$f"
  for kv in "${SB_ENV[@]}"; do
    v=${kv#*=}
    case $v in *\$*|*\'*) echo "sandbox.sh: WARNING: ${kv%%=*} contains \$ or ' and will not reach the sandbox unchanged" >&2 ;; esac
    v=${v//\'/\'\\\'\'}
    printf "export %s='%s'\n" "${kv%%=*}" "$v" >> "$f"
  done
  echo "$f"
}

_sb_singularity() {
  local args=(exec --containall --cleanenv --no-init --pwd "${SB_PWD:-$SB_HOME}" --home "$(_real "$SB_HOME"):$SB_HOME")
  [ "${SB_GPU:-0}" = 1 ] && args+=(--nv)
  [ "${SB_NET:-host}" = none ] && args+=(--net --network none)
  local d
  for d in "${SB_RO[@]}"; do args+=(--bind "$(_real "$d"):$d:ro"); done
  for d in "${SB_RW[@]}"; do [ "$d" = "$SB_HOME" ] || args+=(--bind "$(_real "$d"):$d"); done
  local ef; ef=$(_sb_envfile)
  args+=(--env-file "$ef")
  # A preflight check captures stdout/stderr with $(...). The delayed cleanup
  # must not hold those pipes open after the command has already finished.
  ( sleep 20; rm -f "$ef" ) </dev/null >/dev/null 2>&1 &
  singularity "${args[@]}" "$SIF_IMAGE" "$@"
}

_sb_bwrap() {
  local args=(--ro-bind / / --dev-bind /dev /dev --proc /proc --tmpfs /tmp
              --tmpfs "$REAL_WORK"
              --unshare-pid --unshare-ipc --die-with-parent --clearenv --chdir "${SB_PWD:-$SB_HOME}")
  [ "${SB_NET:-host}" = none ] && args+=(--unshare-net)
  local d
  for d in "${SB_RO[@]}"; do args+=(--ro-bind "$(_real "$d")" "$(_real "$d")"); done
  for d in "${SB_RW[@]}"; do args+=(--bind "$(_real "$d")" "$(_real "$d")"); done
  for kv in "${SB_ENV[@]}"; do args+=(--setenv "${kv%%=*}" "${kv#*=}"); done
  ${BWRAP:-bwrap} "${args[@]}" "$@"
}

_sb_none() {
  ( cd "${SB_PWD:-$SB_HOME}" && env -i "${SB_ENV[@]}" "$@" )
}
