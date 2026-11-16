# Sourced by llm/<provider>/env.sh. Puts the agent CLIs and Node on PATH and loads API keys from $RAGENG_ROOT/.env.
# Versions used in the paper:
#   harness/claude-code       Claude Code 2.1.296
#   harness/codex             Codex CLI 0.162.1 (Node launcher -> needs node on PATH)
#   harness/deepseek-harness  DeepSeek Harness (dsh) 0.2.1-rc.2
#   harness/opencode          OpenCode 1.19.3
#   $SOFTWARE_DIR/node        Node.js v24 LTS
_C=${RAGENG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export PATH="$_C/harness/claude-code/bin:$_C/harness/codex/bin:$_C/harness/deepseek-harness/bin:$_C/harness/opencode/bin:${SOFTWARE_DIR:-$_C/software}/node/bin:$PATH"
[ -f "$_C/.env" ] && { set -a; . "$_C/.env"; set +a; }
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export DISABLE_AUTOUPDATER=1
unset _C
