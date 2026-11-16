# Provider template for agents/run_agent.sh (sourced on the host, never inside the sandbox).
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"   # agent CLIs + node on PATH, keys from $RAGENG_ROOT/.env
# DeepSeek V4.1 Flash (API alias deepseek-flash) through the official DeepSeek API (key: DEEPSEEK_API_KEY).
# Used by the DeepSeek Harness agent (dsh-deepseek-reprompt) and by the four anti-cheating judges (Codex CLI).
# --- Codex -> DeepSeek Responses API: codex_home/config.toml (copied into a fresh CODEX_HOME per run / judge session)
# --- DeepSeek Harness: dsh_home/cordis.patch.yml (copied into a fresh DSH_HOME per run)
# --- Claude Code -> DeepSeek Anthropic-compatible endpoint (not used by the paper's nine agents)
export ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
export ANTHROPIC_AUTH_TOKEN="${DEEPSEEK_API_KEY:-}"
export ANTHROPIC_MODEL='deepseek-flash[1m]'
export ANTHROPIC_DEFAULT_OPUS_MODEL='deepseek-flash[1m]'
export ANTHROPIC_DEFAULT_SONNET_MODEL='deepseek-flash[1m]'
export ANTHROPIC_DEFAULT_HAIKU_MODEL=deepseek-flash
export CLAUDE_CODE_SUBAGENT_MODEL=deepseek-flash
export CLAUDE_CODE_EFFORT_LEVEL=high
export CLAUDE_CODE_AUTO_COMPACT_WINDOW=786432
export OPENCODE_DISABLE_AUTOUPDATE=1
