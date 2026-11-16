# Provider template for agents/run_agent.sh (sourced on the host, never inside the sandbox).
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"   # agent CLIs + node on PATH, keys from $RAGENG_ROOT/.env
# Qwen3.8 Flash + Claude Code (agent claude-qwen38-reprompt). Model and Claude Code settings are those of the paper
# runs; the endpoint is a template: set QWEN_ANTHROPIC_BASE_URL to an Anthropic-Messages-compatible endpoint that
# serves qwen3.8-flash (key: DASHSCOPE_API_KEY).
export ANTHROPIC_BASE_URL=${QWEN_ANTHROPIC_BASE_URL:?set QWEN_ANTHROPIC_BASE_URL (Anthropic-compatible endpoint for qwen3.8-flash)}
export ANTHROPIC_AUTH_TOKEN="${DASHSCOPE_API_KEY:-}"
export ANTHROPIC_MODEL=qwen3.8-flash ANTHROPIC_DEFAULT_OPUS_MODEL=qwen3.8-flash ANTHROPIC_DEFAULT_SONNET_MODEL=qwen3.8-flash
export ANTHROPIC_DEFAULT_HAIKU_MODEL=qwen3.8-flash CLAUDE_CODE_SUBAGENT_MODEL=qwen3.8-flash
export CLAUDE_CODE_AUTO_COMPACT_WINDOW=1000000
export CLAUDE_CODE_EFFORT_LEVEL=high CLAUDE_CODE_MAX_OUTPUT_TOKENS=16384
export API_TIMEOUT_MS=600000 CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 DISABLE_AUTOUPDATER=1 ENABLE_TOOL_SEARCH=false
