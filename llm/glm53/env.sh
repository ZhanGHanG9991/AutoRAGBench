# Provider template for agents/run_agent.sh (sourced on the host, never inside the sandbox).
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"   # agent CLIs + node on PATH, keys from $RAGENG_ROOT/.env
# GLM-5.3 + Claude Code (agent claude-glm53-reprompt). Model and Claude Code settings are those of the paper runs;
# the endpoint is a template: any Anthropic-Messages-compatible endpoint that serves glm-5.3 (default: Zhipu).
export ANTHROPIC_BASE_URL=${GLM_ANTHROPIC_BASE_URL:-https://open.bigmodel.cn/api/anthropic}
export ANTHROPIC_AUTH_TOKEN="${GLM_API_KEY:-}"
export ANTHROPIC_MODEL=glm-5.3 ANTHROPIC_DEFAULT_OPUS_MODEL=glm-5.3 ANTHROPIC_DEFAULT_SONNET_MODEL=glm-5.3
export ANTHROPIC_DEFAULT_HAIKU_MODEL=glm-5.3 CLAUDE_CODE_SUBAGENT_MODEL=glm-5.3
export CLAUDE_CODE_AUTO_COMPACT_WINDOW=200000
export CLAUDE_CODE_EFFORT_LEVEL=high CLAUDE_CODE_MAX_OUTPUT_TOKENS=16384
export API_TIMEOUT_MS=600000 CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 DISABLE_AUTOUPDATER=1 ENABLE_TOOL_SEARCH=false
