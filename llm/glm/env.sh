# Provider template for agents/run_agent.sh (sourced on the host, never inside the sandbox).
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"   # agent CLIs + node on PATH, keys from $RAGENG_ROOT/.env
# GLM-5.3-Flash through Zhipu's Anthropic-compatible endpoint (key: GLM_API_KEY). Agent: claude-glm-reprompt.
export ANTHROPIC_BASE_URL=${GLM_ANTHROPIC_BASE_URL:-https://open.bigmodel.cn/api/anthropic}
export ANTHROPIC_AUTH_TOKEN="${GLM_API_KEY:-}"
export ANTHROPIC_MODEL='glm-5.3-flash[1m]'
export ANTHROPIC_DEFAULT_OPUS_MODEL='glm-5.3-flash[1m]'
export ANTHROPIC_DEFAULT_SONNET_MODEL='glm-5.3-flash[1m]'
export ANTHROPIC_DEFAULT_HAIKU_MODEL=glm-5.3-flash
export CLAUDE_CODE_SUBAGENT_MODEL=glm-5.3-flash
export CLAUDE_CODE_EFFORT_LEVEL=high
export CLAUDE_CODE_AUTO_COMPACT_WINDOW=1000000
export API_TIMEOUT_MS=3000000
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
