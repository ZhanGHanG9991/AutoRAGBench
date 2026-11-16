# Provider template for agents/run_agent.sh (sourced on the host, never inside the sandbox).
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"   # agent CLIs + node on PATH, keys from $RAGENG_ROOT/.env
# Gemini 3.8 Flash + OpenCode (agent opencode-gemini38-reprompt), OpenAI-compatible chat endpoint (key: GEMINI_API_KEY).
# Model settings (reasoning effort high, 1M context, 16k output) are those of the paper runs: opencode/opencode.json.
export GEMINI_OPENAI_BASE_URL=${GEMINI_OPENAI_BASE_URL:-https://generativelanguage.googleapis.com/v1beta/openai}
export OPENCODE_DISABLE_AUTOUPDATE=1
