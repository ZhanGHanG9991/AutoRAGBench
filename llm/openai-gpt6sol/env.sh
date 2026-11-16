# Provider template for agents/run_agent.sh (sourced on the host, never inside the sandbox).
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"   # agent CLIs + node on PATH, keys from $RAGENG_ROOT/.env
# gpt-6-sol + Codex CLI (agent codex-gpt6sol-reprompt) through the OpenAI Responses API (key: OPENAI_API_KEY).
# Model settings (reasoning effort, context window, auto-compaction) are those of the paper runs: codex_home/config.toml.
