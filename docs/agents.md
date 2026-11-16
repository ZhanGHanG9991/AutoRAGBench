# Agent configurations

The nine configurations evaluated in the paper (Table II). Every configuration runs with reasoning effort *high*,
the same task README and prompt (`materials/`), the same sandbox, and the reprompt policy of
`agents/harness/common.sh`: when the agent ends its session with at least 15 minutes left, the same session is resumed
with a fixed continuation message ("You still have Xh Ym remaining. Please continue improving your result
and maximize performance."); five resumed sessions in a row that end within 60 seconds stop the loop.

| Paper name | `agents/registry.json` | Harness (version) | Model (API id) | Provider template | Key |
|-|-|-|-|-|-|
| GPT-6.1 Sol (Codex CLI) | `codex-gpt61sol-reprompt` | Codex CLI 0.162.1 | `gpt-6.1-sol` | `llm/openai-gpt61sol/` | `OPENAI_API_KEY` |
| GPT-6 Sol (Codex CLI) | `codex-gpt6sol-reprompt` | Codex CLI 0.162.1 | `gpt-6-sol` | `llm/openai-gpt6sol/` | `OPENAI_API_KEY` |
| GPT-5.6 Sol (Codex CLI) | `codex-gpt56sol-reprompt` | Codex CLI 0.162.1 | `gpt-5.6-sol` | `llm/openai-gpt56sol/` | `OPENAI_API_KEY` |
| GPT-5.6 Luna (Codex CLI) | `codex-gpt56luna-reprompt` | Codex CLI 0.162.1 | `gpt-5.6-luna` | `llm/openai-gpt56luna/` | `OPENAI_API_KEY` |
| GLM-5.3 (Claude Code) | `claude-glm53-reprompt` | Claude Code 2.1.296 | `glm-5.3` | `llm/glm53/` | `GLM_API_KEY` |
| GLM-5.3-Flash (Claude Code) | `claude-glm-reprompt` | Claude Code 2.1.296 | `glm-5.3-flash[1m]` | `llm/glm/` | `GLM_API_KEY` |
| DeepSeek V4.1 Flash (DeepSeek Harness) | `dsh-deepseek-reprompt` | DeepSeek Harness 0.2.1-rc.2 | `deepseek-flash` (DeepSeek-V4.1-Flash-1017) | `llm/deepseek/` | `DEEPSEEK_API_KEY` |
| Gemini 3.8 Flash (OpenCode) | `opencode-gemini38-reprompt` | OpenCode 1.19.3 | `gemini-3.8-flash` | `llm/gemini38/` | `GEMINI_API_KEY` |
| Qwen3.8 Flash (Claude Code) | `claude-qwen38-reprompt` | Claude Code 2.1.296 | `qwen3.8-flash` | `llm/qwen38/` | `DASHSCOPE_API_KEY` |

Settings taken from the paper runs:

* **Codex CLI** (`llm/openai-*/codex_home/`): Responses API, `model_context_window = 272000`,
  `model_auto_compact_token_limit = 244800`, web search disabled, Codex's `goals` feature disabled (it lets the model
  set itself a token budget that ends the session early). `models.json` is the model catalog Codex reads.
* **Claude Code** (`llm/glm*/env.sh`, `llm/qwen38/env.sh`): `CLAUDE_CODE_EFFORT_LEVEL=high`; built-in WebSearch and
  WebFetch tools disabled (`agents/harness/claude_code.sh`); auto-compaction window 1,000,000 tokens (GLM-5.3-Flash,
  Qwen3.8 Flash) or 200,000 tokens (GLM-5.3); 16,384 output tokens for GLM-5.3 and Qwen3.8 Flash.
* **DeepSeek Harness** (`llm/deepseek/dsh_home/cordis.patch.yml`): reasoning effort high, web tool, session upload and
  telemetry disabled.
* **OpenCode** (`llm/gemini38/opencode/opencode.json`): OpenAI-compatible chat provider, reasoning effort high,
  1M-token context, 16,384 output tokens, `webfetch` and `websearch` denied.

The base URLs in these templates point to each provider's public API (OpenAI, Zhipu, DeepSeek, Google's
OpenAI-compatible Gemini endpoint). Qwen3.8 Flash through Claude Code needs an Anthropic-Messages-compatible endpoint;
set `QWEN_ANTHROPIC_BASE_URL`. The model ids and harness settings above are those of the paper's runs.

The anti-cheating judges (`judges/judges.json`) run Codex CLI with DeepSeek V4.1 Flash (`llm/deepseek/`):
`api_usage`, `data_lookup` and `integrity` with three votes each (majority), `general` with one vote.
