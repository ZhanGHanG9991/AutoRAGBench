# Evaluation protocol (`fixed-evidence-v1`)

## Build

* One run = one agent configuration × one task. `scripts/new_run.sh` creates `runs/<id>/task/` (corpus hard links,
  `train.jsonl`, rendered `README.md`, `task.json`, `self_check`, `timer.sh`, a minimal `/retrieve` example) and an empty
  `runs/<id>/workspace/`.
* `slurm/run.sbatch` starts Qwen3.5-4B under vLLM (development and retrieval generator) behind the build gateway,
  runs preflight checks (`scripts/preflight.sh`: generator, provider API, GPU and CLI inside the sandbox, hidden paths
  invisible, the whole `self_check` chain), then the agent (`agents/run_agent.sh`) inside the sandbox until the
  six-hour deadline (`BUILD_SECONDS=21600`). The sandbox sees only the run directory, the shared Python environments,
  the agent CLI and, read-only, the evaluation tools behind `self_check`. Internet access is allowed during the build;
  Hugging Face downloads go through `scripts/hf_filter.py` (models allowed, datasets refused).
* Agents test their systems with `task/self_check` on training questions: it starts the service exactly as the
  evaluation does, extracts evidence and answers with Qwen3.5-4B through the same evidence packing.

## Freeze and anti-cheating checks

* `scripts/freeze.py` makes the workspace read-only and hashes every file; the trace is redacted
  (`traces/sanitize.py`) and parsed (`traces/parse_trace.py`); `audit/` records where workspace text comes from, which
  models and external API references the workspace contains, and whether test question text appears anywhere the
  agent wrote (`audit/leak_scan.py`, paths only).
* Four LLM judges (`judges/`, Codex CLI + DeepSeek V4.1 Flash) read the build evidence, never the test:
  `api_usage`, `data_lookup`, `integrity` (three votes each, majority) and `general` (one vote). A flagged run is not
  evaluated (`judge_flagged`); in the paper, a confirmed violation led to a rerun, and a rerun that still failed
  received the naive BM25 RAG score.

## Evaluation (`multireader/`)

1. **Extract** (`pipeline.extract`): the frozen service runs offline in the sandbox (`scripts/eval_system.sh`,
   `scripts/run_service.sh`, `evaluator.py`), one hidden test question at a time, `POST /retrieve` →
   `{"passages": [{doc_id, title, text}, ...]}` (≤ 256 passages, ≤ 500,000 characters). The metered gateway allows at
   most 4 Qwen3.5-4B calls, 32,000 total tokens and 60 seconds per question; the whole extraction has 4 hours.
2. **Validate** (`contract.verify_corpus`): every passage must come from its cited corpus document with the
   document's title. Invalid evidence, budget violations and failed requests make the question invalid (score 0).
3. **Seal** (`pipeline.seal`): the evidence is packed in order into one common prefix of at most 32,000 characters and
   12,000 Qwen tokens that fits all five readers' 32,768-token contexts with 512 tokens reserved for the answer
   (`contract.pack_for_readers`; the last passage may be truncated) and hashed.
4. **Read** (`pipeline.read`): Gemma-4-E4B, Gemma-3-4B, Phi-4-mini, SmolLM3-3B and Llama-3.2-3B each answer every
   question from the same sealed evidence: one user turn (`contract.messages`), native chat template, temperature 0,
   at most 512 output tokens, thinking disabled. Empty, truncated or failed generations are invalid.
5. **Score** (`tasks/score.py`): the task metric per reader (EM: NQ, MH-RAG; token F1, maximum over references:
   HotpotQA, MuSiQue, QASPER; label accuracy: PubMedQA, FEVER; Hits@1: WebQSP, CWQ; answers lowercased, punctuation
   and English articles removed, whitespace normalized). `reader_macro_primary` in `records/<id>/multireader_scores.json`
   is the mean over the five readers; the overall score is the unweighted mean over the nine tasks.

Retrieval latency is measured per question from request to returned evidence; `records/<id>/multireader_scores.json`
→ `retrieval` holds the extraction summary. The paper caps latency at 60 s per question and counts timeouts and
missing measurements as 60 s (`tasks/common.mean_latency`).

## Differences between this repository and the cluster setup of the paper

* Paths: everything is relative to the repository (`RAGENG_ROOT`), models live under `models/`, data under
  `data/formal-v2/`. Placeholders `@RAGENG_ROOT@` and `@RAG_BASE_PY@` are filled in by `scripts/new_run.sh`.
* Agent providers: `llm/` holds templates for the providers' public APIs (see `docs/agents.md`).
* Scheduling: the paper's runs were submitted through a GPU-pool wrapper around `slurm/run.sbatch`; submitting
  `slurm/run.sbatch` directly is equivalent.
* Only the `fixed-evidence-v1` protocol is included; the earlier `/answer` protocol (pilot data) is not.
