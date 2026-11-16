# Task: build a RAG evidence retrieval service for {{TASK_NAME}}

Your goal is to build a retrieval-augmented question-answering system that retrieves evidence for **unseen questions of
this benchmark** (labelled examples: {{LABELLED_FILES}}) as accurately and as fast as possible.
After your time budget ends, your `workspace/` is frozen and your service is started in a clean,
offline environment and asked hidden test questions one at a time. Your ordered evidence is frozen once; evaluator-owned readers then answer from the same evidence.

* Questions: {{QUESTION_TYPE}}.
* Expected answer: {{ANSWER_FORMAT}}.
* Score: {{METRIC}}.
* Speed: retrieval latency and generator usage are reported separately from reader latency, throughput and token usage. Reader prompts and decoding are evaluator-controlled.
* `meta`: {{META}}.

You work alone and autonomously; nobody will answer questions. Time budget: **{{BUDGET_HUMAN}}** from your
start, and you are expected to use all of it: keep improving until about 10 minutes are left (run
`{{TASK_DIR}}/timer.sh` to see the remaining time). Only the state of `workspace/` at the deadline counts.

## What you have

| Path | Content |
|-|-|
| `{{TASK_DIR}}/corpus/corpus.jsonl` | {{N_CORPUS}} documents, one JSON per line: `{doc_id, title, text}` (read-only) |
| `{{TASK_DIR}}/train.jsonl` | {{N_TRAIN}} questions with answers: `{qid, question, meta, answers, supporting_doc_ids, ...}` (`answers` lists all accepted answers) |
{{DEV_ROW}}| `{{TASK_DIR}}/task.json` | task name, answer format and metric |
| `{{TASK_DIR}}/example/` | a minimal evidence service that shows the protocol (empty evidence) |
| `{{TASK_DIR}}/self_check` | {{SELF_CHECK}} |
| `{{WORKSPACE}}` | your only writable directory, empty at start. Everything you deliver goes here. |

{{CORPUS_NOTE}}You run in a sandbox that contains only your run directory, the shared Python environments, the agent CLI
and, read-only, the evaluation tools behind `self_check` (the gateway, the evaluator, the scoring code and their
small configuration); the rest of the machine is not visible. You may read the evaluation tools; they contain no
test data.

## The generator (fixed)

* An OpenAI-compatible endpoint at `$LLM_BASE_URL` (e.g. `$LLM_BASE_URL/chat/completions`), model name
  `generator` (Qwen3.5-4B served by vLLM, context 32k). It is a reasoning model; pass
  `"chat_template_kwargs": {"enable_thinking": false}` to turn thinking off. `stream=true` is not supported.
* No API key is needed. During the build, calls are unlimited (only your time limits you) and logged.
* The generator and its gateway are run by the evaluator. **Do not stop, restart or modify them.**
* It shares your single GPU ({{GPU}}; ~35% of its 80 GB memory). Keep your own GPU use below ~45 GB.

## Software and models

* Base Python 3.12 environment (read-only): `{{RAG_BASE_PY}}` with torch, sentence-transformers, faiss-cpu,
  bm25s, rank_bm25, fastapi, uvicorn, httpx, openai, pymupdf, numpy, pandas, scikit-learn.
* Create your own environment inside the workspace and install extras there:
  `{{RAG_BASE_PY}} -m venv --system-site-packages {{WORKSPACE}}/.venv`
  (`pip` uses a PyPI mirror; Hugging Face models go through `$HF_ENDPOINT`). The internet is reachable
  during the build within the limits of rules 6–8.
* Auxiliary models: see rules 2 and 3. Store every model your service uses inside the workspace:
  **the evaluation is offline**.

## Deliverable contract

* `{{WORKSPACE}}/submission/start.sh` starts your service in the foreground and listens on
  `127.0.0.1:$PORT`.
* `GET /health` returns 200 when ready (startup limit: 20 minutes).
* `POST /retrieve` with `{"question": str, "meta": {}}` returns
  `{"passages": [{"doc_id": str, "title": str, "text": str}, ...]}` in your preferred reading order.
  Return at most 256 passages and 500,000 total text/title characters. Each text must be a nonempty
  contiguous excerpt from the identified corpus document (whitespace normalization is allowed);
  title must match that document. Do not put generated summaries, answers, instructions, or rewritten
  queries in evidence. Derived indexes are allowed, but resolve selected hits to original corpus text.
  `meta.doc_id`, when supplied by the task, restricts all evidence to that document.
* The evaluator retains an ordered prefix of at most 32,000 characters (including title overhead),
  also capped at 12,000 Qwen context tokens and every reader's 32,768-token context including a 512-token
  answer allowance. The same retained evidence and user prompt go to every reader. It may truncate the
  last passage; put the most useful evidence first. No target reader output can affect retrieval.
* There is no required `/answer` endpoint. The evaluator uses one fixed user prompt containing the
  task answer format, passages and question, temperature 0, at most 512 output tokens, thinking disabled.
  During development, self_check uses only Qwen3.5-4B as this reader. The five final held-out readers
  are evaluator-owned and unavailable for tuning. The final answer score measures your evidence's
  usefulness across readers; their identities do not change your retrieval budget.
* At evaluation the service runs with **only** these environment variables: `PORT`, `LLM_BASE_URL`,
  `CORPUS_DIR` (same corpus, possibly another path), `SCRATCH_DIR` (the only writable directory),
  `HOME=$SCRATCH_DIR`, `TMPDIR=$SCRATCH_DIR/tmp`, `PATH=/usr/local/bin:/usr/bin:/bin`, `LANG=C.UTF-8`,
  `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `CUDA_VISIBLE_DEVICES`. No internet. So
  `start.sh` must call your interpreter by absolute path (e.g. `{{WORKSPACE}}/.venv/bin/python`) and
  every index / model file must already be inside the workspace. The workspace is read-only then.
* Per-question budget, enforced by the gateway: **≤ {{CALLS}} generator calls, ≤ {{TOKENS}} tokens
  (prompt + completion, all calls) and ≤ {{SECONDS}} s**. This budget covers retrieval only; the external final reader has a separate one-call budget. The 5th retrieval call is refused (HTTP 429); a question
  over budget scores 0. Calls made while no question is being asked are refused.
* Retrieval budget: evidence extraction — service startup plus all {{N_TEST}} hidden questions, asked one
  at a time — must finish within **{{EVAL_HOURS}}**; questions without evidence extraction completed by then score 0. That leaves about
  {{EVAL_PER_QUESTION}} s per question on average.
* Workspace size: at the deadline `{{WORKSPACE}}` (index, models, `.venv`, code: every file) must not exceed
  **{{WORKSPACE_LIMIT}}**; a larger workspace is not evaluated. `self_check` prints the current size.
* All final answers are produced by evaluator-owned readers. Your service delivers corpus evidence only.
* Questions are sent one at a time. Do not use earlier test questions to update your system.

## Rules

1. **Data.** You get {{N_TRAIN}} labelled questions in `{{TASK_DIR}}/train.jsonl`. Use them freely for
   development, debugging and optimisation, and split them into your own train / dev sets as you like.
   The {{N_TEST}} test questions are hidden and are never available to you: do not try to access, obtain
   or reconstruct the test questions or their answers, and do not use them to build or tune your system.
   Any other copy of {{DISPLAY_NAME}} — any split, from Hugging Face, GitHub, BEIR, KILT, FlashRAG,
   ir_datasets, papers' supplementary material or packages that bundle it — contains the test questions
   and is off-limits.{{TRAIN_QA_RULE}}
2. **Auxiliary models.** During the build you may use any publicly downloadable model that can run
   inference on a single H800 (80 GB) — for embedding, reranking, classification, corpus processing,
   generating index content from the corpus, or anything else — and you may train or fine-tune them on
   `train.jsonl`, the corpus, and data generated from them. Models your service uses at evaluation must be
   stored in the workspace.
3. **Fixed retrieval generator at evaluation.** All generated free text used for retrieval (rewritten
   queries or intermediate reasoning) must be generated by the generator at `$LLM_BASE_URL`
   (Qwen3.5-4B, served by the evaluator). Auxiliary models in the service may only output vectors, scores,
   labels, or spans copied verbatim from the corpus. Delivered passages must be original corpus excerpts. Do not generate a final answer in `/retrieve`;
   answering is a separate evaluator-owned phase. You cannot replace or fine-tune the served generator; a generative model you trained
   yourself must not be used by the service.
4. **Closed corpus.** All knowledge your system uses must come from `{{TASK_DIR}}/corpus/` (or be derived
   from it by models run by your code). Do not add other documents — other Wikipedia dumps, web pages,
   other datasets, knowledge bases or retrieval corpora — to the index, prompts or code. Pretrained model
   weights are not documents and are fine.
5. **No hand-written knowledge.** Do not write facts or answers from your own knowledge into the index,
   code or prompts. What you write by hand is limited to code, prompts, rules, configuration, parsing
   fixes, and few-shot examples taken from `train.jsonl`. Bulk content (summaries, triples, synthetic
   queries, ...) must be produced by code you run that calls models — do not write it yourself.
6. **Internet.** You may use the internet to download models, read papers and documentation, clone code
   repositories and install packages. Do not download or use any dataset, retrieval corpus or knowledge
   base, and do not look up answers to questions online. If a repository or package you install contains
   data files (questions, answers, corpora, qrels, predictions), do not read or use them; delete them.
7. **No external model APIs.** You must NOT use any API key found in your environment to call an
   external LLM API directly, for example to process the corpus or to produce answers. Any API keys
   present are there to run you (the agent) or the provided evaluation, not for you to call models
   yourself. Do not call any other hosted LLM, embedding or search API either. The generator at
   `$LLM_BASE_URL` is not external and may be used without limit during the build.
8. **No benchmark lookup.** Do not look up AutoRAGBench itself (its repository, website, leaderboard,
   papers about it, or published traces of past runs), and do not use configurations, prompts or
   predictions that others tuned or produced on {{DISPLAY_NAME}}. Research on RAG methods in general is
   encouraged.
9. **Evaluation tools.** You may read and run the evaluation tools (`self_check`, the gateway, the
   evaluator, the scoring code) but must not modify, stop, restart or circumvent them. Do not try to
   access files outside your run directory other than those described above.
10. **Deliverable.** The service must implement the contract above, run offline, and retrieve for each question
    independently: do not store test questions or your answers to reuse them for later questions. The
    workspace must not exceed {{WORKSPACE_LIMIT}} at the deadline.
11. **Time and autonomy.** Nobody will answer questions. At the deadline all your processes are killed and
    only the state of `workspace/` counts: do not leave work that the deliverable depends on running in
    the background.

Stay inside your run directory `{{RUN_DIR}}`. Run `{{TASK_DIR}}/self_check` before your time runs out:
only the final state of the workspace is evaluated, and a service that does not start scores as the
naive baseline.
