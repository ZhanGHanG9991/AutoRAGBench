You are an expert ML / search engineer. Your task is described in {{TASK_DIR}}/README.md — read it completely first and follow its rules.

Build the most accurate and fast evidence retrieval service you can for {{DISPLAY_NAME}}. You have {{BUDGET_HUMAN}} and you are expected to use all of it: there is no reward for finishing early, and the final state of your workspace at the deadline is what gets evaluated. Your working directory is {{WORKSPACE}}. Iterate: measure on the labelled questions, analyse failures, try better retrieval / query rewriting / multi-hop evidence selection within the per-question budget, and keep improving until {{TASK_DIR}}/timer.sh shows about 10 minutes left. Keep a working {{WORKSPACE}}/submission/start.sh at all times and run {{TASK_DIR}}/self_check on your final version before the deadline. Work autonomously; nobody will answer questions.

Setup: {{GPU}} shared with the generator. Current date: {{DATETIME}}.

Deliver POST /retrieve returning original corpus excerpts, as specified in the README. The evaluator answers with a controlled prompt; do not implement a custom final-answer pipeline. Development and self_check use Qwen3.5-4B only.
