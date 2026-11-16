# Fixed evidence reader evaluation

`fixed-evidence-v1` keeps Qwen3.5-4B as the development and retrieval generator.
The agent delivers `POST /retrieve` returning ordered corpus excerpts. The evaluator
extracts once, validates document IDs/text against the corpus, seals a common evidence
prefix, then answers with five separate local readers. No reader answer feeds back into
retrieval.

Build preparation:

```bash
export RAGENG_ROOT=$PWD RAGENG_EVAL_PROTOCOL=fixed-evidence-v1
export BUILD_SECONDS=21600 EVAL_SECONDS=14400
bash scripts/new_run.sh <unique-run-id> <task>
sbatch --gres=gpu:1 --cpus-per-task=12 --mem=240G slurm/run.sbatch <unique-run-id> <agent>
```

The existing `slurm/run.sbatch` runs build, freeze, all integrity judges, and then
`python -m multireader.run --run-id <id>`. The build sandbox sees only the evidence
contract and Qwen tokenizer files, not the target reader models or hidden data.
`self_check` tests retrieval followed by the evaluator-controlled Qwen reader.

Independent evaluation of an already frozen compatible delivery:

```bash
python -m multireader.pipeline extract --experiment <id> --cases <cases.json>
python -m multireader.pipeline seal --experiment <id> \
  --readers gemma4_e4b,gemma3_4b,phi4_mini,smollm3_3b,llama32_3b
python -m multireader.pipeline read --experiment <id> --reader gemma4_e4b
```

Extract and read require an allocated Slurm GPU. Each case JSON entry contains
`id`, `task`, `run_id`, `source_run`, and `agent` (boolean). Agent deliveries require
their source run's judge gate to have passed. Existing legacy deliveries need an
explicit independently recorded adapter; never overwrite a frozen source workspace.

Evidence, hidden questions and raw predictions stay in `hidden/multireader/<id>`.
Text-free scores, timings, hashes and model metadata are in `records/multireader/<id>`.
`records/<run-id>/multireader_scores.json` holds the complete reader panel.
Run `python -m multireader.report --campaign <campaign>` for per-run CSV and task means.
Incomplete or judge-flagged runs are shown explicitly, with no best-run selection.

Protocol budgets: retrieval <=4 Qwen calls / 32,000 total tokens / 60 seconds per
question; evidence prefix <=32,000 characters and 12,000 Qwen context tokens, fitting
every reader's 32,768-token context; answer <=512 tokens, temperature 0, thinking off.
Packing uses all readers' tokenizers, while development self-check uses only Qwen's.
The reader uses concurrency 4; request latency and throughput exclude startup and
synthetic warm-up. This is a new evaluation protocol, so its scores are not directly
interchangeable with legacy custom `/answer` pipelines.

Validation: `python -m unittest multireader.test_contract -v`, build sandbox preflight,
all five GPU model smoke tests, and a shared training-example panel before test runs.
