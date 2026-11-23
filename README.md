# AutoRAGBench

**A benchmark for LLM agents automating RAG system development.**

An LLM agent (a model plus its harness, e.g. Codex CLI or Claude Code) gets a corpus, labelled question–answer
examples, one H800 GPU and six hours, and builds a retrieval service from scratch. The frozen service is evaluated
offline on a hidden test set: it returns ordered corpus evidence for each question (`POST /retrieve`), the evaluator
validates and packs that evidence once, and five held-out readers answer from the same evidence. This repository
contains the benchmark code and the data of the paper's nine tasks.

Project page: <https://zhanghang9991.github.io/AutoRAGBench_Website>

## Contents

| Path | What it is |
|-|-|
| `tasks/` | the nine tasks: data construction, metadata, metrics |
| `configs/` | paths and budgets, reader models, data sampling rules and fingerprints |
| `materials/`, `selfcheck/` | what the agent receives: task README, prompt, a minimal `/retrieve` example, `self_check` |
| `gateway/`, `multireader/` | metered generator gateway and the fixed-evidence evaluation (extract → validate → pack → five readers → score) |
| `agents/`, `llm/` | agent launcher (runs the agent in a container), harness adapters, the nine agent configurations |
| `judges/`, `audit/`, `traces/` | anti-cheating checks: LLM judges, leak / provenance / workspace scans, trace parsing |
| `baselines/` | closed-book, naive BM25, Reference RAG, Human-SOTA and automated RAG baselines (`baselines/README.md`) |
| `scripts/`, `slurm/` | run creation, container launch, services, freezing; Slurm jobs |
| `examples/minimal_eval.py` | CPU-only walk through retrieval → evidence validation → packing → metric |
| `docs/` | protocol, agent configurations, data card |

## Data

```
data/formal-v2/<task>/public/
  corpus/corpus.jsonl     {doc_id, title, text} per line
  corpus/triples.tsv      head<TAB>relation<TAB>tail (WebQSP, CWQ only)
  train.jsonl             {qid, question, meta, answers, supporting_doc_ids?, ...}
  data_manifest.json      counts, seed, sampling rules, sha256 of corpus and train
hidden/formal-v2/<task>/  test set; never mounted into a container
  test_questions.jsonl    {qid, question, meta}
  test_gold.jsonl         {qid, answers, supporting_doc_ids?, ...}
  split_manifest.json, hidden_manifest.json
```

| Category | Task (directory) | Corpus records | Train | Test | Metric | Corpus size |
|-|-|-:|-:|-:|-|-:|
| Open-domain QA | NQ (`nq_open`) | 21,015,324 | 10,000 | 1,000 | EM | 15.1 GB |
| Document-based multi-hop QA | HotpotQA (`hotpotqa`) | 5,233,235 | 10,000 | 1,000 | F1 | 1.9 GB |
| | MuSiQue (`musique`) | 117,534 | 10,000 | 1,000 | F1 | 64 MB |
| | MH-RAG (`multihop_rag`) | 609 | 1,680 | 500 | EM | 7 MB |
| Domain-specific QA | PubMedQA (`pubmedqa`) | 273,518 | 500 | 500 | Acc. | 403 MB |
| | QASPER (`qasper`) | 1,585 | 3,598 | 1,000 | F1 | 40 MB |
| Fact verification | FEVER (`fever`) | 5,396,106 | 10,000 | 1,000 | Acc. | 3.3 GB |
| Knowledge-graph QA | WebQSP (`webqsp`) | 665,334 | 2,945 | 1,000 | Hits@1 | 330 MB + 252 MB triples |
| | CWQ (`cwq`) | 1,310,816 | 10,000 | 1,000 | Hits@1 | 598 MB + 448 MB triples |

**Download** from <https://drive.google.com/drive/folders/1Gy0x_-PuFCC4wX0WV-P7wmZeWVWowT2t>: one archive per task
plus `test.tar.gz` (test questions and answers of all nine tasks), about 8 GB in total.

| Archive | Size | Archive | Size |
|-|-:|-|-:|
| `nq_open.tar.gz` | 5.3 GB | `webqsp.tar.gz` | 117 MB |
| `fever.tar.gz` | 1.3 GB | `musique.tar.gz` | 28 MB |
| `hotpotqa.tar.gz` | 749 MB | `qasper.tar.gz` | 13 MB |
| `cwq.tar.gz` | 213 MB | `multihop_rag.tar.gz` | 2.7 MB |
| `pubmedqa.tar.gz` | 142 MB | `test.tar.gz` | 1.2 MB |

```bash
sha256sum -c SHA256SUMS.txt                                   # in the download folder
for f in *.tar.gz; do tar -xzf "$f" -C /path/to/AutoRAGBench; done
python scripts/check_data.py [--sha256]                       # in the repository root
```

`docs/data.md` describes sources, splits and sampling. `tasks/prepare.py` with the sampling rules in `configs/data_rules/`
rebuilds the data from the raw benchmark releases (`slurm/prepare.sbatch`).

## Running the benchmark

The paper's runs used Slurm with one H800 (80 GB), 12 CPU cores and 240 GB of memory per run, a Singularity
container for the agent, and a 6-hour build budget (`BUILD_SECONDS=21600` in `configs/pilot.env`).

### 1. Install

```bash
bash scripts/install_envs.sh vllm        # envs/vllm: evaluator side (vLLM 0.31.0)
bash scripts/install_envs.sh rag-base    # envs/rag-base: read-only base environment of agents and baselines
singularity build envs/images/rageng.sif docker://python:3.12-bookworm   # container image (SANDBOX=bwrap: bubblewrap instead)
cp .env.example .env                     # fill in only the keys you need; keys are read from the environment
```

Models go under `models/` (a symlink is fine), named as in `configs/readers.json`: `Qwen3.5-4B` (development and
retrieval generator) and the readers `gemma-4-E4B-it`, `gemma-3-4b-it`, `Phi-4-mini-instruct`, `SmolLM3-3B`,
`Llama-3.2-3B-Instruct`. Agent CLIs go under `harness/<name>/bin/` with Node.js under `$SOFTWARE_DIR/node/bin`
(versions in `docs/agents.md`). The judges use `DEEPSEEK_API_KEY`.

### 2. Run an agent

```bash
export RAGENG_ROOT=$PWD
RUN=gpt61sol-hotpotqa-r1
bash scripts/new_run.sh $RUN hotpotqa             # runs/$RUN/{task,workspace,home}, records/$RUN/
sbatch -p <partition> --gres=gpu:1 --cpus-per-task=12 --mem=240G --time=12:00:00 \
  slurm/run.sbatch $RUN codex-gpt61sol-reprompt
```

One job does everything: start the generator and gateway, run the agent in the container until the deadline, freeze
the workspace, run the four judges, and, if they pass, evaluate the frozen service on the hidden test with the five
readers. The agent configurations are the keys of `agents/registry.json`:

| Configuration | Model + harness | Configuration | Model + harness |
|-|-|-|-|
| `codex-gpt61sol-reprompt` | GPT-6.1 Sol + Codex CLI | `claude-glm53-reprompt` | GLM-5.3 + Claude Code |
| `codex-gpt6sol-reprompt` | GPT-6 Sol + Codex CLI | `claude-glm-reprompt` | GLM-5.3-Flash + Claude Code |
| `codex-gpt56sol-reprompt` | GPT-5.6 Sol + Codex CLI | `claude-qwen38-reprompt` | Qwen3.8 Flash + Claude Code |
| `codex-gpt56luna-reprompt` | GPT-5.6 Luna + Codex CLI | `opencode-gemini38-reprompt` | Gemini 3.8 Flash + OpenCode |
| `dsh-deepseek-reprompt` | DeepSeek V4.1 Flash + DeepSeek Harness | | |

### 3. Run a baseline or your own system

A system that is not an agent is a workspace template with `submission/start.sh` (and optionally
`submission/build.sh`, which builds the index once before freezing). The same job then skips the agent and the judges:

```bash
bash scripts/new_run.sh bm25-hotpotqa hotpotqa $RAGENG_ROOT/baselines/naive_bm25    # closed_book | naive_bm25 | reference_rag
sbatch -p <partition> --gres=gpu:1 --cpus-per-task=12 --mem=240G --time=12:00:00 \
  slurm/run.sbatch bm25-hotpotqa naive_bm25
```

To evaluate your own retriever, start from the minimal example and implement its service:

```bash
cp -r materials/example_fixed my_system         # my_system/submission/{start.sh,server.py}
# start.sh serves on 127.0.0.1:$PORT, reads the corpus from $CORPUS_DIR, writes only to $SCRATCH_DIR, no network:
#   GET  /health   -> 200 when ready (within 20 minutes)
#   POST /retrieve {"question": str, "meta": {}}
#               -> {"passages": [{"doc_id": str, "title": str, "text": str}, ...]}   best first, at most 256
bash scripts/new_run.sh mine-hotpotqa hotpotqa $PWD/my_system
sbatch -p <partition> --gres=gpu:1 --cpus-per-task=12 --mem=240G --time=12:00:00 \
  slurm/run.sbatch mine-hotpotqa my-system
```

Passages must be verbatim corpus excerpts: a question whose evidence does not match the corpus gets no evidence. The
full contract and budgets (≤4 generator calls, ≤32,000 tokens and ≤60 s per question) are in
`materials/README_fixed.md` and `docs/protocol.md`.

### 4. Results

| File in `records/<run_id>/` | Content |
|-|-|
| `run_result.json` | `outcome`: `evaluated_multireader`, `judge_flagged` or `workspace_over_limit` |
| `multireader_scores.json` | per-reader scores and `reader_macro_primary`, the mean over the five readers (the per-task number of Table II) |
| `judge_gate.json`, `agent_meta.json`, `run_job.log` | judge verdicts, agent exit status, job log |

The overall score of a configuration is the unweighted mean over the nine tasks. Test evidence and reader predictions
stay under `hidden/multireader/<run_id>/`. If a job fails after the agent finished (exit code 4), resubmit with
`slurm/run.sbatch <run_id> <system> --eval-only`; the frozen workspace is reused.

Tests of the evaluation code:

```bash
python -m unittest multireader.test_contract multireader.test_ports multireader.test_efficiency multireader.test_build_status traces.test_agent_segments
python examples/minimal_eval.py --task multihop_rag --n 50      # CPU-only BM25 -> evidence check, no API keys
```

## Citation
```bibtex
@article{zhang2026autoragbench,
  title={AutoRAGBench: A Benchmark for LLM Agents Automating RAG System Development},
  author={Zhang, Hang and Pan, Yuzhi and Zhong, Ziyao and Wang, Chaokun and Cao, Shuo and Huang, Zeyu and Xue, Yue and Niu, Fang and Xie, Bofan and Ge, Guanchen and others},
  journal={arXiv preprint arXiv:2611.25582},
  year={2026}
}
```

## License

Code: MIT (`LICENSE`). Adapted third-party baselines keep their upstream licenses; each names its upstream repository
and commit. The data derive from NQ / DPR Wikipedia, HotpotQA, MuSiQue, MultiHop-RAG, PubMedQA, QASPER, FEVER, WebQSP
and ComplexWebQuestions (via the RoG subgraphs) under their original licenses.
