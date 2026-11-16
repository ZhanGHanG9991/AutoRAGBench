"""HAMMER search for one AutoRAGBench task (runs inside the Singularity build sandbox, see build.sbatch).

Stage 1  query selection: upstream L2FaissSelector (Experiment/query_selection/run_selection_graph.py) on the
         formal-v2 public training split, k = 30%, its main() settings (tau 0.249, 150 samples/node, top-50
         excluded); embedding model BAAI/bge-large-en-v1.5 (the class default; the repository .env points to an
         OpenAI embedding API, which AutoRAGBench forbids).
Stage 2  memory-guided MCTS: upstream EnhancedMCTSRAGOptimizer.optimization_engine.suggest_parameters(), the
         search space of the technical report (Table 9), local Qwen3.5-4B memory LLM and generator.
         Reward of a rollout = memory-refined score of the native task metric on the selected questions.
Output   the configuration with the maximum reward (paper, Algorithm 3), its index, every trial, the SE-Bank.
The dev and test splits are never read here.
"""
import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("AUTORAGBENCH_ROOT", Path(__file__).resolve().parents[1]))
H = ROOT / "humansota/HAMMER"
sys.path[:0] = [str(Path(__file__).resolve().parent), str(H / "rageng"), str(ROOT)]
import hammer_rageng as hr


class SearchDeadline(BaseException):
    """Raised before a new rollout once the wall-clock budget is spent (passes upstream `except Exception`)."""


def rows(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def write_json(path, value):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1, default=str))
    tmp.replace(path)


def select_queries(train, seed):
    """Upstream greedy query selection (Algorithm 1 implementation of the repository)."""
    import numpy as np
    from types import SimpleNamespace
    random.seed(seed)
    np.random.seed(seed)
    from Experiment.query_selection import run_selection_graph as rsg
    # The script passes a numpy vector as QueryBundle.embedding; llama-index-core 0.10.61 (pydantic v1 events)
    # only accepts a list. Same vector, as a list.
    query_bundle = rsg.QueryBundle
    rsg.QueryBundle = lambda query_str, embedding=None, **kw: query_bundle(
        query_str=query_str, embedding=None if embedding is None else [float(x) for x in embedding], **kw)
    model_path = os.environ.get("HAMMER_QUERY_EMBED_MODEL", "BAAI/bge-large-en-v1.5")
    selector = rsg.L2FaissSelector(model_name=model_path)
    k = round(0.3 * len(train))
    _, idx, graph = selector.select_queries([SimpleNamespace(question=r["question"]) for r in train], k,
                                            similarity_threshold=0.249, num_samples_per_node=150, exclude_top_k=50)
    del selector
    import torch
    torch.cuda.empty_cache()
    idx = sorted(idx)
    return [train[i] for i in idx], {"k": k, "selected_train_indices": idx, "tau": 0.249, "samples_per_node": 150,
                                     "exclude_top_k": 50, "embedding_model": "BAAI/bge-large-en-v1.5",
                                     "embedding_model_path": model_path, "seed": seed,
                                     "graph_edges": sum(len(v) for v in graph.values()) // 2}


def run(args):
    if os.environ.get("HAMMER_BUILD_SANDBOX") != "singularity":
        raise RuntimeError("the search must run through build.sbatch (Singularity sandbox)")
    unexpected = {k for k in os.environ if k.endswith("API_KEY")}
    if unexpected:
        raise RuntimeError("unexpected credentials in the search environment: " + ",".join(sorted(unexpected)))
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.chdir(out)   # upstream writes Experiment/graph_memory/... and its CSV relative to the working directory
    started = time.monotonic()
    deadline = started + args.seconds
    public = Path(args.public).resolve()
    cache = Path(args.cache).resolve()
    if "formal-v2" not in str(public):
        raise RuntimeError("formal campaign refuses a non-formal-v2 public path")
    hr.setup(H, os.environ["HAMMER_LLM_BASE_URL"], cache, args.task)
    space = json.loads((H / "rageng/search_space.json").read_text())
    embedders = args.embedding_models.split(",") if args.embedding_models else space["embedding_models"]
    os.environ["Optional_EMBED_MODEL"] = ",".join(embedders)   # read by RAGSearchSpace

    t0 = time.monotonic()
    train_path = Path(args.train).resolve() if args.train else public / "train.jsonl"
    selected, selection = select_queries(rows(train_path), args.seed)
    selection["train_path"] = str(train_path)
    selection["seconds"] = time.monotonic() - t0
    write_json(out / "query_selection.json", selection)
    corpus, qa = hr.write_upstream_dataset(args.task, selected, public / "corpus/corpus.jsonl", out / "data")
    hr.set_question_metas([r.get("meta", {}) for r in selected])

    from hammer.tuner import main_tuner_mcts as mt
    mt.MODEL_MAXWORKERS = hr.GENERATOR_WORKERS
    mt.get_dataset_prompt = lambda name: hr.dataset_description(args.task)

    class RagEngDatasetConfig(mt.SimpleDatasetConfig):
        def __init__(self):   # SimpleDatasetConfig.__init__ only knows the paper's dataset files
            self.dataset_name = "rageng-" + args.task
            self.train_size = len(selected)
            self.name = "mcts-" + self.dataset_name
            self.corpus_file, self.qa_file = str(corpus), str(qa)
            self.dataset = mt.create_simple_dataset(self.corpus_file, self.qa_file, self.dataset_name)
            self.max_workers = mt.MODEL_MAXWORKERS
            self.search_space = mt.SimpleSearchSpace()
            self.model_config = {"extra": "forbid", "yaml_file": None}
            self.toy_mode = False
            self.timeouts = mt.SimpleTimeoutConfig()
            self.optimization = mt.SimpleOptimizationConfig()

    # Native metric as the measured score of a rollout (HAMMER's own F1 is kept as hammer_f1).
    original_evaluate = mt.MCTSEvaluationManager.evaluate_flow
    last = {}

    def evaluate_flow(self, flow):
        objective, results = original_evaluate(self, flow)
        logs = results.get("qa_execution_logs") or []
        if results.get("failed") or len(logs) != len(selected):
            results["train_answer_f1"] = 0.0
            last.update(predictions=None, failure=results.get("exception_message", "evaluation failed"))
            return 0.0, results
        answers = []
        for log, gold in zip(logs, selected):
            assert log["question"] == gold["question"], "QA log order differs from the selected questions"
            answers.append(hr.final_answer(log["predicted_answer"]) if log.get("predicted_answer") else None)
        scores = hr.native_scores(args.task, answers, selected)
        for log, s in zip(logs, scores):
            log["hammer_f1_score"], log["f1_score"] = log.get("f1_score"), s
        results["hammer_train_answer_f1"] = results.get("train_answer_f1")
        results["train_answer_f1"] = sum(scores) / len(scores)
        last.update(predictions=[{"qid": g["qid"], "answer": a, "score": s, "response": l.get("predicted_answer")}
                                 for g, a, s, l in zip(selected, answers, scores, logs)], failure=None)
        return results["train_answer_f1"], results

    mt.MCTSEvaluationManager.evaluate_flow = evaluate_flow

    trials = []
    original_single = mt.EnhancedMCTSRAGOptimizer.evaluate_single_flow

    def evaluate_single_flow(self, params, use_simulation=True):
        if time.monotonic() >= deadline:
            raise SearchDeadline()
        n = len(trials) + 1
        t = time.monotonic()
        memory_before = dict(memory_llm.usage)
        record = {"trial": n, "params": params, "started_s": t - started}
        try:
            objective, results, flow_json = original_single(self, params, use_simulation)
            record.update(status="ok", reward=objective, measured=results["real_f1_score"],
                          hammer_f1=results.get("hammer_train_answer_f1"), failure=last.get("failure"),
                          generator_tokens=results.get("total_tokens"), n_api_failed=results.get("api_failed_count"))
        except Exception as e:
            record.update(status="error", reward=None, measured=None, failure=repr(e)[:2000])
            raise
        finally:
            record["seconds"] = time.monotonic() - t
            record["memory_llm"] = {k: memory_llm.usage[k] - memory_before[k] for k in memory_llm.usage}
            trials.append(record)
            with (out / "trials.jsonl").open("a") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            if last.get("predictions"):
                write_json(out / f"trial-{n:02d}-predictions.json", last["predictions"])
            last.clear()
            print("TRIAL_RESULT " + json.dumps({k: record.get(k) for k in ("trial", "status", "measured", "reward",
                  "seconds")}), flush=True)
        return objective, results, flow_json

    mt.EnhancedMCTSRAGOptimizer.evaluate_single_flow = evaluate_single_flow

    memory_llm = hr.LocalQwenClient(out / "memory_llm_exchanges.jsonl")
    optimizer = mt.EnhancedMCTSRAGOptimizer(dataset_config=RagEngDatasetConfig(), api_key="unused",
                                            api_base="https://api.deepseek.com", optimization_target="train_answer_f1",
                                            csv_file=str(out / "Experiment/mcts_results.csv"))
    engine = optimizer.optimization_engine
    for agent in {id(a): a for a in (optimizer.insight_agent, engine.insight_agent,
                                     optimizer.simulation_evaluator.insight_agent)}.values():
        agent.llm_client, agent.model_name = memory_llm, hr.LocalQwenClient.MODEL
    optimizer.simulation_evaluator.gpt_client = memory_llm
    optimizer.simulation_evaluator.model_name = hr.LocalQwenClient.MODEL
    optimizer.set_mcts_iterations(args.iterations)

    write_json(out / "search_budget.json", {"iterations": args.iterations, "seconds": args.seconds,
               "stop_policy": "no new rollout after the deadline", "generator_workers": hr.GENERATOR_WORKERS,
               "memory_llm": "local Qwen3.5-4B (thinking off)", "generator": "Qwen3.5-4B (thinking off)",
               "memory_llm_deviation": "local Qwen3.5-4B replaces existing DeepSeek and paper Qwen2.5-72B",
               "embedding_models": embedders, "tuning_questions": len(selected), "tuning_split": "train (query-selected)",
               "build_sandbox": os.environ.get("HAMMER_BUILD_SANDBOX"), "upstream_commit": hr.UPSTREAM_COMMIT})
    stopped = "iterations"
    upstream_choice = None
    try:
        upstream_choice = engine.suggest_parameters()
    except SearchDeadline:
        stopped = "deadline"
    engine.graph_memory.save_all_layers()

    ok = [t for t in trials if t["status"] == "ok"]
    if not ok:
        raise RuntimeError("no successful rollout")
    best = max(ok, key=lambda t: (t["reward"], t["measured"], -t["trial"]))
    # Export the index files the chosen configuration uses (cache hits; nothing is recomputed).
    hr._state["used"] = set()
    flow = optimizer.flow_builder.build_flow(optimizer._ensure_required_params_for_callback(best["params"]))
    best_dir = out / "best"
    (best_dir / "index").mkdir(parents=True)
    for name in sorted(hr._state["used"]):
        shutil.copy2(cache / name, best_dir / "index" / name)
    del flow
    deployed = optimizer._ensure_required_params_for_callback(best["params"])
    # Parity check material (build.sbatch replays these tuning questions through the frozen service).
    predictions = json.loads((out / f"trial-{best['trial']:02d}-predictions.json").read_text())
    with (best_dir / "parity_questions.jsonl").open("w") as f:
        for r in selected[:30]:
            f.write(json.dumps({"qid": r["qid"], "question": r["question"], "meta": r.get("meta", {})}, ensure_ascii=False) + "\n")
    write_json(best_dir / "parity_expected.json", {p["qid"]: p["answer"] for p in predictions[:30]})
    write_json(best_dir / "artifact.json", {"task": args.task, "params": deployed, "index_files": sorted(hr._state["used"]),
               "trial": best["trial"], "upstream_commit": hr.UPSTREAM_COMMIT})
    write_json(out / "recommended.json", {"trial": best["trial"], "reward": best["reward"], "measured_on_selected_train":
               best["measured"], "metric": hr._state["info"]["primary_metric"], "params": deployed,
               "selection_rule": "maximum reward (paper Algorithm 3)", "upstream_most_visited_path": upstream_choice,
               "hidden_test_used": False, "dev_used": False})
    write_json(out / "completed.json", {"task": args.task, "n_trials": len(trials), "n_ok": len(ok), "stopped_by": stopped,
               "seconds": time.monotonic() - started, "memory_llm_usage": memory_llm.usage,
               "graph_memory": engine.get_graph_memory_stats(), "hidden_test_used": False})
    print("SEARCH_DONE " + json.dumps({"best_trial": best["trial"], "reward": best["reward"], "measured": best["measured"],
                                       "n_trials": len(trials), "stopped_by": stopped}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--public", required=True)
    p.add_argument("--train", default="")
    p.add_argument("--cache", required=True)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--seconds", type=int, default=6000)
    p.add_argument("--seed", type=int, default=20261022)
    p.add_argument("--embedding-models", default="", help="smoke tests only: override the Table 9 embedding list")
    run(p.parse_args())
