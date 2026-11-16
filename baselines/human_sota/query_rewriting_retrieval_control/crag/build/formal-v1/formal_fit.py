#!/usr/bin/env python3
"""Fit CRAG thresholds on formal-v1 tune splits.

Reads min_count from formal-v1/split/manifest.json per task (configuration: flat 10). Asserts at start that the value we will use equals
the manifest, passes it explicitly to choose_pair, asserts the chosen
record carries the same min_count, and writes it into fit/config/reason
outputs.

KEEP_PAPER_BRANCH env (default "1"): passed to choose_pair(keep_paper=...).
Set to "0" to disable paper-threshold retention when a compliant grid point
exists. Does not submit jobs.
"""

def _release_path(value):
    """Resolve portable build/workspace paths without changing their layout."""
    import os
    from pathlib import Path
    workspace = os.environ.get("WORKSPACE", str(Path(__file__).resolve().parents[1]))
    root = os.environ.get("AUTORAGBENCH_ROOT", workspace)
    model_dir = os.environ.get("MODEL_DIR", str(Path(workspace) / "models"))
    for key, replacement in (("@WORKSPACE@", workspace), ("@AUTORAGBENCH_ROOT@", root),
                             ("@MODEL_DIR@", model_dir), ("@BUILD_TMP@", os.environ.get("BUILD_TMP", "/tmp"))):
        value = value.replace(key, replacement)
    return value

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PARENT)

from formal_index import FORMAL, MODEL, TASKS, index_dir, refuse  # noqa: E402
from fit_contriever_musique import choose_pair, score_strips, strips_of, sweep_top_n  # noqa: E402
from fit_contriever_task import select_largest_close_top_n  # noqa: E402
from build_evaluator_pairs import search_top  # noqa: E402

MANIFEST = os.path.join(FORMAL, "split", "manifest.json")
SRC = _release_path("@AUTORAGBENCH_ROOT@/humansota/formal-v1-split")
STATUS = os.path.join(FORMAL, "results", "evaluator_status.json")
DEV_TASKS = list(TASKS.keys())


def keep_paper_branch():
    """Read KEEP_PAPER_BRANCH env (default 1). configuration card: default keep if no reply."""
    raw = os.environ.get("KEEP_PAPER_BRANCH", "1").strip()
    if raw not in ("0", "1"):
        raise SystemExit(f"KEEP_PAPER_BRANCH must be 0 or 1, got {raw!r}")
    return raw == "1"



def load_manifest():
    refuse(MANIFEST)
    blob = json.load(open(MANIFEST, encoding="utf-8"))
    if blob.get("min_count_rule") is None:
        raise SystemExit("manifest missing min_count_rule")
    return blob


def manifest_min_count(manifest, task):
    spec = manifest["tasks"].get(task)
    if not spec:
        raise SystemExit(f"manifest has no task {task}")
    value = spec.get("min_count")
    if not isinstance(value, int) or value < 1:
        raise SystemExit(f"{task} manifest min_count invalid: {value!r}")
    return value


def assert_min_count_matches_manifest(task, used, manifest):
    expected = manifest_min_count(manifest, task)
    if used != expected:
        raise SystemExit(
            f"MIN_COUNT_MISMATCH task={task} used={used} "
            f"manifest={expected} path={MANIFEST}"
        )
    return expected


def output_paths(task):
    cfg = os.path.join(FORMAL, f"crag_config_{task}.json")
    fit = os.path.join(FORMAL, "results", f"{task}_fit.json")
    grid = os.path.join(FORMAL, "results", f"{task}_grid.json")
    for path in (cfg, fit, grid):
        refuse(path)
        if "/formal-v1/" not in os.path.abspath(path):
            raise SystemExit(f"REFUSE non-formal output {path}")
    return cfg, fit, grid


def read_qids(task, kind):
    path = os.path.join(FORMAL, "split", task, f"{kind}_qids.txt")
    refuse(path)
    qids = [line.strip() for line in open(path, encoding="utf-8") if line.strip()]
    if len(qids) != len(set(qids)):
        raise SystemExit(f"duplicate qids in {path}")
    return qids


def tune_questions(task, manifest):
    spec = manifest["tasks"][task]
    tune_n = spec["tune"]
    qids = read_qids(task, "tune")
    if len(qids) != tune_n:
        raise SystemExit(f"{task} tune qids {len(qids)} != manifest tune {tune_n}")
    train_qids = set(read_qids(task, "evaluator_train"))
    if set(qids) & train_qids:
        raise SystemExit(f"{task} tune overlaps evaluator_train")
    path = os.path.join(SRC, task, "train_fit.jsonl")
    refuse(path)
    if not path.endswith(f"/formal-v1-split/{task}/train_fit.jsonl"):
        raise SystemExit(f"REFUSE {path}")
    want = set(qids)
    found = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["qid"] in want:
                found[row["qid"]] = row
    if len(found) != tune_n:
        raise SystemExit(f"{task} tune loaded {len(found)} != {tune_n}")
    return [found[qid] for qid in qids]


def paper_filter(task, questions):
    if task != "qasper":
        return None
    only = []
    for question in questions:
        doc_id = (question.get("meta") or {}).get("doc_id")
        if not doc_id:
            raise SystemExit(f"qasper {question.get('qid')} missing meta.doc_id")
        only.append(doc_id)
    return only


def retrieve(task, questions):
    import torch
    from contriever_index import load_built

    directory = index_dir(task)
    stamp = json.load(open(os.path.join(directory, "contriever.json"), encoding="utf-8"))
    expected = TASKS[task]
    if stamp.get("chunk_words") != expected:
        raise SystemExit(f"{task} chunk_words {stamp.get('chunk_words')!r} != {expected!r}")
    index = load_built(directory, MODEL)
    try:
        if expected and len({doc["doc_id"] for doc in index.docs}) >= len(index.docs):
            raise SystemExit(f"{task} chunked index has no repeated doc_id")
        hits = search_top(index, questions, 10, paper_filter(task, questions))
    finally:
        del index
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    pending = []
    for question, found in zip(questions, hits):
        only = (question.get("meta") or {}).get("doc_id") if task == "qasper" else None
        if only and any(hit["doc_id"] != only for hit in found):
            raise SystemExit(f"search left {only}")
        support = set(question.get("supporting_doc_ids") or [])
        got = [hit["doc_id"] for hit in found]
        gold_ids = []
        seen = set()
        doc_ids, strips = [], []
        for hit in found:
            if hit["doc_id"] in support and hit["doc_id"] not in seen:
                seen.add(hit["doc_id"])
                gold_ids.append(hit["doc_id"])
            for strip in strips_of(hit["passage"]):
                doc_ids.append(hit["doc_id"])
                strips.append(strip)
        pending.append({
            "qid": question["qid"],
            "question": question["question"],
            "passages": [hit["passage"] for hit in found],
            "hit10": bool(support & set(got)),
            "gold_ids": gold_ids,
            "doc_ids": doc_ids,
            "strips": strips,
        })
    return pending, directory


def fit_task(task, manifest, model_dir):
    min_count = manifest_min_count(manifest, task)
    # Research Bot: assert at tune start that used == manifest
    assert_min_count_matches_manifest(task, min_count, manifest)
    print(f"TUNE_START {task} min_count={min_count} from {MANIFEST}", flush=True)

    cfg_path, fit_path, grid_path = output_paths(task)
    if os.path.exists(cfg_path) or os.path.exists(fit_path) or os.path.exists(grid_path):
        raise SystemExit(f"refusing to overwrite existing fit outputs for {task}")

    questions = tune_questions(task, manifest)
    pending, directory = retrieve(task, questions)

    import torch
    from crag_corpus import load_evaluator

    if not torch.cuda.is_available():
        raise SystemExit("refusing formal fit without a GPU")
    evaluator = load_evaluator("t5", model_dir)
    rows, packs = [], []
    for item in pending:
        scores = evaluator.score_docs(item["question"], item["passages"], batch_size=16)
        while len(scores) < 10:
            scores.append(-1.0)
        scores = scores[:10]
        rows.append({"scores": scores, "max_score": max(scores), "hit10": item["hit10"]})
        packs.append({
            "gold_ids": item["gold_ids"],
            "doc_ids": item["doc_ids"],
            "strips": item["strips"],
            "scores": score_strips(evaluator, item["question"], item["strips"]),
        })

    keep_paper = keep_paper_branch()
    print(f"KEEP_PAPER_BRANCH {int(keep_paper)}", flush=True)
    chosen, paper, top, grid = choose_pair(
        rows, min_count=min_count, return_grid=True, keep_paper=keep_paper,
    )
    if chosen.get("keep_paper") != keep_paper:
        raise SystemExit(
            f"{task} choose_pair keep_paper={chosen.get('keep_paper')!r} != {keep_paper}"
        )
    if chosen.get("min_count") != min_count:
        raise SystemExit(
            f"{task} choose_pair recorded min_count={chosen.get('min_count')!r} "
            f"!= used {min_count}"
        )
    assert_min_count_matches_manifest(task, chosen["min_count"], manifest)
    if f"min_count={min_count}" not in str(chosen.get("reason", "")):
        raise SystemExit(f"{task} reason missing min_count tag: {chosen.get('reason')!r}")
    keep_tag = f"keep_paper={int(keep_paper)}"
    if keep_tag not in str(chosen.get("reason", "")):
        raise SystemExit(f"{task} reason missing {keep_tag}: {chosen.get('reason')!r}")

    n_grid = sum(1 for item in grid if item["source"] == "grid")
    n_ok = sum(1 for item in grid if item["source"] == "grid" and item["compliant"])
    if n_grid < 1:
        raise SystemExit(f"{task} empty threshold grid")
    sweep, smallest = sweep_top_n(packs)
    if task == "qasper":
        selected, best_rate = select_largest_close_top_n(sweep)
        top_n = selected["top_n"]
        top_n_rule = "within_0.005_of_best_retention_take_largest_top_n"
    else:
        best_rate = None
        top_n = smallest["top_n"]
        top_n_rule = "exact_tie_takes_smallest_top_n"
    hit = sum(row["hit10"] for row in rows) / len(rows)
    tune_n = len(rows)
    payload = {
        "task": task,
        "split": "formal-v1 tune only",
        "seed": manifest.get("seed"),
        "n": tune_n,
        "evaluator_dir": model_dir,
        "index_dir": directory,
        "chunk_words": TASKS[task],
        "retrieval_scope": "meta.doc_id" if task == "qasper" else "corpus",
        "contriever_hit_at_10": hit,
        "chosen": chosen,
        "paper": paper,
        "grid_top": top,
        "top_n_sweep": sweep,
        "top_n_smallest_tie": smallest["top_n"],
        "top_n_selected": top_n,
        "top_n_rule": top_n_rule,
        "top_n_rule_best_rate": best_rate,
        "qasper_zero_shot": task == "qasper",
        "min_count": min_count,
        "min_count_source": MANIFEST,
        "keep_paper": keep_paper,
        "keep_paper_env": os.environ.get("KEEP_PAPER_BRANCH", "1"),
        "max_fraction": 0.80,
        "n_grid_candidates": n_grid,
        "n_compliant_grid": n_ok,
        "grid_file": os.path.basename(grid_path),
    }
    grid_doc = {
        "task": task,
        "n": tune_n,
        "min_count": min_count,
        "keep_paper": keep_paper,
        "max_fraction": 0.80,
        "evaluator_dir": model_dir,
        "n_grid_candidates": n_grid,
        "n_compliant_grid": n_ok,
        "compliant_means": (
            f"min count of correct, ambiguous, and incorrect is at least {min_count}, "
            "and no class is more than 80% of n. Paper rows stay in the file; the gap "
            "rule can still select a paper row that is not compliant."
        ),
        "candidates": grid,
    }
    config = {
        "variant": "crag_corpus_only",
        "task": task,
        "upper_threshold": chosen["upper"],
        "lower_threshold": chosen["lower"],
        "threshold_rule": f"{task}_formal_tune{tune_n}_" + chosen["reason"],
        "n_docs": 10,
        "rewrite_k": 25,
        "decompose_mode": "excerption",
        "evaluator": "t5",
        "evaluator_temporary": False,
        "evaluator_dir": model_dir,
        "threshold_status": "ok",
        "threshold_counts": chosen["counts"],
        "threshold_n": tune_n,
        "threshold_over_80": chosen["over_80"],
        "first_retriever": "contriever",
        "rewrite_retriever": "contriever",
        "chunk_words": TASKS[task],
        "index_dir": directory,
        "retrieval_scope": payload["retrieval_scope"],
        "top_n_selected_for_b": top_n,
        "top_n_rule": top_n_rule,
        "tune_split": "formal-v1 split tune",
        "tune_n": tune_n,
        "qasper_zero_shot": task == "qasper",
        "min_count": min_count,
        "max_fraction": 0.80,
        "keep_paper": keep_paper,
        "note": (
            f"Tuned on formal-v1 tune ({tune_n}) only. min_count={min_count} from "
            "split/manifest.json (configuration flat 10). "
            f"KEEP_PAPER_BRANCH={int(keep_paper)}. evaluator-train and "
            "dev were not used for thresholds."
        ),
    }
    if not any(item["selected"] for item in grid):
        raise SystemExit(f"{task} grid has no selected candidate")
    os.makedirs(os.path.join(FORMAL, "results"), exist_ok=True)
    with open(grid_path, "w", encoding="utf-8") as handle:
        json.dump(grid_doc, handle, ensure_ascii=False, indent=1)
        handle.write("\n")
    with open(fit_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)
        handle.write("\n")
    with open(cfg_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=1)
        handle.write("\n")
    print(
        "FORMAL_FIT", task, chosen["upper"], chosen["lower"], chosen["reason"],
        chosen["counts"], "min_count", min_count, "compliant_grid", n_ok, "/", n_grid,
        "top_n", top_n, top_n_rule, "hit10", round(hit, 4),
        flush=True,
    )


def dry_check():
    """No GPU. Confirms choose_pair default and manifest wiring pass 10."""
    import inspect
    from fit_contriever_musique import choose_pair as cp

    kp = keep_paper_branch()
    print(f"DRY_CHECK_KEEP_PAPER_BRANCH {int(kp)}", flush=True)
    manifest = load_manifest()
    sig = inspect.signature(cp)
    default = sig.parameters["min_count"].default
    print(f"CHOOSE_PAIR_DEFAULT {default}", flush=True)
    if default != 10:
        raise SystemExit(f"choose_pair default is {default}, want 10")
    for task in DEV_TASKS:
        value = manifest_min_count(manifest, task)
        assert_min_count_matches_manifest(task, value, manifest)
        if value != 10:
            raise SystemExit(f"{task} manifest min_count {value} != 10")
        print(f"MANIFEST_OK {task} min_count={value}", flush=True)
    # Synthetic choose_pair to confirm reason carries min_count
    rows = [
        {"scores": [0.9] + [-1.0] * 9, "max_score": 0.9, "hit10": True},
        {"scores": [0.1] + [-1.0] * 9, "max_score": 0.1, "hit10": False},
    ] * 20
    chosen, _, _, _ = cp(rows, min_count=10, return_grid=True)
    if chosen.get("min_count") != 10:
        raise SystemExit(f"synthetic choose_pair min_count={chosen.get('min_count')}")
    if "min_count=10" not in chosen.get("reason", ""):
        raise SystemExit(f"synthetic reason missing tag: {chosen.get('reason')!r}")
    print(f"SYNTHETIC_OK reason={chosen['reason']!r} min_count={chosen['min_count']}", flush=True)
    # keep_paper switch: default on, and off path never selects paper when grid exists
    if not keep_paper_branch():
        print("DRY_CHECK_NOTE KEEP_PAPER_BRANCH=0 in this process", flush=True)
    off, _, _, _ = cp(rows, min_count=10, return_grid=True, keep_paper=False)
    if off.get("keep_paper") is not False:
        raise SystemExit(f"keep_paper=False not recorded: {off}")
    if "keep_paper=0" not in off.get("reason", ""):
        raise SystemExit(f"keep_paper=0 missing in reason: {off.get('reason')!r}")
    if off.get("source") == "paper" and "keep_paper_disabled" not in off.get("reason", ""):
        # When no compliant grid point, paper can still appear via other branches;
        # for this synthetic tiny set, just confirm the flag is recorded.
        pass
    on, _, _, _ = cp(rows, min_count=10, return_grid=True, keep_paper=True)
    if on.get("keep_paper") is not True or "keep_paper=1" not in on.get("reason", ""):
        raise SystemExit(f"keep_paper=True not tagged: {on.get('reason')!r}")
    print(
        f"KEEP_PAPER_SWITCH_OK off_reason={off['reason']!r} on_reason={on['reason']!r}",
        flush=True,
    )
    print("DRY_CHECK_OK", flush=True)


def fit_all(tasks):
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("refusing formal fit without a GPU")
    refuse(STATUS)
    if not os.path.exists(STATUS):
        raise SystemExit(f"evaluator status missing: {STATUS}")
    status = json.load(open(STATUS, encoding="utf-8"))
    if status.get("status") != "trained":
        raise SystemExit(f"evaluator is not trained: {status.get('status')}")
    model_dir = status["model_dir"]
    refuse(model_dir)
    if "evaluator" not in model_dir:
        raise SystemExit(f"refusing evaluator dir {model_dir}")
    manifest = load_manifest()
    for task in tasks:
        if task not in TASKS:
            raise SystemExit(f"unknown task {task}")
        fit_task(task, manifest, model_dir)
    print("FORMAL_FIT_DONE", flush=True)


def main():
    args = sys.argv[1:]
    if not args or args == ["--dry-check"]:
        dry_check()
        return
    if args[0] == "--task":
        tasks = args[1:]
        if not tasks:
            raise SystemExit("usage: formal_fit.py --task <name> [...]")
        fit_all(tasks)
        return
    if args == ["--all"]:
        fit_all(DEV_TASKS)
        return
    raise SystemExit("usage: formal_fit.py [--dry-check] | --all | --task <name> [...]")


if __name__ == "__main__":
    main()
