"""Fit one text task's Contriever thresholds and top_n on its public train.

200 questions, seed 20261017, same decision rule as MuSiQue. Writes a new
config and does not touch crag_config.json or the MuSiQue Contriever config.
QASPER is chunked at 200 words and retrieved inside meta.doc_id.

--chunk-words 200 --suffix _w200 builds a new multihop_rag index and config.
It does not replace the published whole-document multihop_rag files.
--qasper-tie rewrites only crag_config_contriever_qasper_top16.json from the
saved train sweep. It does not refit thresholds and does not edit the old files.
The default sweep still takes the smallest top_n on an exact tie.
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

import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from fit_contriever_musique import (  # noqa: E402
    ORIGINAL_TOP_N,
    choose_pair,
    refuse,
    score_strips,
    strips_of,
    sweep_top_n,
)

ROOT = _release_path("@AUTORAGBENCH_ROOT@")
C = os.path.join(ROOT, "humansota/CorrectiveRAG")
TASKS = ["fever", "hotpotqa", "multihop_rag", "nq_open", "popqa", "pubmedqa", "qasper"]
PROTECTED = {
    os.path.join(C, "rageng/crag_config.json"),
    os.path.join(C, "rageng/crag_config_contriever_musique.json"),
}


ALLOWED_SUFFIX = ("", "_w200", "_top16")


def published_cfg(task):
    return os.path.join(C, "rageng", f"crag_config_contriever_{task}.json")


def task_paths(task, chunk_words=None, suffix=""):
    if task not in TASKS:
        raise SystemExit(f"refusing task {task}")
    if suffix not in ALLOWED_SUFFIX:
        raise SystemExit(f"refusing suffix {suffix}")
    train = os.path.join(ROOT, "data", task, "public", "train.jsonl")
    corpus = os.path.join(ROOT, "data", task, "public", "corpus")
    published_index = os.path.join(C, "rageng/index", f"{task}-contriever")
    if task == "qasper":
        if chunk_words not in (None, 200):
            raise SystemExit("qasper chunk_words stays 200")
        if suffix == "_w200":
            raise SystemExit("qasper already has a w200 index; use --qasper-tie")
        chunk_words = 200
        index = os.path.join(C, "rageng/index/qasper-contriever-w200")
    elif chunk_words is None:
        if suffix:
            raise SystemExit("a whole-document fit cannot take a suffix")
        index = published_index
    else:
        if chunk_words != 200 or suffix != "_w200" or task != "multihop_rag":
            raise SystemExit("chunked rerun is only multihop_rag --chunk-words 200 --suffix _w200")
        index = os.path.join(C, "rageng/index/multihop_rag-contriever-w200")
        if os.path.realpath(index) == os.path.realpath(published_index):
            raise SystemExit("refusing to rebuild the published multihop_rag index")
    fit_out = os.path.join(C, "rageng/results", f"contriever_{task}{suffix}_fit.json")
    cfg_out = os.path.join(C, "rageng", f"crag_config_contriever_{task}{suffix}.json")
    for path in (train, corpus, index, fit_out, cfg_out):
        refuse(path)
    protected = {os.path.realpath(p) for p in PROTECTED}
    protected.add(os.path.realpath(published_cfg(task)))
    if suffix and os.path.realpath(cfg_out) in protected:
        raise SystemExit(f"refusing protected config {cfg_out}")
    if not suffix and os.path.realpath(cfg_out) in {os.path.realpath(p) for p in PROTECTED}:
        raise SystemExit(f"refusing protected config {cfg_out}")
    if "/public/" not in train or not train.endswith("/train.jsonl"):
        raise SystemExit(f"refusing non-public train file {train}")
    return train, corpus, index, chunk_words, fit_out, cfg_out


def load_sample(train):
    questions = [json.loads(line) for line in open(train, encoding="utf-8")]
    if len(questions) < 200:
        raise SystemExit(f"{train} has {len(questions)} rows")
    return random.Random(20261017).sample(questions, 200)


def doc_filter(task, question):
    doc_id = ((question.get("meta") or {}).get("doc_id") or None)
    if task == "qasper":
        if not doc_id:
            raise SystemExit(f"qasper question {question.get('qid')} missing meta.doc_id")
        return doc_id
    return None


def reuse_or_none(task, fit_out, cfg_out, chunk_words):
    if not os.path.exists(cfg_out):
        return None
    cfg = json.load(open(cfg_out, encoding="utf-8"))
    if cfg.get("task") != task or cfg.get("threshold_status") != "ok" or cfg.get("evaluator") != "t5":
        raise SystemExit(f"refusing to overwrite {cfg_out}")
    if cfg.get("evaluator_temporary"):
        raise SystemExit(f"refusing temporary evaluator config {cfg_out}")
    if cfg.get("chunk_words") != chunk_words:
        raise SystemExit(f"refusing to overwrite {cfg_out}: chunk_words differ")
    if not os.path.exists(fit_out):
        raise SystemExit(f"config exists but fit record is missing: {fit_out}")
    print(
        "CONFIG_REUSED", task, cfg["upper_threshold"], cfg["lower_threshold"],
        "top_n", cfg["top_n_selected_for_b"], "over_80", cfg.get("threshold_over_80"),
        flush=True,
    )
    return cfg


def select_largest_close_top_n(sweep, tol=0.005):
    """Among sweep rows within tol of the best rate, take the largest top_n.

    This is the qasper-only rule. sweep_top_n still takes the smallest top_n
    on an exact tie and is not used here.
    """
    if not sweep:
        raise SystemExit("empty top_n sweep")
    best_rate = max(row["rate"] for row in sweep)
    close = [row for row in sweep if best_rate - row["rate"] <= tol + 1e-12]
    return max(close, key=lambda row: row["top_n"]), best_rate


def qasper_tie():
    """Copy qasper thresholds and select top_n from the saved train sweep."""
    old_fit_path = os.path.join(C, "rageng/results/contriever_qasper_fit.json")
    old_cfg_path = published_cfg("qasper")
    new_fit_path = os.path.join(C, "rageng/results/contriever_qasper_top16_fit.json")
    new_cfg_path = os.path.join(C, "rageng/crag_config_contriever_qasper_top16.json")
    index = os.path.join(C, "rageng/index/qasper-contriever-w200")
    for path in (old_fit_path, old_cfg_path, new_fit_path, new_cfg_path, index):
        refuse(path)
    if os.path.realpath(new_cfg_path) == os.path.realpath(old_cfg_path):
        raise SystemExit("refusing to overwrite the published qasper config")
    old_fit_bytes = open(old_fit_path, "rb").read()
    old_cfg_bytes = open(old_cfg_path, "rb").read()
    old_fit = json.loads(old_fit_bytes)
    old_cfg = json.loads(old_cfg_bytes)
    chosen_row, best_rate = select_largest_close_top_n(old_fit["top_n_sweep"])
    if chosen_row["top_n"] != 16:
        raise SystemExit(f"tie rule selected {chosen_row['top_n']}, expected 16")
    if old_fit.get("top_n_selected") != 1:
        raise SystemExit(f"saved smallest-tie top_n is {old_fit.get('top_n_selected')}, expected 1")
    if old_cfg.get("upper_threshold") != old_fit["chosen"]["upper"]:
        raise SystemExit("old qasper config and fit upper thresholds disagree")
    if old_cfg.get("lower_threshold") != old_fit["chosen"]["lower"]:
        raise SystemExit("old qasper config and fit lower thresholds disagree")
    if old_cfg.get("top_n_selected_for_b") != 1:
        raise SystemExit("published qasper config is no longer top_n 1")
    rule = "within_0.005_of_best_retention_take_largest_top_n"
    note = (
        "Qasper only. Thresholds are copied from crag_config_contriever_qasper.json and are not refit. "
        "top_n 16 is chosen from the saved train retention sweep: rates within 0.005 of the best "
        "take the largest top_n. The rule was added after the dev run that used the smallest-top_n tie. "
        "Dev scores are not used to choose top_n. Does not replace the old qasper config."
    )
    payload = dict(old_fit)
    payload["top_n_selected"] = 16
    payload["top_n_selected_by_smallest_tie"] = old_fit["top_n_selected"]
    payload["top_n_rule"] = rule
    payload["top_n_rule_tolerance"] = 0.005
    payload["top_n_rule_best_rate"] = best_rate
    payload["top_n_rule_scope"] = "qasper only"
    payload["top_n_note"] = note
    payload["source_fit"] = old_fit_path
    payload["index_dir"] = index
    config = dict(old_cfg)
    config["top_n_selected_for_b"] = 16
    config["top_n_rule"] = rule
    config["index_dir"] = index
    config["note"] = note
    if os.path.exists(new_fit_path) or os.path.exists(new_cfg_path):
        have_fit = json.load(open(new_fit_path, encoding="utf-8"))
        have_cfg = json.load(open(new_cfg_path, encoding="utf-8"))
        if have_cfg.get("top_n_selected_for_b") != 16 or have_fit.get("top_n_selected") != 16:
            raise SystemExit("qasper top16 files exist but do not select 16")
        if have_cfg.get("upper_threshold") != old_cfg["upper_threshold"]:
            raise SystemExit("existing qasper top16 config changed the upper threshold")
        print("QASPER_TIE_REUSED", 16, have_cfg["upper_threshold"], have_cfg["lower_threshold"])
    else:
        with open(new_fit_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
            f.write("\n")
        with open(new_cfg_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=1)
            f.write("\n")
        print("QASPER_TIE", 16, config["upper_threshold"], config["lower_threshold"])
        print("WROTE", new_fit_path)
        print("WROTE", new_cfg_path)
    if open(old_fit_path, "rb").read() != old_fit_bytes:
        raise SystemExit("published qasper fit changed")
    if open(old_cfg_path, "rb").read() != old_cfg_bytes:
        raise SystemExit("published qasper config changed")
    print("QASPER_OLD_UNCHANGED")


def fit(task, chunk_words=None, suffix=""):
    import torch
    from contriever_index import build_or_load
    from crag_corpus import load_evaluator

    if not torch.cuda.is_available():
        raise SystemExit("refusing --task fit without a GPU")
    if suffix == "_top16":
        raise SystemExit("qasper top_n 16 is --qasper-tie, not a GPU fit")
    train, corpus, index_dir, chunk_words, fit_out, cfg_out = task_paths(
        task, chunk_words=chunk_words, suffix=suffix
    )
    if reuse_or_none(task, fit_out, cfg_out, chunk_words):
        return
    model = os.path.join(C, "models/contriever-msmarco")
    evaluator_dir = os.path.join(C, "models/evaluator")
    for path in (model, evaluator_dir):
        refuse(path)
    picked = load_sample(train)
    index = build_or_load(corpus, index_dir, model, batch_size=32, chunk_words=chunk_words)
    if chunk_words and len({d["doc_id"] for d in index.docs}) >= len(index.docs):
        raise SystemExit(f"{task} index has no repeated doc_id; chunking did not split documents")
    pending = []
    for q in picked:
        only = doc_filter(task, q)
        hits = index.search(q["question"], 10, only_doc_id=only)
        if only and any(h["doc_id"] != only for h in hits):
            raise SystemExit(f"search left {only}")
        support = set(q.get("supporting_doc_ids") or [])
        got = [h["doc_id"] for h in hits]
        gold_ids = []
        seen = set()
        doc_ids, strips = [], []
        for hit in hits:
            if hit["doc_id"] in support and hit["doc_id"] not in seen:
                seen.add(hit["doc_id"])
                gold_ids.append(hit["doc_id"])
            for strip in strips_of(hit["passage"]):
                doc_ids.append(hit["doc_id"])
                strips.append(strip)
        pending.append({
            "qid": q["qid"],
            "question": q["question"],
            "passages": [h["passage"] for h in hits],
            "hit10": bool(support & set(got)),
            "n_support": len(support),
            "n_support_in_top10": len(support & set(got)),
            "gold_ids": gold_ids,
            "doc_ids": doc_ids,
            "strips": strips,
        })
    del index
    torch.cuda.empty_cache()
    evaluator = load_evaluator("t5", evaluator_dir)
    rows = []
    packs = []
    for item in pending:
        scores = evaluator.score_docs(item["question"], item["passages"], batch_size=16)
        while len(scores) < 10:
            scores.append(-1.0)
        scores = scores[:10]
        strip_scores = score_strips(evaluator, item["question"], item["strips"])
        rows.append({
            "scores": scores,
            "max_score": max(scores),
            "hit10": item["hit10"],
        })
        packs.append({
            "gold_ids": item["gold_ids"],
            "doc_ids": item["doc_ids"],
            "strips": item["strips"],
            "scores": strip_scores,
        })
        print(
            f"FIT {item['qid']} hit={int(item['hit10'])} "
            f"gold_in={item['n_support_in_top10']}/{item['n_support']} max={max(scores):.4f}",
            flush=True,
        )
    chosen, paper, top = choose_pair(rows, min_count=20)
    sweep, best_top = sweep_top_n(packs)
    hit = sum(r["hit10"] for r in rows) / len(rows)
    gap = chosen.get("hit_gap_correct_minus_incorrect")
    correct_hit = (chosen.get("hit10") or {}).get("correct")
    incorrect_hit = (chosen.get("hit10") or {}).get("incorrect")
    separation_empty = (
        gap is None
        or (
            correct_hit is not None
            and incorrect_hit is not None
            and abs(correct_hit - incorrect_hit) < 1e-12
        )
    )
    payload = {
        "task": task,
        "split": f"pilot {task} public train only",
        "seed": 20261017,
        "n": 200,
        "first_retriever": "contriever",
        "rewrite_retriever": "contriever",
        "chunk_words": chunk_words,
        "index_dir": index_dir,
        "suffix": suffix,
        "retrieval_scope": "meta.doc_id" if task == "qasper" else "corpus",
        "pooling": "mean",
        "similarity": "dot",
        "normalized": False,
        "contriever_hit_at_10": hit,
        "separation_empty": separation_empty,
        "paper": paper,
        "grid_top": top,
        "chosen": chosen,
        "top_n_sweep": sweep,
        "top_n_selected": best_top["top_n"],
        "top_n_same_as_original": best_top["top_n"] == ORIGINAL_TOP_N,
        "top_n_note": (
            "Sweep is first-round Contriever top-10 only. The generator rewrite is not "
            "used to pick top_n. A strip shorter than 4 words scores -1. Ties take the "
            "smallest top_n. Original excerption width is 6. The dev run sets this width."
        ),
    }
    os.makedirs(os.path.dirname(fit_out), exist_ok=True)
    if os.path.exists(fit_out) or os.path.exists(cfg_out):
        raise SystemExit("fit outputs appeared during the run; refusing to overwrite")
    with open(fit_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    config = {
        "variant": "crag_corpus_only",
        "task": task,
        "upper_threshold": chosen["upper"],
        "lower_threshold": chosen["lower"],
        "threshold_rule": f"{task}_train_contriever_" + chosen["reason"],
        "n_docs": 10,
        "rewrite_k": 25,
        "decompose_mode": "excerption",
        "evaluator": "t5",
        "evaluator_temporary": False,
        "threshold_status": "ok",
        "threshold_counts": chosen["counts"],
        "threshold_n": 200,
        "threshold_over_80": chosen["over_80"],
        "first_retriever": "contriever",
        "rewrite_retriever": "contriever",
        "chunk_words": chunk_words,
        "index_dir": index_dir,
        "suffix": suffix,
        "retrieval_scope": payload["retrieval_scope"],
        "top_n_selected_for_b": best_top["top_n"],
        "separation_empty": separation_empty,
        "note": (
            "Train-only fit. Contriever first pass and Contriever rewrite. "
            "Does not replace crag_config.json or crag_config_contriever_musique.json."
            + (
                " Does not replace crag_config_contriever_multihop_rag.json or index/multihop_rag-contriever."
                if suffix == "_w200" else ""
            )
        ),
    }
    with open(cfg_out, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=1)
    print("FIT_HIT10", hit)
    print(
        "FIT_CHOSEN", chosen["upper"], chosen["lower"], chosen["reason"],
        chosen["counts"], "over_80", chosen["over_80"],
        "separation_empty", separation_empty,
    )
    print("FIT_TOP_N", best_top)
    print("WROTE", fit_out)
    print("WROTE", cfg_out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--chunk-words", type=int)
    parser.add_argument("--suffix", default="", choices=ALLOWED_SUFFIX)
    parser.add_argument("--qasper-tie", action="store_true")
    args = parser.parse_args()
    if args.qasper_tie:
        if args.task not in (None, "qasper") or args.chunk_words or args.suffix not in ("", "_top16"):
            raise SystemExit("--qasper-tie does not take a chunk size or another task")
        qasper_tie()
        return
    if not args.task:
        raise SystemExit("--task is required")
    fit(args.task, chunk_words=args.chunk_words, suffix=args.suffix)


if __name__ == "__main__":
    main()
