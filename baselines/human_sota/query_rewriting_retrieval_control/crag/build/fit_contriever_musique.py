"""Fit MuSiQue Contriever thresholds and refinement width on train only.

--check-bm25 replays the saved BM25 probe and must select 0.592 / -0.995.
--fit builds or loads the Contriever index, scores the same 200 train
questions, and writes a separate config. It does not read dev or test and
does not rewrite rageng/crag_config.json.
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

ROOT = _release_path("@AUTORAGBENCH_ROOT@")
C = os.path.join(ROOT, "humansota/CorrectiveRAG")
TRAIN = os.path.join(ROOT, "data/musique/public/train.jsonl")
CORPUS = os.path.join(ROOT, "data/musique/public/corpus")
BM25_INDEX = os.path.join(C, "rageng/index/musique-pilot")
PROBE = os.path.join(C, "rageng/results/probe_musique_train200.json")
CONTRIEVER = os.path.join(C, "models/contriever-msmarco")
DENSE_INDEX = os.path.join(C, "rageng/index/musique-contriever")
EVALUATOR = os.path.join(C, "models/evaluator")
FIT_OUT = os.path.join(C, "rageng/results/contriever_musique_fit.json")
CFG_OUT = os.path.join(C, "rageng/crag_config_contriever_musique.json")

PAPER = [
    ("popqa_shell", 0.592, -0.995),
    ("popqa_paper", 0.59, -0.99),
    ("pubhealth_arc", 0.5, -0.91),
    ("biography", 0.95, -0.91),
]
TOP_NS = [1, 2, 3, 4, 6, 8, 10, 12, 16]
ORIGINAL_TOP_N = 6


def refuse(path):
    if "hidden" in path or path.rstrip("/").endswith("test.jsonl") or "/test/" in path:
        raise SystemExit(f"refusing {path}")


def action_of(scores, upper, lower):
    if any(s >= upper for s in scores):
        return "correct"
    if any(s >= lower for s in scores):
        return "ambiguous"
    return "incorrect"


def summarize(rows, upper, lower):
    counts = {"correct": 0, "ambiguous": 0, "incorrect": 0}
    hits = {"correct": 0, "ambiguous": 0, "incorrect": 0}
    for row in rows:
        name = action_of(row["scores"], upper, lower)
        counts[name] += 1
        hits[name] += int(row["hit10"])
    n = len(rows)
    hit10 = {k: (hits[k] / counts[k] if counts[k] else None) for k in counts}
    hc, hi = hit10["correct"], hit10["incorrect"]
    return {
        "upper": upper,
        "lower": lower,
        "n": n,
        "counts": counts,
        "fractions": {k: counts[k] / n for k in counts},
        "hit10": hit10,
        "hit_gap_correct_minus_incorrect": None if hc is None or hi is None else hc - hi,
    }


def quantile_grid(sorted_values, ps):
    n = len(sorted_values)
    return [sorted_values[min(n - 1, int(round(p * (n - 1))))] for p in ps]


def _gap(item):
    gap = item["hit_gap_correct_minus_incorrect"]
    return -9.0 if gap is None else gap


def _band_ok(item, min_count):
    counts = item["counts"]
    return min(counts.values()) >= min_count and max(counts.values()) / item["n"] <= 0.80


def _band_label(min_count):
    if min_count == 20:
        return "20_160"
    return f"min{min_count}_frac80"


def _grid_record(item, min_count):
    return {
        "upper": item["upper"],
        "lower": item["lower"],
        "source": item["source"],
        "name": item.get("name"),
        "gap": item["hit_gap_correct_minus_incorrect"],
        "counts": item["counts"],
        "fractions": item["fractions"],
        "hit10": item["hit10"],
        "n": item["n"],
        "compliant": _band_ok(item, min_count),
    }


def choose_pair(rows, min_count=10, return_grid=False, keep_paper=True):
    """Same grid as the BM25 refit, then the rule that kept the PopQA shell.

    If keep_paper is True (default) and the best in-bound grid point is more
    than 70% Ambiguous and its gap beats the best paper pair by less than
    0.05, keep that paper pair. Otherwise take the higher gap. If every grid
    point breaks the band, flag it, take the smallest max-class fraction, and
    still return a pair.

    keep_paper=False disables every branch that would select a paper pair when
    a compliant grid point exists (formal-v1 KEEP_PAPER_BRANCH=0). Paper rows
    stay in the returned grid for reporting. When no compliant grid point
    exists, the band-break fallback is unchanged.

    min_count is the minimum number of questions in each action. The default
    10 matches configuration formal-v1 (flat 10). formal-v1 must still
    pass the value from split/manifest.json and assert it. Historical pilot
    BM25 reproduction passes 20 explicitly. A class may still be at most
    80% of n. Paper pairs are not dropped for failing that band.
    """
    if min_count < 1:
        raise ValueError(min_count)
    paper_items = []
    grid = []
    for name, upper, lower in PAPER:
        item = summarize(rows, upper, lower)
        item["name"] = name
        item["source"] = "paper"
        paper_items.append(item)
        grid.append(_grid_record(item, min_count))
    usable = [it for it in paper_items if it["hit_gap_correct_minus_incorrect"] is not None]
    paper_best = max(usable, key=_gap) if usable else None

    maxima = sorted(r["max_score"] for r in rows)
    uppers = sorted(set([0.5, 0.59, 0.592, 0.95] + quantile_grid(maxima, [0.6, 0.7, 0.75, 0.8, 0.85, 0.9])))
    lowers = sorted(set([-0.995, -0.99, -0.915, -0.91] + quantile_grid(maxima, [0.05, 0.1, 0.15, 0.2, 0.3])))
    bounded = []
    relaxed = []
    for upper in uppers:
        for lower in lowers:
            if not (lower < upper and upper >= 0.0 and lower <= -0.5):
                continue
            item = summarize(rows, upper, lower)
            item["source"] = "grid"
            relaxed.append(item)
            grid.append(_grid_record(item, min_count))
            if not _band_ok(item, min_count):
                continue
            mono = item["hit10"]["correct"] >= item["hit10"]["ambiguous"] >= item["hit10"]["incorrect"]
            bounded.append((_gap(item), bool(mono), item))
    bounded.sort(key=lambda x: (x[0], x[1]), reverse=True)
    band = _band_label(min_count)
    over_80 = False
    if paper_best is None and not bounded:
        over_80 = True
        chosen = min(relaxed, key=lambda it: (max(it["counts"].values()) / it["n"], -_gap(it)))
        reason = f"no_paper_gap_and_every_grid_point_outside_{band}_flag_and_continue"
    elif paper_best is None:
        chosen = bounded[0][2]
        reason = "no_paper_pair_has_all_three_classes_use_grid"
    elif not bounded:
        over_80 = True
        chosen = min(relaxed, key=lambda it: (max(it["counts"].values()) / it["n"], -_gap(it)))
        reason = f"every_grid_point_outside_{band}_flag_and_continue"
    else:
        grid_best = bounded[0][2]
        advantage = _gap(grid_best) - _gap(paper_best)
        if not keep_paper:
            chosen = grid_best
            reason = "keep_paper_disabled_use_grid"
        elif grid_best["fractions"]["ambiguous"] > 0.70 and advantage < 0.05:
            chosen = paper_best
            reason = "grid_ambiguous_over_0.70_and_gap_advantage_under_0.05_keep_paper"
        elif _gap(grid_best) > _gap(paper_best):
            chosen = grid_best
            reason = "grid_higher_gap"
        else:
            chosen = paper_best
            reason = "paper_gap_at_least_grid"
    chosen = dict(chosen)
    keep_tag = "keep_paper=1" if keep_paper else "keep_paper=0"
    chosen["reason"] = f"{reason};min_count={min_count};{keep_tag}"
    chosen["over_80"] = over_80
    chosen["min_count"] = min_count
    chosen["keep_paper"] = bool(keep_paper)
    top = [{"gap": g, "mono": m, "upper": it["upper"], "lower": it["lower"], "counts": it["counts"],
            "hit10": it["hit10"]} for g, m, it in bounded[:8]]
    for record in grid:
        record["selected"] = (
            record["upper"] == chosen["upper"]
            and record["lower"] == chosen["lower"]
            and record["source"] == chosen["source"]
        )
    if return_grid:
        return chosen, paper_items, top, grid
    return chosen, paper_items, top


def load_train_sample():
    refuse(TRAIN)
    if os.path.basename(TRAIN) != "train.jsonl" or "/public/" not in TRAIN:
        raise SystemExit(f"refusing non-public train file {TRAIN}")
    questions = [json.loads(line) for line in open(TRAIN, encoding="utf-8")]
    return random.Random(20261017).sample(questions, 200)


def check_bm25():
    from bm25_index import build_or_load, search

    blob = json.load(open(PROBE, encoding="utf-8"))
    saved = blob["per_question_scores"]
    picked = load_train_sample()
    if len(saved) != 200:
        raise SystemExit(f"probe rows {len(saved)}")
    docs, retriever = build_or_load(CORPUS, BM25_INDEX)
    rows = []
    for q, scores in zip(picked, saved):
        hits = search(docs, retriever, q["question"], 10)
        support = set(q.get("supporting_doc_ids") or [])
        got = {h["doc_id"] for h in hits}
        rows.append({
            "scores": scores,
            "max_score": max(scores),
            "hit10": bool(support & got),
        })
    hit = sum(r["hit10"] for r in rows) / len(rows)
    expected = blob["summary"]["bm25_hit_at_10"]
    if abs(hit - expected) > 1e-9:
        raise SystemExit(f"sample misaligned: recomputed hit {hit} != saved {expected}")
    chosen, paper, top = choose_pair(rows, min_count=20)
    print("CHECK_BM25_HIT", hit)
    print("CHECK_BM25_CHOSEN", chosen["upper"], chosen["lower"], chosen["reason"], chosen["counts"])
    print("CHECK_BM25_PAPER_GAPS", [(p["name"], p["hit_gap_correct_minus_incorrect"], p["counts"]) for p in paper])
    if top:
        print("CHECK_BM25_GRID_BEST", top[0])
    if chosen["upper"] != 0.592 or chosen["lower"] != -0.995:
        raise SystemExit("rule did not reproduce the MuSiQue BM25 pair 0.592 / -0.995")
    print("CHECK_BM25_OK")


def strips_of(passage):
    from crag_corpus import extract_strips_from_psg, normalize_doc

    flat = normalize_doc(passage)
    if not flat:
        return []
    return [s.replace("\n", " ") for s in extract_strips_from_psg(psg=flat, mode="excerption")]


def score_strips(evaluator, query, strips):
    scores = [-1.0] * len(strips)
    live_i = [i for i, strip in enumerate(strips) if len(strip.split()) >= 4]
    if not live_i:
        return scores
    got = evaluator.score_docs(query, [strips[i] for i in live_i], batch_size=16)
    for i, score in zip(live_i, got):
        scores[i] = score
    return scores


def bridge_of(n_gold, kept_gold):
    if n_gold <= 0:
        return None
    if n_gold >= 2:
        return kept_gold >= 2
    return kept_gold >= 1


def sweep_top_n(packs):
    """packs: per question, gold doc ids in the first top-10 and scored strips.

    The rate is, among questions that retrieved at least one gold doc, the
    share that still keep two when two or more were retrieved, or the one
    when only one was retrieved. Ties take the smallest top_n.
    """
    rows = []
    for top_n in TOP_NS:
        ok = eligible = 0
        for pack in packs:
            order = sorted(range(len(pack["scores"])), key=lambda i: pack["scores"][i], reverse=True)
            kept_text = "; ".join(pack["strips"][i] for i in order[:top_n])
            kept_docs = set()
            for doc_id, strip in zip(pack["doc_ids"], pack["strips"]):
                if len(strip.split()) < 4:
                    continue
                if strip in kept_text and doc_id in pack["gold_ids"]:
                    kept_docs.add(doc_id)
            flag = bridge_of(len(pack["gold_ids"]), len(kept_docs))
            if flag is None:
                continue
            eligible += 1
            ok += int(flag)
        rate = ok / eligible if eligible else 0.0
        rows.append({"top_n": top_n, "ok": ok, "eligible": eligible, "rate": rate})
    best = sorted(rows, key=lambda r: (-r["rate"], r["top_n"]))[0]
    return rows, best


def fit():
    import torch
    from contriever_index import build_or_load as load_contriever
    from crag_corpus import load_evaluator

    if not torch.cuda.is_available():
        raise SystemExit("refusing --fit without a GPU")
    for path in (TRAIN, CORPUS, CONTRIEVER, EVALUATOR, DENSE_INDEX):
        refuse(path)
    picked = load_train_sample()
    index = load_contriever(CORPUS, DENSE_INDEX, CONTRIEVER, batch_size=32)
    pending = []
    for q in picked:
        hits = index.search(q["question"], 10)
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
    evaluator = load_evaluator("t5", EVALUATOR)
    rows = []
    packs = []
    for item in pending:
        scores = evaluator.score_docs(item["question"], item["passages"], batch_size=16)
        if len(scores) != 10:
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
    payload = {
        "split": "pilot musique train only",
        "seed": 20261017,
        "n": 200,
        "first_retriever": "contriever",
        "rewrite_retriever": "bm25",
        "pooling": "mean",
        "similarity": "dot",
        "normalized": False,
        "contriever_hit_at_10": hit,
        "paper": paper,
        "grid_top": top,
        "chosen": chosen,
        "top_n_sweep": sweep,
        "top_n_selected": best_top["top_n"],
        "top_n_same_as_original": best_top["top_n"] == ORIGINAL_TOP_N,
        "top_n_note": (
            "Sweep is first-round Contriever top-10 only. Rewrite retrieval stays BM25 "
            "and is not used to pick top_n, because the rewrite query comes from the generator. "
            "A strip shorter than 4 words scores -1, matching select_relevants. "
            "Ties take the smallest top_n. Original excerption width is 6."
        ),
    }
    os.makedirs(os.path.dirname(FIT_OUT), exist_ok=True)
    with open(FIT_OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    config = {
        "variant": "crag_corpus_only",
        "upper_threshold": chosen["upper"],
        "lower_threshold": chosen["lower"],
        "threshold_rule": "musique_train_contriever_" + chosen["reason"],
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
        "top_n_selected_for_b": best_top["top_n"],
        "note": (
            "Train-only fit for Contriever first-round retrieval on pilot MuSiQue. "
            "Does not replace rageng/crag_config.json. Group A leaves CRAG_TOP_N unset. "
            "Group B sets CRAG_TOP_N only when the sweep did not select 6."
        ),
    }
    with open(CFG_OUT, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=1)
    print("FIT_HIT10", hit)
    print("FIT_CHOSEN", chosen["upper"], chosen["lower"], chosen["reason"], chosen["counts"], "over_80", chosen["over_80"])
    print("FIT_TOP_N", best_top)
    print("WROTE", FIT_OUT)
    print("WROTE", CFG_OUT)


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-bm25", action="store_true")
    mode.add_argument("--fit", action="store_true")
    args = parser.parse_args()
    if args.check_bm25:
        check_bm25()
    else:
        fit()


if __name__ == "__main__":
    main()
