"""Fine-tune google/t5-large to map a question to A, B or C.

Holdout is 10% of the silver file, by qid. The checkpoint is chosen on that
holdout (macro-F1). Dev questions are scored only after the choice is frozen,
and only to report the route mix. Test is not read.
"""
import argparse
import hashlib
import json
import os
import random

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

from classify import predict_letters
from guard import assert_task, public_jsonl

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(os.path.dirname(HERE), "models", "t5-large")
OUT = os.path.join(os.path.dirname(HERE), "models", "classifier")


def holdout(task, qid):
    digest = hashlib.md5(f"{task}\n{qid}".encode()).hexdigest()
    return int(digest, 16) % 10 == 0


def load_rows(tasks, labels_dir=""):
    rows = []
    base = labels_dir or os.path.join(HERE, "labels")
    if labels_dir:
        text = os.path.abspath(labels_dir).replace("\\", "/")
        parts = text.split("/")
        if "test" in parts or "hidden" in parts or "eval_test" in parts:
            raise SystemExit(f"refuse {labels_dir}")
        if not text.endswith("/formal-v1/labels"):
            raise SystemExit(f"refuse labels dir {labels_dir}")
    for task in tasks:
        path = os.path.join(base, f"{task}_train.jsonl")
        if not os.path.exists(path):
            raise SystemExit(f"missing labels {path}")
        if "test" in path.split("/") or "hidden" in path.split("/"):
            raise SystemExit(f"refuse {path}")
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("source") == "error" or rec.get("label") not in ("A", "B", "C"):
                    continue
                rec["task"] = task
                rows.append(rec)
    return rows


def encode(tokenizer, rows):
    questions = [r["question"] for r in rows]
    targets = [r["label"] for r in rows]
    src = tokenizer(questions, max_length=256, truncation=True, padding=True, return_tensors="pt")
    tgt = tokenizer(targets, max_length=4, truncation=True, padding=True, return_tensors="pt")
    labels = tgt.input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return src.input_ids, src.attention_mask, labels


def metrics(rows, preds):
    """Confusion rows are gold A/B/C, columns are predicted A/B/C."""
    idx = {letter: i for i, letter in enumerate("ABC")}
    matrix = [[0, 0, 0] for _ in range(3)]
    for rec, pred in zip(rows, preds):
        matrix[idx[rec["label"]]][idx[pred]] += 1
    per_class = {}
    f1s = []
    for letter in "ABC":
        i = idx[letter]
        tp = matrix[i][i]
        support = sum(matrix[i])
        predicted = sum(matrix[r][i] for r in range(3))
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[letter] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        f1s.append(f1)
    correct = sum(matrix[i][i] for i in range(3))
    return {
        "n": len(rows),
        "accuracy": correct / len(rows) if rows else 0.0,
        "macro_f1": sum(f1s) / 3,
        "confusion_gold_rows_ABC": matrix,
        "per_class": per_class,
    }


def balanced_order(rows, epoch):
    """Each class is drawn up to the largest class, with replacement."""
    rng = random.Random(1000 + epoch)
    buckets = {letter: [rec for rec in rows if rec["label"] == letter] for letter in "ABC"}
    if any(not buckets[letter] for letter in "ABC"):
        raise SystemExit(f"a class is missing from train: {[ (k, len(v)) for k, v in buckets.items() ]}")
    width = max(len(buckets[letter]) for letter in "ABC")
    mixed = []
    for letter in "ABC":
        pool = buckets[letter]
        mixed.extend(pool[rng.randrange(len(pool))] for _ in range(width))
    rng.shuffle(mixed)
    return mixed


def evaluate(model, tokenizer, rows):
    preds = predict_letters(model, tokenizer, [rec["question"] for rec in rows])
    return metrics(rows, preds), preds


def fmt_report(report):
    parts = [f"acc={report['accuracy']:.3f} macro_f1={report['macro_f1']:.3f}"]
    for letter in "ABC":
        stat = report["per_class"][letter]
        parts.append(
            f"{letter}:P={stat['precision']:.2f}/R={stat['recall']:.2f}/n={stat['support']}"
        )
    parts.append(f"confusion={report['confusion_gold_rows_ABC']}")
    return " ".join(parts)


# Single-hop tasks. A joint model that sends these to C has collapsed.
# musique being mostly C matches its silver labels and is not collapse.
SINGLE_HOP = ("nq_open", "fever", "pubmedqa")


def label_mix(rows):
    by_task = {}
    for rec in rows:
        by_task.setdefault(rec["task"], {letter: 0 for letter in "ABC"})
        by_task[rec["task"]][rec["label"]] += 1
    return by_task


def dev_route_mix(model, tokenizer, tasks, train_by_task):
    """Route counts on public dev questions. Not used to pick the checkpoint."""
    mixes = {}
    for task in tasks:
        path = public_jsonl(task, "dev")
        questions = []
        with open(path) as handle:
            for line in handle:
                if line.strip():
                    questions.append(json.loads(line)["question"])
        if len(questions) != 100:
            raise SystemExit(f"{task} dev has {len(questions)} rows")
        preds = predict_letters(model, tokenizer, questions)
        mixes[task] = {letter: preds.count(letter) for letter in "ABC"}
        print(
            f"DEV_ROUTES {task} dev={mixes[task]} train={train_by_task.get(task)}",
            flush=True,
        )
    return mixes


def collapsed(report, dev_mixes, train_by_task):
    """Holdout class vanished, or a single-hop dev set was sent to C.

    Multi-hop tasks, musique included, may be mostly C. That is expected.
    """
    for letter, stat in report["per_class"].items():
        if stat["support"] >= 15 and stat["recall"] < 0.15:
            return True, f"holdout {letter} recall {stat['recall']:.2f}"
    if report["macro_f1"] < 0.40:
        return True, f"holdout macro_f1 {report['macro_f1']:.3f}"
    for task in SINGLE_HOP:
        mix = dev_mixes.get(task) or {}
        total = sum(mix.values()) or 1
        c_share = mix.get("C", 0) / total
        train_counts = train_by_task.get(task) or {}
        train_n = sum(train_counts.values()) or 1
        train_c = train_counts.get("C", 0) / train_n
        if c_share >= 0.40 and c_share >= train_c + 0.25:
            return True, f"{task} dev {c_share:.0%} C but train C is {train_c:.0%}"
    return False, "ok"


def fit(tokenizer, train, valid, device, lr, epochs, batch, dest, n_param, tasks):
    model = T5ForConditionalGeneration.from_pretrained(BASE)
    if n_param >= 1_000_000_000:
        raise SystemExit(f"classifier has {n_param} parameters, over the 1B cap")
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best = -1.0
    best_epoch = 0
    stale = 0
    os.makedirs(dest, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        order = balanced_order(train, epoch)
        total = 0.0
        steps = 0
        for start in range(0, len(order), batch):
            batch_rows = order[start : start + batch]
            ids, mask, labels = encode(tokenizer, batch_rows)
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            opt.zero_grad()
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item())
            steps += 1
        report, _preds = evaluate(model, tokenizer, valid)
        print(
            f"EPOCH lr={lr:g} {epoch} loss={total / max(steps, 1):.4f} {fmt_report(report)}",
            flush=True,
        )
        if report["macro_f1"] >= best:
            best = report["macro_f1"]
            best_epoch = epoch
            stale = 0
            model.save_pretrained(dest)
            tokenizer.save_pretrained(dest)
            with open(os.path.join(dest, "train_meta.json"), "w") as handle:
                json.dump({
                    "tasks": tasks,
                    "n_train": len(train),
                    "n_valid": len(valid),
                    "best_valid_macro_f1": best,
                    "best_valid_acc": report["accuracy"],
                    "per_class": report["per_class"],
                    "confusion_gold_rows_ABC": report["confusion_gold_rows_ABC"],
                    "epoch": epoch,
                    "params": n_param,
                    "base": "google-t5/t5-large",
                    "lr": lr,
                    "sampling": "balanced",
                    "decode": "abc_logits",
                    "selected_on": "train_holdout_macro_f1",
                }, handle, indent=1)
        else:
            stale += 1
            if stale >= 3:
                print(f"EARLY_STOP lr={lr:g} best_epoch={best_epoch}", flush=True)
                break
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return best, best_epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--lrs", type=float, nargs="+", default=[3e-5, 1e-4])
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--labels-dir", default="")
    ap.add_argument("--skip-dev", action="store_true")
    args = ap.parse_args()
    lrs = [args.lr] if args.lr is not None else args.lrs
    out = args.out
    tasks = [assert_task(t) for t in args.tasks]
    rows = load_rows(tasks, args.labels_dir)
    train = [r for r in rows if not holdout(r["task"], r["qid"])]
    valid = [r for r in rows if holdout(r["task"], r["qid"])]
    label_counts = {letter: sum(rec["label"] == letter for rec in train) for letter in "ABC"}
    print(
        f"TRAIN n={len(train)} valid={len(valid)} tasks={tasks} train_labels={label_counts}",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    probe = T5ForConditionalGeneration.from_pretrained(BASE)
    n_param = sum(p.numel() for p in probe.parameters())
    print(f"PARAMS {n_param}", flush=True)
    del probe
    if n_param >= 1_000_000_000:
        raise SystemExit(f"classifier has {n_param} parameters, over the 1B cap")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_score = -1.0
    best_lr = None
    for lr in lrs:
        dest = os.path.join(out, f"lr-{lr:g}")
        score, epoch = fit(tokenizer, train, valid, device, lr, args.epochs, args.batch, dest, n_param, tasks)
        print(f"LR_DONE lr={lr:g} best_macro_f1={score:.3f} epoch={epoch}", flush=True)
        if score >= best_score:
            best_score = score
            best_lr = lr
    chosen = os.path.join(out, f"lr-{best_lr:g}")
    model = T5ForConditionalGeneration.from_pretrained(chosen).to(device)
    report, _preds = evaluate(model, tokenizer, valid)
    print(f"SELECTED lr={best_lr:g} {fmt_report(report)}", flush=True)
    train_by_task = label_mix(rows)
    if args.skip_dev:
        mixes = {}
        print("DEV_SKIPPED selection uses the train holdout only", flush=True)
    else:
        mixes = dev_route_mix(model, tokenizer, tasks, train_by_task)
    bad, why = collapsed(report, mixes, train_by_task)
    # Copy the chosen checkpoint to the directory the server loads.
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    with open(os.path.join(out, "train_meta.json"), "w") as handle:
        json.dump({
            "tasks": tasks,
            "n_train": len(train),
            "n_valid": len(valid),
            "best_valid_macro_f1": report["macro_f1"],
            "best_valid_acc": report["accuracy"],
            "per_class": report["per_class"],
            "confusion_gold_rows_ABC": report["confusion_gold_rows_ABC"],
            "dev_routes": mixes,
            "train_labels_by_task": train_by_task,
            "params": n_param,
            "base": "google-t5/t5-large",
            "lr": best_lr,
            "sampling": "balanced",
            "decode": "abc_logits",
            "selected_on": "train_holdout_macro_f1",
            "collapsed": bad,
            "collapsed_reason": why,
            **({
                "data": "formal-v1-split/train_fit",
                "dev_used_for_selection": False,
            } if args.skip_dev else {}),
        }, handle, indent=1)
    print(f"SAVED {out} macro_f1={report['macro_f1']:.3f} collapsed={bad} {why}", flush=True)
    if bad:
        raise SystemExit(f"classifier still collapsed: {why}")


if __name__ == "__main__":
    main()
