"""Minimal, CPU-only walk through the AutoRAGBench evaluation path on the released formal-v2 data.

  python examples/minimal_eval.py --task multihop_rag --n 50
  python examples/minimal_eval.py --task multihop_rag --split test --n 500        # the test set of Table II
  python examples/minimal_eval.py --task qasper --n 50 \
      --reader-base-url http://127.0.0.1:8000/v1 --reader-model generator      # optional reader

Steps (the same code the benchmark uses, without Slurm, sandboxes or GPUs):
  1. Build the naive BM25 baseline (baselines/ragcore.py: ~200-word passages, BM25, top 5) on the task corpus.
  2. Retrieve evidence for N questions: by default of the fixed ~10% train holdout (tasks.common.is_holdout); with
     --split test, of the test set (hidden/formal-v2/<task>/test_questions.jsonl + test_gold.jsonl).
  3. Validate the evidence against the corpus (multireader.contract.verify_corpus) and pack it into the shared
     evidence prefix (multireader.contract.pack_passages: 32,000 characters; with --tokenizers also the 12,000
     Qwen-token and 32,768-token reader limits of multireader.contract.pack_for_readers).
  4. With --reader-base-url: ask one reader through any OpenAI-compatible endpoint with the evaluator's prompt
     (multireader.contract.messages, temperature 0, at most 512 tokens), then score with the task metric
     (tasks.score.score). Without a reader the script reports evidence validity and supporting-document recall.
"""
import argparse
import json
import os
import pathlib
import shutil
import sys
import tempfile
import time
import urllib.request

ROOT = pathlib.Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
from multireader.contract import clean_answer, messages, pack_for_readers, pack_passages, verify_corpus  # noqa: E402
from tasks import registry  # noqa: E402
from tasks.common import is_holdout, read_jsonl  # noqa: E402
from tasks.score import score  # noqa: E402


def load_baseline(task, workdir):
    """Copy the naive BM25 baseline into a workspace the way scripts/new_run.sh does, then import it."""
    sub = workdir / "submission"
    sub.mkdir(parents=True)
    shutil.copy(ROOT / "baselines/ragcore.py", sub / "ragcore.py")
    (sub / "task.json").write_text(json.dumps(registry.info(task)))
    sys.path.insert(0, str(sub))
    import ragcore  # noqa: E402
    return ragcore


def ask(base_url, model, key, msg, template_kwargs):
    body = {"model": model, "messages": msg, "temperature": 0, "max_tokens": 512}
    if template_kwargs:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    with urllib.request.urlopen(req, timeout=120) as r:
        choice = json.load(r)["choices"][0]
    return clean_answer(choice["message"].get("content") or ""), choice.get("finish_reason")


def main():
    if os.name == "nt" and not sys.flags.utf8_mode:  # the data and the baseline read files as UTF-8 text
        import subprocess  # os.execv does not wait on Windows; run a UTF-8-mode child instead
        sys.exit(subprocess.run([sys.executable, "-X", "utf8"] + sys.argv).returncode)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="multihop_rag", choices=registry.TASKS)
    ap.add_argument("--data", default=str(ROOT / "data/formal-v2"), help="data/<version> directory")
    ap.add_argument("--split", choices=["holdout", "test"], default="holdout",
                    help="train holdout (public data) or the test set (hidden/<version>/<task>/)")
    ap.add_argument("--n", type=int, default=50, help="number of questions (first N)")
    ap.add_argument("--k", type=int, default=5, help="passages per question (naive BM25: 5)")
    ap.add_argument("--reader-base-url", help="OpenAI-compatible endpoint, e.g. a vLLM server of one reader")
    ap.add_argument("--reader-model", default="generator")
    ap.add_argument("--reader-key-env", default="READER_API_KEY", help="env var holding the endpoint key (optional)")
    ap.add_argument("--no-chat-template-kwargs", action="store_true",
                    help="do not send chat_template_kwargs.enable_thinking=false (non-vLLM endpoints)")
    ap.add_argument("--tokenizers", help="comma-separated tokenizer dirs, Qwen3.5-4B first, then the readers")
    ap.add_argument("--out", help="write per-question results (JSONL) here")
    a = ap.parse_args()

    pub = pathlib.Path(a.data) / a.task / "public"
    corpus_dir, info = pub / "corpus", registry.info(a.task)
    if not (corpus_dir / "corpus.jsonl").exists():
        sys.exit(f"missing {corpus_dir}/corpus.jsonl: download the formal-v2 data first (README, 'Data')")
    if a.split == "test":
        hid = ROOT / "hidden" / pathlib.Path(a.data).name / a.task
        if not (hid / "test_questions.jsonl").exists():
            sys.exit(f"missing {hid}/test_questions.jsonl: download the formal-v2 test set first (README, 'Data')")
        gold = {g["qid"]: g for g in read_jsonl(hid / "test_gold.jsonl")}
        qs = [{**q, **gold[q["qid"]]} for q in read_jsonl(hid / "test_questions.jsonl")][: a.n]
    else:
        qs = [q for q in read_jsonl(pub / "train.jsonl") if is_holdout(q["qid"])][: a.n]
    print(f"task={a.task} metric={info['primary_metric']} questions={len(qs)} "
          f"({'test' if a.split == 'test' else 'train holdout'})")

    work = pathlib.Path(tempfile.mkdtemp(prefix="autoragbench-minimal-"))
    try:
        ragcore = load_baseline(a.task, work)
        t0 = time.time()
        ragcore.build(str(corpus_dir), str(work / "index"))
        R = ragcore.Retriever(str(corpus_dir), index=str(work / "index"))
        print(f"[1] naive BM25 index: {R.n} passages in {time.time() - t0:.1f}s")

        rows, lat = [], []
        for q in qs:
            t = time.time()
            idx = R.search(q["question"], k=a.k, doc_id=(q.get("meta") or {}).get("doc_id"))
            lat.append(time.time() - t)
            rows.append({"qid": q["qid"], "question": q["question"], "meta": q.get("meta", {}),
                         "passages": [{"doc_id": str(R.doc_list[int(R.doc_of[i])]), "title": R.store.get(i)[0],
                                       "text": R.store.get(i)[1]} for i in idx]})
        print(f"[2] retrieval: mean latency {sum(lat) / len(lat) * 1000:.1f} ms/question")

        scope = "claimed" if info.get("uses_meta_doc_id") else "any"
        check = verify_corpus(rows, corpus_dir / "corpus.jsonl", line_scope=scope)
        print(f"[3] evidence validation: {json.dumps(check)}")
        toks = None
        if a.tokenizers:
            from transformers import AutoTokenizer
            toks = [AutoTokenizer.from_pretrained(p, local_files_only=True) for p in a.tokenizers.split(",")]
        fmt = info["answer_format"]
        for r in rows:
            r["packed"] = (pack_for_readers(r["question"], r["passages"], fmt, toks[0], toks, batch=True) if toks
                           else pack_passages(r["passages"]))
        chars = [sum(len(p["text"]) + len(p["title"]) for p in r["packed"]) for r in rows]
        print(f"    packed evidence: mean {sum(chars) / len(chars):.0f} characters "
              f"({'token-checked' if toks else 'character budget only; pass --tokenizers for token limits'})")

        gold = {q["qid"]: q for q in qs}
        sup = [(set(gold[r["qid"]].get("supporting_doc_ids") or []), {p["doc_id"] for p in r["packed"]}) for r in rows]
        sup = [(s, got) for s, got in sup if s]
        if sup:
            print(f"    supporting-document recall@{a.k}: {sum(len(s & got) / len(s) for s, got in sup) / len(sup):.3f} "
                  f"over {len(sup)} questions with labelled supporting documents")

        if a.reader_base_url:
            key = os.environ.get(a.reader_key_env, "EMPTY")
            preds = {}
            for r in rows:
                rec = {"qid": r["qid"], "error": r.get("error"), "over_budget": False, "answer": ""}
                if not rec["error"]:
                    try:
                        rec["answer"], finish = ask(a.reader_base_url, a.reader_model, key,
                                                    messages(r["question"], r["packed"], fmt), not a.no_chat_template_kwargs)
                        if not rec["answer"]:
                            rec["error"] = "empty_generation"
                        elif finish == "length":
                            rec["error"] = "truncated_generation"
                    except Exception as e:  # counted as an invalid prediction, as in the evaluator
                        rec["error"] = type(e).__name__
                preds[r["qid"]] = rec
                r["answer"], r["error"] = rec["answer"], rec["error"]
            res = score(a.task, preds, qs)
            print(f"[4] reader {a.reader_model}: {info['primary_metric']} = {res['primary']:.4f} "
                  f"(invalid {res['n_invalid']}/{res['n']})")
        else:
            print("[4] no reader endpoint given (--reader-base-url): answering and scoring skipped")
        if a.out:
            with open(a.out, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps({k: r.get(k) for k in ("qid", "question", "packed", "answer", "error")},
                                       ensure_ascii=False) + "\n")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
