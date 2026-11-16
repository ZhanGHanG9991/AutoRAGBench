"""Render a materials/ template for one run: render.py <template> <run_dir> <task> > out.

Placeholders ({{NAME}}) come from configs/pilot.env (exported by new_run.sh), tasks/<task>/info.json
and the run directory."""
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tasks import registry  # noqa: E402


def count(p):
    with open(p) as f:
        return sum(1 for _ in f)


def human(secs):
    h, m = divmod(secs // 60, 60)
    return f"{h} h" + (f" {m} min" if m else "")


def n_test(e, task):
    """Size of the hidden test (the count only; the store itself is never visible to the agent)."""
    p = Path(e.get("HIDDEN_DIR", "")) / e.get("DATA_VERSION", "") / task / "test_questions.jsonl"
    if e.get("HIDDEN_DIR") and p.is_file():
        return count(p)
    if e.get("N_TEST"):
        return int(e["N_TEST"])
    # without the hidden store: the test sizes of the data version (configs/data_rules/<version>.test_counts.json)
    sizes = Path(__file__).resolve().parents[1] / "configs/data_rules" / f"{e.get('DATA_VERSION', '')}.test_counts.json"
    if sizes.is_file() and task in json.loads(sizes.read_text()):
        return json.loads(sizes.read_text())[task]
    raise SystemExit(f"no hidden test at {p}: set N_TEST")


def main():
    template, run, task = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    info = registry.info(task)
    e = os.environ
    secs = int(e["BUILD_SECONDS"])
    eval_secs = int(e["EVAL_SECONDS"])
    nt = n_test(e, task) if "{{N_TEST}}" in Path(template).read_text() else 0
    values = {
        "TASK_NAME": info["display_name"], "DISPLAY_NAME": info["display_name"],
        "QUESTION_TYPE": info["question_type"], "ANSWER_FORMAT": info["answer_format"],
        "METRIC": info["metric_description"], "META": info["meta"],
        "BUDGET_HUMAN": human(secs),
        # evaluation budget (separate from the build budget): startup (<= 20 min) + all test questions
        "N_TEST": nt, "EVAL_HOURS": human(eval_secs),
        "EVAL_PER_QUESTION": (eval_secs - 1200) // nt if nt else "",
        "WORKSPACE_LIMIT": f"{e['WORKSPACE_LIMIT_GB']} GB",
        "TASK_DIR": str(run / "task"), "WORKSPACE": str(run / "workspace"), "RUN_DIR": str(run),
        "RAG_BASE_PY": e["RAG_BASE_PY"], "CALLS": e["BUDGET_CALLS"], "TOKENS": e["BUDGET_TOKENS"],
        "SECONDS": e["BUDGET_SECONDS"], "GPU": e.get("GPU_NAME", "1 x NVIDIA H800 80GB"),
        "N_CORPUS": count(run / "task/corpus/corpus.jsonl"), "N_TRAIN": count(run / "task/train.jsonl"),
        # pilot data has a dev file; formal data versions give train only (no hint to hold anything out)
        **({"LABELLED_FILES": "`train.jsonl` / `dev.jsonl`",
            "TRAIN_QA_RULE": "",
            "DEV_ROW": f"| `{run / 'task'}/dev.jsonl` | {count(run / 'task/dev.jsonl')} more labelled questions, same format, used by `self_check` |\n",
            "SELF_CHECK": "runs your service with the official evaluation protocol, on `dev.jsonl`, and prints the score "
                          "(`self_check <file> <n>` for the first n questions of a file)"}
           if (run / "task/dev.jsonl").exists() else
           {"LABELLED_FILES": "`train.jsonl`", "DEV_ROW": "",
            "TRAIN_QA_RULE": " Train questions and answers may serve as few-shot examples, for tuning and for\n"
                             "   training auxiliary models, but the service must not look answers up in stored train\n"
                             "   question-answer pairs.",
            "SELF_CHECK": "runs your service with the official evaluation protocol on a labelled question file (same "
                          "format as `train.jsonl`) and prints the score: `self_check <file> [n]` (first n questions)"}),
        # knowledge-graph tasks describe their corpus (and triples.tsv) in one extra paragraph; others render nothing
        "CORPUS_NOTE": info["corpus_note"].replace("{{N_TRIPLES}}", f"{count(run / 'task/corpus/triples.tsv'):,}") + "\n\n"
                       if info.get("corpus_note") else "",
        "DATETIME": subprocess.run(["date", "-u"], capture_output=True, text=True, env={**os.environ, "LC_ALL": "C"}).stdout.strip(),
    }
    values["SELF_CHECK"] += (
        ". Delivery freeze also removes filesystem write permission bits from every workspace file and directory. "
        "The self_check read-only mount does not change those source mode bits. "
        "If you copy runtime metadata or lock files to SCRATCH_DIR using copy2/cp -p, explicitly restore "
        "owner write permission on the scratch copies (and directories), or use a content-only copy. "
        "Do not chmod or write the delivered workspace; recreate lock files in SCRATCH_DIR."
    )
    text = Path(template).read_text()
    for k, v in values.items():
        text = text.replace("{{" + k + "}}", str(v))
    left = [w for w in text.split("{{")[1:]]
    if left:
        raise SystemExit(f"unfilled placeholders: {[w.split('}}')[0] for w in left]}")
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
