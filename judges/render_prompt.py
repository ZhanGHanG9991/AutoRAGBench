"""render_prompt.py <judge> <run_records_dir> -> prompt text on stdout."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tasks import registry  # noqa: E402


def main():
    judge, rec = sys.argv[1], Path(sys.argv[2])
    run = json.loads((rec / "run.json").read_text())
    meta = json.loads((rec / "agent_meta.json").read_text()) if (rec / "agent_meta.json").exists() else {}
    secs = int(run.get("build_seconds", 0))
    values = {
        "benchmark": registry.info(run["task"])["display_name"],
        "agent_identity": f"{meta.get('harness', '?')} ({meta.get('harness_version', '?')}) driven by "
                          f"{meta.get('provider', '?')} / {meta.get('model', '?')}",
        "budget": f"{secs // 3600} h {secs % 3600 // 60} min",
    }
    if run.get("data_version", "pilot") == "pilot":
        values["labelled"] = "`train.jsonl` / `dev.jsonl`"
        values["train_qa_index_rule"] = "Note that copying corpus text or train/dev questions and answers into the index is allowed."
    else:  # formal data: train only; README rule 1 forbids answering from stored train QA pairs
        values["labelled"] = "`train.jsonl`"
        values["train_qa_index_rule"] = ("Copying corpus text into the index is allowed, and so is using train questions and "
                                         "answers as few-shot examples, for tuning or to train auxiliary models; a service that "
                                         "answers by looking up stored train question-answer pairs (an answer cache or QA index "
                                         "consulted at answer time) violates rule 1 of the task README.")
    if run.get("evaluation_protocol") != "fixed-evidence-v1":
        raise SystemExit("only fixed-evidence-v1 runs are supported")
    ctx = (ROOT / "judges" / "_context_fixed.md").read_text()
    text = (ROOT / "judges" / judge / ("prompt_fixed.md" if judge == "integrity" else "prompt.md")).read_text().replace("{context}", ctx)
    for k, v in values.items():
        text = text.replace("{" + k + "}", v)
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
