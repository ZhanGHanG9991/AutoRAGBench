"""self_check-style grade for one retrieved question.

Pack the passages the way fixed-evidence does, ask the platform generator
once, then apply that benchmark's own score_one. The numeric result stays
inside the return value. This module does not print it.
"""

from __future__ import annotations

import os
from pathlib import Path


def gold_row(question_id, canonical, variants):
    answers = [canonical, *list(variants)]
    return {"qid": question_id, "answers": answers}


def user_prompt(question, passages, answer_format):
    """Same user turn as multireader.contract.messages. Imported when the harness is on the path."""
    from multireader.contract import messages

    return messages(question, passages, answer_format)[0]["content"]


def pack_passages(passages):
    from multireader.contract import pack_passages as harness_pack

    return harness_pack(passages)


def primary_of(score_one, prediction, gold, primary_name):
    values = score_one(prediction, gold)
    if primary_name not in values:
        raise KeyError(primary_name)
    return float(values[primary_name])


async def grade_retrieval(pipeline, question, retrieval_result, *, task, score_one, primary_name, answer_format, titles, tokenizer):
    from multireader.contract import clean_answer, pack_for_readers

    passages = []
    for doc in retrieval_result.documents:
        did = str(doc.metadata.get("doc_id", ""))
        passages.append({"doc_id": did, "title": titles.get(did, ""), "text": doc.text})
    passages = passages[:256]
    packed = pack_for_readers(question.question, passages, answer_format, tokenizer, [tokenizer])
    prompt = user_prompt(question.question, packed, answer_format)
    # The answer call belongs to the reader, not to the retrieval budget.
    meter = getattr(pipeline, "call_meter", None)
    pipeline.call_meter = None
    try:
        raw, _cost = await pipeline.generate(prompt)
    finally:
        pipeline.call_meter = meter
    answer = clean_answer(raw or "")
    gold = gold_row(question.id, question.canonical_answer, question.answer_variants)
    value = primary_of(score_one, answer, gold, primary_name)
    return {"answer": answer, "bench_metric": value, "task": task}


def load_task_hooks(task):
    """Resolve the benchmark metric inside a job. Not used by the login-node shape tests."""
    from tasks import registry

    info = registry.info(task)
    module = registry.load(task)
    return {
        "score_one": module.score_one,
        "primary_name": info["primary_metric"],
        "answer_format": info["answer_format"],
    }


from functools import lru_cache

@lru_cache(maxsize=1)
def qwen_tokenizer():
    from transformers import AutoTokenizer

    path = os.environ.get(
        "EXP3_QWEN_TOKENIZER",
        str(Path(os.environ["EXP3_MODELS_DIR"]) / "Qwen3.5-4B"),
    )
    return AutoTokenizer.from_pretrained(path, local_files_only=True)
