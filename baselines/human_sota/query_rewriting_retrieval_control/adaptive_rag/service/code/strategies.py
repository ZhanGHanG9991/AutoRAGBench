"""A / B / C routes.

A is one closed-book call. B is one retrieval and one call.
C is IRCoT cut down to at most three reasoning calls plus one answer call.
The classifier is not a generator call and does not live here.
"""
import re
import time

ANSWER_RE = re.compile(r"answer is:?\s*(.+)", re.I)
MAX_REASON = 3
TOP_K = 5
MAX_PARAS = 8
TOKEN_SOFT = 24000
CALL_LIMIT = 4
TOKEN_LIMIT = 32000
SECOND_LIMIT = 60


def clean_answer(text):
    text = re.sub(r"^(final answer|answer)\s*:\s*", "", (text or "").strip(), flags=re.I)
    return text.split("\n")[0].strip().strip('"').strip()


def answer_prompt(answer_format, question, context=None):
    head = f"Answer format: {answer_format}.\n"
    if context:
        return (
            f"Use the passages to answer the question.\n{head}Reply with the answer only.\n\n"
            f"Passages:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
    return f"{head}Reply with the answer only.\n\nQuestion: {question}\nAnswer:"


def reason_prompt(answer_format, question, context, previous):
    prev = "\n".join(f"- {line}" for line in previous) if previous else "(none yet)"
    return (
        "Answer the following question by reasoning step-by-step.\n"
        "Write exactly one short sentence.\n"
        "If the passages already contain the answer, end that sentence with "
        "\"So the answer is: <short answer>.\"\n"
        "If they do not, state the single missing fact to look up. Do not add a second sentence.\n\n"
        f"Answer format: {answer_format}.\n\n"
        f"Passages:\n{context}\n\n"
        f"Question: {question}\n"
        f"Reasoning so far:\n{prev}\n"
        "Next sentence:"
    )


def retrieval_query(sentence, question):
    sentence = ANSWER_RE.sub(" ", sentence or "")
    sentence = re.sub(r"\s+", " ", sentence).strip(" .")
    if len(sentence.split()) < 3:
        return question
    return sentence[:300]


class Budget:
    def __init__(self):
        self.calls = 0
        self.tokens = 0
        self.t0 = time.monotonic()
        self.error = None

    @property
    def seconds(self):
        return time.monotonic() - self.t0

    def can_call(self, reserve_s=6):
        return (
            self.calls < CALL_LIMIT
            and self.tokens < TOKEN_SOFT
            and self.seconds < (SECOND_LIMIT - reserve_s)
            and self.error is None
        )

    def over(self):
        return self.calls > CALL_LIMIT or self.tokens > TOKEN_LIMIT or self.seconds > SECOND_LIMIT

    def snapshot(self):
        return {
            "calls": self.calls,
            "tokens": self.tokens,
            "seconds": round(self.seconds, 3),
            "error": self.error,
            "over_budget": self.over(),
        }


def _call(gen, budget, prompt, max_tokens):
    if not budget.can_call():
        return ""
    try:
        text, tokens = gen(prompt, max_tokens)
    except Exception as exc:  # network / HTTP / timeout inside one question
        budget.error = f"{type(exc).__name__}: {exc}"
        return ""
    budget.calls += 1
    budget.tokens += int(tokens or 0)
    return text or ""


def _keep(retriever, ids, query, doc_id):
    found = retriever.search(query, k=TOP_K, doc_id=doc_id)
    for i in found:
        if i not in ids:
            ids.append(i)
    return ids[-MAX_PARAS:]


def run_a(gen, budget, answer_format, question):
    text = _call(gen, budget, answer_prompt(answer_format, question), 32)
    return clean_answer(text), text


def run_b(gen, retriever, budget, answer_format, question, doc_id):
    ids = _keep(retriever, [], question, doc_id)
    text = _call(gen, budget, answer_prompt(answer_format, question, retriever.fmt(ids)), 32)
    return clean_answer(text), text


def run_c(gen, retriever, budget, answer_format, question, doc_id):
    ids = []
    reasoning = []
    last = ""
    query = question
    for _ in range(MAX_REASON):
        if not budget.can_call(reserve_s=8):
            break
        ids = _keep(retriever, ids, query, doc_id)
        last = _call(
            gen,
            budget,
            reason_prompt(answer_format, question, retriever.fmt(ids), reasoning),
            64,
        )
        if not last:
            break
        reasoning.append(clean_answer(last) or last.strip().split("\n")[0])
        if ANSWER_RE.search(last):
            break
        query = retrieval_query(last, question)
    if budget.can_call(reserve_s=5):
        ids = ids[-MAX_PARAS:]
        text = _call(gen, budget, answer_prompt(answer_format, question, retriever.fmt(ids)), 32)
        if text:
            return clean_answer(text), text
    match = ANSWER_RE.search(last)
    if match:
        return clean_answer(match.group(1)), last
    return clean_answer(last), last
