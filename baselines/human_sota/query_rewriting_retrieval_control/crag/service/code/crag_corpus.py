"""Corpus-only CRAG.

Document scores and knowledge refinement call the upstream T5 evaluator
(scripts/CRAG_Inference.py process_flag, scripts/utils.py select_relevants,
scripts/internal_knowledge_preparation.py extract_strips_from_psg) unchanged.
Web search is replaced by one generator rewrite plus a second in-corpus retrieval.
"""
import os
import sys

import torch
from transformers import T5ForSequenceClassification, T5Tokenizer

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.environ.get("CRAG_SCRIPTS") or os.path.abspath(os.path.join(_HERE, "..", "scripts"))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from internal_knowledge_preparation import extract_strips_from_psg  # noqa: E402
from utils import select_relevants  # noqa: E402


def process_flag(scores, n_docs, threshold1, threshold2):
    """Verbatim from scripts/CRAG_Inference.py. threshold1 is the upper logit, threshold2 the lower logit."""
    flags = []
    for score in scores:
        if score >= threshold1:
            flags.append("2")
        elif score >= threshold2:
            flags.append("1")
        else:
            flags.append("0")
    tmp_flag = []
    identification_flag = []
    for i, f in enumerate(flags):
        tmp_flag.append(f)
        if i % n_docs == n_docs - 1:
            if "2" in tmp_flag:
                identification_flag.append(2)
            elif "1" in tmp_flag:
                identification_flag.append(1)
            else:
                identification_flag.append(0)
            tmp_flag = []
    return identification_flag


ACTION_NAME = {2: "correct", 1: "ambiguous", 0: "incorrect"}

# PopQA published run in run_crag_inference.sh. The shell passes lower_threshold 0.995
# and CRAG_Inference.main negates it before process_flag.
PAPER_UPPER = 0.592
PAPER_LOWER = -0.995


def normalize_doc(passage):
    """Match data_process.py: a doc is one line, newlines and tabs become spaces.

    CRAG_Inference.inference scores that line. An empty doc (the line ends with
    '[SEP]') is not scored.
    """
    if passage is None:
        return ""
    return str(passage).replace("\n", " ").replace("\t", " ").strip()


def format_evaluator_input(query, passage):
    """query + ' [SEP] ' + doc, the string CRAG_Inference.inference tokenizes."""
    doc = normalize_doc(passage)
    if not doc:
        return None
    return query + " [SEP] " + doc


class Evaluator:
    """Official CRAG T5-large retrieval evaluator. Not a temporary stand-in."""
    tag = "t5"
    temporary = False

    def __init__(self, model_path, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForSequenceClassification.from_pretrained(model_path, num_labels=1)
        self.model.to(self.device).eval()

    def score_docs(self, query, passages, batch_size=8):
        """One raw logit per passage.

        CRAG_Inference.inference tokenizes with padding='max_length' and
        max_length=512, then takes float(outputs['logits']) — the regression
        logit, not the sign and not a probability. train_evaluator.py and
        select_relevants also set truncation=True at 512; inference() omits
        that flag, so over-long inputs would error. We truncate to the same 512.
        """
        scores = []
        texts = [format_evaluator_input(query, p) for p in passages]
        live = [t for t in texts if t is not None]
        live_scores = []
        for i in range(0, len(live), batch_size):
            batch = live[i : i + batch_size]
            enc = self.tokenizer(
                batch, return_tensors="pt", padding="max_length", truncation=True, max_length=512
            )
            with torch.no_grad():
                logits = self.model(
                    enc["input_ids"].to(self.device), attention_mask=enc["attention_mask"].to(self.device)
                ).logits
            # num_labels=1, so each row is the single logit the upstream float() reads.
            live_scores.extend(float(x) for x in logits.view(-1).detach().cpu())
        it = iter(live_scores)
        for t in texts:
            scores.append(-1.0 if t is None else next(it))
        return scores

    def refine(self, query, passages, decompose_mode, top_n=None):
        strips = []
        for p in passages:
            flat = normalize_doc(p)
            if flat:
                strips += extract_strips_from_psg(psg=flat, mode=decompose_mode)
        if not strips:
            return ""
        if top_n is None:
            top_n = 3 if decompose_mode == "selection" else 6
        text, _idxs = select_relevants(
            strips=strips, query=query, tokenizer=self.tokenizer, model=self.model,
            device=self.device, top_n=top_n,
        )
        return text.replace("\n", " ")


def decide_action(scores, n_docs, upper, lower):
    padded = list(scores[:n_docs])
    while len(padded) < n_docs:
        padded.append(-1.0)
    flag = process_flag(padded, n_docs, upper, lower)
    return (flag[0] if flag else 0), padded


class BGEEvaluator:
    """Temporary stand-in. Same three-way interface, different scorer. Not the CRAG T5."""
    tag = "bge-reranker-base"
    temporary = True

    def __init__(self, model_path, device="cuda"):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device).eval()

    def score_docs(self, query, passages, batch_size=8):
        scores = [-1.0] * len(passages)
        pairs, idxs = [], []
        for i, p in enumerate(passages):
            if p and str(p).strip():
                pairs.append([query, p])
                idxs.append(i)
        for i in range(0, len(pairs), batch_size):
            enc = self.tokenizer(
                pairs[i : i + batch_size], padding=True, truncation=True, max_length=512, return_tensors="pt"
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self.model(**enc).logits.view(-1).detach().cpu().tolist()
            for j, s in enumerate(logits):
                scores[idxs[i + j]] = float(s)
        return scores

    def refine(self, query, passages, decompose_mode):
        strips = []
        for p in passages:
            if p and str(p).strip():
                strips += extract_strips_from_psg(psg=p, mode=decompose_mode)
        if not strips:
            return ""
        top_n = 3 if decompose_mode == "selection" else 6
        scored = sorted(zip(self.score_docs(query, strips), strips), key=lambda x: x[0], reverse=True)
        return "; ".join(s for _, s in scored[:top_n]).replace("\n", " ")


def load_evaluator(kind, path, device="cuda"):
    if kind == "bge":
        return BGEEvaluator(path, device)
    if kind == "t5":
        return Evaluator(path, device)
    raise SystemExit(f"unknown evaluator kind {kind}")


def build_knowledge(evaluator, query, first_passages, second_passages, upper, lower, n_docs, decompose_mode, top_n=None):
    """first_passages: initial top-n. second_passages: rewrite retrieval, or None if not yet fetched.

    Returns knowledge, info. Caller runs the rewrite retrieval only when the action is not Correct,
    then calls again with second_passages set. To keep one code path, pass second_passages always
    when already retrieved.
    """
    scores = evaluator.score_docs(query, first_passages[:n_docs])
    action, padded = decide_action(scores, n_docs, upper, lower)
    info = {
        "action": action,
        "action_name": ACTION_NAME[action],
        "scores": [round(s, 4) for s in padded],
        "max_score": round(max(padded), 4) if padded else None,
    }
    if action == 2:
        info["knowledge"] = evaluator.refine(query, first_passages[:n_docs], decompose_mode, top_n)
        return info
    if second_passages is None:
        info["need_rewrite"] = True
        return info
    external = evaluator.refine(query, second_passages, decompose_mode, top_n)
    if action == 0:
        info["knowledge"] = external  # discard the original documents
    else:
        internal = evaluator.refine(query, first_passages[:n_docs], decompose_mode, top_n)
        # String format written by scripts/combined_knowledge_preparation.py
        info["knowledge"] = "Knowledge1: " + internal + " [sep] Knowledge2: " + external
    return info


ANSWER_PROMPT = (
    "Refer to the following documents, follow the instruction and answer the question.\n\n"
    "Documents: {docs}\n\n"
    "Instruction: Answer the question: {question}\n\n"
    "Give only the short answer phrase, with no explanation."
)

REWRITE_PROMPT = (
    "Rewrite the question as a short search query that would retrieve the missing facts "
    "from a document collection. Output only the query, not an answer.\n\n"
    "Question: {question}\nQuery:"
)


def chat(client, prompt, max_tokens):
    """One generator call. Returns the text and prompt+completion tokens."""
    resp = client.chat.completions.create(
        model=os.environ.get("GENERATOR_NAME", "generator"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    text = (resp.choices[0].message.content or "").strip()
    usage = resp.usage
    tokens = 0
    if usage is not None:
        tokens = int(usage.prompt_tokens or 0) + int(usage.completion_tokens or 0)
    return text, tokens
