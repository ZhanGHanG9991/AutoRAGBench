"""CPU T5 classifier. It returns only A, B or C.

The choice is the class token with the highest logit at the first decode step,
matching the official run_classifier.py. The three ids are the single
SentencePiece tokens for "A", "B" and "C".
"""
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

from guard import bias_label

LETTERS = ("A", "B", "C")


def option_token_ids(tokenizer):
    ids = []
    for letter in LETTERS:
        token_ids = tokenizer(letter).input_ids
        if not token_ids:
            raise SystemExit(f"tokenizer returned no ids for {letter}")
        ids.append(int(token_ids[0]))
    if len(set(ids)) != 3:
        raise SystemExit(f"A/B/C are not three distinct tokens: {ids}")
    return ids


@torch.no_grad()
def predict_letters(model, tokenizer, questions, batch_size=8):
    """Return one of A/B/C per question. Argmax over the three class logits."""
    if not questions:
        return []
    device = next(model.parameters()).device
    opt_ids = option_token_ids(tokenizer)
    was_training = model.training
    model.eval()
    letters = []
    for start in range(0, len(questions), batch_size):
        chunk = questions[start : start + batch_size]
        src = tokenizer(chunk, max_length=256, truncation=True, padding=True, return_tensors="pt")
        src = {key: value.to(device) for key, value in src.items()}
        first = model.generate(
            **src,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
        ).scores[0]
        choice = torch.stack([first[:, token_id] for token_id in opt_ids], dim=1)
        letters.extend(LETTERS[i] for i in choice.argmax(dim=1).tolist())
    if was_training:
        model.train()
    return letters


class Classifier:
    def __init__(self, path, task):
        self.task = task
        self.fallback = bias_label(task)
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = T5ForConditionalGeneration.from_pretrained(path)
        n_param = sum(p.numel() for p in self.model.parameters())
        if n_param >= 1_000_000_000:
            raise SystemExit(f"classifier has {n_param} parameters")
        self.option_ids = option_token_ids(self.tok)
        self.model.eval()
        self.params = n_param

    def predict(self, question):
        try:
            letter = predict_letters(self.model, self.tok, [question])[0]
        except Exception:
            return self.fallback, True
        if letter in LETTERS:
            return letter, False
        return self.fallback, True
