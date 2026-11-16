"""MedCPT query/article encoders and the cross-encoder.

AutoRAG's embedding slot loads one SentenceTransformer for both sides.
MedCPT uses a query encoder and an article encoder. This adapter only
changes how those weights are loaded and how a string is encoded. It does
not score questions.

Official MedCPT pooling is the [CLS] vector of the last hidden state.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MEDCPT_BI = "exp3/medcpt-bi-encoder"
MEDCPT_CROSS_MARK = "MedCPT-Cross-Encoder"


def _cls(model, tokenizer, texts, device):
    import torch

    batch = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.no_grad():
        hidden = model(**batch).last_hidden_state
    return hidden[:, 0, :].detach().cpu().numpy().astype(np.float32)


class MedCPTEmbedder:
    """encode() indexes documents. encode_queries() encodes questions."""

    def __init__(self, query_path, article_path, device="cpu"):
        from transformers import AutoModel, AutoTokenizer

        self.device = device
        self.max_seq_length = 512
        self.query_tokenizer = AutoTokenizer.from_pretrained(query_path, local_files_only=True)
        self.article_tokenizer = AutoTokenizer.from_pretrained(article_path, local_files_only=True)
        self.query_model = AutoModel.from_pretrained(query_path, local_files_only=True).to(device).eval()
        self.article_model = AutoModel.from_pretrained(article_path, local_files_only=True).to(device).eval()
        # _count_embedding_tokens in AutoRAG reads embedder.tokenizer.
        self.tokenizer = self.article_tokenizer

    def encode(self, texts, show_progress_bar=False, batch_size=32, **_ignored):
        return self.encode_documents(texts)

    def encode_documents(self, texts):
        return _cls(self.article_model, self.article_tokenizer, texts, self.device)

    def encode_queries(self, texts, show_progress_bar=False, batch_size=32, **_ignored):
        if isinstance(texts, str):
            texts = [texts]
        return _cls(self.query_model, self.query_tokenizer, texts, self.device)


class MedCPTCrossEncoder:
    """predict([(query, document), ...]) -> one logit per pair."""

    def __init__(self, model_path, device="cpu"):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        self.model.to(device).eval()

    def predict(self, pairs, show_progress_bar=False, batch_size=16, **_ignored):
        import torch

        queries = [pair[0] for pair in pairs]
        documents = [pair[1] for pair in pairs]
        batch = self.tokenizer(
            queries,
            documents,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        batch = {key: value.to(self.device) for key, value in batch.items()}
        with torch.no_grad():
            logits = self.model(**batch).logits
        values = logits.detach().cpu().numpy()
        if values.ndim == 2 and values.shape[1] == 1:
            values = values[:, 0]
        return values.astype(np.float32)


def adapter_paths(model_name: str):
    path = Path(model_name)
    marker = path / "adapter.json"
    if path.name == "medcpt-bi-encoder" and marker.is_file():
        spec = json.loads(marker.read_text())
        root = marker.parent
        return str((root / spec["query"]).resolve()), str((root / spec["article"]).resolve())
    return None


def maybe_medcpt_embedder(model_name: str, device=None):
    found = adapter_paths(model_name)
    if found is None and model_name != MEDCPT_BI:
        return None
    if found is None:
        raise FileNotFoundError("medcpt-bi-encoder adapter.json is missing")
    if device is None:
        device = "cpu"
    return MedCPTEmbedder(found[0], found[1], device=device)


def maybe_medcpt_cross_encoder(model_name: str, device=None):
    if MEDCPT_CROSS_MARK not in str(model_name):
        return None
    if device is None:
        device = "cpu"
    return MedCPTCrossEncoder(model_name, device=device)
