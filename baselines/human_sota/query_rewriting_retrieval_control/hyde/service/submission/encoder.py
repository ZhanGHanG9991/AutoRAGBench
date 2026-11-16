"""Frozen facebook/contriever mean-pooling encoder used by the HyDE adapter."""
from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class Encoder:
    def __init__(self, model_dir: str, *, build_mode: bool = False) -> None:
        torch.set_num_threads(4)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_dir, local_files_only=True).to("cuda").eval()
        self.model.requires_grad_(False)
        self.build_mode = build_mode
        self.last: list[np.ndarray] = []

    def batch(self, texts: list[str]) -> np.ndarray:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {key: value.to("cuda", non_blocking=True) for key, value in encoded.items()}
        autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.build_mode)
        with torch.inference_mode(), autocast:
            hidden = self.model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            vectors = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return vectors.float().cpu().numpy()

    def encode(self, text: str) -> np.ndarray:
        vector = self.batch([text])[0]
        self.last.append(vector)
        return vector
