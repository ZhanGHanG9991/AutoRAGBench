"""Frozen gte-large-en-v1.5. CLS token, then L2 normalisation, matching the official encoder."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def load_encoder(path: str, device: str):
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    kwargs = {"trust_remote_code": True}
    try:
        model = AutoModel.from_pretrained(path, **kwargs)
    except Exception:
        model = AutoModel.from_pretrained(
            path, trust_remote_code=True, unpad_inputs=False, use_memory_efficient_attention=False
        )
    model.eval()
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    return tok, model, n_params


@torch.no_grad()
def embed_texts(tok, model, texts, batch: int = 32, max_length: int = 256) -> np.ndarray:
    device = next(model.parameters()).device
    dim = int(model.config.hidden_size)
    out = np.zeros((len(texts), dim), np.float16)
    start = 0
    while start < len(texts):
        chunk = [t if t else " " for t in texts[start : start + batch]]
        encoded = tok(
            chunk, max_length=max_length, padding=True, truncation=True, return_tensors="pt"
        )
        encoded = {
            "input_ids": encoded["input_ids"].to(device),
            "attention_mask": encoded["attention_mask"].to(device),
        }
        use_cuda = encoded["input_ids"].is_cuda
        try:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_cuda):
                hidden = model(**encoded).last_hidden_state[:, 0]
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or len(chunk) == 1:
                raise
            torch.cuda.empty_cache()
            print(f"oom at batch {len(chunk)}, splitting", flush=True)
            mid = max(1, len(chunk) // 2)
            batch = mid
            left = embed_texts(tok, model, texts[start : start + mid], batch=mid, max_length=max_length)
            right = embed_texts(tok, model, texts[start + mid : start + len(chunk)], batch=mid, max_length=max_length)
            out[start : start + len(chunk)] = np.concatenate([left, right], axis=0)
            start += len(chunk)
            continue
        hidden = F.normalize(hidden.float(), p=2, dim=1)
        out[start : start + len(chunk)] = hidden.half().cpu().numpy()
        if start and start % (max(batch, 1) * 20) == 0:
            print(f"encoded {start + len(chunk):,}/{len(texts):,}", flush=True)
        start += len(chunk)
    return out
