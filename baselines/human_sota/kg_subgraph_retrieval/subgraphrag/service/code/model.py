"""SubgraphRAG retriever: frozen text vectors + DDE + a two-layer MLP.

The math follows Graph-COM/SubgraphRAG ``retrieve/src/model/retriever.py``.
DDE is mean aggregation along edges (and along reversed edges). That is the
same update as PyG ``MessagePassing(aggr='mean')`` with no parameters.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def mean_aggregate(x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    if src.numel() == 0:
        return out
    out.index_add_(0, dst, x.index_select(0, src))
    deg = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    deg.index_add_(0, dst, torch.ones(src.shape[0], device=x.device, dtype=x.dtype))
    mask = deg > 0
    out[mask] = out[mask] / deg[mask].unsqueeze(1)
    return out


class Retriever(nn.Module):
    def __init__(self, emb_size: int, topic_pe: bool = True, num_rounds: int = 2, num_reverse_rounds: int = 2):
        super().__init__()
        self.non_text_entity_emb = nn.Embedding(1, emb_size)
        self.topic_pe = topic_pe
        self.num_rounds = num_rounds
        self.num_reverse_rounds = num_reverse_rounds
        pred_in = 4 * emb_size
        if topic_pe:
            pred_in += 2 * 2
        pred_in += 2 * 2 * (num_rounds + num_reverse_rounds)
        self.pred_in = pred_in
        self.pred = nn.Sequential(
            nn.Linear(pred_in, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, 1),
        )

    def dde(self, topic_oh: torch.Tensor, h_ids: torch.Tensor, t_ids: torch.Tensor):
        feats = []
        h = topic_oh
        for _ in range(self.num_rounds):
            h = mean_aggregate(h, h_ids, t_ids)
            feats.append(h)
        h = topic_oh
        for _ in range(self.num_reverse_rounds):
            h = mean_aggregate(h, t_ids, h_ids)
            feats.append(h)
        return feats

    def forward(self, h_ids, r_ids, t_ids, q_emb, entity_embs, n_non_text, relation_embs, topic_oh):
        device = topic_oh.device
        if n_non_text:
            idx = torch.zeros(n_non_text, dtype=torch.long, device=device)
            extra = self.non_text_entity_emb(idx)
            h_e = torch.cat([entity_embs, extra], dim=0) if entity_embs.numel() else extra
        else:
            h_e = entity_embs
        parts = [h_e]
        if self.topic_pe:
            parts.append(topic_oh)
        if h_ids.numel():
            parts.extend(self.dde(topic_oh, h_ids, t_ids))
        else:
            zeros = torch.zeros_like(topic_oh)
            parts.extend([zeros for _ in range(self.num_rounds + self.num_reverse_rounds)])
        h_e = torch.cat(parts, dim=1)
        if h_ids.numel() == 0:
            return h_e.new_zeros((0, 1))
        # Score triples in chunks so a 100k-triple question does not build one giant concat.
        q = q_emb.reshape(1, -1)
        outs = []
        step = 8192
        for s in range(0, h_ids.shape[0], step):
            sl = slice(s, s + step)
            hh = h_ids[sl]
            block = torch.cat(
                [
                    q.expand(hh.shape[0], -1),
                    h_e.index_select(0, hh),
                    relation_embs.index_select(0, r_ids[sl]),
                    h_e.index_select(0, t_ids[sl]),
                ],
                dim=1,
            )
            outs.append(self.pred(block))
        return torch.cat(outs, dim=0)
