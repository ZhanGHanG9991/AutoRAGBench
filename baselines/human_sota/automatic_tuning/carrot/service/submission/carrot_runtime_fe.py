"""Six-reranker CARROT runtime for fixed-evidence evaluation."""
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import faiss
import httpx
import numpy as np
import torch
from FlagEmbedding import BGEM3FlagModel

BASE = Path(__file__).resolve().parent
WORKSPACE = BASE.parent
UPSTREAM = WORKSPACE / "upstream"
sys.path[:0] = [str(BASE), str(UPSTREAM)]

from runtime import BudgetMCTS, checked, token_count  # noqa: E402
from agents.network_enhanced import SiameseNetworkEnhanced  # noqa: E402
from profile_native import ORDER, load_native_bank  # noqa: E402


class Pipeline:
    def __init__(self, workspace, task):
        self.root, self.task = Path(workspace), task
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device != "cuda":
            raise RuntimeError("formal CARROT six-reranker runtime requires CUDA")
        torch.set_num_threads(min(8, int(os.environ.get("OMP_NUM_THREADS", "8"))))
        self.embedder = BGEM3FlagModel(str(self.root / "models/bge-m3"), use_fp16=True, device="cuda")
        self.loader, self.rerankers, self.model_parameters = load_native_bank(self.root / "models")
        self.model_parameters["bge-m3"] = checked(self.embedder.model)
        self.chunks = json.loads((self.root / "index/chunks.json").read_text())
        self.embeddings = np.load(self.root / "index/embeddings.npy", mmap_mode="r")
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(np.asarray(self.embeddings, dtype="float32"))
        self.by_doc = {}
        for i, chunk in enumerate(self.chunks):
            self.by_doc.setdefault(str(chunk["doc_id"]), []).append(i)
        artifact = json.loads((self.root / "artifact.json").read_text())
        self.classes = artifact["label_encoder_classes"]
        if sorted(ORDER) != self.classes:
            raise RuntimeError("CARROT label encoder classes do not match the six-reranker catalog")
        self.network = SiameseNetworkEnhanced(embed_dim=1024, num_classes=6, use_chunk_fusion=True).cuda()
        self.network.load_state_dict(torch.load(self.root / "predictor.pt", map_location="cuda", weights_only=True))
        self.network.eval()

    def encode(self, texts):
        result = self.embedder.encode(texts, batch_size=32, max_length=8192,
                                      return_dense=True, return_sparse=False, return_colbert_vecs=False)
        return np.asarray(result["dense_vecs"], dtype="float32")

    def retrieve(self, question, meta, top_k=60):
        query = self.encode([question])[0]
        if self.task == "qasper":
            doc_id = str((meta or {}).get("doc_id") or "")
            if not doc_id or doc_id not in self.by_doc:
                raise ValueError("QASPER requires a valid meta.doc_id")
            candidates = np.asarray(self.by_doc[doc_id])
            chosen = candidates[np.argsort(-(self.embeddings[candidates] @ query))[:top_k]].tolist()
        else:
            _, ids = self.index.search(query.reshape(1, -1), min(top_k, len(self.chunks)))
            chosen = [int(i) for i in ids[0] if i >= 0]
        nodes = [SimpleNamespace(node=SimpleNamespace(text=self.chunks[i]["text"]), chunk_id=i) for i in chosen]
        # Match the released training preprocessing: re-encode all top-60 chunks and mean pool them.
        chunk_embedding = self.encode([node.node.text for node in nodes]).mean(axis=0)
        return nodes, query, chunk_embedding

    def predict_config(self, query_embedding, chunk_embedding):
        with torch.no_grad():
            logits, params = self.network(
                torch.tensor(query_embedding[None], dtype=torch.float32, device="cuda"),
                torch.tensor(chunk_embedding[None], dtype=torch.float32, device="cuda"))
        name = self.classes[int(logits[0].argmax())]
        raw = params[0].cpu().numpy().astype(float)
        return {"reranker": name, "C": float(np.clip(raw[0], 2.0, 4.0)),
                "iterations": int(np.clip(round(raw[1]), 10, 30)),
                "lambda": float(np.clip(raw[2], 0.0, 1.0))}, raw.tolist()

    def search(self, question, nodes, config, budget=1024):
        started = time.perf_counter()
        model, tokenizer = self.rerankers[config["reranker"]]
        mcts = BudgetMCTS(nodes_query=nodes, model=model, tokenizer=tokenizer,
                          C=config["C"], lambda_=config["lambda"], budget=budget)
        best = mcts.search(question, max_iterations=config["iterations"])
        ids = [node.chunk_id for node in best.state]
        if not ids:
            raise RuntimeError("CARROT MCTS selected no chunks")
        if token_count(best.concat_text) > budget:
            raise RuntimeError("CARROT MCTS returned an over-budget context")
        return ids, {"algorithm": "upstream.core.mcts.MCTS.search", "configuration": config,
                     "budget_cl100k": budget, "context_cl100k": token_count(best.concat_text),
                     "selected_chunks": ids, "selected_reward": float(best.reranker_score),
                     "selected_depth": best.layer, "mcts_iterations": mcts.total_trials,
                     "mcts_seconds": time.perf_counter() - started}

    def retrieve_passages(self, question, meta, top_k=60, budget=1024):
        started = time.perf_counter()
        nodes, query, chunk_embedding = self.retrieve(question, meta, top_k)
        config, raw = self.predict_config(query, chunk_embedding)
        ids, stats = self.search(question, nodes, config, budget)
        passages, seen = [], set()
        for chunk_id in ids:
            chunk = self.chunks[chunk_id]
            text = chunk.get("evidence_text", chunk["text"])
            key = (str(chunk["doc_id"]), text)
            if not text or key in seen:
                continue
            seen.add(key)
            passages.append({"doc_id": str(chunk["doc_id"]), "title": chunk.get("title", "") or "", "text": text})
        if not passages:
            raise RuntimeError("CARROT returned empty evidence")
        stats.update({"predictor_raw_params": raw, "passages": len(passages),
                      "selected_doc_ids": list(dict.fromkeys(p["doc_id"] for p in passages)),
                      "retrieval_total_seconds": time.perf_counter() - started})
        return passages, stats

    def generate(self, question, context, base_url):
        hints = {
            "musique": "a short answer phrase, no explanation",
            "pubmedqa": "exactly one word: yes, no, or maybe",
            "qasper": "a concise extracted span; Yes/No for yes-no questions; Unanswerable if the paper does not answer it",
        }
        prompt = (f"Use the passages to answer the question.\nAnswer format: {hints[self.task]}.\n"
                  f"Reply with the answer only.\n\nPassages:\n{context}\n\nQuestion: {question}\nAnswer:")
        with httpx.Client(timeout=50, trust_env=False) as client:
            response = client.post(base_url.rstrip("/") + "/chat/completions", json={
                "model": "generator", "temperature": 0, "max_tokens": 256,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": prompt}]})
        response.raise_for_status()
        body = response.json()
        return (body["choices"][0]["message"].get("content") or "").strip(), body.get("usage", {})
