#!/usr/bin/env python3
"""Resident bge-m3 encoder. CLS pooling plus L2 norm, over a unix socket.

Run with the rag-base (or vLLM) Python, which has torch and transformers.
The LightRAG 3.11 environment does not.
"""
import argparse
import json
import os
import socket
import struct

import torch
from transformers import AutoModel, AutoTokenizer


def read_exact(conn, nbytes: int) -> bytes:
    chunks = []
    left = nbytes
    while left:
        part = conn.recv(left)
        if not part:
            raise RuntimeError("client closed the connection")
        chunks.append(part)
        left -= len(part)
    return b"".join(chunks)


def encode(model, tokenizer, texts, device):
    batch = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=8192,
        return_tensors="pt",
    )
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.no_grad():
        hidden = model(**batch).last_hidden_state[:, 0]
        hidden = torch.nn.functional.normalize(hidden, p=2, dim=1)
    return hidden.float().cpu().tolist()


def serve(model, tokenizer, sock_path, device):
    if os.path.exists(sock_path):
        os.remove(sock_path)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(sock_path)
    os.chmod(sock_path, 0o600)
    server.listen(4)
    print(f"BGE_READY sock={sock_path}", flush=True)
    while True:
        conn, _addr = server.accept()
        try:
            (nbytes,) = struct.unpack(">I", read_exact(conn, 4))
            if nbytes > 32 * 1024 * 1024:
                raise RuntimeError("request too large")
            request = json.loads(read_exact(conn, nbytes).decode("utf-8"))
            texts = request.get("texts") or []
            vectors = encode(model, tokenizer, texts, device) if texts else []
            body = json.dumps({"dim": 1024, "n": len(vectors), "vectors": vectors}).encode("utf-8")
        except Exception as exc:
            body = json.dumps({"error": f"{type(exc).__name__}: {exc}"}).encode("utf-8")
        try:
            conn.sendall(struct.pack(">I", len(body)) + body)
        finally:
            conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--sock", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(args.device).eval()
    serve(model, tokenizer, args.sock, args.device)


if __name__ == "__main__":
    main()
