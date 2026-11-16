"""Length-prefixed JSON client for the resident bge-m3 server."""
import json
import socket
import struct

import numpy as np


def embed_texts(sock_path: str, texts: list[str], timeout: float = 120.0) -> np.ndarray:
    payload = json.dumps({"texts": list(texts)}, ensure_ascii=False).encode("utf-8")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect(sock_path)
        sock.sendall(struct.pack(">I", len(payload)) + payload)
        header = _read_exact(sock, 4)
        (nbytes,) = struct.unpack(">I", header)
        body = _read_exact(sock, nbytes)
    response = json.loads(body.decode("utf-8"))
    if response.get("error"):
        raise RuntimeError(response["error"])
    rows = np.asarray(response["vectors"], dtype=np.float32)
    if rows.ndim != 2 or rows.shape[0] != len(texts):
        raise RuntimeError(f"bad embedding shape {rows.shape} for {len(texts)} texts")
    return rows


def _read_exact(sock: socket.socket, nbytes: int) -> bytes:
    chunks = []
    left = nbytes
    while left:
        part = sock.recv(left)
        if not part:
            raise RuntimeError("bge server closed the connection")
        chunks.append(part)
        left -= len(part)
    return b"".join(chunks)
