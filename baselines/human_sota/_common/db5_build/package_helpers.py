#!/usr/bin/env python3
"""Assemble isolated formal-v2 packages from successful campaign builds.

This stage does not freeze.  The GPU freeze worker performs public retrieval
before and after freezing and publishes the immutable catalog only on parity.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import stat
import subprocess as sp
from pathlib import Path

R = Path(os.environ.get("AUTORAGBENCH_ROOT", Path(__file__).resolve().parents[1]))
CAMP = R / "humansota/db5-formal-v2-20261105"
METHODS = ("bookrag", "hammer", "lego", "carrot", "clue")
TASKS = ("musique", "pubmedqa", "qasper")
EVIDENCE = {
    "bookrag": "Original formal-v2 leaf/chapter text selected by BookRAG section selection, graph/PPR and Skyline, preserving first retrieval order and deduplicating exact repeats; Map/Reduce answer generation is not run.",
    "hammer": "Original formal-v2 chunks returned by the final HAMMER configuration after decomposition retrieval, fusion and reranking, preserving the processed-node order and deduplicating exact repeats.",
    "lego": "Original formal-v2 passages that provide the triples on the selected LEGO reasoning paths, preserving path order and first occurrence; generated path text is not returned.",
    "carrot": "Original formal-v2 chunks in the final learned-config CARROT MCTS state, preserving MCTS state order; title-augmented model text is never returned as evidence.",
    "clue": "Original formal-v2 chunks corresponding to the final Clue/Q-Iter reranked chunk IDs, preserving upstream order; Title/Content transport wrappers are removed.",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(*cmd, **kwargs):
    sp.run([str(x) for x in cmd], check=True, **kwargs)


def sync_dir(source, target):
    target.mkdir(parents=True, exist_ok=True)
    run("rsync", "-a", "--exclude=__pycache__", str(source) + "/", str(target) + "/")


def hardlink_tree(source, target):
    if target.exists():
        raise FileExistsError(target)
    run("cp", "-al", source, target)


def hardlink_resolved_tree(source, target):
    """Hardlink file contents while dereferencing source-file symlinks."""
    source = Path(source)
    if target.exists():
        raise FileExistsError(target)
    target.mkdir(parents=True)
    for item in source.rglob("*"):
        rel = item.relative_to(source)
        dst = target / rel
        if item.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
        elif item.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.hardlink_to(item.resolve(strict=True))
        else:
            raise RuntimeError(f"unsupported GTE package entry: {item}")


def hardlink_file(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    target.hardlink_to(source)


def copy_file(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def selected_build(method, task):
    pointer = CAMP / "builds" / method / task / "SELECTED_BUILD"
    if not pointer.is_file():
        raise FileNotFoundError(f"successful build pointer missing: {pointer}")
    build = Path(pointer.read_text().strip())
    if not (build / "BUILD_COMPLETE").is_file():
        raise RuntimeError(f"incomplete build selected: {build}")
    manifest = json.loads((build / "build_manifest.json").read_text())
    if manifest["method"] != method or manifest["task"] != task or manifest["data_version"] != "formal-v2":
        raise RuntimeError(f"build manifest mismatch: {build}")
    return build, manifest


def write_start(path, body):
    path.write_text("#!/bin/bash\nset -euo pipefail\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def package_bookrag(ws, build, task):
    B = R / "humansota/BookRAG"
    for name in ("bookindex.pkl", "entities.npy", "index_stats.json"):
        copy_file(build / "index" / name, ws / "index" / name)
    copy_file(CAMP / "source_overrides/bookrag/runtime.py", ws / "adapter/runtime.py")
    copy_file(B / "rageng/engine.py", ws / "adapter/engine.py")
    copy_file(CAMP / "campaign/adapters/bookrag_engine_fe.py", ws / "adapter/bookrag_engine_fe.py")
    copy_file(CAMP / "campaign/adapters/bookrag_server.py", ws / "adapter/bookrag_server.py")
    hardlink_tree(B / "rageng/vendor", ws / "adapter/vendor")
    hardlink_tree(B / "rageng/tiktoken_cache", ws / "adapter/tiktoken_cache")
    hardlink_tree(B / ".venv", ws / "venv")
    (ws / "models").mkdir(exist_ok=True)
    for name in ("bge-base-en-v1.5", "bge-reranker-base"):
        hardlink_tree(B / "models" / name, ws / "models" / name)
    write_start(ws / "submission/start.sh", 'W=$(cd "$(dirname "$0")/.." && pwd)\n'
                'export PYTHONPATH="$W/adapter/vendor:$W/adapter" TIKTOKEN_CACHE_DIR="$W/adapter/tiktoken_cache"\n'
                'export PYTHONHASHSEED=20261022 ANONYMIZED_TELEMETRY=False TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1\n'
                'exec "$W/venv/bin/python" "$W/adapter/bookrag_server.py"\n')


def package_hammer(ws, build, task):
    H = R / "humansota/HAMMER"
    C = R / "humansota/ClueRAG"
    best = build / "search/best"
    sync_dir(best / "index", ws / "index")
    copy_file(best / "artifact.json", ws / "artifact.json")
    # Reuse the exact converted public corpus from the completed search.  The
    # HAMMER FlowBuilder needs it to reproduce the selected cache key.
    hardlink_file(build / "search/data/corpus.json", ws / "data/corpus.json")
    hardlink_file(build / "search/data/qa.jsonl", ws / "data/qa.jsonl")
    for folder in ("hammer", "docs"):
        sync_dir(H / folder, ws / "upstream" / folder)
    for folder in ("results", "studies", "hammer/data/sota"):
        (ws / "upstream" / folder).mkdir(parents=True, exist_ok=True)
    hardlink_tree(H / ".venv", ws / ".venv")
    hardlink_tree(H / "rageng/.hf-home", ws / "hf-home")
    (ws / "models").mkdir(exist_ok=True)
    hardlink_tree(C / "models/bge-m3", ws / "models/bge-m3")
    copy_file(CAMP / "source_overrides/hammer/hammer_rageng.py", ws / "submission/hammer_rageng.py")
    copy_file(CAMP / "campaign/adapters/hammer_server.py", ws / "submission/hammer_server.py")
    write_start(ws / "submission/start.sh", 'W=$(cd "$(dirname "$0")/.." && pwd)\n'
                'export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false HF_HOME="$W/hf-home"\n'
                'export LLAMA_INDEX_CACHE_DIR="$W/hf-home/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HAMMER_LOCAL_BGE_M3="$W/models/bge-m3"\n'
                'cd "$SCRATCH_DIR"\nexec "$W/.venv/bin/python" "$W/submission/hammer_server.py"\n')


def package_lego(ws, build, task):
    L = R / "humansota/LEGO-GraphRAG"
    for name in ("graph.pkl", "node_emb.npy", "triple_pids.pkl", "doc_nodes.pkl", "passages.jsonl", "graph_stats.json"):
        copy_file(build / "index" / name, ws / "index" / name)
    hardlink_tree(L / "models", ws / "models")
    for name in ("lego_text.py", "config.py"):
        copy_file(L / "rageng" / name, ws / "submission" / name)
    copy_file(CAMP / "campaign/adapters/lego_server.py", ws / "submission/lego_server.py")
    hardlink_tree(L / "Instance", ws / "submission/lego_instance")
    selection = json.loads((build / "selection.json").read_text())
    (ws / "submission/lego.json").write_text(json.dumps({"instance": selection["instance"],
                                                          "ctx": selection["ctx"], "n_passages": 8}))
    (ws / "pylib").mkdir(exist_ok=True)
    write_start(ws / "submission/start.sh", 'W=$(cd "$(dirname "$0")/.." && pwd)\n'
                'export PYTHONPATH="$W/pylib" PYTHONDONTWRITEBYTECODE=1\n'
                'exec "$W/.venv/bin/python" "$W/submission/lego_server.py"\n')


def package_carrot(ws, build, task):
    C = R / "humansota/CARROT"
    sync_dir(build / "index", ws / "index")
    copy_file(build / "training/predictor.pt", ws / "predictor.pt")
    for folder in ("core", "agents", "data"):
        sync_dir(C / folder, ws / "upstream" / folder)
    # The profile6 catalog contains relative symlinks.  Copying that directory
    # verbatim makes them point outside an isolated run, so resolve each entry
    # and hardlink the real local model tree into the package.
    model_catalog = C / "rageng/profile6/models"
    (ws / "models").mkdir(exist_ok=True)
    for model in model_catalog.iterdir():
        hardlink_tree(model.resolve(), ws / "models" / model.name)
    hardlink_tree(CAMP / "model-cache/huggingface", ws / "hf-home")
    hardlink_resolved_tree(CAMP / "model-cache/gte-multilingual-reranker-base-offline", ws / "gte-offline")
    hardlink_tree(C / ".venv", ws / ".venv")
    hardlink_tree(C / "rageng/tiktoken_cache", ws / "tiktoken_cache")
    for name in ("runtime.py", "optimize.py", "profile_native.py"):
        copy_file(CAMP / "source_overrides/carrot" / name, ws / "submission" / name)
    copy_file(CAMP / "campaign/adapters/carrot_runtime_fe.py", ws / "submission/carrot_runtime_fe.py")
    copy_file(CAMP / "campaign/adapters/carrot_server.py", ws / "submission/carrot_server.py")
    provenance = json.loads((build / "training/provenance.json").read_text())
    artifact = {"task": task, "budget": 1024, "top_k": 60, "epochs": 60, "seed": 20261022,
                "train_qids": provenance["train_qids"], "train_sha256": provenance["train_sha256"],
                "catalog": provenance["rerankers_order"],
                "label_encoder_classes": sorted(provenance["rerankers_order"]), "hidden_test_used": False}
    (ws / "artifact.json").write_text(json.dumps(artifact, indent=2))
    write_start(ws / "submission/start.sh", 'W=$(cd "$(dirname "$0")/.." && pwd)\n'
                'export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8\n'
                'export TIKTOKEN_CACHE_DIR="$W/tiktoken_cache" CARROT_UPSTREAM="$W/upstream" CARROT_MODEL_ROOT="$W/models" HF_HOME="$W/hf-home" CARROT_GTE_OFFLINE_DIR="$W/gte-offline"\n'
                'export CARROT_RAGENG_ROOT="$W"\nexec "$W/.venv/bin/python" "$W/submission/carrot_server.py"\n')


def package_clue(ws, build, task):
    C = R / "humansota/ClueRAG"
    sync_dir(build / "index/store", ws / "index/store")
    copy_file(build / "index/COSINE_1.00/graph_cache.json", ws / "index/graph_cache.json")
    for name in ("chunks.json", "index_stats.json"):
        copy_file(build / "index" / name, ws / "index" / name)
    (ws / "models").mkdir(exist_ok=True)
    for name in ("bge-m3", "bge-reranker-v2-m3"):
        hardlink_tree(C / "models" / name, ws / "models" / name)
    copy_file(C / "rageng/cluerag_text.py", ws / "submission/cluerag_text.py")
    copy_file(CAMP / "source_overrides/clue/local_backend.py", ws / "submission/local_backend.py")
    copy_file(Path(__file__).with_name("clue_startup_v1.py"), ws / "submission/clue_server.py")
    for folder in ("dataset", "generation", "index", "llm", "retrieval", "utils"):
        sync_dir(C / folder, ws / "submission/cluerag_upstream" / folder)
    (ws / "submission/cluerag.json").write_text('{"alpha":1.0,"max_tokens":1024}\n')
    (ws / "pylib").mkdir(exist_ok=True)
    hardlink_tree(C / "nltk_data", ws / "pylib/nltk_data")
    write_start(ws / "submission/start.sh", 'W=$(cd "$(dirname "$0")/.." && pwd)\n'
                'export PYTHONPATH="$W/pylib" NLTK_DATA="$W/pylib/nltk_data" TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=20261022\n'
                'exec "$W/.venv/bin/python" "$W/submission/clue_server.py"\n')


PACKAGERS = {"bookrag": package_bookrag, "hammer": package_hammer, "lego": package_lego,
             "carrot": package_carrot, "clue": package_clue}

