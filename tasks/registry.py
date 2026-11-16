"""Task registry: tasks/<name>/{task.py, info.json}."""
import importlib
import json
from pathlib import Path

TASKS_DIR = Path(__file__).resolve().parent
# the nine AutoRAGBench tasks (formal-v2)
TASKS = ["musique", "hotpotqa", "nq_open", "pubmedqa", "fever", "qasper", "webqsp", "cwq", "multihop_rag"]


def load(name):
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; known: {TASKS}")
    return importlib.import_module(f"tasks.{name}.task")


def info(name):
    return json.loads((TASKS_DIR / name / "info.json").read_text())
