"""Run the archived encoder/index functions against an explicit public corpus."""
import argparse
import importlib.util
import os
from pathlib import Path
import sys
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--workspace', type=Path, required=True)
p.add_argument('--corpus', type=Path, required=True)
p.add_argument('--task', required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
w = a.workspace.resolve()
sys.path.insert(0, str(w / 'submission'))
os.environ.setdefault('AUTORAGBENCH_ROOT', str(w))
spec = importlib.util.spec_from_file_location('index_builder', w / 'build_sources/build_index.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
assert a.task in builder.TASKS
a.out.mkdir(parents=True, exist_ok=False)
start = time.monotonic()
encoder = builder.Encoder(str(w / 'models/contriever'), build_mode=True)
plan = builder.prepare(a.task, a.corpus.resolve(), a.out, encoder.tokenizer)
builder.encode(a.task, a.corpus.resolve(), a.out, plan, encoder)
builder.verify(a.task, a.out, plan, time.monotonic() - start)
