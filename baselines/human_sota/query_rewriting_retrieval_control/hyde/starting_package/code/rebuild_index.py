"""Rebuild the HyDE contriever corpus index into a new directory (never overwrites the shipped index).
Same encoding as the shipped index: facebook/contriever attention-mask mean pooling, max_length 512,
document text = title + ' ' + text (text only when the title is empty), bfloat16 autocast, stored float16,
exact inner product at query time. line_offsets.npy holds n+1 uint64 byte offsets into corpus.jsonl."""
import json, os, sys, hashlib, datetime
from pathlib import Path
import numpy as np

W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / 'submission'))
from encoder import Encoder

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=False)
corpus = Path(os.environ['CORPUS_DIR']) / 'corpus.jsonl'
assert 'hidden' not in corpus.parts and 'test' not in corpus.name
offsets = [0]
h = hashlib.sha256()
with corpus.open('rb') as f:
    for line in f:
        h.update(line)
        offsets.append(offsets[-1] + len(line))
n = len(offsets) - 1
np.save(out / 'line_offsets.npy', np.asarray(offsets, dtype=np.uint64))
enc = Encoder(str(W / 'models/contriever'), build_mode=True)
vec = np.lib.format.open_memmap(out / 'vectors.npy', mode='w+', dtype=np.float16, shape=(n, 768))
batch = int(os.environ.get('HYDE_BUILD_BATCH', '256'))
buf = []
row = 0


def flush():
    global buf, row
    if buf:
        vec[row:row + len(buf)] = enc.batch(buf).astype(np.float16)
        row += len(buf)
        buf = []


with corpus.open() as f:
    for line in f:
        d = json.loads(line)
        title = str(d.get('title') or '')
        text = str(d.get('text', ''))
        buf.append(title + ' ' + text if title else text)
        if len(buf) >= batch:
            flush()
flush()
vec.flush()
assert row == n
shipped = json.loads((W / 'index/manifest.json').read_text())
keys = ['task', 'data_version', 'chunked', 'chunk_tokens', 'dimension', 'dtype', 'pooling', 'max_length',
        'normalized', 'similarity', 'model', 'model_revision']
m = {k: shipped[k] for k in keys}
m.update(prepared=True, corpus=str(corpus), source_sha256=h.hexdigest(), n_documents=n, n_vectors=n,
         completed=datetime.datetime.now().isoformat(), rebuilt_by='code/rebuild_index.py')
(out / 'manifest.json').write_text(json.dumps(m, indent=2))
(out / 'COMPLETE').write_text(json.dumps({'task': m['task'], 'n_vectors': n}))
print('HYDE_INDEX_REBUILT', n, flush=True)
