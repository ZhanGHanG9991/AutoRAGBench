import json
import os
import sqlite3
import time
import traceback
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
import httpx
from runtime import model_bundle, load_rag
from fe_corpus import corpus_evidence, check_limits

W = Path(__file__).resolve().parents[1]
INFO = json.loads((W/'submission/task.json').read_text())
mapping = json.loads((W/'index/mapping.json').read_text())
embed, ner, _ = model_bundle(W/'models')
current = {}
client = httpx.Client(timeout=50, trust_env=False)


class SqliteMetadata:
    """Read-only Mapping interface matching passage_metadata.json."""

    def __init__(self, path):
        self.connection = sqlite3.connect(
            f"file:{Path(path).resolve()}?mode=ro", uri=True, check_same_thread=False
        )

    def __getitem__(self, passage_hash):
        row = self.connection.execute(
            "SELECT doc_id, word_offset FROM metadata WHERE passage_hash=?",
            (passage_hash,),
        ).fetchone()
        if row is None:
            raise KeyError(passage_hash)
        return {'doc_id': row[0], 'word_offset': row[1]}


def get_rag(key):
    if key not in mapping:
        raise ValueError('Unknown/missing meta.doc_id')
    if key not in current:
        current.clear()
        idx = W/'index/graphs'/mapping[key]
        compact_metadata = idx/'passage_metadata.sqlite'
        metadata = (SqliteMetadata(compact_metadata) if compact_metadata.exists()
                    else json.loads((idx/'passage_metadata.json').read_text()))
        current[key] = (load_rag(idx, embed, ner), metadata)
    return current[key]


# The fixed 60-second question budget is part of the evaluation contract.  On
# FEVER the frozen official graph's entity-seeded PRPACK warmup itself took
# 45.97--62.03 seconds in three independent starts, and all first scored
# queries timed out.  Use LinearRAG's own exact dense retrieval component for
# this dataset instead of starting an iterative graph step that cannot finish
# within the contract.  This is deterministic and is reported as a budget
# adaptation; no index, embedding, corpus text, or reader setting is changed.
BUDGET_DENSE_FALLBACK = INFO['name'] == 'fever'

if INFO['name'] != 'qasper' and not BUDGET_DENSE_FALLBACK:
    warm_rag, _ = get_rag('global')
    warm_started = time.monotonic()
    warm_rag.search('Who was Albert Einstein?')
    assert warm_rag.trace.get('ppr_calls', 0) >= 1
    print('LINEARRAG_WARMUP ' + json.dumps({
        'task': INFO['name'], 'seconds': time.monotonic()-warm_started,
        'ppr_calls': warm_rag.trace.get('ppr_calls', 0)}), flush=True)
elif BUDGET_DENSE_FALLBACK:
    warm_rag, _ = get_rag('global')
    # Materialize the corpus verifier before health so the first scored query
    # does not pay one-time provenance-table initialization.
    corpus_evidence()
    print('LINEARRAG_BUDGET_ADAPTATION ' + json.dumps({
        'task': INFO['name'], 'mode': 'exact_dense_top5',
        'reason': 'entity_seeded_prpack_exceeds_60s_contract'}), flush=True)


def search_ids(rag, question):
    if not BUDGET_DENSE_FALLBACK:
        ids, scores = rag.search(question)
        return ids, scores, False
    t = time.monotonic()
    emb = rag.config.embedding_model.encode(
        question, normalize_embeddings=True, show_progress_bar=False,
        batch_size=rag.config.batch_size)
    dense_idx, dense_scores = rag.dense_passage_retrieval(emb)
    ids = [rag.passage_hash_ids[i] for i in dense_idx]
    rag.trace = {
        'ppr_calls': 0,
        'budget_dense_fallback': True,
        'query_embed_and_dense_seconds': time.monotonic() - t,
        'retrieval_seconds': time.monotonic() - t,
    }
    return ids, dense_scores, True


def retrieve(question, meta):
    t = time.monotonic()
    key = str(meta.get('doc_id', '')) if INFO['name'] == 'qasper' else 'global'
    rag, metadata = get_rag(key)
    ids, scores, adapted = search_ids(rag, question)
    selected = ids[:5]
    if INFO['name'] == 'qasper':
        assert all(str(metadata[h]['doc_id']) == key for h in selected)
    evidence = corpus_evidence()
    passages = [evidence.chunk_passage(
        rag.passage_embedding_store.hash_id_to_text[h], str(metadata[h]['doc_id'])) for h in selected]
    print('LINEARRAG_FIXED_EVIDENCE ' + json.dumps({
        'task': INFO['name'], 'selected': len(selected), 'calls': 0,
        'seconds': time.monotonic()-t, 'budget_adapted': adapted, **rag.trace}), flush=True)
    return {'passages': check_limits(passages)}


def answer(question, meta):
    t = time.monotonic()
    key = str(meta.get('doc_id', '')) if INFO['name'] == 'qasper' else 'global'
    rag, metadata = get_rag(key)
    ids, scores, adapted = search_ids(rag, question)
    selected = ids[:5]
    if INFO['name'] == 'qasper':
        assert all(metadata[h]['doc_id'] == key for h in selected)
    context = '\n\n'.join(rag.passage_embedding_store.hash_id_to_text[h] for h in selected)
    messages = [
        {'role': 'system', 'content': 'Answer the question using the provided passages. Return only the final answer. ' + INFO['answer_format']},
        {'role': 'user', 'content': context + '\n\nQuestion: ' + question}]
    response = client.post(os.environ['LLM_BASE_URL'].rstrip('/') + '/chat/completions',
        json={'model': 'generator', 'messages': messages, 'temperature': 0,
              'max_tokens': 512, 'chat_template_kwargs': {'enable_thinking': False}})
    response.raise_for_status()
    data = response.json()
    result = data['choices'][0]['message']['content'].strip()
    if not result:
        raise ValueError('Empty generator output')
    trace = dict(task=INFO['name'], calls=1, model='generator', budget_adapted=adapted,
                 usage=data.get('usage'), seconds=time.monotonic()-t, **rag.trace)
    print('LINEARRAG_TRACE ' + json.dumps(trace), flush=True)
    return {'answer': result, 'citations': [metadata[h]['doc_id'] for h in selected]}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == '/health' else 404)
        self.end_headers()

    def do_POST(self):
        try:
            if self.path not in ('/answer', '/retrieve'):
                raise ValueError('Unknown route')
            q = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            fn = retrieve if self.path == '/retrieve' else answer
            body = json.dumps(fn(q['question'], q.get('meta') or {})).encode()
            self.send_response(200)
        except Exception:
            traceback.print_exc()
            body = b'{"error":"answer_failed"}'
            self.send_response(500)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(body)


HTTPServer(('127.0.0.1', int(os.environ['PORT'])), Handler).serve_forever()
