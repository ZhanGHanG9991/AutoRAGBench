"""AutoRAGBench I/O adapters; upstream HippoRAG implements all graph retrieval."""
import hashlib
import importlib
import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests
import torch
from transformers import AutoModel, AutoTokenizer
from hipporag import HippoRAG, Chunk
from hipporag.utils.config_utils import BaseConfig
from hipporag.utils.llm_utils import fix_broken_generated_json

IDENTITY = 'rageng-formal-v1-qwen35-4b-bge-base-v3-schema-chunk220'


def normalize_invalid_unicode(text):
    """Replace only unpaired UTF-16 surrogates with U+FFFD.

    JSON occasionally decodes a model-emitted ``\\udxxx`` escape into a lone
    surrogate, which Python cannot persist as UTF-8.  Valid surrogate pairs
    are combined into their original Unicode scalar; every other character is
    preserved byte-for-byte at the Python string level.
    """
    output = []
    index = 0
    changed = False
    while index < len(text):
        codepoint = ord(text[index])
        if 0xD800 <= codepoint <= 0xDBFF:
            if index + 1 < len(text):
                following = ord(text[index + 1])
                if 0xDC00 <= following <= 0xDFFF:
                    output.append(chr(
                        0x10000 + ((codepoint - 0xD800) << 10)
                        + following - 0xDC00
                    ))
                    index += 2
                    changed = True
                    continue
            output.append('\uFFFD')
            changed = True
        elif 0xDC00 <= codepoint <= 0xDFFF:
            output.append('\uFFFD')
            changed = True
        else:
            output.append(text[index])
        index += 1
    return ''.join(output), changed


def normalize_openie_unicode(ner_results, triple_results):
    """Make parsed OpenIE values valid UTF-8 without changing valid text."""
    repairs = 0
    for result in ner_results.values():
        normalized = []
        changed_result = False
        for entity in result.unique_entities:
            value, changed = normalize_invalid_unicode(entity)
            normalized.append(value)
            repairs += int(changed)
            changed_result |= changed
        result.unique_entities = normalized
        if changed_result:
            result.metadata['adapter_repair'] = 'invalid_unicode_scalar'
    for result in triple_results.values():
        normalized = []
        changed_result = False
        for triple in result.triples:
            values = []
            for item in triple:
                value, changed = normalize_invalid_unicode(item)
                values.append(value)
                repairs += int(changed)
                changed_result |= changed
            normalized.append(values)
        result.triples = normalized
        if changed_result:
            result.metadata['adapter_repair'] = 'invalid_unicode_scalar'
    return repairs

def config(index):
    return BaseConfig(save_dir=str(index), llm_name='generator',
        llm_base_url='http://127.0.0.1/rageng-gateway',
        embedding_model_name='BAAI/bge-base-en-v1.5',
        embedding_max_seq_len=512, embedding_model_dtype='float32',
        embedding_batch_size=64, openie_max_workers=16,
        openie_ner_max_tokens=1024, openie_triple_max_tokens=2048,
        response_format={'type':'json_object'}, max_retry_attempts=0,
        linking_top_k=5, retrieval_top_k=5, qa_top_k=5,
        damping=0.5, passage_node_weight=0.05, save_openie=False)

class GatewayLLM:
    def __init__(self, url, cfg, cache=None):
        u = urlparse(url)
        if u.scheme != 'http' or u.hostname not in ('127.0.0.1','localhost'):
            raise ValueError('Only the evaluator-owned loopback gateway is allowed')
        self.url = url.rstrip('/') + '/chat/completions'
        self.global_config = cfg
        self.llm_name = 'generator'
        self.cache = Path(cache) if cache else None
        self.lock = threading.Lock()
        self.cache_db = None
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)
            self.cache_db = sqlite3.connect(
                self.cache / 'cache.sqlite3', timeout=60, check_same_thread=False)
            self.cache_db.execute('PRAGMA journal_mode=WAL')
            self.cache_db.execute('PRAGMA synchronous=NORMAL')
            self.cache_db.execute(
                'CREATE TABLE IF NOT EXISTS responses '
                '(key TEXT PRIMARY KEY, payload TEXT NOT NULL)')
            self.cache_db.commit()
        self.calls = 0
        self.tokens = 0
        self.events = []
        self.deadline = None

    @staticmethod
    def _valid_structured_output(text, field, finish_reason=None):
        """Match the upstream OpenIE parsers before accepting a cache entry."""
        # Upstream repairs token-limit truncation before parsing.  Validate that
        # repaired form here so repairable legacy responses are not rejected.
        if finish_reason == 'length':
            text = fix_broken_generated_json(text)
        decoder = json.JSONDecoder()
        for start, character in enumerate(text):
            if character != '{':
                continue
            try:
                payload, _ = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                continue
            value = payload.get(field) if isinstance(payload, dict) else None
            if not isinstance(value, list):
                continue
            if field == 'named_entities' and all(isinstance(x, str) for x in value):
                return True
            if field == 'triples' and all(
                    isinstance(x, list) and len(x) == 3 and
                    all(isinstance(y, str) for y in x) for x in value):
                return True
        # The official NER parser also accepts an exact bare JSON list.
        if field == 'named_entities':
            try:
                value = json.loads(text.strip())
            except json.JSONDecodeError:
                return False
            return isinstance(value, list) and all(isinstance(x, str) for x in value)
        return False

    @classmethod
    def _normalize_finish_reason(cls, text, field, metadata, max_new_tokens):
        """Correct gateways that report ``stop`` after exhausting max tokens.

        HippoRAG's official parser repairs truncated JSON only when the finish
        reason is ``length``.  Some OpenAI-compatible vLLM responses observed
        during the build instead report another reason while usage proves that
        the configured completion limit was reached.  Normalize only that
        objectively identifiable case, and only for otherwise-invalid
        structured output, so the unchanged official repair path is used.
        """
        metadata = dict(metadata)
        if (field is not None and metadata.get('finish_reason') != 'length' and
                metadata.get('completion_tokens', 0) >= max_new_tokens and
                not cls._valid_structured_output(
                    text, field, metadata.get('finish_reason'))):
            metadata['finish_reason'] = 'length'
        return metadata

    @staticmethod
    def _bounded_retry_body(body, field):
        """Constrain only the last offline retry against runaway JSON strings."""
        retry_body = dict(body)
        retry_body['response_format'] = json.loads(json.dumps(body['response_format']))
        values = retry_body['response_format']['json_schema']['schema'][
            'properties'][field]
        values['maxItems'] = 128
        if field == 'named_entities':
            values['items']['maxLength'] = 256
        else:
            values['items']['items']['maxLength'] = 256
        return retry_body

    @classmethod
    def _salvage_complete_prefix(cls, text, field):
        """Return only fully delimited list items preceding a broken item.

        This is a final offline-only repair for a model output that exhausted
        every inference retry.  It never invents or edits an extracted value:
        an item is retained only when Python's JSON decoder accepts it and the
        following non-whitespace character is a legal comma or list close.
        The malformed item and everything after it are discarded.
        """
        marker = '"' + field + '"'
        marker_at = text.find(marker)
        if marker_at < 0:
            return None
        start = text.find('[', marker_at + len(marker))
        if start < 0:
            return None
        decoder = json.JSONDecoder()
        values, cursor = [], start + 1
        while cursor < len(text):
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            if cursor < len(text) and text[cursor] == ']':
                break
            try:
                value, end = decoder.raw_decode(text, cursor)
            except json.JSONDecodeError:
                break
            probe = end
            while probe < len(text) and text[probe].isspace():
                probe += 1
            if probe >= len(text) or text[probe] not in ',]':
                break
            valid = (isinstance(value, str) if field == 'named_entities' else
                     isinstance(value, list) and len(value) == 3 and
                     all(isinstance(x, str) for x in value))
            if not valid:
                break
            values.append(value)
            if text[probe] == ']':
                break
            cursor = probe + 1
        if not values:
            return None
        repaired = json.dumps({field: values}, ensure_ascii=False)
        return repaired if cls._valid_structured_output(repaired, field) else None

    @classmethod
    def _salvage_unterminated_single_entity(cls, text, field):
        """Repair a one-item NER array whose closing quote follows ``]}``.

        The model occasionally emits ``[\"entity]}`` followed by the missing
        quote.  Accept only that narrow delimiter transposition, decode the
        existing string bytes as JSON, and never apply it to triples.
        """
        if field != 'named_entities':
            return None
        marker = '"named_entities"'
        marker_at = text.find(marker)
        start = text.find('[', marker_at + len(marker)) if marker_at >= 0 else -1
        if start < 0:
            return None
        cursor = start + 1
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text) or text[cursor] != '"':
            return None
        close = text.find(']}"', cursor + 1)
        if close < 0:
            return None
        raw_value = text[cursor + 1:close]
        try:
            value = json.loads('"' + raw_value + '"')
        except json.JSONDecodeError:
            return None
        repaired = json.dumps({field: [value]}, ensure_ascii=False)
        return repaired if cls._valid_structured_output(repaired, field) else None

    def _quarantine_cached_response(self, key, payload, reason):
        """Preserve a malformed response for audit while preventing its reuse."""
        if not self.cache_db:
            return
        with self.lock:
            self.cache_db.execute(
                'CREATE TABLE IF NOT EXISTS invalid_responses '
                '(key TEXT NOT NULL, payload TEXT NOT NULL, reason TEXT NOT NULL, '
                'quarantined_at REAL NOT NULL)')
            self.cache_db.execute(
                'INSERT INTO invalid_responses(key,payload,reason,quarantined_at) '
                'VALUES (?,?,?,?)', (key, payload, reason, time.time()))
            self.cache_db.execute('DELETE FROM responses WHERE key=?', (key,))
            self.cache_db.commit()

    def infer(self, messages, max_new_tokens=512, **kwargs):
        body = dict(model='generator', messages=messages, temperature=0,
                    max_tokens=max_new_tokens, chat_template_kwargs={'enable_thinking':False})
        field = None
        if kwargs.get('response_format'):
            # The upstream prompts request these exact structures. Qwen otherwise
            # sometimes emits typed entity objects instead of entity strings.
            is_ner = messages[0]['content'].startswith('Your task is to extract named entities')
            field = 'named_entities' if is_ner else 'triples'
            item = {'type':'string'} if is_ner else {'type':'array','items':{'type':'string'},'minItems':3,'maxItems':3}
            schema = {'type':'object','properties':{field:{'type':'array','items':item}},'required':[field],'additionalProperties':False}
            body['response_format']={'type':'json_schema','json_schema':{'name':field,'strict':True,'schema':schema}}
        key = hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()
        if self.cache_db:
            with self.lock:
                row = self.cache_db.execute(
                    'SELECT payload FROM responses WHERE key=?', (key,)).fetchone()
            if row:
                data = json.loads(row[0])
                data['metadata'] = self._normalize_finish_reason(
                    data['text'], field, data['metadata'], max_new_tokens)
                if field is None or self._valid_structured_output(
                        data['text'], field, data['metadata'].get('finish_reason')):
                    return data['text'], data['metadata'], True
                self._quarantine_cached_response(
                    key, row[0], f'malformed cached {field} response')
        # Read legacy per-call caches so an interrupted pre-SQLite build resumes
        # without repeating model inference.  New responses use SQLite only.
        path = self.cache / (key+'.json') if self.cache else None
        if path and path.exists():
            data = json.loads(path.read_text())
            data['metadata'] = self._normalize_finish_reason(
                data['text'], field, data['metadata'], max_new_tokens)
            if field is None or self._valid_structured_output(
                    data['text'], field, data['metadata'].get('finish_reason')):
                # Do not migrate every legacy hit with a separate NFS commit.  A
                # full build can contain hundreds of thousands of these files.
                return data['text'], data['metadata'], True
            self._quarantine_cached_response(
                key, json.dumps(data), f'malformed legacy {field} response')
        timeout = 120 if self.cache else 45
        if self.deadline:
            timeout = min(timeout, self.deadline-time.monotonic())
            if timeout <= 0: raise TimeoutError('Per-question deadline reached')
        # Offline OpenIE may retry a transient local gateway/server failure.
        # Online answering (cache=None) remains strictly one gateway attempt per
        # logical call so the evaluator's four-call budget cannot be bypassed.
        # Rare passages can make the fixed local model repeat malformed JSON
        # even across the original four attempts.  Extra attempts are confined
        # to offline OpenIE and remain deterministic; online Recognition Memory
        # and final answering still perform exactly one request per logical call.
        attempts = 8 if self.cache else 1
        t = time.monotonic()
        for attempt in range(attempts):
            with self.lock:
                if not self.cache and self.calls >= 4:
                    raise RuntimeError('Call budget reached')
                self.calls += 1
            try:
                request_body = body
                if self.cache and attempt:
                    # Greedy decoding can deterministically repeat the same
                    # malformed quote/repetition loop.  Keep the official
                    # OpenIE prompt, schema and parser unchanged, but make
                    # offline retries reproducibly non-greedy so they can
                    # recover instead of replaying an identical failure.
                    request_body = dict(body)
                    request_body['temperature'] = min(
                        0.25, 0.1 + 0.05 * ((attempt - 1) // 2))
                    request_body['seed'] = (int(key[:8], 16) + attempt) % (2**31)
                    if attempt >= 3 and field is not None:
                        # Last-resort decoding guard for observed unescaped
                        # multi-kilobyte entity strings and repeated triples.
                        # Messages and the official OpenIE parser are unchanged.
                        request_body = self._bounded_retry_body(request_body, field)
                resp = requests.post(self.url, json=request_body, timeout=timeout)
                resp.raise_for_status()
            except requests.RequestException:
                if attempt + 1 == attempts:
                    raise
                time.sleep(attempt + 1)
                continue
            data = resp.json()
            choice = data['choices'][0]
            text = choice['message']['content'] or ''
            meta = dict(data.get('usage',{}),finish_reason=choice.get('finish_reason'))
            meta = self._normalize_finish_reason(
                text, field, meta, max_new_tokens)
            with self.lock:
                self.tokens += meta.get('total_tokens',0)
                self.events.append(dict(seconds=time.monotonic()-t, **meta))
            if field is None or self._valid_structured_output(
                    text, field, meta.get('finish_reason')):
                break
            if attempt + 1 == attempts:
                if self.cache_db:
                    self._quarantine_cached_response(
                        key, json.dumps(dict(text=text, metadata=meta)),
                        f'malformed generated {field} response after {attempts} attempts')
                repaired = self._salvage_complete_prefix(text, field)
                repair_name = 'complete_json_prefix'
                if repaired is None:
                    repaired = self._salvage_unterminated_single_entity(
                        text, field)
                    repair_name = 'unterminated_single_entity'
                if repaired is not None:
                    text = repaired
                    meta['adapter_repair'] = repair_name
                    break
                raise ValueError(
                    f'Gateway returned malformed {field} output after {attempts} attempts')
            time.sleep(attempt + 1)
        if self.cache_db:
            payload = json.dumps(dict(text=text,metadata=meta))
            with self.lock:
                self.cache_db.execute(
                    'INSERT OR REPLACE INTO responses(key,payload) VALUES (?,?)',
                    (key, payload))
                self.cache_db.commit()
        return text, meta, False

class BGE:
    query_instruction_mode = 'shared'
    embedding_dim = 768
    def __init__(self, path):
        self.tokenizer = AutoTokenizer.from_pretrained(path,local_files_only=True)
        self.model = AutoModel.from_pretrained(path,local_files_only=True).eval().cuda()
        self.parameters = sum(p.numel() for p in self.model.parameters())
        assert self.parameters <= 1_000_000_000
        for p in self.model.parameters(): p.requires_grad_(False)

    def batch_encode(self, texts, instruction=None, **kwargs):
        single = isinstance(texts,str)
        texts = [texts] if single else texts
        if instruction:
            texts = ['Represent this sentence for searching relevant passages: '+t for t in texts]
        rows = []
        with torch.inference_mode():
            for i in range(0,len(texts),64):
                inp = self.tokenizer(texts[i:i+64],padding=True,truncation=True,max_length=512,return_tensors='pt').to('cuda')
                out = self.model(**inp).last_hidden_state[:,0]
                rows.append(torch.nn.functional.normalize(out,p=2,dim=1).cpu().numpy())
        a = np.concatenate(rows) if rows else np.zeros((0,768),dtype=np.float32)
        return a[0] if single else a

def readonly_manifest_check(manifest_path, manifest, *args, **kwargs):
    # Frozen artifacts cannot acquire upstream's adjacent writable FileLock.
    # Compare exactly the same upstream-generated manifest, without writing.
    stored = json.loads(Path(manifest_path).read_text())
    if stored != manifest: raise RuntimeError('Frozen index manifest mismatch')

def load_rag(index, llm, embed, readonly=False):
    if readonly:
        importlib.import_module('hipporag.HippoRAG').validate_or_create_index_manifest = readonly_manifest_check
    rag = HippoRAG(global_config=config(index),extraction_llm=llm,
                   embedding_model=embed,index_identity=IDENTITY)
    # Upstream DSPy demos contain external benchmark questions. Keep its exact
    # system instructions/parser/operator, omit those examples under this protocol.
    rag.rerank_filter.message_template = rag.rerank_filter.message_template[:1]
    return rag

def chunks(row, task=None):
    """Deterministic, query-independent corpus organization.

    Freebase corpora are already line-delimited triples.  Keeping complete
    lines together is a data-reader adaptation only: OpenIE still processes
    every resulting passage and the upstream HippoRAG graph builder remains
    unchanged.
    """
    title = row.get('title', '')
    source_id = str(row['doc_id'])
    if task in ('webqsp', 'cwq'):
        lines = [line.strip() for line in row['text'].splitlines() if line.strip()]
        groups, current, words = [], [], 0
        for line in lines or ['']:
            n = max(1, len(line.split()))
            if current and words + n > 220:
                groups.append(current)
                current, words = [], 0
            current.append(line)
            words += n
        if current:
            groups.append(current)
        return [Chunk(content=title + '\n' + '\n'.join(group), source_id=source_id,
            metadata={'doc_id': source_id, 'chunk_number': i})
            for i, group in enumerate(groups)]

    words = row['text'].split()
    result = []
    for i in range(0, max(1, len(words)), 220):
        content = title + '\n' + ' '.join(words[i:i + 220])
        result.append(Chunk(content=content, source_id=source_id,
            metadata={'doc_id': source_id, 'chunk_number': i // 220}))
    return result


def retrieval_query(question, meta, task):
    """Expose benchmark-provided KG anchors without adding outside knowledge."""
    if task in ('webqsp', 'cwq'):
        topics = [str(x).strip() for x in (meta or {}).get('topic_entities', []) if str(x).strip()]
        if topics:
            return question + '\nTopic entities: ' + '; '.join(topics)
    return question

def instrument(rag):
    rag.trace = {}
    for name in ('get_fact_scores','rerank_facts','dense_passage_retrieval','graph_search_with_fact_entities','run_ppr'):
        original = getattr(rag,name)
        def wrapped(*args,_fn=original,_name=name,**kwargs):
            rag.trace[_name] = rag.trace.get(_name,0)+1
            result = _fn(*args,**kwargs)
            if _name == 'rerank_facts':
                rag.trace['facts_after_filter'] = len(result[1])
                rag.trace['fact_filter_log'] = result[2]
            return result
        setattr(rag,name,wrapped)
    return rag
