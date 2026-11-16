"""Corpus provenance for fixed evidence; no generated content is admitted."""
import json, os, sqlite3, threading
from pathlib import Path

def normalized(s): return ' '.join(s.split())

class CorpusEvidence:
    def __init__(self, corpus=None):
        self.path=Path(corpus or os.environ['CORPUS_DIR'])/'corpus.jsonl'
        scratch=Path(os.environ.get('FE_CORPUS_CACHE_DIR') or os.environ['SCRATCH_DIR']);scratch.mkdir(parents=True,exist_ok=True)
        self.db=scratch/'fe-corpus.sqlite'
        self.lock=threading.RLock()
        # A dev launcher may share this provenance-only cache across methods.
        # Serialize initialization; lookup semantics and corpus bytes are fixed.
        import fcntl
        with (scratch/'fe-corpus.lock').open('a') as init_lock:
            fcntl.flock(init_lock,fcntl.LOCK_EX)
            self.con=sqlite3.connect(self.db,check_same_thread=False,timeout=120)
            self.con.execute('PRAGMA cache_size=-131072')
            self.con.execute('PRAGMA temp_store=MEMORY')
            self.con.execute('PRAGMA journal_mode=MEMORY')
            self.con.execute('PRAGMA synchronous=OFF')
            self.con.execute('CREATE TABLE IF NOT EXISTS docs (id TEXT,title TEXT,off INTEGER)')
            self.con.execute('CREATE TABLE IF NOT EXISTS metadata (path TEXT,size INTEGER,mtime INTEGER)')
            st=self.path.stat();signature=(str(self.path.resolve()),st.st_size,st.st_mtime_ns)
            old=self.con.execute('SELECT * FROM metadata').fetchone()
            if old is not None and old!=signature:raise ValueError('stale corpus provenance index')
            if old is None:
                if self.con.execute('SELECT 1 FROM docs LIMIT 1').fetchone():raise ValueError('incomplete corpus provenance index')
                with self.path.open('rb') as f:
                    batch=[]
                    while True:
                        off=f.tell();line=f.readline()
                        if not line:break
                        d=json.loads(line);batch.append((str(d['doc_id']),d.get('title',''),off))
                        if len(batch)>=10000:
                            self.con.executemany('INSERT INTO docs VALUES (?,?,?)',batch);batch=[]
                    self.con.executemany('INSERT INTO docs VALUES (?,?,?)',batch)
                # Sorting indexes once avoids random B-tree writes for every
                # inserted document on the shared filesystem.
                self.con.execute('CREATE UNIQUE INDEX IF NOT EXISTS ids ON docs(id)')
                self.con.execute('CREATE INDEX IF NOT EXISTS titles ON docs(title)')
                self.con.execute('INSERT INTO metadata VALUES (?,?,?)',signature);self.con.commit()
    def document(self,doc_id=None,title=None):
        with self.lock:
            row=self.con.execute('SELECT off FROM docs WHERE '+('id=?' if doc_id is not None else 'title=?'),(str(doc_id) if doc_id is not None else title,)).fetchone()
            if row is None:raise ValueError('evidence source absent from corpus')
            with self.path.open('rb') as f:f.seek(row[0]);return json.loads(f.readline())
    def chunk_passage(self,text,doc_id):
        d=self.document(doc_id=doc_id)
        if normalized(text) not in normalized(d['text']):
            # Index construction prepended title + two newlines. Remove only
            # that known metadata prefix, and still validate the whole body.
            title=d.get('title','').strip()
            prefix=title+'\n\n' if title else ''
            if prefix and text.startswith(prefix):text=text[len(prefix):]
            else:raise ValueError('indexed chunk is not a corpus substring or exact title-prefixed body')
        return self.passage(text,doc_id=doc_id)

    def passage(self,text,doc_id=None,title=None):
        d=self.document(doc_id,title)
        if not text.strip() or normalized(text) not in normalized(d['text']):
            raise ValueError('evidence text is not a corpus substring')
        return {'doc_id':str(d['doc_id']),'title':d.get('title',''),'text':text}

_INSTANCE=None
_LOCK=threading.Lock()
def corpus_evidence():
    global _INSTANCE
    with _LOCK:
        if _INSTANCE is None:_INSTANCE=CorpusEvidence()
    return _INSTANCE

def check_limits(passages):
    if len(passages)>256 or sum(len(p['title'])+len(p['text']) for p in passages)>500000:
        raise ValueError('selected method evidence exceeds fixed-evidence-v1 contract; no silent truncation')
    return passages
