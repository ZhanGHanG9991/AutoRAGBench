"""Map method-selected chunks to exact spans of the allowed public corpus."""
import json, os, re
from pathlib import Path

def tokens(text):
    return [(m.group().casefold(), m.start(), m.end()) for m in re.finditer(r'\w+', text)]

class Evidence:
    def __init__(self, hippo=False):
        self.docs = {}
        self.hippo = {} if hippo else None
        with (Path(os.environ['CORPUS_DIR'])/'corpus.jsonl').open() as f:
            for line in f:
                d=json.loads(line);self.docs[str(d['doc_id'])]=d
                if hippo:
                    words=list(re.finditer(r'\S+', d['text']))
                    for i in range(0,len(words),220):
                        sub=words[i:i+220]
                        raw=d['text'][sub[0].start():sub[-1].end()]
                        key=' '.join((d.get('title','')+'\n'+raw).split())
                        self.hippo.setdefault(key,{'doc_id':str(d['doc_id']),'title':d.get('title',''),'text':raw})
    def hippo_passage(self,text):
        key=' '.join(text.split())
        if key not in self.hippo:raise ValueError('Hippo passage has no exact public corpus mapping')
        return self.hippo[key]
    def leaf_passage(self,doc_id,text):
        d=self.docs[str(doc_id)];body=d['text'];raw=tokens(body);target=[v[0] for v in tokens(text)]
        title=[v[0] for v in tokens(d.get('title',''))]
        # Upstream split_text normalizes punctuation and can prepend the title.
        # Locate the same token sequence and return the original continuous span.
        options=[target]
        if title and target[:len(title)]==title:options.append(target[len(title):])
        raw_values=[v[0] for v in raw]
        for needle in options:
            if not needle:continue
            for i,value in enumerate(raw_values):
                if value==needle[0] and raw_values[i:i+len(needle)]==needle:
                    return {'doc_id':str(d['doc_id']),'title':d.get('title',''),'text':body[raw[i][1]:raw[i+len(needle)-1][2]]}
        raise ValueError('RAPTOR leaf has no exact public corpus mapping')

def bounded(passages):
    # A conservative UTF-8 byte bound also bounds byte-level tokenizer tokens.
    out=[];seen=set();nbytes=0;nchars=0
    for p in passages:
        key=(p['doc_id'],p['text'])
        if key in seen:continue
        add=len((p['title']+p['text']).encode('utf-8'));chars=len(p['title'])+len(p['text'])
        if nbytes+add>12000 or nchars+chars>32000 or len(out)>=256:break
        seen.add(key);out.append(p);nbytes+=add;nchars+=chars
    return out
