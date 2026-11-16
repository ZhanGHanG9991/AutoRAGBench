import hashlib,json,pathlib,re

PROTOCOL='fixed-evidence-v1'
# Qwen3.5-4B tokenizer files under <repository>/models (a symlink is fine; the build sandbox binds them at this path)
QWEN_TOKENIZER=str(pathlib.Path(__file__).absolute().parents[1]/'models'/'Qwen3.5-4B')

def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))

def sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()

def validate_passages(passages):
    if not isinstance(passages,list) or len(passages)>256:raise ValueError('passages must be a list of at most 256 items')
    clean=[]
    for p in passages:
        if not isinstance(p,dict) or any(not isinstance(p.get(k),str) for k in ['doc_id','text']):raise ValueError('passage requires string doc_id and text')
        if not p['doc_id'] or not p['text'].strip():raise ValueError('empty passage')
        title=p.get('title','')
        if not isinstance(title,str):raise ValueError('title must be a string')
        clean.append({'doc_id':p['doc_id'],'title':title,'text':p['text']})
    if sum(len(p['text'])+len(p['title']) for p in clean)>500000:raise ValueError('evidence too large')
    return clean

def normalized(text):return ' '.join(text.split())
def unescape_quotes(text):return text.replace('\\"','"')

def pack_passages(passages,max_chars=32000):
    """Shared deterministic prefix budget, independent of the answer model."""
    packed=[];left=max_chars
    for p in validate_passages(passages):
        overhead=len(p['title'])+16
        n=min(len(p['text']),max(0,left-overhead))
        if n and p['text'][:n].strip():packed.append({**p,'text':p['text'][:n]})
        left-=overhead+n
        if n<len(p['text']):break
    return packed

def pack_for_readers(question,passages,answer_format,qwen_tokenizer,reader_tokenizers,*,batch=False):
    """The same prefix for every reader; tokenizers only enforce context limits."""
    packed=[]
    def fits(candidate):
        context='\n'.join(p['title']+'\n'+p['text'] for p in candidate)
        if len(qwen_tokenizer.encode(context,add_special_tokens=False))>12000:return False
        msg=messages(question,candidate,answer_format)
        return all(len(t.apply_chat_template(msg,tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False))+512<=32768 for t in reader_tokenizers)
    if not fits([]):raise ValueError('Question itself exceeds shared context limit')
    candidates=pack_passages(passages)
    first_bad=None
    if batch and len(candidates)>1:
        # Evaluate precisely the same prefixes as the scalar algorithm. Do not
        # infer that every shorter prefix fits merely because the full one fits:
        # token counts can change at text boundaries. Binary search below stays
        # scalar and follows the original sequence of candidate substrings.
        contexts=['\n'.join(p['title']+'\n'+p['text'] for p in candidates[:i]) for i in range(1,len(candidates)+1)]
        ids=qwen_tokenizer(contexts,add_special_tokens=False,padding=False,truncation=False,return_attention_mask=False)['input_ids']
        if len(ids)!=len(candidates) or any(not isinstance(row,list) for row in ids):raise ValueError('Tokenizer batch shape mismatch')
        first_bad=next((i for i,row in enumerate(ids) if len(row)>12000),len(candidates))
        conversations=[messages(question,candidates[:i],answer_format) for i in range(1,first_bad+1)]
        for tok in reader_tokenizers:
            if not first_bad:break
            ids=tok.apply_chat_template(conversations[:first_bad],tokenize=True,add_generation_prompt=True,
                                       enable_thinking=False,return_dict=False,padding=False,truncation=False)
            if len(ids)!=first_bad or any(not isinstance(row,list) for row in ids):raise ValueError('Chat tokenizer batch shape mismatch')
            first_bad=next((i for i,row in enumerate(ids) if len(row)+512>32768),first_bad)
    for i,p in enumerate(candidates):
        if (i<first_bad if first_bad is not None else fits(packed+[p])):packed.append(p);continue
        lo,hi=0,len(p['text'])
        while lo<hi:
            mid=(lo+hi+1)//2
            if fits(packed+[{**p,'text':p['text'][:mid]}]):lo=mid
            else:hi=mid-1
        if lo and p['text'][:lo].strip():packed.append({**p,'text':p['text'][:lo]})
        break
    return packed

def lines_of(text):return [normalized(x) for x in text.split('\n') if x.strip()]

def verify_corpus(rows,corpus_file,line_scope=None):
    """One streaming scan; validate only requested passages, never gold answers.
    line_scope: a passage that is not one contiguous excerpt still counts when its title is
    the cited document's title and every non-empty line is a whole corpus line -- of the cited document ('claimed') or of
    any document ('any'; a second scan, only when some passage failed). None keeps the contiguous-excerpt rule alone."""
    wanted={}
    for row in rows:
        for p in row.get('passages',[]):wanted.setdefault(p['doc_id'],[]).append(p)
    found=set();bad=set();repaired=0;line_level=set();deferred={}
    with open(corpus_file,encoding='utf-8') as f:
        for line in f:
            d=json.loads(line);did=str(d['doc_id'])
            if did not in wanted:continue
            found.add(did);text=normalized(d['text']);title=normalized(d.get('title',''))
            for p in wanted[did]:
                if normalized(p['text']) in text and normalized(p['title'])==title:continue
                # A JSON quote over-escape (\\" for ") is a serialisation slip, not invented evidence (2026-11-02):
                # rewrite such a passage to the corpus form; anything that still differs is rejected.
                t,h=unescape_quotes(p['text']),unescape_quotes(p['title'])
                if normalized(t) in text and normalized(h)==title:p['text'],p['title']=t,d.get('title','');repaired+=1
                elif line_scope=='claimed' and normalized(p['title'])==title and set(lines_of(p['text']))<=set(lines_of(d['text'])):line_level.add(sha(p))
                elif line_scope=='any' and normalized(p['title'])==title:deferred[sha(p)]=set(lines_of(p['text']))
                else:bad.add(sha(p))
    if deferred:
        need=set().union(*deferred.values());seen=set()
        with open(corpus_file,encoding='utf-8') as f:
            for line in f:
                seen.update(x for x in lines_of(json.loads(line)['text']) if x in need)
        for k,ls in deferred.items():
            if ls<=seen:line_level.add(k)
            else:bad.add(k)
    line_level-=bad
    for row in rows:
        if any(p['doc_id'] not in found or sha(p) in bad for p in row.get('passages',[])):
            row['error']='evidence_not_from_corpus';row['passages']=[]
    return {'requested_doc_ids':len(wanted),'found_doc_ids':len(found),'bad_passages':len(bad),
            'repaired_escaping':repaired,'line_scope':line_scope,'line_level_passages':len(line_level),
            'invalid_rows':sum(r.get('error')=='evidence_not_from_corpus' for r in rows)}

def messages(question,passages,answer_format):
    context='\n\n'.join(f"[{i+1}] {p['title']}\n{p['text']}" for i,p in enumerate(passages))
    # Identical semantic prompt for every reader, one user turn works across native templates.
    head=('Use the passages to answer the question.\n' if context else '')
    prompt=head+f'Answer format: {answer_format}.\nReply with the answer only.\n\n'
    if context:prompt+='Passages:\n'+context+'\n\n'
    return [{'role':'user','content':prompt+'Question: '+question+'\nAnswer:'}]

def clean_answer(text):
    text=text.strip()
    if text.startswith('```') and text.endswith('```'):
        text=re.sub(r'^```(?:text|answer)?\s*','',text).removesuffix('```').strip()
    return re.sub(r'^(?:final answer|answer)\s*:\s*','',text,flags=re.I).strip()
