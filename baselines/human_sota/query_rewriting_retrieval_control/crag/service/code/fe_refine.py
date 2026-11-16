"""Use upstream refine scores and exact strip order, with character provenance."""
from fe_corpus import corpus_evidence, check_limits

def tracked_strips(flat,mode):
    chars=list(zip(flat,range(len(flat))))
    if mode=='selection':return [chars]
    if mode!='excerption':raise ValueError('unsupported approved decomposition')
    # Exactly extract_strips_from_psg: split ?, split . , merge short pieces,
    # then join groups of three using a synthetic space.
    def split(parts,sep):
        out=[]
        for part in parts:
            text=''.join(c for c,_ in part);start=0
            while True:
                end=text.find(sep,start)
                if end<0:out.append(part[start:]);break
                out.append(part[start:end]);start=end+len(sep)
        return out
    origins=split(split([chars],'?'),'. ')
    strips=[]
    for part in origins:
        text=''.join(c for c,_ in part)
        if any(text==''.join(c for c,_ in x) for x in strips):continue
        if not strips or len(text.split())>5:strips.append(part)
        else:strips[-1]+=part
    groups=[]
    for start in range(0,len(strips),3):
        group=[]
        for part in strips[start:start+3]:
            if group:group.append((' ',None))
            group+=part
        groups.append(group)
    return groups

def refine_hits(evaluator,question,hits,mode,top_n):
    from crag_corpus import normalize_doc, extract_strips_from_psg, select_relevants
    candidates=[];origins=[]
    for hit in hits:
        flat=normalize_doc(hit['passage'])
        if not flat:continue
        strips=tracked_strips(flat,mode)
        assert [''.join(c for c,_ in x) for x in strips]==extract_strips_from_psg(psg=flat,mode=mode)
        for strip in strips:candidates.append(''.join(c for c,_ in strip));origins.append((hit,flat,strip))
    if not candidates:return []
    _,indices=select_relevants(strips=candidates,query=question,tokenizer=evaluator.tokenizer,model=evaluator.model,device=evaluator.device,top_n=top_n)
    evidence=corpus_evidence();passages=[]
    for idx in [int(x.strip()) for x in indices.split(',') if x.strip()][:top_n]:
        hit,flat,strip=origins[idx]
        # Indexed passage starts with title; title is metadata, never corpus text.
        title=normalize_doc(hit.get('title',''))
        raw=hit['passage'];body_start=0
        if title and normalize_doc(raw).startswith(title):body_start=len(title)
        spans=[];start=last=None
        for _char,pos in strip:
            if pos is None or pos<body_start:
                if start is not None:spans.append((start,last+1));start=last=None
                continue
            if start is not None and pos!=last+1:spans.append((start,last+1));start=None
            if start is None:start=pos
            last=pos
        if start is not None:spans.append((start,last+1))
        for a,b in spans:
            text=flat[a:b].strip()
            if text:passages.append(evidence.passage(text,doc_id=hit['doc_id']))
    return passages
