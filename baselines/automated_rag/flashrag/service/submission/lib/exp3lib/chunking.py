"""Token windows retaining exact source character offsets."""
def chunk_spans(text, tokenizer, size=256, overlap=32):
    if not 0 <= overlap < size:
        raise ValueError("invalid chunk parameters")
    offsets=tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                      truncation=False)["offset_mapping"]
    offsets=[(int(a),int(b)) for a,b in offsets if b>a]
    for i in range(0,len(offsets),size-overlap):
        window=offsets[i:i+size]
        a,b=window[0][0],window[-1][1]
        if text[a:b].strip(): yield a,b
        if i+size >= len(offsets): break
