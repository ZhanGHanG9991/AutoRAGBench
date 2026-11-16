"""Download the pinned public encoder only; verify each file against official metadata."""
import hashlib, json, urllib.request
from pathlib import Path

repo='facebook/contriever'
revision='2bd46a25019aeea091fd42d1f0fd4801675cf699'
base=Path(__file__).resolve().parent/'models'
dest=base/'contriever'; dest.mkdir(parents=True,exist_ok=True)
url=f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true'
with urllib.request.urlopen(url,timeout=60) as r: manifest=json.load(r)
assert manifest['sha']==revision
(base/'contriever.manifest.json').write_text(json.dumps(manifest,indent=2))
files=['config.json','special_tokens_map.json','tokenizer.json','tokenizer_config.json','vocab.txt','pytorch_model.bin']
checks=[]
for name in files:
    metadata=next(f for f in manifest['siblings'] if f['rfilename']==name)
    target=dest/name
    if not target.exists():
        partial=dest/(name+'.partial')
        with urllib.request.urlopen(f'https://huggingface.co/{repo}/resolve/{revision}/{name}',timeout=900) as r,partial.open('wb') as f:
            while block:=r.read(1024*1024): f.write(block)
        target=partial
    b=target.read_bytes(); assert len(b)==metadata['size'],name
    if metadata.get('lfs'):
        actual=hashlib.sha256(b).hexdigest(); expected=metadata['lfs']['sha256']
    else:
        actual=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(); expected=metadata['blobId']
    assert actual==expected,(name,actual,expected)
    if target.name.endswith('.partial'): target.rename(dest/name)
    checks.append({'file':name,'bytes':len(b),'hash':actual,'verified':True})
(base/'verification.json').write_text(json.dumps(checks,indent=2))
print('MODEL_ALL_HASHES_VERIFIED')
