import os,runpy,sys
from pathlib import Path
import dotenv
dotenv.load_dotenv=lambda *a,**k:False
_chdir=os.chdir
os.chdir=lambda path:_chdir(os.environ["SCRATCH_DIR"] if os.fspath(path)=="/tmp" else path)
import qwen_tok
qwen_tok.QwenPieceTokenizer.__init__.__defaults__=(str(Path(__file__).resolve().parents[1]/"models/qwen"),)
target=sys.argv.pop(1)
import query_server as q,hashlib,shutil
_build=q.build_rag
def build(storage,*args,**kwargs):
 dest=Path(os.environ["SCRATCH_DIR"])/"private-storages"/hashlib.sha256(storage.encode()).hexdigest()
 if not dest.exists():shutil.copytree(storage,dest)
 return _build(str(dest),*args,**kwargs)
q.build_rag=build
q.main()
