"""Full public-corpus LightRAG entity/relation graph rebuild via local generator."""
import asyncio,json,os,sys
from pathlib import Path
import dotenv
dotenv.load_dotenv=lambda *a,**k:False
from qwen_tok import QwenPieceTokenizer
from query_server import build_rag
from lightrag.kg.shared_storage import initialize_pipeline_status
W=Path(__file__).resolve().parents[1]
async def main():
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
    qwen=QwenPieceTokenizer(str(W/'models/qwen'))
    rag,stats,client=build_rag(str(out/'storage'),os.environ['LLM_BASE_URL'],os.environ['BGE_SOCK'],qwen)
    await rag.initialize_storages();await initialize_pipeline_status()
    corpus=Path(os.environ['CORPUS_DIR'])/'corpus.jsonl'
    with corpus.open() as f:
        for line in f:
            d=json.loads(line);text=d.get('title','')+'\n\n'+d['text']
            await rag.ainsert(text,ids=[str(d['doc_id'])],file_paths=[str(d['doc_id'])])
    await rag.finalize_storages();await client.close()
    (out/'COMPLETE').write_text('Full public corpus graph and vectors built.\n')
asyncio.run(main())
