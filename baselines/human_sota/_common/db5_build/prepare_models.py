"""Download public auxiliary model weights; no benchmark data are downloaded."""
import argparse
import json
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download

REPOS = {
    'bookrag': ['BAAI/bge-base-en-v1.5', 'BAAI/bge-reranker-base'],
    'clue': ['BAAI/bge-m3', 'BAAI/bge-reranker-v2-m3'],
    'lego': ['sentence-transformers/all-MiniLM-L6-v2', 'BAAI/bge-reranker-v2-m3'],
    'carrot': ['BAAI/bge-m3', 'jinaai/jina-reranker-v1-turbo-en',
               'jinaai/jina-reranker-v2-base-multilingual', 'BAAI/bge-reranker-v2-m3',
               'BAAI/bge-reranker-large', 'BAAI/bge-reranker-base',
               'Alibaba-NLP/gte-multilingual-reranker-base'],
}
NAMES = {'bookrag': 'BookRAG', 'clue': 'ClueRAG', 'lego': 'LEGO-GraphRAG', 'carrot': 'CARROT', 'hammer': 'HAMMER'}
GTE_REVISION = '40ced75c3017eb27626c9d4ea981bde21a2662f4'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--method', choices=NAMES, required=True)
    args = parser.parse_args()
    repo = args.root / 'humansota' / NAMES[args.method]
    campaign = args.root / 'humansota/db5-formal-v2-20261105'
    cache = campaign / 'model-cache/huggingface'
    model_dir = repo / ('rageng/profile6/models' if args.method == 'carrot' else 'models')
    for name in REPOS.get(args.method, []):
        snapshot_download(name, local_dir=model_dir / name.split('/')[-1], cache_dir=cache)
    if args.method == 'bookrag':
        # setup_env.sh also supports predownloaded benchmark model directories.
        destination = args.root / 'envs/models'
        for name in REPOS['bookrag']:
            path = destination / name.split('/')[-1]
            if not path.exists():
                shutil.copytree(model_dir / path.name, path)
    if args.method == 'carrot':
        code = Path(snapshot_download('Alibaba-NLP/new-impl', revision=GTE_REVISION, cache_dir=cache))
        local = campaign / 'model-cache/gte-multilingual-reranker-base-offline'
        shutil.copytree(model_dir / 'gte-multilingual-reranker-base', local, dirs_exist_ok=True, symlinks=False)
        for name in ['configuration.py', 'modeling.py']:
            shutil.copy2(code / name, local / name)
        config_path = local / 'config.json'
        config = json.loads(config_path.read_text())
        config['auto_map'] = {k: v.split('--')[-1] for k, v in config['auto_map'].items()}
        config_path.write_text(json.dumps(config, indent=2))
    if args.method == 'hammer':
        snapshot_download('BAAI/bge-large-en-v1.5',
                          revision='d4aa6901d3a41ba39fb536a557fa166f842b0e09',
                          local_dir=campaign / 'model-cache/bge-large-en-v1.5')
        snapshot_download('BAAI/bge-m3', local_dir=args.root / 'humansota/ClueRAG/models/bge-m3')
        # The original build/download_models.sh populates the remaining search-space cache.
    if args.method in {'bookrag', 'carrot'}:
        import os
        os.environ['TIKTOKEN_CACHE_DIR'] = str(repo / 'rageng/tiktoken_cache')
        import tiktoken
        tiktoken.get_encoding('cl100k_base')

if __name__ == '__main__':
    main()
