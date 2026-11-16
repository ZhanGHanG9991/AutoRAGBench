"""Stage released adapters alongside pinned, patched upstream checkouts."""
import argparse
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
HUMAN = HERE.parents[1]
METHODS = {
    'bookrag': ('BookRAG', 'hierarchical_indexing/bookrag'),
    'carrot': ('CARROT', 'automatic_tuning/carrot'),
    'clue': ('ClueRAG', 'graph_indexing/clue_rag'),
    'hammer': ('HAMMER', 'automatic_tuning/hammer'),
    'lego': ('LEGO-GraphRAG', 'graph_indexing/lego_graphrag'),
}

def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def overlay(method, task, workspace):
    source = HUMAN / METHODS[method][1] / 'service'
    layout = json.loads((source / 'layout.json').read_text())
    for target, relative in layout[task].items():
        copy(source / relative, workspace / target)
    (workspace / 'submission/start.sh').chmod(0o755)

def stage(root, method, task):
    upstream, relative = METHODS[method]
    release = HUMAN / relative
    repo = root / 'humansota' / upstream
    if not (repo / '.git').exists():
        raise FileNotFoundError('Clone the pinned upstream repository at ' + str(repo))
    campaign = root / 'humansota/db5-formal-v2-20261105'
    target = repo / 'rageng'
    for source in (release / 'build').iterdir():
        if source.is_file():
            copy(source, target / source.name)
    layout = json.loads((release / 'service/layout.json').read_text())[task]
    for rel, source in layout.items():
        if rel.endswith('.py'):
            name = Path(rel).name
            src = release / 'service' / source
            if not (release / 'build' / name).exists():
                copy(src, target / name)
            copy(src, campaign / 'source_overrides' / method / name)
            copy(src, campaign / 'campaign/adapters' / name)
    # Build-only overrides are distinct from the frozen serving modules.
    for name in ['optimize.py', 'devgrid.py', 'profile_native.py']:
        if (release / 'build' / name).exists():
            copy(release / 'build' / name, campaign / 'source_overrides' / method / name)
    for name in ['carrot_build_index.py', 'select_lego.py']:
        if (release / 'build' / name).exists():
            copy(release / 'build' / name, campaign / 'campaign/build' / name)
    copy(HERE / 'write_manifest.py', campaign / 'campaign/build/write_manifest.py')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--task', required=True)
    args = parser.parse_args()
    stage(args.root.resolve(), args.method, args.task)
