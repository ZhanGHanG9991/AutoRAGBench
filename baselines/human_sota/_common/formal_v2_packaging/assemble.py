#!/usr/bin/env python3
"""Assemble one portable source workspace from common files and task overlays."""
import argparse
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
HUMAN_SOTA = HERE.parents[1]

def copy_layer(source, destination, skip=()):
    if not source.is_dir():
        return
    for path in sorted(source.rglob('*')):
        relative = path.relative_to(source)
        if not path.is_file() or any(part in skip for part in relative.parts):
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == '.json':
            text = path.read_text(encoding='utf-8').replace('@WORKSPACE@', destination.as_posix())
            target.write_text(text, encoding='utf-8')
        else:
            shutil.copy2(path, target)
        if target.suffix in ('.sh', '.sbatch'):
            target.chmod(target.stat().st_mode | 0o111)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('method', choices=('crag', 'adaptive_rag', 'subgraphrag', 'lightrag'))
    parser.add_argument('task')
    parser.add_argument('workspace', type=Path)
    parser.add_argument('--variant', choices=('crag_exc', 'crag_sel'))
    parser.add_argument('--starting-package', action='store_true')
    args = parser.parse_args()
    catalog = json.loads((HERE / 'catalog.json').read_text())
    entry = catalog[args.method]
    tasks = entry['starting_tasks'] if args.starting_package else entry['tasks']
    if args.task not in tasks:
        parser.error('task is not available for this package')
    if args.method == 'crag' and args.variant is None:
        parser.error('CRAG requires --variant')
    if args.method != 'crag' and args.variant is not None:
        parser.error('--variant is only valid for CRAG')
    source = HUMAN_SOTA / entry['folder'] / ('starting_package' if args.starting_package else 'service')
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=False)
    excluded = ('configs', '__pycache__', 'crag_exc', 'crag_sel')
    copy_layer(source, workspace, excluded)
    copy_layer(source / 'configs' / args.task, workspace, excluded)
    if args.variant:
        copy_layer(source / 'configs' / args.task / args.variant, workspace)
    if not (workspace / 'submission/start.sh').is_file():
        raise RuntimeError('assembled workspace has no launcher')
    print('ASSEMBLED', args.method, args.task, 'source files only')

if __name__ == '__main__':
    main()
