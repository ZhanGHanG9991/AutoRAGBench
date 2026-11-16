"""Assemble rebuilt artifacts, then overlay the selected frozen service code."""
import argparse
import json
import os
import shutil
from pathlib import Path
from stage import HUMAN, METHODS, overlay

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    args = parser.parse_args()
    os.environ['AUTORAGBENCH_ROOT'] = str(args.root.resolve())
    import package_helpers as pack
    ws = args.workspace.resolve()
    ws.mkdir(parents=True, exist_ok=False)
    (ws / 'submission').mkdir()
    # Package helpers preserve the index/model layout of the evaluated workspaces.
    pack.PACKAGERS[args.method](ws, args.build.resolve(), args.task)
    repo = args.root.resolve() / 'humansota' / METHODS[args.method][0]
    if args.method in {'clue', 'lego'}:
        shutil.copytree(repo / '.venv', ws / '.venv', symlinks=False)
    # Rebuilt artifacts must agree with the evaluated configuration before overlay.
    service = HUMAN / METHODS[args.method][1] / 'service'
    layout = json.loads((service / 'layout.json').read_text())[args.task]
    if args.method == 'hammer':
        expected = json.loads((service / layout['artifact.json']).read_text())
        actual = json.loads((ws / 'artifact.json').read_text())
        for key in ['params', 'index_files']:
            if actual[key] != expected[key]:
                raise RuntimeError('Rebuilt HAMMER artifact differs from evaluated ' + key)
    if args.method == 'lego':
        expected = json.loads((service / layout['submission/lego.json']).read_text())
        actual = json.loads((ws / 'submission/lego.json').read_text())
        if actual != expected:
            raise RuntimeError('Rebuilt LEGO selection differs from evaluated configuration')
    overlay(args.method, args.task, ws)
    print('Assembled', args.method, args.task, ws)
