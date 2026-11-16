"""Assemble the task's frozen code layout without copying experiment artifacts."""
import argparse
import json
import pathlib
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', type=pathlib.Path, required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--workspace', type=pathlib.Path, required=True)
    parser.add_argument('--upstream', type=pathlib.Path, required=True)
    parser.add_argument('--starting-package', action='store_true')
    args = parser.parse_args()
    method = args.method.resolve()
    spec = json.loads((method / 'build/layout.json').read_text())
    section = 'starting_package' if args.starting_package else 'service'
    if args.task not in spec[section]:
        parser.error(f'Task {args.task} is unavailable in {section}')
    upstream = args.upstream.resolve()
    commit = subprocess.check_output(
        ['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != spec['upstream_commit']:
        parser.error('Upstream checkout does not match the pinned commit')
    sources = dict(spec[section][args.task])
    sources.update({'build_sources/' + dest: source
                    for dest, source in spec['build_sources'].items()})
    for source in sources.values():
        if not (method / source).is_file():
            raise FileNotFoundError(method / source)
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=False)
    for dest, source in sources.items():
        target = workspace / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(method / source, target)
    # Only source modules are needed from upstream; datasets and outputs stay out.
    for source, dest in spec['upstream_copy'].items():
        source = upstream / source
        target = workspace / dest
        if source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        else:
            for module in source.rglob('*.py'):
                if '__pycache__' in module.parts:
                    continue
                output = target / module.relative_to(source)
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(module, output)
    for script in workspace.rglob('*.sh'):
        script.chmod(script.stat().st_mode | 0o111)
    print(f'ASSEMBLED {args.task} {section}; install dependencies and rebuild models/index next')


if __name__ == '__main__':
    main()
