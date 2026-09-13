#!/usr/bin/env python3
"""Check module metadata before consumer build and runtime release gates."""

import json
from pathlib import Path
import subprocess
import sys


def go_json(*args):
    return json.loads(subprocess.check_output(['go', *args], text=True))


def main():
    failed = False

    def reject(message):
        nonlocal failed
        print('error: ' + message, file=sys.stderr)
        failed = True

    if not any(Path(name).is_file() for name in ('LICENSE', 'LICENSE.md', 'LICENSE.txt', 'COPYING')):
        reject('repository license is missing')
    workspace = go_json('work', 'edit', '-json', 'go.work')
    members = workspace.get('Use') or []
    if not members:
        reject('workspace has no modules')
    for member in members:
        directory = Path(member['DiskPath'])
        if directory.is_absolute() or '..' in directory.parts:
            reject(f'workspace member must be inside the repository: {directory}')
            continue
        if directory == Path('examples'):
            continue  # Development examples are not released as a module.
        modfile = directory / 'go.mod'
        module = go_json('mod', 'edit', '-json', str(modfile))
        expected = 'github.com/tmc/lib/' + directory.as_posix()
        if module['Module']['Path'] != expected:
            reject(f'{modfile}: module path must be {expected}')
        for replacement in module.get('Replace') or []:
            reject(f"{modfile}: remove replacement for {replacement['Old']['Path']} before release")
        for requirement in module.get('Require') or []:
            version = requirement['Version']
            if version == 'v0.0.0' or '00010101000000-000000000000' in version:
                reject(f"{modfile}: {requirement['Path']} has placeholder version {version}")
    if failed:
        return 1
    print('Release metadata checks passed; consumer builds and native runtime gates are still required.')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f'error: checking release metadata: {error}', file=sys.stderr)
        sys.exit(1)
