"""Lightweight repository smoke checks.

Run from the repository root with:
    python scripts/smoke_check.py

Use --imports after installing requirements to also import the implementation modules.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def check_paths() -> None:
    required = [
        ROOT / 'resources',
        ROOT / 'resources' / '__init__.py',
        ROOT / 'resources' / 'aerofoils.py',
        ROOT / 'resources' / 'cases.py',
        ROOT / 'resources' / 'data.py',
        ROOT / 'resources' / 'nnetwork.py',
        ROOT / 'resources' / 'saved.py',
        ROOT / 'dat',
        ROOT / 'dat-saved',
        ROOT / 'jupyter',
        ROOT / 'main.py',
        ROOT / 'requirements.txt',
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f'Missing expected project paths: {missing}')


def check_notebooks() -> None:
    for notebook in sorted((ROOT / 'jupyter').glob('*.ipynb')):
        with notebook.open(encoding='utf-8') as handle:
            json.load(handle)


def check_imports() -> None:
    modules = [
        'resources.aerofoils',
        'resources.cases',
        'resources.data',
        'resources.nnetwork',
        'resources.saved',
        'main',
    ]
    for module in modules:
        importlib.import_module(module)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run lightweight repository smoke checks.')
    parser.add_argument('--imports', action='store_true', help='Import project modules after dependency installation.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    check_paths()
    check_notebooks()
    if args.imports:
        check_imports()
    print('Smoke checks passed.')


if __name__ == '__main__':
    main()
