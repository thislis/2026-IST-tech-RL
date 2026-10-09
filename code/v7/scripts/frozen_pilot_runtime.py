"""Restore the exact registered pilot source for evaluating pre-fix checkpoints."""

# Locate shared modules from the physical versioned layout.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents
                    if (p / "code/shared/project_paths.py").is_file())
_layout_sys.path.insert(0, str(_layout_root / "code/shared"))
from project_paths import activate as _activate_layout
_activate_layout()
from project_paths import project_root, project_path

import fcntl
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = project_root()


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_files(root):
    files = sorted(list((root/'blackout_rl').rglob('*.py')) + list((root/'scripts').glob('*v7*')))
    return {str(p.relative_to(root)): sha256(p) for p in files if p.is_file()}


def resolve():
    registration = json.loads((project_path('logs/v7/reports/pilot_registration.json', root=ROOT)).read_text())
    expected = registration['source_files']
    if source_files(ROOT) == expected:
        return ROOT
    raise RuntimeError(
        "Archived pilot sources use the retired dependency layout. "
        "Their files are preserved in logs/v7/frozen_runtimes; "
        "use a new registration with the current runtime instead of replaying this cache."
    )
