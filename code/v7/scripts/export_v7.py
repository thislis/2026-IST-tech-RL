#!/usr/bin/env python3
"""Fail closed until the official submission reset/dependency contract is resolved."""

# Locate shared modules from the physical versioned layout.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents
                    if (p / "code/shared/project_paths.py").is_file())
_layout_sys.path.insert(0, str(_layout_root / "code/shared"))
from project_paths import activate as _activate_layout
_activate_layout()

import argparse
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True);p.parse_args()
    p.error('v7 research checkpoints are not submission-certified: explicit episode/step/reset and isolated two-file loading remain unverified; export is disabled')
