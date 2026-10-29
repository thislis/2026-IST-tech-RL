#!/usr/bin/env python3
"""Location-independent v12 entry point; installs or starts nothing on import."""
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'code/shared/project_paths.py').is_file())
sys.path.insert(0, str(ROOT / 'code/shared'))
from project_paths import activate
activate()
from blackout_v12.runner import main

if __name__ == '__main__':
    main()
