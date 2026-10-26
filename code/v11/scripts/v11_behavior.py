#!/usr/bin/env python3
"""Location-independent behavior report and trace inspection command."""
from pathlib import Path
import sys

root = next(p for p in Path(__file__).resolve().parents if (p / "code/shared/project_paths.py").is_file())
sys.path.insert(0, str(root / "code/shared"))
from project_paths import activate
activate()
from blackout_v11.behavior_reports import main

if __name__ == "__main__":
    main()
