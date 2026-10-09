#!/usr/bin/env python3
"""Run a module or versioned script with the shared project packages enabled."""
from pathlib import Path
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "shared"))
from project_paths import activate, project_path

activate()
if len(sys.argv) < 2:
    raise SystemExit("Usage: python code/run.py -m MODULE [args] | SCRIPT [args]")
if sys.argv[1] == "-m":
    if len(sys.argv) < 3:
        raise SystemExit("-m requires a module")
    module = sys.argv[2]
    sys.argv = sys.argv[2:]
    runpy.run_module(module, run_name="__main__", alter_sys=True)
else:
    script = project_path(sys.argv[1])
    sys.argv = [str(script), *sys.argv[2:]]
    runpy.run_path(str(script), run_name="__main__")
