#!/usr/bin/env python3

# Locate shared modules from the physical versioned layout.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents
                    if (p / "code/shared/project_paths.py").is_file())
_layout_sys.path.insert(0, str(_layout_root / "code/shared"))
from project_paths import activate as _activate_layout
_activate_layout()
from project_paths import project_root, project_path

import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(project_root()))
from blackout_rl.v7_registry import load_config,check_config
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    graph,_=check_config(load_config(a.config));print(json.dumps(graph.audit(),indent=2))
