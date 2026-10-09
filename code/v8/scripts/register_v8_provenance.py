#!/usr/bin/env python3
"""Original game restored. Former custom-environment experiment entrypoint retired."""

# Locate shared modules from the physical versioned layout.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents
                    if (p / "code/shared/project_paths.py").is_file())
_layout_sys.path.insert(0, str(_layout_root / "code/shared"))
from project_paths import activate as _activate_layout
_activate_layout()
from project_paths import project_root, project_path

from pathlib import Path
import sys
ROOT=project_root()
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

def main():
    import json
    from blackout_rl.v8.provided_environment import verify_original, RETIRED_MESSAGE
    if any(a in sys.argv[1:] for a in ('--check','--status')):
        print(json.dumps(verify_original(),ensure_ascii=False,indent=2))
        return
    print(RETIRED_MESSAGE,file=sys.stderr)
    raise SystemExit(2)

if __name__=='__main__':main()
