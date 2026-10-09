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

import argparse,json
from pathlib import Path
import sys
sys.path.insert(0,str(project_root()))
from blackout_rl.v8.statistics import compare
from blackout_rl.v8.checkpoints import atomic_json

def main():
    p=argparse.ArgumentParser(description='Compare complete registered run,map,side W/N arrays.')
    p.add_argument('--candidate',required=True);p.add_argument('--reference',required=True);p.add_argument('--output',required=True)
    p.add_argument('--fixed-reference',action='store_true');p.add_argument('--replicates',type=int,default=20000);p.add_argument('--seed',type=int,default=0);a=p.parse_args()
    result=compare(json.loads(Path(a.candidate).read_text()),json.loads(Path(a.reference).read_text()),reference_fixed=a.fixed_reference,replicates=a.replicates,seed=a.seed)
    result['input_files']={p:__import__('hashlib').sha256(Path(p).read_bytes()).hexdigest() for p in (a.candidate,a.reference)}
    atomic_json(a.output,result);print(a.output)
if __name__=='__main__':main()
