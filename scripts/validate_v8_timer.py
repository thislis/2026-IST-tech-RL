#!/usr/bin/env python3
"""Original game restored. Former custom-environment experiment entrypoint retired."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
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
