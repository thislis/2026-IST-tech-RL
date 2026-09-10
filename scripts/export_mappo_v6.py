#!/usr/bin/env python3
"""Export a v6 checkpoint to self-contained policy.py + checkpoint.pt; no training."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from blackout_rl.mappo_v6_artifacts import export_v6

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_v6(args.checkpoint, args.output), indent=2))
