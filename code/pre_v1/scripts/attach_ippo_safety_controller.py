#!/usr/bin/env python3
"""Attach the audited counter-strategy safety controller to an IPPO checkpoint."""

from __future__ import annotations

# Locate shared modules from the physical versioned layout.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents
                    if (p / "code/shared/project_paths.py").is_file())
_layout_sys.path.insert(0, str(_layout_root / "code/shared"))
from project_paths import activate as _activate_layout
_activate_layout()
from project_paths import project_root, project_path


import argparse
from pathlib import Path
import sys


ROOT = project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from blackout_rl import load_checkpoint, save_checkpoint  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    _, payload = load_checkpoint(args.input, device="cpu")
    payload.pop("optimizer_state", None)
    payload.pop("teacher_replay_state", None)
    payload["inference_guardrail"] = {
        "version": "scripted_counter_v1",
        "mode": "planner_override",
        "roles": ["worker", "worker", "worker", "guard", "guard"],
        "chase_radius_cells": 48,
        "neural_core": "parameter-sharing IPPO categorical actor-critic",
        "reason": "closed-loop safety after offline imitation and DAgger covariate shift",
    }
    payload["source"] = {
        **payload.get("source", {}),
        "safety_controller_source": str(args.input.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(args.output, payload)
    print(f"saved={args.output} guardrail={payload['inference_guardrail']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
