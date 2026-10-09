#!/usr/bin/env python3
"""Run the canonical side-swapped scripted evaluation for an IPPO checkpoint."""

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

from eval.paired_series import parse_seeds  # noqa: E402
from scripts.train_ippo_vs_scripted import evaluate_checkpoint  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--build", type=Path, default=project_path('artifacts/builds/BlackOut.app', root=ROOT))
    parser.add_argument(
        "--opponent-config",
        type=Path,
        default=project_path('code/shared/configs/policies/scripted_battery_v1.json', root=ROOT),
    )
    parser.add_argument("--seeds", type=parse_seeds, required=True)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--time-scale", type=float, default=100.0)
    parser.add_argument("--max-episode-steps", type=int, default=22_000)
    args = parser.parse_args()
    result = evaluate_checkpoint(
        build=args.build.resolve(),
        checkpoint=args.checkpoint.resolve(),
        opponent_config=args.opponent_config.resolve(),
        seeds=args.seeds,
        workers=args.workers,
        time_scale=args.time_scale,
        max_episode_steps=args.max_episode_steps,
        global_step=0,
        output=args.output.resolve(),
    )
    summary = result["summary"]
    print(
        f"W-D-L={summary['wins']}-{summary['draws']}-{summary['losses']} "
        f"win_rate={summary['win_rate']:.3f} "
        f"mean_score_diff={summary['mean_model_score_diff']:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
