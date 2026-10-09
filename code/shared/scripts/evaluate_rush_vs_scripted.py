#!/usr/bin/env python3
"""Evaluate a five-worker rush controller against the fixed scripted baseline."""

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


from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys


ROOT = project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from blackout_rl import ContractBlackOutEnv, ScriptedTeamController  # noqa: E402
from blackout_rl.policy import PolicyArtifact  # noqa: E402
from blackout_rl.team_state import Role  # noqa: E402
from eval.evaluator import evaluate_episode, summarize_episodes  # noqa: E402


BUILD = project_path('artifacts/builds/BlackOut.app', root=ROOT)
CONFIG = project_path('code/shared/configs/policies/scripted_battery_v1.json', root=ROOT)
GAME_COMMIT = "d2220a7d01be88d413f551efd529f4758833be8b"
PYTHON_API_COMMIT = "6ba7d9993cf1bdefe1ed480c8efbcabcb923f539"


def _job(seed: int) -> list[dict]:
    env = ContractBlackOutEnv(env_path=str(BUILD), no_graphics=False, time_scale=50.0)
    rush_artifact = PolicyArtifact.from_file("scripted-five-worker-rush-v1", "policy-config", CONFIG)
    opponent_artifact = PolicyArtifact.from_file("scripted-battery-v1", "policy-config", CONFIG)
    episodes = []
    try:
        for pair_index, team in enumerate((0, 1)):
            episodes.append(
                evaluate_episode(
                    env,
                    build=BUILD,
                    game_commit=GAME_COMMIT,
                    python_api_commit=PYTHON_API_COMMIT,
                    seed=seed,
                    model_team=team,
                    model_policy=ScriptedTeamController(
                        team,
                        seed=900001 + seed,
                        enable_special_items=False,
                        roles=(Role.WORKER, Role.WORKER, Role.WORKER, Role.GUARD, Role.GUARD),
                        chase_radius_cells=48,
                    ),
                    opponent_policy=ScriptedTeamController(
                        1 - team,
                        seed=900002 + seed,
                        enable_special_items=False,
                    ),
                    model_artifact=rush_artifact,
                    opponent_artifact=opponent_artifact,
                    pair_id=f"rush-vs-scripted-{seed}",
                    pair_index=pair_index,
                    max_steps=22_000,
                )
            )
    finally:
        env.close()
    return episodes


def main() -> int:
    with ProcessPoolExecutor(max_workers=5) as executor:
        episodes = [episode for pair in executor.map(_job, range(7171, 7176)) for episode in pair]
    print(summarize_episodes(episodes))
    print([(e["score"]["model"], e["score"]["opponent"], e["model_result"]) for e in episodes])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
