#!/usr/bin/env python3
"""Paired-seed side-swapped evaluation for a registered deterministic IPPO checkpoint."""

from __future__ import annotations

import argparse
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from blackout_rl import ContractBlackOutEnv, DeterministicCheckpointPolicy, RandomPolicy
from blackout_rl.logging_schema import SERIES_SCHEMA_VERSION, validate_series_log, write_json
from blackout_rl.policy import PolicyArtifact
from eval.evaluator import evaluate_episode, summarize_episodes
from eval.paired_series import parse_seeds


GAME_COMMIT = "d2220a7d01be88d413f551efd529f4758833be8b"
PYTHON_API_COMMIT = "6ba7d9993cf1bdefe1ed480c8efbcabcb923f539"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--opponent-policy", required=True, type=Path)
    parser.add_argument("--policy-id", default="ippo-bc-warm-start-v1")
    parser.add_argument("--seeds", required=True, type=parse_seeds)
    parser.add_argument("--model-policy-seed", type=int, default=1315001)
    parser.add_argument("--opponent-policy-seed", type=int, default=1315002)
    parser.add_argument("--time-scale", type=float, default=50.0)
    parser.add_argument("--max-steps", type=int, default=22_000)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    model_artifact = PolicyArtifact.from_file(args.policy_id, "registered-ippo", args.checkpoint)
    opponent_artifact = PolicyArtifact.from_file(
        "random-v1", "policy-config", args.opponent_policy
    )
    episodes = []
    env = ContractBlackOutEnv(
        env_path=str(args.build.expanduser().resolve()),
        no_graphics=False,
        time_scale=args.time_scale,
    )
    try:
        for seed in args.seeds:
            for pair_index, model_team in enumerate((0, 1)):
                episodes.append(
                    evaluate_episode(
                        env,
                        build=args.build,
                        game_commit=GAME_COMMIT,
                        python_api_commit=PYTHON_API_COMMIT,
                        seed=seed,
                        model_team=model_team,
                        model_policy=DeterministicCheckpointPolicy(
                            args.checkpoint,
                            team=model_team,
                            seed=args.model_policy_seed + seed,
                        ),
                        opponent_policy=RandomPolicy(args.opponent_policy_seed + seed),
                        model_artifact=model_artifact,
                        opponent_artifact=opponent_artifact,
                        pair_id=f"ippo-seed-{seed}",
                        pair_index=pair_index,
                        max_steps=args.max_steps,
                    )
                )
    finally:
        env.close()
    payload = {
        "schema_version": SERIES_SCHEMA_VERSION,
        "series_id": str(uuid.uuid4()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "series_config": {
            "seeds": args.seeds,
            "held_out": True,
            "side_swap": True,
            "model_policy_seed_base": args.model_policy_seed,
            "opponent_policy_seed_base": args.opponent_policy_seed,
            "time_scale": args.time_scale,
            "max_steps": args.max_steps,
        },
        "episodes": episodes,
        "summary": summarize_episodes(episodes),
    }
    validate_series_log(payload)
    write_json(args.output, payload)
    print(payload["summary"])


if __name__ == "__main__":
    main()
