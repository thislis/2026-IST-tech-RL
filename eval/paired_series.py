#!/usr/bin/env python3
"""Run side-swapped BlackOut episode pairs and write a validated series log."""

from __future__ import annotations

import argparse
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from blackout_rl import ContractBlackOutEnv
from blackout_rl.logging_schema import SERIES_SCHEMA_VERSION, validate_series_log, write_json
from blackout_rl.policy import PolicyArtifact, RandomPolicy
from eval.evaluator import evaluate_episode, summarize_episodes


GAME_COMMIT = "d2220a7d01be88d413f551efd529f4758833be8b"
PYTHON_API_COMMIT = "6ba7d9993cf1bdefe1ed480c8efbcabcb923f539"


def parse_seeds(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds or any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError("seeds must be a comma-separated list of non-negative integers")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--seeds", required=True, type=parse_seeds)
    parser.add_argument("--model-policy", required=True, type=Path)
    parser.add_argument("--opponent-policy", required=True, type=Path)
    parser.add_argument("--model-policy-seed", type=int, default=81001)
    parser.add_argument("--opponent-policy-seed", type=int, default=81002)
    parser.add_argument("--time-scale", type=float, default=50.0)
    parser.add_argument("--max-steps", type=int, default=22_000)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    build = args.build.expanduser().resolve()
    model_artifact = PolicyArtifact.from_file("model-random-v1", "policy-config", args.model_policy)
    opponent_artifact = PolicyArtifact.from_file("opponent-random-v1", "policy-config", args.opponent_policy)
    series_id = str(uuid.uuid4())
    episodes = []
    env = ContractBlackOutEnv(env_path=str(build), no_graphics=False, time_scale=args.time_scale)
    try:
        for seed in args.seeds:
            pair_id = f"seed-{seed}"
            for pair_index, model_team in enumerate((0, 1)):
                # Reset role-specific streams for each side of the pair. The
                # model keeps its stream when moving from A to B, as does the opponent.
                model_policy = RandomPolicy(args.model_policy_seed + seed)
                opponent_policy = RandomPolicy(args.opponent_policy_seed + seed)
                episodes.append(
                    evaluate_episode(
                        env,
                        build=build,
                        game_commit=GAME_COMMIT,
                        python_api_commit=PYTHON_API_COMMIT,
                        seed=seed,
                        model_team=model_team,
                        model_policy=model_policy,
                        opponent_policy=opponent_policy,
                        model_artifact=model_artifact,
                        opponent_artifact=opponent_artifact,
                        pair_id=pair_id,
                        pair_index=pair_index,
                        max_steps=args.max_steps,
                    )
                )
    finally:
        env.close()

    payload = {
        "schema_version": SERIES_SCHEMA_VERSION,
        "series_id": series_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "series_config": {
            "seeds": args.seeds,
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
    print(f"wrote {len(episodes)} episodes to {args.output}")
    print(payload["summary"])


if __name__ == "__main__":
    main()
