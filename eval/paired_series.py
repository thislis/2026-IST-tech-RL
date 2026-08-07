#!/usr/bin/env python3
"""Run side-swapped BlackOut episode pairs and write a validated series log."""

from __future__ import annotations

import argparse
import sys
import uuid
from concurrent.futures import ProcessPoolExecutor
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
    if len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("seeds must be unique")
    return seeds


def _evaluate_seed_job(job: dict) -> list[dict]:
    """Evaluate one side-swapped pair in an isolated process/Unity instance."""
    build = Path(job["build"])
    model_artifact = PolicyArtifact.from_file(
        "model-random-v1", "policy-config", job["model_policy"]
    )
    opponent_artifact = PolicyArtifact.from_file(
        "opponent-random-v1", "policy-config", job["opponent_policy"]
    )
    seed = int(job["seed"])
    pair_id = f"seed-{seed}"
    episodes = []
    env = ContractBlackOutEnv(
        env_path=str(build),
        no_graphics=False,
        time_scale=float(job["time_scale"]),
    )
    try:
        for pair_index, model_team in enumerate((0, 1)):
            model_policy = RandomPolicy(int(job["model_policy_seed"]) + seed)
            opponent_policy = RandomPolicy(int(job["opponent_policy_seed"]) + seed)
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
                    max_steps=int(job["max_steps"]),
                )
            )
    finally:
        env.close()
    return episodes


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
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="independent Unity processes used for seed pairs",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    build = args.build.expanduser().resolve()
    if args.workers < 1:
        parser.error("--workers must be positive")
    series_id = str(uuid.uuid4())
    jobs = [
        {
            "build": str(build),
            "seed": seed,
            "model_policy": str(args.model_policy),
            "opponent_policy": str(args.opponent_policy),
            "model_policy_seed": args.model_policy_seed,
            "opponent_policy_seed": args.opponent_policy_seed,
            "time_scale": args.time_scale,
            "max_steps": args.max_steps,
        }
        for seed in args.seeds
    ]
    worker_count = min(args.workers, len(jobs))
    if worker_count == 1:
        pairs = [_evaluate_seed_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            pairs = list(executor.map(_evaluate_seed_job, jobs))
    episodes = [episode for pair in pairs for episode in pair]

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
            "workers": worker_count,
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
