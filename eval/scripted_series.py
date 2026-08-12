#!/usr/bin/env python3
"""Paired held-out evaluation for scripted battery and special-item variants."""

from __future__ import annotations

import argparse
import sys
import uuid
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from blackout_rl import ContractBlackOutEnv, RandomPolicy, ScriptedTeamController
from blackout_rl.logging_schema import SERIES_SCHEMA_VERSION, validate_series_log, write_json
from blackout_rl.policy import PolicyArtifact
from eval.evaluator import evaluate_episode, summarize_episodes
from eval.paired_series import parse_seeds


GAME_COMMIT = "d2220a7d01be88d413f551efd529f4758833be8b"
PYTHON_API_COMMIT = "6ba7d9993cf1bdefe1ed480c8efbcabcb923f539"


class InstrumentedScriptedController(ScriptedTeamController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.event_counts: Counter[str] = Counter()

    def act(self, observations, agents):
        actions = super().act(observations, agents)
        self.event_counts.update(event["kind"] for event in self.last_events)
        return actions


def _evaluate_seed_job(job: dict) -> list[dict]:
    build = Path(job["build"])
    special_items = bool(job["special_items"])
    variant = "special" if special_items else "battery"
    model_artifact = PolicyArtifact.from_file(
        f"scripted-{variant}-v1", "policy-config", job["model_policy"]
    )
    opponent_artifact = PolicyArtifact.from_file(
        "random-v1", "policy-config", job["opponent_policy"]
    )
    seed = int(job["seed"])
    pair_id = f"{variant}-seed-{seed}"
    episodes = []
    env = ContractBlackOutEnv(
        env_path=str(build),
        no_graphics=False,
        time_scale=float(job["time_scale"]),
    )
    try:
        for pair_index, model_team in enumerate((0, 1)):
            model_policy = InstrumentedScriptedController(
                model_team,
                seed=int(job["model_policy_seed"]) + seed,
                enable_special_items=special_items,
            )
            opponent_policy = RandomPolicy(int(job["opponent_policy_seed"]) + seed)
            episode = evaluate_episode(
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
            episode["scripted_diagnostics"] = {
                "variant": variant,
                "special_items_enabled": special_items,
                "event_counts": dict(model_policy.event_counts),
            }
            episodes.append(episode)
    finally:
        env.close()
    return episodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--seeds", required=True, type=parse_seeds)
    parser.add_argument("--model-policy", required=True, type=Path)
    parser.add_argument("--opponent-policy", required=True, type=Path)
    parser.add_argument("--special-items", action="store_true")
    parser.add_argument("--model-policy-seed", type=int, default=514001)
    parser.add_argument("--opponent-policy-seed", type=int, default=514002)
    parser.add_argument("--time-scale", type=float, default=50.0)
    parser.add_argument("--max-steps", type=int, default=22_000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be positive")
    build = args.build.expanduser().resolve()
    jobs = [
        {
            "build": str(build),
            "seed": seed,
            "model_policy": str(args.model_policy.resolve()),
            "opponent_policy": str(args.opponent_policy.resolve()),
            "special_items": args.special_items,
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
    event_totals = Counter()
    for episode in episodes:
        event_totals.update(episode["scripted_diagnostics"]["event_counts"])
    payload = {
        "schema_version": SERIES_SCHEMA_VERSION,
        "series_id": str(uuid.uuid4()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "series_config": {
            "seeds": args.seeds,
            "held_out": True,
            "side_swap": True,
            "variant": "special" if args.special_items else "battery",
            "special_items_enabled": args.special_items,
            "model_policy_seed_base": args.model_policy_seed,
            "opponent_policy_seed_base": args.opponent_policy_seed,
            "time_scale": args.time_scale,
            "max_steps": args.max_steps,
            "workers": worker_count,
        },
        "episodes": episodes,
        "summary": {
            **summarize_episodes(episodes),
            "scripted_event_totals": dict(event_totals),
        },
    }
    validate_series_log(payload)
    write_json(args.output, payload)
    print(f"wrote {len(episodes)} episodes to {args.output}")
    print(payload["summary"])


if __name__ == "__main__":
    main()
