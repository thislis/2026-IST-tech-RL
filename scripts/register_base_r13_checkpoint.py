#!/usr/bin/env python3
"""Create and register the BASE-R12 BC warm-start checkpoint with full provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from blackout_rl import (
    ExperimentIdentity,
    IPPOActorCritic,
    ScriptedTrajectoryDataset,
    SubmissionPolicy,
    behavior_clone,
    checkpoint_payload,
    register_checkpoint,
)


def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", nargs="+", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1212)
    parser.add_argument("--run-id", default="base-r13-bc-warm-start-seed-1212")
    parser.add_argument("--opponent-id", default="scripted-battery-v1")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dataset = ScriptedTrajectoryDataset(args.trajectory)
    actor_critic = IPPOActorCritic()
    samples_seen = behavior_clone(
        actor_critic, dataset, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, seed=args.seed,
    )
    policy = SubmissionPolicy()
    policy.actor_critic.load_state_dict(actor_critic.state_dict())
    config = {
        "algorithm": "BC_warm_start_for_IPPO",
        "model": dict(actor_critic.model_config),
        "behavior_cloning": {
            "trajectory_paths": [str(path) for path in args.trajectory],
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "demonstration_samples_seen": samples_seen,
        },
        "downstream": {
            "algorithm": "IPPO",
            "reward_mode": "score_delta",
            "opponent_frozen": True,
        },
    }
    identity = ExperimentIdentity(
        run_id=args.run_id,
        config=config,
        seed=args.seed,
        git_sha=git_sha(),
        opponent_id=args.opponent_id,
    )
    payload = checkpoint_payload(
        policy,
        global_step=0,
        training_seed=args.seed,
        source={
            "game_commit": "d2220a7d01be88d413f551efd529f4758833be8b",
            "python_api_commit": "6ba7d9993cf1bdefe1ed480c8efbcabcb923f539",
        },
        experiment=identity.checkpoint_metadata(),
    )
    entry = register_checkpoint(args.checkpoint, payload, registry_path=args.registry)
    print(json.dumps(entry, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
