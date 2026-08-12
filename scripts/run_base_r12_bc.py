#!/usr/bin/env python3
"""Run the reproducible BASE-R12 scripted-trajectory BC warm-start experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from blackout_rl import IPPOActorCritic
from blackout_rl.behavior_cloning import (
    ScriptedTrajectoryDataset,
    run_bc_warm_start_experiment,
    write_bc_experiment,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", nargs="+", type=Path, required=True)
    parser.add_argument("--held-out", nargs="+", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--environment-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1212)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    train = ScriptedTrajectoryDataset(args.train)
    held_out = ScriptedTrajectoryDataset(args.held_out)
    model = IPPOActorCritic()
    _, result = run_bc_warm_start_experiment(
        model,
        train,
        held_out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        downstream_environment_steps=args.environment_steps,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_bc_experiment(args.output, result)
    print(result.to_dict())


if __name__ == "__main__":
    main()
