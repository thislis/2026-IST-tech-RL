#!/usr/bin/env python3
"""AGENT-14 equal-budget live navigation-shaping experiment."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from blackout_rl import (  # noqa: E402
    ContractBlackOutEnv,
    FrozenScriptedOpponent,
    PPOConfig,
    ParallelRolloutCollector,
    RewardMode,
    SubmissionPolicy,
    TeamTrainingReward,
    TrainingRewardConfig,
    checkpoint_payload,
    load_checkpoint,
    ppo_update,
    save_checkpoint,
)
from blackout_rl.curriculum import NavigationShapedTeamReward  # noqa: E402
from blackout_rl.evaluation_protocol import SeedSplits  # noqa: E402
from blackout_rl.ippo_training import select_training_device  # noqa: E402
from scripts.train_ippo_vs_scripted import evaluate_checkpoint  # noqa: E402


def _save_actor(path: Path, model, *, step: int, seed: int, arm: str) -> None:
    config = dict(model.model_config)
    config.pop("n_actions", None)
    policy = SubmissionPolicy(**config)
    policy.actor_critic.load_state_dict(model.to("cpu").state_dict())
    save_checkpoint(
        path,
        checkpoint_payload(
            policy,
            global_step=step,
            training_seed=seed,
            source={"phase": "phase3-agent14", "arm": arm},
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        default=ROOT / "checkpoints/phase3_repr_flatten_legacy.pt",
    )
    args = parser.parse_args()
    if args.steps <= 0 or args.rollout_steps <= 0 or args.steps % args.rollout_steps:
        raise ValueError("steps must be a positive multiple of rollout-steps")

    build = ROOT / "builds/BlackOut.app"
    opponent_config = ROOT / "configs/policies/scripted_battery_v1.json"
    splits = SeedSplits.from_dict(
        json.loads((ROOT / "configs/seed_splits_v1.json").read_text())
    )
    train_seed = splits.train[0]
    dev_seeds = list(splits.seeds_for("dev", purpose="model_selection"))
    device = select_training_device(args.device)
    ppo_config = PPOConfig(update_epochs=2, minibatch_size=128, target_kl=0.03)
    base_reward = TrainingRewardConfig(
        mode=RewardMode.SCORE_DELTA,
        terminal_win_reward=1.0,
    )
    arm_results: dict[str, dict] = {}

    for arm_index, arm in enumerate(("score_terminal_only", "navigation_annealed")):
        torch.manual_seed(141400)
        np.random.seed(141400)
        if device.resolved == "mps":
            torch.mps.manual_seed(141400)
        policy, _ = load_checkpoint(args.initial_checkpoint, device=device.resolved)
        model = policy.actor_critic
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        if arm == "score_terminal_only":
            reward_transform = TeamTrainingReward(0, base_reward)
        else:
            reward_transform = NavigationShapedTeamReward(
                0,
                base_config=base_reward,
                gamma=0.999,
                initial_weight=0.25,
                anneal_steps=args.steps,
            )
        env = ContractBlackOutEnv(
            env_path=str(build), no_graphics=False, time_scale=50.0
        )
        collector = ParallelRolloutCollector(
            env,
            model,
            FrozenScriptedOpponent(1, seed=141401),
            learning_team=0,
            device=device.resolved,
            reward_transform=reward_transform,
        )
        history = []
        try:
            for update, global_step in enumerate(
                range(args.rollout_steps, args.steps + 1, args.rollout_steps), start=1
            ):
                buffer = collector.collect(
                    args.rollout_steps,
                    seed=train_seed if update == 1 else None,
                )
                batch = buffer.as_batch(gamma=0.999, gae_lambda=0.95)
                diagnostics = ppo_update(model, optimizer, batch, config=ppo_config)
                history.append(
                    {
                        "update": update,
                        "global_step": global_step,
                        "mean_agent_reward": float(batch.reward.mean()),
                        "episodes_completed": collector.episodes_completed,
                        "navigation_reward_sum": float(
                            getattr(reward_transform, "navigation_reward_sum", 0.0)
                        ),
                        "navigation_weight": float(
                            reward_transform.annealer.weight(global_step)
                            if hasattr(reward_transform, "annealer")
                            else 0.0
                        ),
                        "diagnostics": diagnostics.to_dict(),
                    }
                )
        finally:
            env.close()

        checkpoint = ROOT / f"checkpoints/phase3_agent14_{arm}.pt"
        _save_actor(checkpoint, model, step=args.steps, seed=141400, arm=arm)
        evaluation_path = ROOT / f"logs/phase3_agent14_{arm}_dev.json"
        evaluation = evaluate_checkpoint(
            build=build,
            checkpoint=checkpoint,
            opponent_config=opponent_config,
            seeds=dev_seeds,
            workers=args.workers,
            time_scale=50.0,
            max_episode_steps=22_000,
            global_step=args.steps,
            output=evaluation_path,
        )
        arm_results[arm] = {
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "training_history": history,
            "evaluation_log": str(evaluation_path.relative_to(ROOT)),
            "dev_summary": evaluation["summary"],
        }

    baseline = arm_results["score_terminal_only"]["dev_summary"]
    candidate = arm_results["navigation_annealed"]["dev_summary"]
    promote = (
        candidate["win_rate"] > baseline["win_rate"]
        or (
            candidate["win_rate"] == baseline["win_rate"]
            and candidate["mean_model_score_diff"] >= baseline["mean_model_score_diff"]
        )
    )
    selected = "navigation_annealed" if promote else "score_terminal_only"
    output = {
        "schema_version": "blackout.phase3_agent14.v1",
        "initial_checkpoint": str(args.initial_checkpoint.relative_to(ROOT)),
        "train_seed": train_seed,
        "dev_seeds": dev_seeds,
        "same_environment_steps": True,
        "environment_steps_per_arm": args.steps,
        "ppo_config": asdict(ppo_config),
        "device": asdict(device),
        "arms": arm_results,
        "decision": {
            "promote_navigation_shaping": promote,
            "selected_arm": selected,
            "selected_checkpoint": arm_results[selected]["checkpoint"],
        },
    }
    output_path = ROOT / "logs/phase3_agent14_navigation.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"output": str(output_path), "decision": output["decision"]}, indent=2))


if __name__ == "__main__":
    main()
