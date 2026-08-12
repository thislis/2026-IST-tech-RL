#!/usr/bin/env python3
"""Train parameter-sharing IPPO against scripted agents until a held-out target."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from blackout_rl import (  # noqa: E402
    ContractBlackOutEnv,
    DeterministicCheckpointPolicy,
    ExperimentIdentity,
    FrozenScriptedOpponent,
    PPODiagnosticLogger,
    PPOConfig,
    ParallelRolloutCollector,
    RewardMode,
    ScriptedTeamController,
    SubmissionPolicy,
    TeamTrainingReward,
    TrainingRewardConfig,
    checkpoint_payload,
    concatenate_rollout_batches,
    load_checkpoint,
    online_imitation_update,
    ppo_update,
    save_checkpoint,
)
from blackout_rl.ippo_training import passed_win_rate, required_wins, select_training_device  # noqa: E402
from blackout_rl.logging_schema import SERIES_SCHEMA_VERSION, validate_series_log, write_json  # noqa: E402
from blackout_rl.policy import PolicyArtifact  # noqa: E402
from eval.evaluator import evaluate_episode, summarize_episodes  # noqa: E402
from eval.paired_series import parse_seeds  # noqa: E402


GAME_COMMIT = "d2220a7d01be88d413f551efd529f4758833be8b"
PYTHON_API_COMMIT = "6ba7d9993cf1bdefe1ed480c8efbcabcb923f539"


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _evaluation_job(job: dict[str, object]) -> list[dict]:
    """Evaluate one held-out seed with the checkpoint on both physical sides."""

    torch.set_num_threads(1)
    build = Path(str(job["build"]))
    checkpoint = Path(str(job["checkpoint"]))
    opponent_config = Path(str(job["opponent_config"]))
    seed = int(job["seed"])
    model_artifact = PolicyArtifact.from_file(
        "win-70-vs-scripted-candidate", "ippo-checkpoint", checkpoint
    )
    opponent_artifact = PolicyArtifact.from_file(
        "scripted-battery-v1", "policy-config", opponent_config
    )
    env = ContractBlackOutEnv(
        env_path=str(build),
        no_graphics=False,
        time_scale=float(job["time_scale"]),
    )
    episodes: list[dict] = []
    try:
        for pair_index, model_team in enumerate((0, 1)):
            episodes.append(
                evaluate_episode(
                    env,
                    build=build,
                    game_commit=GAME_COMMIT,
                    python_api_commit=PYTHON_API_COMMIT,
                    seed=seed,
                    model_team=model_team,
                    model_policy=DeterministicCheckpointPolicy(
                        checkpoint,
                        team=model_team,
                        seed=int(job["model_policy_seed"]) + seed,
                        device="cpu",
                    ),
                    opponent_policy=ScriptedTeamController(
                        1 - model_team,
                        seed=int(job["opponent_policy_seed"]) + seed,
                        enable_special_items=False,
                    ),
                    model_artifact=model_artifact,
                    opponent_artifact=opponent_artifact,
                    pair_id=f"ippo-vs-scripted-seed-{seed}",
                    pair_index=pair_index,
                    max_steps=int(job["max_episode_steps"]),
                )
            )
    finally:
        env.close()
    return episodes


def evaluate_checkpoint(
    *,
    build: Path,
    checkpoint: Path,
    opponent_config: Path,
    seeds: list[int],
    workers: int,
    time_scale: float,
    max_episode_steps: int,
    global_step: int,
    output: Path,
) -> dict:
    """Run a side-swapped, held-out evaluation and persist the evidence log."""

    jobs = [
        {
            "build": str(build),
            "checkpoint": str(checkpoint),
            "opponent_config": str(opponent_config),
            "seed": seed,
            "model_policy_seed": 707001,
            "opponent_policy_seed": 707002,
            "time_scale": time_scale,
            "max_episode_steps": max_episode_steps,
        }
        for seed in seeds
    ]
    worker_count = min(workers, len(jobs))
    if worker_count == 1:
        pairs = [_evaluation_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            pairs = list(executor.map(_evaluation_job, jobs))
    episodes = [episode for pair in pairs for episode in pair]
    payload = {
        "schema_version": SERIES_SCHEMA_VERSION,
        "series_id": str(uuid.uuid4()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "series_config": {
            "seeds": seeds,
            "held_out": True,
            "side_swap": True,
            "opponent": "scripted-battery-v1",
            "model_policy_seed_base": 707001,
            "opponent_policy_seed_base": 707002,
            "time_scale": time_scale,
            "max_steps": max_episode_steps,
            "workers": worker_count,
            "training_global_step": global_step,
        },
        "episodes": episodes,
        "summary": summarize_episodes(episodes),
    }
    validate_series_log(payload)
    write_json(output, payload)
    return payload


def _policy_from_actor_critic(model: torch.nn.Module) -> SubmissionPolicy:
    config = dict(model.model_config)
    config.pop("n_actions", None)
    policy = SubmissionPolicy(**config)
    policy.actor_critic.load_state_dict(model.state_dict())
    return policy


def _save_training_state(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    seed: int,
    run_id: str,
    config: dict,
) -> None:
    policy = _policy_from_actor_critic(model)
    identity = ExperimentIdentity(
        run_id=run_id,
        config=config,
        seed=seed,
        git_sha=_git_sha(),
        opponent_id="scripted-battery-v1",
    )
    payload = checkpoint_payload(
        policy,
        global_step=global_step,
        training_seed=seed,
        optimizer=optimizer,
        source={
            "game_commit": GAME_COMMIT,
            "python_api_commit": PYTHON_API_COMMIT,
        },
        experiment=identity.checkpoint_metadata(),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(path, payload)


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, default=ROOT / "builds/BlackOut.app")
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        default=ROOT / "checkpoints/base_r13_bc_warm_start.pt",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints/win_70_vs_scripted.pt",
    )
    parser.add_argument(
        "--latest-checkpoint",
        type=Path,
        default=ROOT / "checkpoints/ippo_vs_scripted_latest.pt",
    )
    parser.add_argument(
        "--opponent-config",
        type=Path,
        default=ROOT / "configs/policies/scripted_battery_v1.json",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=7070)
    parser.add_argument("--train-seeds", type=parse_seeds, default=[7071, 7072])
    parser.add_argument(
        "--eval-seeds", type=parse_seeds, default=[7171, 7172, 7173, 7174, 7175]
    )
    parser.add_argument("--win-rate", type=float, default=0.70)
    parser.add_argument("--max-env-steps", type=int, default=500_000)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=25_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--score-delta-weight", type=float, default=1.0)
    parser.add_argument("--unity-shaping-weight", type=float, default=0.25)
    parser.add_argument("--teacher-epochs", type=int, default=1)
    parser.add_argument("--teacher-loss-coef", type=float, default=1.0)
    parser.add_argument("--time-scale", type=float, default=50.0)
    parser.add_argument("--max-episode-steps", type=int, default=22_000)
    parser.add_argument("--eval-workers", type=int, default=5)
    parser.add_argument(
        "--diagnostics", type=Path, default=ROOT / "logs/ippo_vs_scripted_ppo.jsonl"
    )
    parser.add_argument(
        "--training-log", type=Path, default=ROOT / "logs/ippo_vs_scripted_training.jsonl"
    )
    parser.add_argument(
        "--evaluation-log", type=Path, default=ROOT / "logs/ippo_vs_scripted_eval.json"
    )
    parser.add_argument("--skip-initial-eval", action="store_true")
    args = parser.parse_args()

    if len(args.train_seeds) != 2:
        parser.error("--train-seeds must contain exactly two seeds, one per physical side")
    if not args.eval_seeds:
        parser.error("--eval-seeds must not be empty")
    if args.eval_workers <= 0 or args.rollout_steps <= 0 or args.eval_every <= 0:
        parser.error("worker, rollout, and evaluation intervals must be positive")
    if args.rollout_steps % 2:
        parser.error("--rollout-steps must be even so both physical sides contribute equally")
    if args.max_env_steps <= 0:
        parser.error("--max-env-steps must be positive")
    if args.teacher_epochs <= 0 or args.teacher_loss_coef <= 0.0:
        parser.error("teacher epochs and loss coefficient must be positive")

    device = select_training_device(args.device)
    print(f"training_device={device.resolved} reason={device.reason}", flush=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.resolved == "mps":
        torch.mps.manual_seed(args.seed)

    initial_policy, initial_payload = load_checkpoint(
        args.initial_checkpoint, device=device.resolved
    )
    model = initial_policy.actor_critic
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    if "optimizer_state" in initial_payload:
        optimizer.load_state_dict(initial_payload["optimizer_state"])
    global_step = int(initial_payload["training"]["global_step"])
    update = 0
    ppo_config = PPOConfig(
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        clip_coef=0.2,
        value_clip_coef=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        target_kl=0.03,
    )
    run_config = {
        "algorithm": "parameter_sharing_IPPO_with_online_scripted_teacher",
        "initial_checkpoint": str(args.initial_checkpoint.resolve()),
        "model": dict(model.model_config),
        "device": asdict(device),
        "training": {
            "learning_rate": args.learning_rate,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "rollout_steps": args.rollout_steps,
            "ppo": asdict(ppo_config),
            "learning_teams": [0, 1],
            "train_seeds": args.train_seeds,
            "max_environment_steps": args.max_env_steps,
            "teacher_epochs": args.teacher_epochs,
            "teacher_loss_coef": args.teacher_loss_coef,
        },
        "reward": {
            "mode": RewardMode.COMBINED.value,
            "score_delta_weight": args.score_delta_weight,
            "unity_shaping_weight": args.unity_shaping_weight,
            "terminal_win_reward": 1.0,
        },
        "opponent": {
            "policy_id": "scripted-battery-v1",
            "special_items": False,
            "frozen": True,
        },
        "stopping": {
            "threshold": args.win_rate,
            "held_out_seeds": args.eval_seeds,
            "side_swapped": True,
            "required_wins": required_wins(2 * len(args.eval_seeds), args.win_rate),
        },
    }
    run_id = f"ippo-win70-vs-scripted-seed-{args.seed}"

    build = args.build.expanduser().resolve()
    args.latest_checkpoint = args.latest_checkpoint.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    evaluation_index = 0

    def save_and_evaluate() -> dict:
        nonlocal evaluation_index
        _save_training_state(
            args.latest_checkpoint,
            model=model,
            optimizer=optimizer,
            global_step=global_step,
            seed=args.seed,
            run_id=run_id,
            config=run_config,
        )
        evaluation_index += 1
        output = args.evaluation_log.with_name(
            f"{args.evaluation_log.stem}_step_{global_step}{args.evaluation_log.suffix}"
        )
        result = evaluate_checkpoint(
            build=build,
            checkpoint=args.latest_checkpoint,
            opponent_config=args.opponent_config.resolve(),
            seeds=args.eval_seeds,
            workers=args.eval_workers,
            time_scale=args.time_scale,
            max_episode_steps=args.max_episode_steps,
            global_step=global_step,
            output=output,
        )
        shutil.copy2(output, args.evaluation_log)
        summary = result["summary"]
        _append_jsonl(
            args.training_log,
            {
                "record_type": "evaluation",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "evaluation_index": evaluation_index,
                "global_step": global_step,
                "wins": summary["wins"],
                "draws": summary["draws"],
                "losses": summary["losses"],
                "win_rate": summary["win_rate"],
                "threshold": args.win_rate,
                "passed": passed_win_rate(
                    wins=summary["wins"],
                    episodes=summary["episodes"],
                    threshold=args.win_rate,
                ),
                "evidence": str(output),
            },
        )
        print(
            f"evaluation step={global_step} W-D-L="
            f"{summary['wins']}-{summary['draws']}-{summary['losses']} "
            f"win_rate={summary['win_rate']:.3f}",
            flush=True,
        )
        return result

    if not args.skip_initial_eval:
        initial_eval = save_and_evaluate()
        summary = initial_eval["summary"]
        if passed_win_rate(
            wins=summary["wins"], episodes=summary["episodes"], threshold=args.win_rate
        ):
            shutil.copy2(args.latest_checkpoint, args.checkpoint)
            print(f"target already met; saved {args.checkpoint}", flush=True)
            return 0

    reward_config = TrainingRewardConfig(
        mode=RewardMode.COMBINED,
        terminal_win_reward=1.0,
        score_delta_weight=args.score_delta_weight,
        unity_shaping_weight=args.unity_shaping_weight,
    )
    envs: list[ContractBlackOutEnv] = []
    collectors: list[ParallelRolloutCollector] = []
    try:
        for learning_team in (0, 1):
            env = ContractBlackOutEnv(
                env_path=str(build),
                worker_id=learning_team,
                no_graphics=False,
                time_scale=args.time_scale,
            )
            envs.append(env)
            collectors.append(
                ParallelRolloutCollector(
                    env,
                    model,
                    FrozenScriptedOpponent(
                        1 - learning_team,
                        seed=args.seed + 100 + learning_team,
                        enable_special_items=False,
                    ),
                    learning_team=learning_team,
                    device=device.resolved,
                    reward_transform=TeamTrainingReward(learning_team, reward_config),
                    teacher=FrozenScriptedOpponent(
                        learning_team,
                        seed=args.seed + 200 + learning_team,
                        enable_special_items=False,
                    ),
                )
            )

        diagnostic_logger = PPODiagnosticLogger(args.diagnostics)
        next_eval = ((global_step // args.eval_every) + 1) * args.eval_every
        started = time.perf_counter()
        initialized_collectors: set[int] = set()
        while global_step < args.max_env_steps:
            remaining = args.max_env_steps - global_step
            steps = min(args.rollout_steps, remaining)
            if steps % 2:
                steps -= 1
            if steps <= 0:
                break
            per_side_steps = steps // 2
            side_batches = []
            for collector_index, collector in enumerate(collectors):
                seed = (
                    args.train_seeds[collector_index]
                    if collector_index not in initialized_collectors
                    else None
                )
                buffer = collector.collect(per_side_steps, seed=seed)
                initialized_collectors.add(collector_index)
                side_batches.append(
                    buffer.as_batch(gamma=args.gamma, gae_lambda=args.gae_lambda)
                )
            batch = concatenate_rollout_batches(side_batches)
            diagnostics = ppo_update(model, optimizer, batch, config=ppo_config)
            imitation = online_imitation_update(
                model,
                optimizer,
                batch,
                epochs=args.teacher_epochs,
                minibatch_size=args.minibatch_size,
                loss_coef=args.teacher_loss_coef,
                max_grad_norm=ppo_config.max_grad_norm,
            )
            global_step += steps
            update += 1
            diagnostic_logger.log(
                diagnostics,
                update=update,
                global_step=global_step,
                config=ppo_config,
            )
            _append_jsonl(
                args.training_log,
                {
                    "record_type": "update",
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "update": update,
                    "global_step": global_step,
                    "learning_teams": [0, 1],
                    "episodes_completed": [
                        collector.episodes_completed for collector in collectors
                    ],
                    "wall_seconds": round(time.perf_counter() - started, 3),
                    "metrics": diagnostics.to_dict(),
                    "online_imitation": asdict(imitation),
                },
            )
            print(
                f"update={update} step={global_step} teams=0+1 "
                f"episodes={[collector.episodes_completed for collector in collectors]} "
                f"policy_loss={diagnostics.policy_loss:.4f} "
                f"value_loss={diagnostics.value_loss:.4f} "
                f"entropy={diagnostics.entropy:.4f}",
                f" teacher_loss={imitation.loss:.4f}"
                f" teacher_accuracy={imitation.accuracy:.3f}",
                flush=True,
            )

            if global_step >= next_eval or global_step >= args.max_env_steps:
                result = save_and_evaluate()
                summary = result["summary"]
                if passed_win_rate(
                    wins=summary["wins"],
                    episodes=summary["episodes"],
                    threshold=args.win_rate,
                ):
                    shutil.copy2(args.latest_checkpoint, args.checkpoint)
                    print(
                        f"target met at step={global_step}; saved {args.checkpoint}", flush=True
                    )
                    return 0
                next_eval += args.eval_every
    finally:
        for env in envs:
            env.close()

    print(
        f"target not met by max_env_steps={args.max_env_steps}; "
        f"latest recovery checkpoint is {args.latest_checkpoint}",
        flush=True,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
