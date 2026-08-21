#!/usr/bin/env python3
"""Train MAPPO against the immutable win_70_vs_scripted checkpoint.

The actor remains decentralized and submission-compatible.  The centralized
critic and optimizer are stored as extra training-only checkpoint fields so a
run can resume without changing the ordinary project checkpoint loader.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
import uuid
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from blackout_rl import (  # noqa: E402
    ContractBlackOutEnv,
    PPOConfig,
    RewardMode,
    SubmissionPolicy,
    TeamTrainingReward,
    TrainingRewardConfig,
    checkpoint_payload,
    load_checkpoint,
    save_checkpoint,
)
from blackout_rl.evaluation_protocol import SeedSplits, paired_seed_bootstrap_ci  # noqa: E402
from blackout_rl.ippo_training import select_training_device  # noqa: E402
from blackout_rl.logging_schema import (  # noqa: E402
    SERIES_SCHEMA_VERSION,
    sha256_file,
    validate_series_log,
    write_json,
)
from blackout_rl.mappo import (  # noqa: E402
    MAPPOActorCritic,
    MAPPOParallelRolloutCollector,
    initialize_mappo_from_checkpoint,
    initialize_mappo_from_ippo,
    mappo_update,
)
from blackout_rl.policy import DeterministicCheckpointPolicy, PolicyArtifact  # noqa: E402
from blackout_rl.self_play import FrozenCheckpointOpponent, snapshot_from_checkpoint  # noqa: E402
from eval.evaluator import evaluate_episode, summarize_episodes  # noqa: E402
from scripts.train_ippo_vs_scripted import GAME_COMMIT, PYTHON_API_COMMIT  # noqa: E402


SCHEMA_VERSION = "blackout.mappo_vs_win70.v1"


def parse_seeds(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds or any(seed < 0 for seed in seeds) or len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("seeds must be unique non-negative comma-separated integers")
    return seeds


def required_target_wins(episodes: int, target_win_rate: float) -> int:
    if episodes <= 0 or not 0.0 <= target_win_rate <= 1.0:
        raise ValueError("invalid target win-rate inputs")
    return math.ceil(episodes * target_win_rate)


def target_reached(summary: Mapping[str, Any], target_win_rate: float) -> bool:
    episodes = int(summary["episodes"])
    return int(summary["wins"]) >= required_target_wins(episodes, target_win_rate)


def promotion_evaluation_eligible(
    evaluation_seeds: Sequence[int], committed_dev_seeds: Sequence[int]
) -> bool:
    return (
        len(evaluation_seeds) == len(committed_dev_seeds)
        and set(evaluation_seeds) == set(committed_dev_seeds)
    )


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _actor_policy(model: MAPPOActorCritic) -> SubmissionPolicy:
    config = dict(model.actor_model.model_config)
    config.pop("n_actions", None)
    policy = SubmissionPolicy(**config)
    policy.actor_critic.load_state_dict(_cpu_tree(model.actor_model.state_dict()))
    return policy


def save_mappo_training_checkpoint(
    path: Path,
    *,
    model: MAPPOActorCritic,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    update: int,
    training_seed: int,
    run_id: str,
    opponent_checkpoint: Path,
    initial_checkpoint: Path,
    config: Mapping[str, Any],
    best_evaluation: Mapping[str, Any] | None,
) -> None:
    policy = _actor_policy(model)
    payload = checkpoint_payload(
        policy,
        global_step=global_step,
        training_seed=training_seed,
        source={
            "phase": "mappo-vs-win70",
            "algorithm": "MAPPO_CTDE",
            "initial_checkpoint": str(initial_checkpoint.resolve()),
            "opponent_checkpoint": str(opponent_checkpoint.resolve()),
            "opponent_sha256": sha256_file(opponent_checkpoint),
        },
    )
    payload["mappo_state"] = _cpu_tree(model.state_dict())
    payload["mappo_optimizer_state"] = _cpu_tree(optimizer.state_dict())
    payload["mappo_training"] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "global_step": global_step,
        "update": update,
        "config": dict(config),
        "best_evaluation": dict(best_evaluation) if best_evaluation is not None else None,
        "torch_rng_state": torch.get_rng_state(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_checkpoint(temporary, payload)
    temporary.replace(path)


def load_mappo_training_checkpoint(
    path: Path, *, device: str
) -> tuple[MAPPOActorCritic, torch.optim.Optimizer, dict[str, Any]]:
    policy, payload = load_checkpoint(path, device=device)
    training = payload.get("mappo_training")
    if not isinstance(training, dict) or training.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("resume checkpoint has no compatible MAPPO training state")
    if "mappo_state" not in payload or "mappo_optimizer_state" not in payload:
        raise ValueError("resume checkpoint is missing critic or optimizer state")
    model = initialize_mappo_from_ippo(policy.actor_critic).to(device)
    model.load_state_dict(payload["mappo_state"], strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["config"]["learning_rate"]))
    optimizer.load_state_dict(payload["mappo_optimizer_state"])
    rng_state = training.get("torch_rng_state")
    if isinstance(rng_state, torch.Tensor):
        torch.set_rng_state(rng_state.cpu())
    return model, optimizer, payload


def _evaluation_job(job: Mapping[str, Any]) -> list[dict[str, Any]]:
    torch.set_num_threads(1)
    build = Path(str(job["build"]))
    candidate = Path(str(job["candidate"]))
    opponent = Path(str(job["opponent"]))
    seed = int(job["seed"])
    candidate_artifact = PolicyArtifact.from_file(
        "mappo-vs-win70-candidate", "mappo-decentralized-actor", candidate
    )
    opponent_artifact = PolicyArtifact.from_file(
        "win-70-vs-scripted", "frozen-hybrid-checkpoint", opponent
    )
    env = ContractBlackOutEnv(
        env_path=str(build),
        no_graphics=False,
        time_scale=float(job["time_scale"]),
    )
    episodes: list[dict[str, Any]] = []
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
                        candidate,
                        team=model_team,
                        seed=int(job["candidate_policy_seed"]) + seed,
                        device="cpu",
                    ),
                    opponent_policy=DeterministicCheckpointPolicy(
                        opponent,
                        team=1 - model_team,
                        seed=int(job["opponent_policy_seed"]) + seed,
                        device="cpu",
                    ),
                    model_artifact=candidate_artifact,
                    opponent_artifact=opponent_artifact,
                    pair_id=f"mappo-vs-win70-seed-{seed}",
                    pair_index=pair_index,
                    max_steps=int(job["max_episode_steps"]),
                )
            )
    finally:
        env.close()
    return episodes


def evaluate_candidate(
    *,
    build: Path,
    candidate: Path,
    opponent: Path,
    seeds: Sequence[int],
    workers: int,
    time_scale: float,
    max_episode_steps: int,
    global_step: int,
    output: Path,
) -> dict[str, Any]:
    jobs = [
        {
            "build": str(build),
            "candidate": str(candidate),
            "opponent": str(opponent),
            "seed": seed,
            "candidate_policy_seed": 850001,
            "opponent_policy_seed": 850002,
            "time_scale": time_scale,
            "max_episode_steps": max_episode_steps,
        }
        for seed in seeds
    ]
    worker_count = min(max(workers, 1), len(jobs))
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
            "seeds": list(seeds),
            "split": "dev",
            "held_out": True,
            "side_swap": True,
            "opponent": str(opponent.resolve()),
            "opponent_sha256": sha256_file(opponent),
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


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(record), ensure_ascii=False, allow_nan=False) + "\n")


def _better(summary: Mapping[str, Any], best: Mapping[str, Any] | None) -> bool:
    if best is None:
        return True
    return (
        float(summary["win_rate"]),
        float(summary["mean_model_score_diff"]),
    ) > (
        float(best["win_rate"]),
        float(best["mean_model_score_diff"]),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a decentralized MAPPO actor against frozen win_70_vs_scripted.pt"
    )
    parser.add_argument("--build", type=Path, default=ROOT / "builds/BlackOut.app")
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        default=ROOT / "checkpoints/phase3_selfplay_gen_08.pt",
    )
    parser.add_argument(
        "--opponent-checkpoint",
        type=Path,
        default=ROOT / "checkpoints/win_70_vs_scripted.pt",
    )
    parser.add_argument(
        "--latest-checkpoint",
        type=Path,
        default=ROOT / "checkpoints/mappo_vs_win70_latest.pt",
    )
    parser.add_argument(
        "--best-checkpoint",
        type=Path,
        default=ROOT / "checkpoints/mappo_vs_win70_best.pt",
    )
    parser.add_argument(
        "--target-checkpoint",
        type=Path,
        default=ROOT / "checkpoints/mappo_win_85_vs_win70.pt",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-latest", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=8500)
    parser.add_argument("--train-seeds", type=parse_seeds)
    parser.add_argument("--eval-seeds", type=parse_seeds)
    parser.add_argument("--target-win-rate", type=float, default=0.85)
    parser.add_argument("--max-env-steps", type=int, default=2_000_000)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--eval-every", type=int, default=25_000)
    parser.add_argument("--save-every", type=int, default=5_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument(
        "--reward-mode",
        choices=tuple(mode.value for mode in RewardMode),
        default=RewardMode.COMBINED.value,
    )
    parser.add_argument("--score-delta-weight", type=float, default=1.0)
    parser.add_argument("--unity-shaping-weight", type=float, default=0.05)
    parser.add_argument("--terminal-win-reward", type=float, default=1.0)
    parser.add_argument("--time-scale", type=float, default=50.0)
    parser.add_argument("--eval-time-scale", type=float, default=100.0)
    parser.add_argument("--max-episode-steps", type=int, default=22_000)
    parser.add_argument("--eval-workers", type=int, default=5)
    parser.add_argument("--skip-initial-eval", action="store_true")
    parser.add_argument(
        "--log-dir", type=Path, default=ROOT / "logs/mappo_vs_win70"
    )
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.resume is not None and args.resume_latest:
        parser.error("--resume and --resume-latest are mutually exclusive")
    if not 0.0 <= args.target_win_rate <= 1.0:
        parser.error("--target-win-rate must be in [0,1]")
    positive = {
        "max-env-steps": args.max_env_steps,
        "rollout-steps": args.rollout_steps,
        "eval-every": args.eval_every,
        "save-every": args.save_every,
        "learning-rate": args.learning_rate,
        "gamma": args.gamma,
        "gae-lambda": args.gae_lambda,
        "update-epochs": args.update_epochs,
        "minibatch-size": args.minibatch_size,
        "target-kl": args.target_kl,
        "time-scale": args.time_scale,
        "eval-time-scale": args.eval_time_scale,
        "max-episode-steps": args.max_episode_steps,
        "eval-workers": args.eval_workers,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        parser.error(f"these values must be positive: {', '.join(invalid)}")
    if args.max_env_steps < args.rollout_steps:
        parser.error("--max-env-steps must cover at least one rollout")
    if not 0.0 < args.gamma <= 1.0 or not 0.0 <= args.gae_lambda <= 1.0:
        parser.error("gamma/GAE lambda are outside valid ranges")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    _validate_args(args, parser)
    for path in (args.build, args.initial_checkpoint, args.opponent_checkpoint):
        if not path.exists():
            parser.error(f"required artifact does not exist: {path}")

    splits = SeedSplits.from_dict(
        json.loads((ROOT / "configs/seed_splits_v1.json").read_text())
    )
    committed_train = tuple(splits.seeds_for("train", purpose="training"))
    committed_dev = tuple(splits.seeds_for("dev", purpose="model_selection"))
    train_seeds = args.train_seeds or list(committed_train)
    eval_seeds = args.eval_seeds or list(committed_dev)
    if not set(train_seeds) <= set(committed_train):
        parser.error("--train-seeds must be a subset of the committed train split")
    if not set(eval_seeds) <= set(committed_dev):
        parser.error("--eval-seeds must be a subset of the committed dev split")
    if set(train_seeds) & set(eval_seeds):
        parser.error("train and evaluation seeds must be disjoint")
    promotion_eligible_eval = promotion_evaluation_eligible(eval_seeds, committed_dev)
    target_wins = required_target_wins(len(eval_seeds) * 2, args.target_win_rate)

    device_selection = select_training_device(args.device)
    device = device_selection.resolved
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device == "mps":
        torch.mps.manual_seed(args.seed)

    resume_path = args.resume
    if args.resume_latest:
        resume_path = args.latest_checkpoint
    if resume_path is not None:
        if not resume_path.is_file():
            parser.error(f"resume checkpoint does not exist: {resume_path}")
        model, optimizer, resume_payload = load_mappo_training_checkpoint(
            resume_path, device=device
        )
        resume_training = resume_payload["mappo_training"]
        global_step = int(resume_training["global_step"])
        update = int(resume_training["update"])
        run_id = str(resume_training["run_id"])
        best_evaluation = resume_training.get("best_evaluation")
        expected_sha = resume_payload["source"].get("opponent_sha256")
        if expected_sha != sha256_file(args.opponent_checkpoint):
            parser.error("resume opponent SHA differs from --opponent-checkpoint")
        print(f"resumed={resume_path} global_step={global_step} update={update}", flush=True)
    else:
        model, _ = initialize_mappo_from_checkpoint(
            str(args.initial_checkpoint), device=device
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        global_step = 0
        update = 0
        run_id = str(uuid.uuid4())
        best_evaluation = None

    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = args.learning_rate

    reward_config = TrainingRewardConfig(
        mode=RewardMode(args.reward_mode),
        terminal_win_reward=args.terminal_win_reward,
        score_delta_weight=args.score_delta_weight,
        unity_shaping_weight=args.unity_shaping_weight,
    )
    reward_config.validate()
    ppo_config = PPOConfig(
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        entropy_coef=args.entropy_coef,
        target_kl=args.target_kl,
    )
    ppo_config.validate()
    run_config = {
        "algorithm": "MAPPO_CTDE",
        "initial_checkpoint": str(args.initial_checkpoint.resolve()),
        "opponent_checkpoint": str(args.opponent_checkpoint.resolve()),
        "opponent_sha256": sha256_file(args.opponent_checkpoint),
        "target_win_rate": args.target_win_rate,
        "target_wins": target_wins,
        "evaluation_episodes": len(eval_seeds) * 2,
        "promotion_eligible_evaluation": promotion_eligible_eval,
        "train_seeds": train_seeds,
        "eval_seeds": eval_seeds,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "rollout_steps": args.rollout_steps,
        "reward": {
            **asdict(reward_config),
            "mode": reward_config.mode.value,
        },
        "ppo": asdict(ppo_config),
        "side_sampling": "alternating_by_update",
        "device": asdict(device_selection),
    }
    args.log_dir.mkdir(parents=True, exist_ok=True)
    training_log = args.log_dir / "training.jsonl"
    summary_log = args.log_dir / "run_summary.json"
    started = time.monotonic()
    opponent_snapshot = snapshot_from_checkpoint(
        args.opponent_checkpoint, generation=0, global_step=0
    )

    def save(path: Path) -> None:
        save_mappo_training_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            global_step=global_step,
            update=update,
            training_seed=args.seed,
            run_id=run_id,
            opponent_checkpoint=args.opponent_checkpoint,
            initial_checkpoint=args.initial_checkpoint,
            config=run_config,
            best_evaluation=best_evaluation,
        )

    def evaluate() -> tuple[dict[str, Any], bool]:
        nonlocal best_evaluation
        save(args.latest_checkpoint)
        evaluation_path = args.log_dir / f"eval_step_{global_step}.json"
        result = evaluate_candidate(
            build=args.build,
            candidate=args.latest_checkpoint,
            opponent=args.opponent_checkpoint,
            seeds=eval_seeds,
            workers=args.eval_workers,
            time_scale=args.eval_time_scale,
            max_episode_steps=args.max_episode_steps,
            global_step=global_step,
            output=evaluation_path,
        )
        summary = result["summary"]
        summary_with_ci = {
            **summary,
            "global_step": global_step,
            "evaluation_log": str(evaluation_path.resolve()),
            "win_rate_ci": paired_seed_bootstrap_ci(
                result["episodes"], metric="win_rate", resamples=10_000, seed=args.seed
            ).to_dict(),
            "score_diff_ci": paired_seed_bootstrap_ci(
                result["episodes"], metric="score_diff", resamples=10_000, seed=args.seed + 1
            ).to_dict(),
        }
        if _better(summary_with_ci, best_evaluation):
            best_evaluation = summary_with_ci
            save(args.best_checkpoint)
        save(args.latest_checkpoint)
        passed = promotion_eligible_eval and target_reached(
            summary, args.target_win_rate
        )
        write_json(
            summary_log,
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "global_step": global_step,
                "update": update,
                "target_win_rate": args.target_win_rate,
                "target_wins": target_wins,
                "best_evaluation": best_evaluation,
                "latest_evaluation": summary_with_ci,
                "target_reached": passed,
                "elapsed_seconds": time.monotonic() - started,
                "config": run_config,
            },
        )
        print(
            f"evaluation step={global_step} wins={summary['wins']}/{summary['episodes']} "
            f"win_rate={summary['win_rate']:.3f} "
            f"score_diff={summary['mean_model_score_diff']:.2f} target={target_wins} "
            f"promotion_eligible={promotion_eligible_eval}",
            flush=True,
        )
        return summary_with_ci, passed

    try:
        if global_step == 0 and not args.skip_initial_eval:
            _, passed = evaluate()
            if passed:
                save(args.target_checkpoint)
                return 0

        last_save_bucket = global_step // args.save_every
        last_eval_bucket = global_step // args.eval_every
        while global_step < args.max_env_steps:
            learner_team = update % 2
            train_seed = train_seeds[update % len(train_seeds)]
            env = ContractBlackOutEnv(
                env_path=str(args.build), no_graphics=False, time_scale=args.time_scale
            )
            collector = MAPPOParallelRolloutCollector(
                env,
                model,
                FrozenCheckpointOpponent(
                    opponent_snapshot,
                    team=1 - learner_team,
                    device="cpu",
                ),
                learning_team=learner_team,
                device=device,
                reward_transform=TeamTrainingReward(learner_team, reward_config),
            )
            try:
                buffer = collector.collect(args.rollout_steps, seed=train_seed)
                batch = buffer.as_batch(gamma=args.gamma, gae_lambda=args.gae_lambda)
                diagnostics = mappo_update(
                    model, optimizer, batch, config=ppo_config
                )
            finally:
                env.close()
            update += 1
            global_step += args.rollout_steps
            record = {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "update": update,
                "global_step": global_step,
                "learner_team": learner_team,
                "train_seed": train_seed,
                "opponent_sha256": opponent_snapshot.checkpoint_sha256,
                "episodes_completed": collector.episodes_completed,
                "mean_agent_reward": float(batch.return_.mean()),
                "metrics": diagnostics.to_dict(),
            }
            _append_jsonl(training_log, record)
            print(
                f"update={update} step={global_step} side={learner_team} "
                f"reward={record['mean_agent_reward']:.4f} "
                f"entropy={diagnostics.entropy:.4f} kl={diagnostics.approximate_kl:.6f}",
                flush=True,
            )

            save_bucket = global_step // args.save_every
            if save_bucket > last_save_bucket:
                save(args.latest_checkpoint)
                last_save_bucket = save_bucket
            eval_bucket = global_step // args.eval_every
            if eval_bucket > last_eval_bucket or global_step >= args.max_env_steps:
                _, passed = evaluate()
                last_eval_bucket = eval_bucket
                if passed:
                    save(args.target_checkpoint)
                    print(f"target_checkpoint={args.target_checkpoint}", flush=True)
                    return 0
    except KeyboardInterrupt:
        save(args.latest_checkpoint)
        print(f"interrupted; resumable checkpoint saved to {args.latest_checkpoint}", flush=True)
        return 130

    save(args.latest_checkpoint)
    print(
        f"target not reached by max_env_steps={args.max_env_steps}; "
        f"resume with --resume-latest",
        flush=True,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
