#!/usr/bin/env python3
"""Train fail-closed planner-residual MAPPO through opponent curriculum."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
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
    FrozenScriptedOpponent,
    PPOConfig,
    PPODiagnostics,
    RewardMode,
    TeamTrainingReward,
    TeacherReplayBuffer,
    TrainingRewardConfig,
    checkpoint_payload,
    load_checkpoint,
    save_checkpoint,
    teacher_replay_update,
)
from blackout_rl.evaluation_protocol import SeedSplits, paired_seed_bootstrap_ci  # noqa: E402
from blackout_rl.ippo_training import OnlineImitationMetrics, select_training_device  # noqa: E402
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
    initialize_planner_residual_head,
    mappo_update,
)
from blackout_rl.mappo_curriculum import (  # noqa: E402
    DEFAULT_MAPPO_CURRICULUM,
    EpisodeOpponentMixture,
    MAPPOCurriculumController,
    MAPPOCurriculumStage,
)
from blackout_rl.policy import (  # noqa: E402
    DeterministicCheckpointPolicy,
    PolicyArtifact,
)
from blackout_rl.self_play import FrozenCheckpointOpponent, snapshot_from_checkpoint  # noqa: E402
from blackout_rl.team_state import Role  # noqa: E402
from eval.evaluator import evaluate_episode, summarize_episodes  # noqa: E402
from scripts.train_ippo_vs_scripted import GAME_COMMIT, PYTHON_API_COMMIT  # noqa: E402
from scripts.train_mappo_vs_win70 import (  # noqa: E402
    CyclicSeedStream,
    _actor_policy,
    _better,
    _cpu_tree,
    evaluate_candidate,
    parse_seeds,
    promotion_evaluation_eligible,
    required_target_wins,
    target_reached,
)


SCHEMA_VERSION = "blackout.mappo_teacher_curriculum.v4"
PLANNER_ROLES = (Role.WORKER, Role.WORKER, Role.WORKER, Role.GUARD, Role.GUARD)
BASE_SCRIPTED_CONFIG = ROOT / "configs/policies/scripted_battery_v1.json"
WEAK_WIN70_CONFIG = ROOT / "configs/policies/weak_win70_v1.json"


def scaled_curriculum(scale: float, rollout_steps: int) -> tuple[MAPPOCurriculumStage, ...]:
    if scale <= 0.0 or rollout_steps <= 0:
        raise ValueError("curriculum scale and rollout steps must be positive")
    stages: list[MAPPOCurriculumStage] = []
    for stage in DEFAULT_MAPPO_CURRICULUM:
        minimum = max(
            rollout_steps if stage.minimum_steps else 0,
            int(math.ceil(stage.minimum_steps * scale / rollout_steps)) * rollout_steps,
        )
        maximum = max(
            minimum or rollout_steps,
            int(math.ceil(stage.maximum_steps * scale / rollout_steps)) * rollout_steps,
        )
        stages.append(
            MAPPOCurriculumStage(
                **{
                    **asdict(stage),
                    "minimum_steps": minimum,
                    "maximum_steps": maximum,
                }
            )
        )
    return tuple(stages)


def bc_coefficient(stage: MAPPOCurriculumStage, stage_steps: int) -> float:
    return stage.bc_coefficient(stage_steps)


def teacher_forcing_probability(
    stage: MAPPOCurriculumStage, stage_steps: int
) -> float:
    return stage.teacher_forcing_probability(stage_steps)


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(record), ensure_ascii=False, allow_nan=False) + "\n")


def _scripted_evaluation_job(job: Mapping[str, Any]) -> list[dict[str, Any]]:
    torch.set_num_threads(1)
    build = Path(str(job["build"]))
    candidate = Path(str(job["candidate"]))
    seed = int(job["seed"])
    opponent_id = str(job["opponent_id"])
    model_artifact = PolicyArtifact.from_file(
        "mappo-teacher-curriculum-v3", "mappo-decentralized-actor", candidate
    )
    opponent_config = (
        BASE_SCRIPTED_CONFIG if opponent_id == "base_scripted" else WEAK_WIN70_CONFIG
    )
    opponent_artifact = PolicyArtifact.from_file(
        opponent_id, "scripted-policy-config", opponent_config
    )
    env = ContractBlackOutEnv(
        env_path=str(build),
        background=True,
        no_graphics=False,
        time_scale=float(job["time_scale"]),
    )
    episodes: list[dict[str, Any]] = []
    try:
        for pair_index, model_team in enumerate((0, 1)):
            opponent_team = 1 - model_team
            if opponent_id == "base_scripted":
                opponent = FrozenScriptedOpponent(
                    opponent_team,
                    seed=880_000 + seed,
                    enable_special_items=False,
                )
            elif opponent_id == "weak_win70":
                opponent = FrozenScriptedOpponent(
                    opponent_team,
                    seed=880_000 + seed,
                    enable_special_items=False,
                    roles=PLANNER_ROLES,
                    chase_radius_cells=12,
                    policy_id="weak-win70-v1",
                )
            else:
                raise ValueError(f"unsupported scripted evaluation opponent: {opponent_id}")
            episodes.append(
                evaluate_episode(
                    env,
                    build=build,
                    game_commit=GAME_COMMIT,
                    python_api_commit=PYTHON_API_COMMIT,
                    seed=seed,
                    model_team=model_team,
                    model_policy=DeterministicCheckpointPolicy(
                        candidate, team=model_team, seed=870_000 + seed, device="cpu"
                    ),
                    opponent_policy=opponent,
                    model_artifact=model_artifact,
                    opponent_artifact=opponent_artifact,
                    pair_id=f"mappo-v3-vs-{opponent_id}-seed-{seed}",
                    pair_index=pair_index,
                    max_steps=int(job["max_episode_steps"]),
                )
            )
    finally:
        env.close()
    return episodes


def evaluate_stage_candidate(
    *,
    build: Path,
    candidate: Path,
    full_opponent: Path,
    opponent_id: str,
    seeds: Sequence[int],
    workers: int,
    time_scale: float,
    max_episode_steps: int,
    global_step: int,
    output: Path,
) -> dict[str, Any]:
    if opponent_id == "full_win70":
        return evaluate_candidate(
            build=build,
            candidate=candidate,
            opponent=full_opponent,
            seeds=seeds,
            workers=workers,
            time_scale=time_scale,
            max_episode_steps=max_episode_steps,
            global_step=global_step,
            output=output,
        )
    jobs = [
        {
            "build": str(build),
            "candidate": str(candidate),
            "seed": seed,
            "opponent_id": opponent_id,
            "time_scale": time_scale,
            "max_episode_steps": max_episode_steps,
        }
        for seed in seeds
    ]
    worker_count = min(max(workers, 1), len(jobs))
    if worker_count == 1:
        pairs = [_scripted_evaluation_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            pairs = list(executor.map(_scripted_evaluation_job, jobs))
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
            "opponent": opponent_id,
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


def save_v4_checkpoint(
    path: Path,
    *,
    model: MAPPOActorCritic,
    optimizer: torch.optim.Optimizer,
    replay: TeacherReplayBuffer,
    global_step: int,
    update: int,
    run_id: str,
    seed: int,
    initial_checkpoint: Path,
    opponent_checkpoint: Path,
    config: Mapping[str, Any],
    runtime_state: Mapping[str, Any],
    best_evaluation: Mapping[str, Any] | None,
    planner_residual: bool = False,
) -> None:
    payload = checkpoint_payload(
        _actor_policy(model),
        global_step=global_step,
        training_seed=seed,
        source={
            "phase": "mappo-planner-residual-curriculum-v4",
            "algorithm": "MAPPO_CTDE_planner_residual",
            "initial_checkpoint": str(initial_checkpoint.resolve()),
            "initial_checkpoint_sha256": sha256_file(initial_checkpoint),
            "opponent_checkpoint": str(opponent_checkpoint.resolve()),
            "opponent_sha256": sha256_file(opponent_checkpoint),
        },
    )
    if planner_residual:
        payload["inference_guardrail"] = {
            "version": "scripted_counter_v1",
            "mode": "planner_residual_v1",
            "roles": [role.value for role in PLANNER_ROLES],
            "chase_radius_cells": 48,
            "fallback_action_index": 0,
            "override_action_indices": list(range(1, 9)),
            "reason": "retain audited planner while PPO learns explicit directional overrides",
        }
    payload["mappo_state"] = _cpu_tree(model.state_dict())
    payload["mappo_optimizer_state"] = _cpu_tree(optimizer.state_dict())
    payload["teacher_replay_state"] = replay.state_dict()
    payload["mappo_v4_training"] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "global_step": global_step,
        "update": update,
        "config": dict(config),
        "runtime_state": dict(runtime_state),
        "best_evaluation": dict(best_evaluation) if best_evaluation is not None else None,
        "torch_rng_state": torch.get_rng_state(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_checkpoint(temporary, payload)
    temporary.replace(path)


def load_v4_checkpoint(
    path: Path,
    *,
    device: str,
    replay_capacity: int,
) -> tuple[MAPPOActorCritic, torch.optim.Optimizer, TeacherReplayBuffer, dict[str, Any]]:
    policy, payload = load_checkpoint(path, device=device)
    training = payload.get("mappo_v4_training")
    if not isinstance(training, dict) or training.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("resume checkpoint has no compatible MAPPO v4 training state")
    model = initialize_mappo_from_ippo(policy.actor_critic).to(device)
    model.load_state_dict(payload["mappo_state"], strict=True)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(training["config"]["learning_rate"])
    )
    optimizer.load_state_dict(payload["mappo_optimizer_state"])
    replay = TeacherReplayBuffer(replay_capacity, seed=int(training["config"]["seed"]) + 1)
    replay.load_state_dict(payload["teacher_replay_state"])
    rng_state = training.get("torch_rng_state")
    if isinstance(rng_state, torch.Tensor):
        torch.set_rng_state(rng_state.cpu())
    return model, optimizer, replay, payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed planner-residual MAPPO against win_70_vs_scripted.pt"
    )
    parser.add_argument("--build", type=Path, default=ROOT / "builds/BlackOut.app")
    parser.add_argument("--initial-checkpoint", type=Path, default=ROOT / "checkpoints/win_70_vs_scripted.pt")
    parser.add_argument("--opponent-checkpoint", type=Path, default=ROOT / "checkpoints/win_70_vs_scripted.pt")
    parser.add_argument("--latest-checkpoint", type=Path, default=ROOT / "checkpoints/mappo_planner_residual_v4_latest.pt")
    parser.add_argument("--best-checkpoint", type=Path, default=ROOT / "checkpoints/mappo_planner_residual_v4_best.pt")
    parser.add_argument("--target-checkpoint", type=Path, default=ROOT / "checkpoints/mappo_win_85_vs_win70.pt")
    parser.add_argument("--snapshot-dir", type=Path, default=ROOT / "checkpoints/mappo_v4_snapshots")
    parser.add_argument("--log-dir", type=Path, default=ROOT / "logs/mappo_planner_residual_v4")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-latest", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=9500)
    parser.add_argument("--train-seeds", type=parse_seeds)
    parser.add_argument("--eval-seeds", type=parse_seeds)
    parser.add_argument("--target-win-rate", type=float, default=0.85)
    parser.add_argument("--max-env-steps", type=int, default=3_000_000)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--curriculum-scale", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=50_000)
    parser.add_argument("--save-every", type=int, default=25_000)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--teacher-replay-capacity", type=int, default=100_000)
    parser.add_argument("--teacher-minibatches", type=int, default=8)
    parser.add_argument("--teacher-minibatch-size", type=int, default=512)
    parser.add_argument("--teacher-class-balance-power", type=float, default=0.0)
    parser.add_argument("--planner-residual", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fallback-logit-bias", type=float, default=4.0)
    parser.add_argument("--initial-stage-min-win-rate", type=float, default=0.60)
    parser.add_argument("--score-delta-weight", type=float, default=1.0)
    parser.add_argument("--unity-shaping-weight", type=float, default=0.25)
    parser.add_argument("--terminal-win-reward", type=float, default=1.0)
    parser.add_argument("--time-scale", type=float, default=50.0)
    parser.add_argument("--eval-time-scale", type=float, default=100.0)
    parser.add_argument("--max-episode-steps", type=int, default=22_000)
    parser.add_argument("--eval-workers", type=int, default=5)
    parser.add_argument("--skip-initial-eval", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.resume is not None and args.resume_latest:
        parser.error("--resume and --resume-latest are mutually exclusive")
    positive = {
        "max-env-steps": args.max_env_steps,
        "rollout-steps": args.rollout_steps,
        "curriculum-scale": args.curriculum_scale,
        "eval-every": args.eval_every,
        "save-every": args.save_every,
        "learning-rate": args.learning_rate,
        "teacher-replay-capacity": args.teacher_replay_capacity,
        "teacher-minibatches": args.teacher_minibatches,
        "teacher-minibatch-size": args.teacher_minibatch_size,
        "fallback-logit-bias": args.fallback_logit_bias,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        parser.error(f"these values must be positive: {', '.join(invalid)}")
    if args.max_env_steps < args.rollout_steps:
        parser.error("--max-env-steps must cover at least one rollout")
    if not 0.0 <= args.target_win_rate <= 1.0:
        parser.error("--target-win-rate must be in [0,1]")
    if not 0.0 < args.gamma <= 1.0 or not 0.0 <= args.gae_lambda <= 1.0:
        parser.error("--gamma must be in (0,1] and --gae-lambda in [0,1]")
    if not 0.0 <= args.teacher_class_balance_power <= 1.0:
        parser.error("--teacher-class-balance-power must be in [0,1]")
    if not 0.0 <= args.initial_stage_min_win_rate <= 1.0:
        parser.error("--initial-stage-min-win-rate must be in [0,1]")
    if args.planner_residual and args.teacher_class_balance_power != 0.0:
        parser.error("planner residual labels use only fallback class 0; class balancing must be 0")


def _collector_counts(collector: MAPPOParallelRolloutCollector) -> dict[str, int]:
    return {
        name: int(getattr(collector, name))
        for name in (
            "environment_steps", "episodes_completed", "terminal_episodes",
            "truncated_episodes", "wins", "draws", "losses",
            "teacher_failures", "teacher_forced_steps", "teacher_labeled_steps",
            "residual_fallback_actions", "residual_override_actions",
        )
    }


def _restore_counts(collector: MAPPOParallelRolloutCollector, state: Mapping[str, Any]) -> None:
    for name in _collector_counts(collector):
        setattr(collector, name, int(state.get(name, 0)))


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    _validate_args(args, parser)
    for path in (
        args.build,
        args.initial_checkpoint,
        args.opponent_checkpoint,
        BASE_SCRIPTED_CONFIG,
        WEAK_WIN70_CONFIG,
    ):
        if not path.exists():
            parser.error(f"required artifact does not exist: {path}")

    splits = SeedSplits.from_dict(json.loads((ROOT / "configs/seed_splits_v1.json").read_text()))
    committed_train = tuple(splits.seeds_for("train", purpose="training"))
    committed_dev = tuple(splits.seeds_for("dev", purpose="model_selection"))
    train_seeds = args.train_seeds or list(committed_train)
    eval_seeds = args.eval_seeds or list(committed_dev)
    if not set(train_seeds) <= set(committed_train) or not set(eval_seeds) <= set(committed_dev):
        parser.error("train/eval seeds must come from their committed splits")
    promotion_eligible = promotion_evaluation_eligible(eval_seeds, committed_dev)
    target_wins = required_target_wins(len(eval_seeds) * 2, args.target_win_rate)
    stages = scaled_curriculum(args.curriculum_scale, args.rollout_steps)

    resume_path = args.latest_checkpoint if args.resume_latest else args.resume
    if resume_path is None:
        conflicts = [
            path for path in (
                args.latest_checkpoint, args.best_checkpoint, args.target_checkpoint,
                args.log_dir / "training.jsonl", args.log_dir / "run_summary.json",
            ) if path.exists()
        ]
        if conflicts:
            parser.error(
                "fresh v4 run would mix with existing artifacts; resume or choose new paths: "
                + ", ".join(str(path) for path in conflicts[:5])
            )

    device_selection = select_training_device(args.device)
    device = device_selection.resolved
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device == "mps":
        torch.mps.manual_seed(args.seed)

    saved_runtime: Mapping[str, Any] = {}
    saved_config: Mapping[str, Any] | None = None
    if resume_path is not None:
        if not resume_path.is_file():
            parser.error(f"resume checkpoint does not exist: {resume_path}")
        try:
            model, optimizer, replay, payload = load_v4_checkpoint(
                resume_path, device=device, replay_capacity=args.teacher_replay_capacity
            )
        except ValueError as exc:
            parser.error(str(exc))
        training = payload["mappo_v4_training"]
        global_step = int(training["global_step"])
        update = int(training["update"])
        run_id = str(training["run_id"])
        best_evaluation = training.get("best_evaluation")
        saved_runtime = training.get("runtime_state") or {}
        if payload["source"].get("opponent_sha256") != sha256_file(args.opponent_checkpoint):
            parser.error("resume opponent SHA differs")
        if payload["source"].get("initial_checkpoint_sha256") != sha256_file(args.initial_checkpoint):
            parser.error("resume initial checkpoint SHA differs")
        saved_config = training["config"]
        print(
            f"resumed={resume_path} global_step={global_step} update={update} "
            f"replay={len(replay)}",
            flush=True,
        )
    else:
        model, _ = initialize_mappo_from_checkpoint(str(args.initial_checkpoint), device=device)
        if args.planner_residual:
            initialize_planner_residual_head(
                model, fallback_logit_bias=args.fallback_logit_bias
            )
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        replay = TeacherReplayBuffer(args.teacher_replay_capacity, seed=args.seed + 1)
        global_step = 0
        update = 0
        run_id = str(uuid.uuid4())
        best_evaluation = None

    if args.planner_residual and global_step == 0 and args.skip_initial_eval:
        parser.error("fresh planner-residual training cannot skip the baseline preflight")

    curriculum_state = saved_runtime.get("curriculum", {})
    curriculum = MAPPOCurriculumController(
        stages,
        stage_index=int(curriculum_state.get("stage_index", 0)),
        stage_start_step=int(curriculum_state.get("stage_start_step", 0)),
    )
    seed_state = saved_runtime.get("seed_streams", {})
    seed_streams = {
        team: (
            CyclicSeedStream.from_state(seed_state[str(team)], expected_seeds=train_seeds)
            if str(team) in seed_state
            else CyclicSeedStream(tuple(train_seeds), cursor=0 if team == 0 else max(1, len(train_seeds) // 2))
        )
        for team in (0, 1)
    }
    cumulative_counts = {
        str(team): dict(saved_runtime.get("collectors", {}).get(str(team), {}))
        for team in (0, 1)
    }
    historical_checkpoint = saved_runtime.get("historical_checkpoint")
    pending_resume_mix_states = dict(saved_runtime.get("opponent_mixtures", {}))

    reward_config = TrainingRewardConfig(
        mode=RewardMode.COMBINED,
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
        "schema_version": SCHEMA_VERSION,
        "algorithm": "MAPPO_CTDE_planner_residual_curriculum",
        "seed": args.seed,
        "initial_checkpoint": str(args.initial_checkpoint.resolve()),
        "opponent_checkpoint": str(args.opponent_checkpoint.resolve()),
        "opponent_sha256": sha256_file(args.opponent_checkpoint),
        "target_win_rate": args.target_win_rate,
        "target_wins": target_wins,
        "max_env_steps": args.max_env_steps,
        "train_seeds": train_seeds,
        "eval_seeds": eval_seeds,
        "promotion_eligible_evaluation": promotion_eligible,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "rollout_steps": args.rollout_steps,
        "curriculum_scale": args.curriculum_scale,
        "curriculum": [asdict(stage) for stage in stages],
        "teacher_replay_capacity": args.teacher_replay_capacity,
        "teacher_minibatches": args.teacher_minibatches,
        "teacher_minibatch_size": args.teacher_minibatch_size,
        "teacher_class_balance_power": args.teacher_class_balance_power,
        "planner_residual": args.planner_residual,
        "fallback_logit_bias": args.fallback_logit_bias,
        "initial_stage_min_win_rate": args.initial_stage_min_win_rate,
        "unity_background": True,
        "teacher": {
            "policy_id": (
                "win70-planner-residual-base-v1"
                if args.planner_residual
                else "win70-planner-teacher-v1"
            ),
            "roles": [role.value for role in PLANNER_ROLES],
            "guard_chase_radius_cells": 48,
            "special_items": False,
            "label_semantics": (
                "action_0_planner_fallback"
                if args.planner_residual
                else "absolute_direction"
            ),
        },
        "opponents": {
            "base_scripted": str(BASE_SCRIPTED_CONFIG.resolve()),
            "weak_win70": str(WEAK_WIN70_CONFIG.resolve()),
            "full_win70": str(args.opponent_checkpoint.resolve()),
            "historical": "frozen snapshot created on entry to robust_historical_mix",
        },
        "eval_every": args.eval_every,
        "save_every": args.save_every,
        "time_scale": args.time_scale,
        "eval_time_scale": args.eval_time_scale,
        "max_episode_steps": args.max_episode_steps,
        "eval_workers": args.eval_workers,
        "reward": {**asdict(reward_config), "mode": reward_config.mode.value},
        "ppo": asdict(ppo_config),
        "device": asdict(device_selection),
    }

    if saved_config is not None:
        immutable_keys = (
            "seed",
            "initial_checkpoint",
            "opponent_checkpoint",
            "opponent_sha256",
            "target_win_rate",
            "target_wins",
            "train_seeds",
            "eval_seeds",
            "learning_rate",
            "gamma",
            "gae_lambda",
            "rollout_steps",
            "curriculum_scale",
            "curriculum",
            "teacher_replay_capacity",
            "teacher_minibatches",
            "teacher_minibatch_size",
            "teacher_class_balance_power",
            "planner_residual",
            "fallback_logit_bias",
            "initial_stage_min_win_rate",
            "teacher",
            "reward",
            "ppo",
        )
        changed = [key for key in immutable_keys if saved_config.get(key) != run_config.get(key)]
        if changed:
            parser.error(
                "resume configuration differs for immutable fields: " + ", ".join(changed)
            )

    for group in optimizer.param_groups:
        group["lr"] = args.learning_rate

    args.log_dir.mkdir(parents=True, exist_ok=True)
    training_log = args.log_dir / "training.jsonl"
    summary_log = args.log_dir / "run_summary.json"
    args.snapshot_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    collectors: dict[int, MAPPOParallelRolloutCollector] = {}
    mixtures: dict[int, EpisodeOpponentMixture] = {}
    envs: list[ContractBlackOutEnv] = []

    def runtime_state() -> dict[str, Any]:
        counts = {
            str(team): (
                _collector_counts(collectors[team]) if team in collectors else cumulative_counts[str(team)]
            )
            for team in (0, 1)
        }
        return {
            "curriculum": curriculum.state_dict(),
            "seed_streams": {str(team): seed_streams[team].state_dict() for team in (0, 1)},
            "collectors": counts,
            "opponent_mixtures": {
                str(team): mixture.state_dict() for team, mixture in mixtures.items()
            },
            "historical_checkpoint": historical_checkpoint,
        }

    def save(path: Path) -> None:
        save_v4_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            replay=replay,
            global_step=global_step,
            update=update,
            run_id=run_id,
            seed=args.seed,
            initial_checkpoint=args.initial_checkpoint,
            opponent_checkpoint=args.opponent_checkpoint,
            config=run_config,
            runtime_state=runtime_state(),
            best_evaluation=best_evaluation,
            planner_residual=args.planner_residual,
        )

    def close_collectors() -> None:
        nonlocal envs
        for team, collector in collectors.items():
            cumulative_counts[str(team)] = _collector_counts(collector)
        collectors.clear()
        mixtures.clear()
        for env in envs:
            env.close()
        envs = []

    def build_policy(policy_id: str, team: int) -> Any:
        if policy_id == "base_scripted":
            return FrozenScriptedOpponent(
                team, seed=args.seed + 100 + team, enable_special_items=False
            )
        if policy_id == "weak_win70":
            return FrozenScriptedOpponent(
                team,
                seed=args.seed + 200 + team,
                enable_special_items=False,
                roles=PLANNER_ROLES,
                chase_radius_cells=12,
                policy_id="weak-win70-v1",
            )
        if policy_id == "full_win70":
            return FrozenCheckpointOpponent(
                snapshot_from_checkpoint(args.opponent_checkpoint, generation=0, global_step=0),
                team=team,
                device="cpu",
            )
        if policy_id == "historical":
            if historical_checkpoint is None or not Path(historical_checkpoint).is_file():
                raise ValueError("historical curriculum stage has no saved snapshot")
            return FrozenCheckpointOpponent(
                snapshot_from_checkpoint(Path(historical_checkpoint), generation=1, global_step=global_step),
                team=team,
                device="cpu",
            )
        raise ValueError(f"unknown opponent policy: {policy_id}")

    def rebuild_collectors() -> None:
        nonlocal pending_resume_mix_states
        close_collectors()
        stage = curriculum.stage
        saved_mix_states = pending_resume_mix_states
        pending_resume_mix_states = {}
        for learner_team in (0, 1):
            opponent_team = 1 - learner_team
            policies = {
                policy_id: build_policy(policy_id, opponent_team)
                for policy_id in stage.opponent_weights
            }
            mixture = EpisodeOpponentMixture(
                policies,
                stage.opponent_weights,
                seed=args.seed + 1_000 * curriculum.stage_index + learner_team,
            )
            if str(learner_team) in saved_mix_states:
                mixture.load_state_dict(saved_mix_states[str(learner_team)])
            env = ContractBlackOutEnv(
                env_path=str(args.build),
                background=True,
                worker_id=learner_team,
                no_graphics=False,
                time_scale=args.time_scale,
            )
            envs.append(env)
            collector = MAPPOParallelRolloutCollector(
                env,
                model,
                mixture,
                learning_team=learner_team,
                device=device,
                reward_transform=TeamTrainingReward(learner_team, reward_config),
                episode_seed_provider=seed_streams[learner_team].next_seed,
                teacher=(
                    None
                    if args.planner_residual
                    else FrozenScriptedOpponent(
                        learner_team,
                        seed=args.seed + 500 + learner_team,
                        enable_special_items=False,
                        roles=PLANNER_ROLES,
                        chase_radius_cells=48,
                        policy_id="win70-planner-teacher-v1",
                    )
                ),
                residual_base=(
                    FrozenScriptedOpponent(
                        learner_team,
                        seed=args.seed + 500 + learner_team,
                        enable_special_items=False,
                        roles=PLANNER_ROLES,
                        chase_radius_cells=48,
                        policy_id="win70-planner-residual-base-v1",
                    )
                    if args.planner_residual
                    else None
                ),
                teacher_forcing_probability=0.0,
                teacher_seed=args.seed + 700 + learner_team,
            )
            _restore_counts(collector, cumulative_counts[str(learner_team)])
            collectors[learner_team] = collector
            mixtures[learner_team] = mixture

    def target_evaluation() -> tuple[dict[str, Any], bool]:
        nonlocal best_evaluation
        save(args.latest_checkpoint)
        output = args.log_dir / f"target_eval_step_{global_step}.json"
        result = evaluate_candidate(
            build=args.build,
            candidate=args.latest_checkpoint,
            opponent=args.opponent_checkpoint,
            seeds=eval_seeds,
            workers=args.eval_workers,
            time_scale=args.eval_time_scale,
            max_episode_steps=args.max_episode_steps,
            global_step=global_step,
            output=output,
        )
        summary = result["summary"]
        enriched = {
            **summary,
            "global_step": global_step,
            "stage": curriculum.stage.name,
            "evaluation_log": str(output.resolve()),
            "win_rate_ci": paired_seed_bootstrap_ci(
                result["episodes"], metric="win_rate", resamples=10_000, seed=args.seed
            ).to_dict(),
            "score_diff_ci": paired_seed_bootstrap_ci(
                result["episodes"], metric="score_diff", resamples=10_000, seed=args.seed + 1
            ).to_dict(),
        }
        if _better(enriched, best_evaluation):
            best_evaluation = enriched
            save(args.best_checkpoint)
        save(args.latest_checkpoint)
        passed = promotion_eligible and target_reached(summary, args.target_win_rate)
        return enriched, passed

    def stage_evaluation() -> dict[str, Any]:
        opponent_id = curriculum.stage.evaluation_opponent
        if opponent_id == "full_win70":
            raise RuntimeError("full target evaluation should be reused for this stage")
        output = args.log_dir / f"stage_{curriculum.stage.name}_eval_step_{global_step}.json"
        return evaluate_stage_candidate(
            build=args.build,
            candidate=args.latest_checkpoint,
            full_opponent=args.opponent_checkpoint,
            opponent_id=opponent_id,
            seeds=eval_seeds,
            workers=args.eval_workers,
            time_scale=args.eval_time_scale,
            max_episode_steps=args.max_episode_steps,
            global_step=global_step,
            output=output,
        )["summary"]

    def write_summary(
        target_summary: Mapping[str, Any], stage_summary: Mapping[str, Any]
    ) -> None:
        write_json(
            summary_log,
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "global_step": global_step,
                "update": update,
                "stage": curriculum.stage.name,
                "stage_steps": curriculum.stage_steps(global_step),
                "target_win_rate": args.target_win_rate,
                "target_wins": target_wins,
                "target_reached": promotion_eligible and target_reached(target_summary, args.target_win_rate),
                "best_evaluation": best_evaluation,
                "latest_target_evaluation": dict(target_summary),
                "latest_stage_evaluation": dict(stage_summary),
                "teacher_replay_size": len(replay),
                "runtime_state": runtime_state(),
                "elapsed_seconds": time.monotonic() - started,
                "config": run_config,
            },
        )

    try:
        if global_step == 0 and not args.skip_initial_eval:
            # Check the cheap/easy baseline before spending time on the full
            # target series. This catches a broken residual checkpoint before
            # any rollout or misleading best-checkpoint update is produced.
            save(args.latest_checkpoint)
            stage_summary = (
                evaluate_candidate(
                    build=args.build,
                    candidate=args.latest_checkpoint,
                    opponent=args.opponent_checkpoint,
                    seeds=eval_seeds,
                    workers=args.eval_workers,
                    time_scale=args.eval_time_scale,
                    max_episode_steps=args.max_episode_steps,
                    global_step=global_step,
                    output=args.log_dir / f"target_eval_step_{global_step}.json",
                )["summary"]
                if curriculum.stage.evaluation_opponent == "full_win70"
                else stage_evaluation()
            )
            if stage_summary["win_rate"] < args.initial_stage_min_win_rate:
                write_json(
                    summary_log,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "run_id": run_id,
                        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "status": "preflight_failed",
                        "global_step": global_step,
                        "required_stage_win_rate": args.initial_stage_min_win_rate,
                        "latest_stage_evaluation": dict(stage_summary),
                        "config": run_config,
                    },
                )
                _append_jsonl(
                    training_log,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "record_type": "preflight_failed",
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                        "global_step": global_step,
                        "required_stage_win_rate": args.initial_stage_min_win_rate,
                        "stage_evaluation": dict(stage_summary),
                    },
                )
                print(
                    "preflight failed: initial policy does not preserve the planner "
                    f"baseline ({stage_summary['win_rate']:.3f} < "
                    f"{args.initial_stage_min_win_rate:.3f})",
                    flush=True,
                )
                return 3
            target_summary, passed = target_evaluation()
            write_summary(target_summary, stage_summary)
            if passed:
                save(args.target_checkpoint)
                return 0

        rebuild_collectors()
        next_eval = ((global_step // args.eval_every) + 1) * args.eval_every
        next_save = ((global_step // args.save_every) + 1) * args.save_every
        while global_step < args.max_env_steps:
            stage = curriculum.stage
            stage_steps = curriculum.stage_steps(global_step)
            forcing = teacher_forcing_probability(stage, stage_steps)
            beta = bc_coefficient(stage, stage_steps)
            learner_team = update % 2
            collector = collectors[learner_team]
            collector.teacher_forcing_probability = forcing
            before = _collector_counts(collector)
            buffer = collector.collect(args.rollout_steps)
            batch = buffer.as_batch(gamma=args.gamma, gae_lambda=args.gae_lambda)
            diagnostics = (
                mappo_update(model, optimizer, batch, config=ppo_config)
                if stage.use_ppo
                else PPODiagnostics(
                    policy_loss=0.0,
                    value_loss=0.0,
                    entropy=0.0,
                    approximate_kl=0.0,
                    clip_fraction=0.0,
                    explained_variance=None,
                    gradient_norm=0.0,
                    epochs_completed=0,
                    minibatches=0,
                    samples=len(batch),
                    early_stopped=False,
                )
            )
            replay.add(batch)
            imitation = (
                teacher_replay_update(
                    model.actor_model,
                    optimizer,
                    replay,
                    minibatches=args.teacher_minibatches,
                    minibatch_size=args.teacher_minibatch_size,
                    loss_coef=beta,
                    max_grad_norm=ppo_config.max_grad_norm,
                    class_balance_power=args.teacher_class_balance_power,
                )
                if beta > 0.0 and len(replay) > 0
                else OnlineImitationMetrics(0.0, 0.0, 0, 0.0)
            )
            update += 1
            global_step += len(buffer)
            after = _collector_counts(collector)
            rollout = {name: after[name] - before[name] for name in after}
            _append_jsonl(
                training_log,
                {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": run_id,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "update": update,
                    "global_step": global_step,
                    "stage": stage.name,
                    "stage_steps": curriculum.stage_steps(global_step),
                    "learner_team": learner_team,
                    "use_ppo": stage.use_ppo,
                    "teacher_forcing_probability": forcing,
                    "bc_loss_coefficient": beta,
                    "teacher_replay_size": len(replay),
                    "rollout": rollout,
                    "opponent_selection_counts": dict(mixtures[learner_team].selection_counts),
                    "mean_return": float(batch.return_.mean()),
                    "ppo": diagnostics.to_dict(),
                    "imitation": asdict(imitation),
                },
            )
            print(
                f"stage={stage.name} update={update} step={global_step} side={learner_team} "
                f"episodes={rollout['episodes_completed']} wins={rollout['wins']} "
                f"forcing={forcing:.3f} beta={beta:.3f} replay={len(replay)} "
                f"bc_acc={imitation.accuracy:.3f} kl={diagnostics.approximate_kl:.6f}",
                flush=True,
            )

            if global_step >= next_save:
                save(args.latest_checkpoint)
                next_save += args.save_every
            if global_step >= next_eval or global_step >= args.max_env_steps:
                target_summary, passed = target_evaluation()
                stage_summary = (
                    target_summary
                    if stage.evaluation_opponent == "full_win70"
                    else stage_evaluation()
                )
                write_summary(target_summary, stage_summary)
                print(
                    f"evaluation stage={stage.name} target_wins={target_summary['wins']}/"
                    f"{target_summary['episodes']} stage_wins={stage_summary['wins']}/"
                    f"{stage_summary['episodes']}",
                    flush=True,
                )
                if passed:
                    save(args.target_checkpoint)
                    print(f"target_checkpoint={args.target_checkpoint}", flush=True)
                    return 0
                reason = curriculum.promotion_reason(stage_summary, global_step=global_step)
                if reason is not None:
                    previous, current = curriculum.promote(global_step=global_step)
                    if current == "robust_historical_mix":
                        save(args.latest_checkpoint)
                        snapshot = args.snapshot_dir / f"full_win70_step_{global_step}.pt"
                        shutil.copy2(args.latest_checkpoint, snapshot)
                        historical_checkpoint = str(snapshot.resolve())
                    _append_jsonl(
                        training_log,
                        {
                            "schema_version": SCHEMA_VERSION,
                            "record_type": "stage_transition",
                            "created_at_utc": datetime.now(timezone.utc).isoformat(),
                            "global_step": global_step,
                            "from_stage": previous,
                            "to_stage": current,
                            "reason": reason,
                            "stage_evaluation": dict(stage_summary),
                        },
                    )
                    print(f"stage_transition={previous}->{current} reason={reason}", flush=True)
                    rebuild_collectors()
                elif curriculum.budget_exhausted(global_step=global_step):
                    _append_jsonl(
                        training_log,
                        {
                            "schema_version": SCHEMA_VERSION,
                            "record_type": "stage_blocked",
                            "created_at_utc": datetime.now(timezone.utc).isoformat(),
                            "global_step": global_step,
                            "stage": stage.name,
                            "reason": "maximum_stage_budget_without_gate",
                            "stage_evaluation": dict(stage_summary),
                        },
                    )
                    save(args.latest_checkpoint)
                    print(
                        f"stage_blocked={stage.name}: maximum budget reached without "
                        "the promotion gate; stopped before increasing difficulty",
                        flush=True,
                    )
                    return 3
                next_eval += args.eval_every
    except KeyboardInterrupt:
        save(args.latest_checkpoint)
        print(f"interrupted; resumable checkpoint saved to {args.latest_checkpoint}", flush=True)
        return 130
    finally:
        close_collectors()

    save(args.latest_checkpoint)
    print(
        f"target not reached by max_env_steps={args.max_env_steps}; resume with --resume-latest",
        flush=True,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
