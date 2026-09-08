#!/usr/bin/env python3
"""Train planner-conditioned, fail-closed MAPPO residual v5."""

from __future__ import annotations

import argparse
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
    TrainingRewardConfig,
    checkpoint_payload,
    load_checkpoint,
    save_checkpoint,
)
from blackout_rl.evaluation_protocol import SeedSplits, paired_seed_bootstrap_ci  # noqa: E402
from blackout_rl.ippo_training import select_training_device  # noqa: E402
from blackout_rl.logging_schema import sha256_file, write_json  # noqa: E402
from blackout_rl.mappo import (  # noqa: E402
    MAPPOActorCritic,
    MAPPOParallelRolloutCollector,
    initialize_mappo_from_checkpoint,
    initialize_mappo_from_ippo,
    initialize_planner_conditioned_residual,
    mappo_update,
)
from blackout_rl.mappo_curriculum import EpisodeOpponentMixture  # noqa: E402
from blackout_rl.mappo_curriculum_v5 import (  # noqa: E402
    DEFAULT_MAPPO_V5_CURRICULUM,
    MAPPOV5CurriculumController,
    MAPPOV5Stage,
)
from blackout_rl.self_play import FrozenCheckpointOpponent, snapshot_from_checkpoint  # noqa: E402
from eval.evaluator import summarize_episodes  # noqa: E402
from scripts.train_mappo_teacher_curriculum_v3 import (  # noqa: E402
    BASE_SCRIPTED_CONFIG,
    PLANNER_ROLES,
    WEAK_WIN70_CONFIG,
    _append_jsonl,
    _collector_counts,
    _restore_counts,
    evaluate_stage_candidate,
)
from scripts.train_mappo_vs_win70 import (  # noqa: E402
    CyclicSeedStream,
    _actor_policy,
    _better,
    _cpu_tree,
    parse_seeds,
    promotion_evaluation_eligible,
    required_target_wins,
    target_reached,
)


SCHEMA_VERSION = "blackout.mappo_planner_residual.v5"
GUARDRAIL_MODE = "planner_residual_v2"


def scaled_curriculum(scale: float, rollout_steps: int) -> tuple[MAPPOV5Stage, ...]:
    if scale <= 0.0 or rollout_steps <= 0:
        raise ValueError("curriculum scale and rollout steps must be positive")
    stages: list[MAPPOV5Stage] = []
    for stage in DEFAULT_MAPPO_V5_CURRICULUM:
        minimum = (
            0
            if stage.minimum_steps == 0
            else max(
                rollout_steps,
                int(math.ceil(stage.minimum_steps * scale / rollout_steps)) * rollout_steps,
            )
        )
        maximum = max(
            minimum or rollout_steps,
            int(math.ceil(stage.maximum_steps * scale / rollout_steps)) * rollout_steps,
        )
        stages.append(
            MAPPOV5Stage(
                **{
                    **asdict(stage),
                    "minimum_steps": minimum,
                    "maximum_steps": maximum,
                }
            )
        )
    return tuple(stages)


def save_v5_checkpoint(
    path: Path,
    *,
    model: MAPPOActorCritic,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    update: int,
    run_id: str,
    seed: int,
    initial_checkpoint: Path,
    opponent_checkpoint: Path,
    config: Mapping[str, Any],
    runtime_state: Mapping[str, Any],
    best_target_evaluation: Mapping[str, Any] | None,
    best_stage_evaluation: Mapping[str, Any] | None,
    fallback_logit_bias: float,
) -> None:
    if model.planner_residual_head is None:
        raise ValueError("v5 checkpoint requires a planner-conditioned residual head")
    payload = checkpoint_payload(
        _actor_policy(model),
        global_step=global_step,
        training_seed=seed,
        source={
            "phase": "mappo-planner-conditioned-residual-v5",
            "algorithm": "MAPPO_CTDE_planner_residual_v2",
            "initial_checkpoint": str(initial_checkpoint.resolve()),
            "initial_checkpoint_sha256": sha256_file(initial_checkpoint),
            "opponent_checkpoint": str(opponent_checkpoint.resolve()),
            "opponent_sha256": sha256_file(opponent_checkpoint),
        },
    )
    payload["inference_guardrail"] = {
        "version": "scripted_counter_v1",
        "mode": GUARDRAIL_MODE,
        "roles": [role.value for role in PLANNER_ROLES],
        "chase_radius_cells": 48,
        "fallback_action_index": 0,
        "override_action_indices": list(range(1, 9)),
        "override_action_semantics": [
            "keep_planner",
            "rotate_left_45",
            "rotate_right_45",
            "rotate_left_90",
            "rotate_right_90",
            "stop",
            "reverse",
            "rotate_left_135",
            "rotate_right_135",
        ],
        "max_overrides_per_step": 1,
        "fallback_logit_bias": fallback_logit_bias,
        "context_version": "planner_role_path_nearest_target_v1",
    }
    payload["planner_residual_v2_state"] = _cpu_tree(
        model.planner_residual_head.state_dict()
    )
    payload["mappo_state"] = _cpu_tree(model.state_dict())
    payload["mappo_optimizer_state"] = _cpu_tree(optimizer.state_dict())
    payload["mappo_v5_training"] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "global_step": global_step,
        "update": update,
        "config": dict(config),
        "runtime_state": dict(runtime_state),
        "best_target_evaluation": (
            dict(best_target_evaluation) if best_target_evaluation is not None else None
        ),
        "best_stage_evaluation": (
            dict(best_stage_evaluation) if best_stage_evaluation is not None else None
        ),
        "torch_rng_state": torch.get_rng_state(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_checkpoint(temporary, payload)
    temporary.replace(path)


def load_v5_checkpoint(
    path: Path, *, device: str
) -> tuple[MAPPOActorCritic, torch.optim.Optimizer, dict[str, Any]]:
    policy, payload = load_checkpoint(path, device=device)
    training = payload.get("mappo_v5_training")
    if not isinstance(training, dict) or training.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("resume checkpoint has no compatible MAPPO v5 training state")
    model = initialize_mappo_from_ippo(policy.actor_critic).to(device)
    initialize_planner_conditioned_residual(
        model,
        fallback_logit_bias=float(training["config"]["fallback_logit_bias"]),
    )
    model.load_state_dict(payload["mappo_state"], strict=True)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(training["config"]["learning_rate"])
    )
    optimizer.load_state_dict(payload["mappo_optimizer_state"])
    rng_state = training.get("torch_rng_state")
    if isinstance(rng_state, torch.Tensor):
        torch.set_rng_state(rng_state.cpu())
    return model, optimizer, payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Planner-conditioned, rollback-safe MAPPO residual v5"
    )
    parser.add_argument("--build", type=Path, default=ROOT / "builds/BlackOut.app")
    parser.add_argument("--initial-checkpoint", type=Path, default=ROOT / "checkpoints/win_70_vs_scripted.pt")
    parser.add_argument("--opponent-checkpoint", type=Path, default=ROOT / "checkpoints/win_70_vs_scripted.pt")
    parser.add_argument("--latest-checkpoint", type=Path, default=ROOT / "checkpoints/mappo_planner_residual_v5_latest.pt")
    parser.add_argument("--target-best-checkpoint", type=Path, default=ROOT / "checkpoints/mappo_planner_residual_v5_target_best.pt")
    parser.add_argument("--stage-best-checkpoint", type=Path, default=ROOT / "checkpoints/mappo_planner_residual_v5_stage_best.pt")
    parser.add_argument("--target-checkpoint", type=Path, default=ROOT / "checkpoints/mappo_win_85_vs_win70_v5.pt")
    parser.add_argument("--snapshot-dir", type=Path, default=ROOT / "checkpoints/mappo_v5_snapshots")
    parser.add_argument("--log-dir", type=Path, default=ROOT / "logs/mappo_planner_residual_v5")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-latest", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=10500)
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
    parser.add_argument("--fallback-logit-bias", type=float, default=6.5)
    parser.add_argument("--initial-stage-min-win-rate", type=float, default=0.60)
    parser.add_argument("--rollback-drop-tolerance", type=float, default=0.10)
    parser.add_argument("--confirmation-replicas", type=int, default=3)
    parser.add_argument("--score-delta-weight", type=float, default=1.0)
    parser.add_argument("--unity-shaping-weight", type=float, default=0.25)
    parser.add_argument("--terminal-win-reward", type=float, default=1.0)
    parser.add_argument("--time-scale", type=float, default=50.0)
    parser.add_argument("--eval-time-scale", type=float, default=100.0)
    parser.add_argument("--max-episode-steps", type=int, default=22_000)
    parser.add_argument("--eval-workers", type=int, default=5)
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
        "fallback-logit-bias": args.fallback_logit_bias,
        "confirmation-replicas": args.confirmation_replicas,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        parser.error("these values must be positive: " + ", ".join(invalid))
    for name, value in (
        ("target-win-rate", args.target_win_rate),
        ("initial-stage-min-win-rate", args.initial_stage_min_win_rate),
        ("rollback-drop-tolerance", args.rollback_drop_tolerance),
    ):
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name} must be in [0,1]")
    if args.max_env_steps < args.rollout_steps:
        parser.error("--max-env-steps must cover at least one rollout")


def _enrich_evaluation(
    payload: Mapping[str, Any], *, global_step: int, stage: str, seed: int, output: Path
) -> dict[str, Any]:
    return {
        **payload["summary"],
        "global_step": global_step,
        "stage": stage,
        "evaluation_log": str(output.resolve()),
        "win_rate_ci": paired_seed_bootstrap_ci(
            payload["episodes"], metric="win_rate", resamples=10_000, seed=seed
        ).to_dict(),
        "score_diff_ci": paired_seed_bootstrap_ci(
            payload["episodes"], metric="score_diff", resamples=10_000, seed=seed + 1
        ).to_dict(),
    }


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

    splits = SeedSplits.from_dict(
        json.loads((ROOT / "configs/seed_splits_v1.json").read_text())
    )
    committed_train = tuple(splits.seeds_for("train", purpose="training"))
    committed_dev = tuple(splits.seeds_for("dev", purpose="model_selection"))
    train_seeds = args.train_seeds or list(committed_train)
    eval_seeds = args.eval_seeds or list(committed_dev)
    if not set(train_seeds) <= set(committed_train):
        parser.error("--train-seeds must come from the committed train split")
    if not set(eval_seeds) <= set(committed_dev):
        parser.error("--eval-seeds must come from the committed dev split")
    promotion_eligible = promotion_evaluation_eligible(eval_seeds, committed_dev)
    target_wins = required_target_wins(len(eval_seeds) * 2, args.target_win_rate)
    stages = scaled_curriculum(args.curriculum_scale, args.rollout_steps)

    resume_path = args.latest_checkpoint if args.resume_latest else args.resume
    if resume_path is None:
        conflicts = [
            path
            for path in (
                args.latest_checkpoint,
                args.target_best_checkpoint,
                args.stage_best_checkpoint,
                args.target_checkpoint,
                args.log_dir / "training.jsonl",
                args.log_dir / "run_summary.json",
            )
            if path.exists()
        ]
        if conflicts:
            parser.error(
                "fresh v5 run would mix with existing artifacts; resume or choose new paths: "
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
            model, optimizer, payload = load_v5_checkpoint(resume_path, device=device)
        except ValueError as exc:
            parser.error(str(exc))
        training = payload["mappo_v5_training"]
        global_step = int(training["global_step"])
        update = int(training["update"])
        run_id = str(training["run_id"])
        saved_runtime = training.get("runtime_state") or {}
        saved_config = training["config"]
        best_target_evaluation = training.get("best_target_evaluation")
        best_stage_evaluation = training.get("best_stage_evaluation")
        print(f"resumed={resume_path} step={global_step} update={update}", flush=True)
    else:
        model, _ = initialize_mappo_from_checkpoint(
            str(args.initial_checkpoint), device=device
        )
        initialize_planner_conditioned_residual(
            model, fallback_logit_bias=args.fallback_logit_bias
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        global_step = 0
        update = 0
        run_id = str(uuid.uuid4())
        best_target_evaluation = None
        best_stage_evaluation = None

    curriculum_state = saved_runtime.get("curriculum", {})
    curriculum = MAPPOV5CurriculumController(
        stages,
        stage_index=int(curriculum_state.get("stage_index", 0)),
        stage_start_step=int(curriculum_state.get("stage_start_step", 0)),
    )
    seed_state = saved_runtime.get("seed_streams", {})
    seed_streams = {
        team: (
            CyclicSeedStream.from_state(seed_state[str(team)], expected_seeds=train_seeds)
            if str(team) in seed_state
            else CyclicSeedStream(
                tuple(train_seeds),
                cursor=0 if team == 0 else max(1, len(train_seeds) // 2),
            )
        )
        for team in (0, 1)
    }
    cumulative_counts = {
        str(team): dict(saved_runtime.get("collectors", {}).get(str(team), {}))
        for team in (0, 1)
    }
    pending_resume_mix_states = dict(saved_runtime.get("opponent_mixtures", {}))
    historical_checkpoint = saved_runtime.get("historical_checkpoint")
    baseline_target_win_rate = saved_runtime.get("baseline_target_win_rate")
    rollback_count = int(saved_runtime.get("rollback_count", 0))

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
        "algorithm": "MAPPO_CTDE_planner_conditioned_residual_v5",
        "seed": args.seed,
        "initial_checkpoint": str(args.initial_checkpoint.resolve()),
        "opponent_checkpoint": str(args.opponent_checkpoint.resolve()),
        "opponent_sha256": sha256_file(args.opponent_checkpoint),
        "target_win_rate": args.target_win_rate,
        "target_wins": target_wins,
        "max_env_steps": args.max_env_steps,
        "train_seeds": train_seeds,
        "eval_seeds": eval_seeds,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "rollout_steps": args.rollout_steps,
        "curriculum_scale": args.curriculum_scale,
        "curriculum": [asdict(stage) for stage in stages],
        "fallback_logit_bias": args.fallback_logit_bias,
        "initial_stage_min_win_rate": args.initial_stage_min_win_rate,
        "rollback_drop_tolerance": args.rollback_drop_tolerance,
        "confirmation_replicas": args.confirmation_replicas,
        "confirmation_episodes": len(eval_seeds) * 2 * args.confirmation_replicas,
        "planner_context": [
            "planner_action_one_hot",
            "worker_guard_role",
            "planner_path_valid",
            "planner_direction",
            "nearest_task_target_direction_distance",
        ],
        "exploration": {
            "max_overrides_per_step": 1,
            "all_zero_behavior_cloning_after_warmup": False,
        },
        "unity_background": True,
        "time_scale": args.time_scale,
        "eval_time_scale": args.eval_time_scale,
        "eval_every": args.eval_every,
        "save_every": args.save_every,
        "max_episode_steps": args.max_episode_steps,
        "eval_workers": args.eval_workers,
        "reward": {**asdict(reward_config), "mode": reward_config.mode.value},
        "ppo": asdict(ppo_config),
        "device": asdict(device_selection),
    }
    if saved_config is not None:
        immutable = (
            "seed", "initial_checkpoint", "opponent_checkpoint", "opponent_sha256",
            "target_win_rate", "train_seeds", "eval_seeds", "learning_rate", "gamma",
            "gae_lambda", "rollout_steps", "curriculum_scale", "curriculum",
            "fallback_logit_bias", "rollback_drop_tolerance", "confirmation_replicas",
            "reward", "ppo",
        )
        changed = [key for key in immutable if saved_config.get(key) != run_config.get(key)]
        if changed:
            parser.error("resume configuration differs: " + ", ".join(changed))

    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.snapshot_dir.mkdir(parents=True, exist_ok=True)
    training_log = args.log_dir / "training.jsonl"
    summary_log = args.log_dir / "run_summary.json"
    started = time.monotonic()
    collectors: dict[int, MAPPOParallelRolloutCollector] = {}
    mixtures: dict[int, EpisodeOpponentMixture] = {}
    envs: list[ContractBlackOutEnv] = []

    def runtime_state() -> dict[str, Any]:
        return {
            "curriculum": curriculum.state_dict(),
            "seed_streams": {
                str(team): seed_streams[team].state_dict() for team in (0, 1)
            },
            "collectors": {
                str(team): (
                    _collector_counts(collectors[team])
                    if team in collectors
                    else cumulative_counts[str(team)]
                )
                for team in (0, 1)
            },
            "opponent_mixtures": {
                str(team): mixture.state_dict() for team, mixture in mixtures.items()
            },
            "historical_checkpoint": historical_checkpoint,
            "baseline_target_win_rate": baseline_target_win_rate,
            "rollback_count": rollback_count,
        }

    def save(path: Path) -> None:
        save_v5_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            global_step=global_step,
            update=update,
            run_id=run_id,
            seed=args.seed,
            initial_checkpoint=args.initial_checkpoint,
            opponent_checkpoint=args.opponent_checkpoint,
            config=run_config,
            runtime_state=runtime_state(),
            best_target_evaluation=best_target_evaluation,
            best_stage_evaluation=best_stage_evaluation,
            fallback_logit_bias=args.fallback_logit_bias,
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
                snapshot_from_checkpoint(
                    args.opponent_checkpoint, generation=0, global_step=0
                ),
                team=team,
                device="cpu",
            )
        if policy_id == "historical":
            if historical_checkpoint is None or not Path(historical_checkpoint).is_file():
                raise ValueError("historical stage has no checkpoint")
            return FrozenCheckpointOpponent(
                snapshot_from_checkpoint(
                    Path(historical_checkpoint), generation=1, global_step=global_step
                ),
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
            policies = {
                policy_id: build_policy(policy_id, 1 - learner_team)
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
            collector = MAPPOParallelRolloutCollector(
                env,
                model,
                mixture,
                learning_team=learner_team,
                device=device,
                reward_transform=TeamTrainingReward(learner_team, reward_config),
                episode_seed_provider=seed_streams[learner_team].next_seed,
                residual_base=FrozenScriptedOpponent(
                    learner_team,
                    seed=args.seed + 500 + learner_team,
                    enable_special_items=False,
                    roles=PLANNER_ROLES,
                    chase_radius_cells=48,
                    policy_id="win70-planner-residual-v2-base",
                ),
                teacher_forcing_probability=stage.planner_forcing,
                teacher_seed=args.seed + 700 + learner_team,
                max_residual_overrides_per_step=1,
            )
            _restore_counts(collector, cumulative_counts[str(learner_team)])
            envs.append(env)
            collectors[learner_team] = collector
            mixtures[learner_team] = mixture

    def evaluate_once(opponent_id: str, output: Path, *, replica: int = 0) -> dict[str, Any]:
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
            candidate_policy_seed=870_000 + replica * 10_000,
            opponent_policy_seed=880_000 + replica * 10_000,
        )

    def confirmation_evaluation(opponent_id: str, label: str) -> dict[str, Any]:
        episodes: list[dict[str, Any]] = []
        replica_paths: list[str] = []
        for replica in range(args.confirmation_replicas):
            replica_output = args.log_dir / (
                f"confirmation_{label}_step_{global_step}_replica_{replica}.json"
            )
            payload = evaluate_once(opponent_id, replica_output, replica=replica)
            replica_paths.append(str(replica_output.resolve()))
            for episode in payload["episodes"]:
                copied = dict(episode)
                copied["pair_id"] = f"{episode['pair_id']}-replica-{replica}"
                episodes.append(copied)
        output = args.log_dir / f"confirmation_{label}_step_{global_step}.json"
        payload = {
            "schema_version": "blackout.paired_series.v1",
            "series_id": str(uuid.uuid4()),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "series_config": {
                "seeds": list(eval_seeds),
                "split": "dev",
                "held_out": True,
                "side_swap": True,
                "opponent": opponent_id,
                "policy_seed_replicas": args.confirmation_replicas,
                "replica_logs": replica_paths,
                "training_global_step": global_step,
            },
            "episodes": episodes,
            "summary": summarize_episodes(episodes),
        }
        write_json(output, payload)
        return _enrich_evaluation(
            payload,
            global_step=global_step,
            stage=curriculum.stage.name,
            seed=args.seed + 100,
            output=output,
        )

    def target_evaluation() -> dict[str, Any]:
        nonlocal best_target_evaluation
        save(args.latest_checkpoint)
        output = args.log_dir / f"target_eval_step_{global_step}.json"
        payload = evaluate_once("full_win70", output)
        enriched = _enrich_evaluation(
            payload,
            global_step=global_step,
            stage=curriculum.stage.name,
            seed=args.seed,
            output=output,
        )
        if _better(enriched, best_target_evaluation):
            best_target_evaluation = enriched
            save(args.target_best_checkpoint)
        save(args.latest_checkpoint)
        return enriched

    def stage_evaluation() -> dict[str, Any]:
        nonlocal best_stage_evaluation
        opponent_id = curriculum.stage.evaluation_opponent
        if opponent_id == "full_win70":
            raise RuntimeError("reuse target evaluation for full-win70 stages")
        output = args.log_dir / f"stage_{curriculum.stage.name}_eval_step_{global_step}.json"
        payload = evaluate_once(opponent_id, output)
        enriched = _enrich_evaluation(
            payload,
            global_step=global_step,
            stage=curriculum.stage.name,
            seed=args.seed + 10,
            output=output,
        )
        if _better(enriched, best_stage_evaluation):
            best_stage_evaluation = enriched
            save(args.stage_best_checkpoint)
        return enriched

    def write_summary(
        target_summary: Mapping[str, Any],
        stage_summary: Mapping[str, Any],
        *,
        status: str = "running",
    ) -> None:
        write_json(
            summary_log,
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": status,
                "global_step": global_step,
                "update": update,
                "stage": curriculum.stage.name,
                "stage_steps": curriculum.stage_steps(global_step),
                "target_win_rate": args.target_win_rate,
                "target_wins": target_wins,
                "best_target_evaluation": best_target_evaluation,
                "best_stage_evaluation": best_stage_evaluation,
                "latest_target_evaluation": dict(target_summary),
                "latest_stage_evaluation": dict(stage_summary),
                "runtime_state": runtime_state(),
                "elapsed_seconds": time.monotonic() - started,
                "config": run_config,
            },
        )

    def restore_target_best(reason: str, confirmed: Mapping[str, Any]) -> None:
        nonlocal rollback_count
        if not args.target_best_checkpoint.is_file():
            raise RuntimeError("rollback requested before a target-best checkpoint exists")
        restored_model, restored_optimizer, _ = load_v5_checkpoint(
            args.target_best_checkpoint, device=device
        )
        model.load_state_dict(restored_model.state_dict(), strict=True)
        optimizer.load_state_dict(restored_optimizer.state_dict())
        rollback_count += 1
        _append_jsonl(
            training_log,
            {
                "schema_version": SCHEMA_VERSION,
                "record_type": "target_rollback",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "global_step": global_step,
                "stage": curriculum.stage.name,
                "reason": reason,
                "confirmed_evaluation": dict(confirmed),
                "restored_checkpoint": str(args.target_best_checkpoint.resolve()),
                "rollback_count": rollback_count,
            },
        )
        save(args.latest_checkpoint)
        rebuild_collectors()

    last_target_summary: Mapping[str, Any] | None = best_target_evaluation
    last_stage_summary: Mapping[str, Any] | None = best_stage_evaluation

    try:
        if global_step == 0:
            save(args.latest_checkpoint)
            initial_stage = stage_evaluation()
            if float(initial_stage["win_rate"]) < args.initial_stage_min_win_rate:
                write_summary(initial_stage, initial_stage, status="preflight_failed")
                print(
                    f"preflight_failed={initial_stage['win_rate']:.3f} < "
                    f"{args.initial_stage_min_win_rate:.3f}",
                    flush=True,
                )
                return 3
            initial_target = target_evaluation()
            baseline_target_win_rate = float(initial_target["win_rate"])
            save(args.target_best_checkpoint)
            write_summary(initial_target, initial_stage)
            last_target_summary = initial_target
            last_stage_summary = initial_stage

        rebuild_collectors()
        next_eval = ((global_step // args.eval_every) + 1) * args.eval_every
        next_save = ((global_step // args.save_every) + 1) * args.save_every
        while global_step < args.max_env_steps:
            stage = curriculum.stage
            stage_steps = curriculum.stage_steps(global_step)
            penalty = stage.override_penalty(stage_steps)
            learner_team = update % 2
            collector = collectors[learner_team]
            collector.teacher_forcing_probability = stage.planner_forcing
            before = _collector_counts(collector)
            buffer = collector.collect(args.rollout_steps)
            batch = buffer.as_batch(gamma=args.gamma, gae_lambda=args.gae_lambda)
            diagnostics = (
                mappo_update(
                    model,
                    optimizer,
                    batch,
                    config=ppo_config,
                    override_penalty_coef=penalty,
                )
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
            update += 1
            global_step += len(buffer)
            after = _collector_counts(collector)
            rollout = {name: after[name] - before[name] for name in after}
            override_total = (
                rollout["residual_fallback_actions"]
                + rollout["residual_override_actions"]
            )
            override_rate = (
                rollout["residual_override_actions"] / override_total
                if override_total
                else 0.0
            )
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
                    "planner_forcing": stage.planner_forcing,
                    "override_penalty": penalty,
                    "override_rate": override_rate,
                    "rollout": rollout,
                    "opponent_selection_counts": dict(
                        mixtures[learner_team].selection_counts
                    ),
                    "mean_return": float(batch.return_.mean()),
                    "ppo": diagnostics.to_dict(),
                },
            )
            print(
                f"stage={stage.name} update={update} step={global_step} "
                f"side={learner_team} episodes={rollout['episodes_completed']} "
                f"wins={rollout['wins']} override={override_rate:.4f} "
                f"penalty={penalty:.5f} kl={diagnostics.approximate_kl:.6f}",
                flush=True,
            )

            if global_step >= next_save:
                save(args.latest_checkpoint)
                next_save += args.save_every
            stage_budget_hit = curriculum.budget_exhausted(global_step=global_step)
            if (
                global_step >= next_eval
                or stage_budget_hit
                or global_step >= args.max_env_steps
            ):
                target_summary = target_evaluation()
                stage_summary = (
                    target_summary
                    if stage.evaluation_opponent == "full_win70"
                    else stage_evaluation()
                )
                last_target_summary = target_summary
                last_stage_summary = stage_summary

                assert baseline_target_win_rate is not None
                rollback_floor = max(
                    0.0, baseline_target_win_rate - args.rollback_drop_tolerance
                )
                if global_step > 0 and float(target_summary["win_rate"]) < rollback_floor:
                    confirmed = confirmation_evaluation("full_win70", "rollback")
                    if float(confirmed["win_rate"]) < rollback_floor:
                        restore_target_best("target_below_baseline_floor", confirmed)
                        write_summary(target_summary, stage_summary, status="rolled_back")
                        next_eval = global_step + args.eval_every
                        continue

                fast_target_pass = promotion_eligible and target_reached(
                    target_summary, args.target_win_rate
                )
                if fast_target_pass:
                    confirmed = confirmation_evaluation("full_win70", "target")
                    required = required_target_wins(
                        int(confirmed["episodes"]), args.target_win_rate
                    )
                    if int(confirmed["wins"]) >= required:
                        save(args.target_checkpoint)
                        write_summary(confirmed, stage_summary, status="target_reached")
                        print(f"target_checkpoint={args.target_checkpoint}", flush=True)
                        return 0

                gate_passed = curriculum.gate_passed(
                    stage_summary, global_step=global_step
                )
                if gate_passed and curriculum.can_promote():
                    confirmed_stage = confirmation_evaluation(
                        stage.evaluation_opponent, f"stage_{stage.name}"
                    )
                    if curriculum.gate_passed(
                        confirmed_stage, global_step=global_step
                    ):
                        previous, current = curriculum.promote(global_step=global_step)
                        if current == "robust_historical_mix":
                            snapshot = args.snapshot_dir / f"full_win70_step_{global_step}.pt"
                            shutil.copy2(args.latest_checkpoint, snapshot)
                            historical_checkpoint = str(snapshot.resolve())
                        best_stage_evaluation = None
                        _append_jsonl(
                            training_log,
                            {
                                "schema_version": SCHEMA_VERSION,
                                "record_type": "stage_transition",
                                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                                "global_step": global_step,
                                "from_stage": previous,
                                "to_stage": current,
                                "reason": "replicated_dev_confirmation_gate",
                                "confirmation": dict(confirmed_stage),
                            },
                        )
                        print(f"stage_transition={previous}->{current}", flush=True)
                        save(args.latest_checkpoint)
                        next_eval = global_step + args.eval_every
                        write_summary(target_summary, confirmed_stage)
                        last_stage_summary = confirmed_stage
                        if global_step >= args.max_env_steps:
                            break
                        rebuild_collectors()
                        continue

                write_summary(target_summary, stage_summary)
                if stage_budget_hit:
                    save(args.latest_checkpoint)
                    _append_jsonl(
                        training_log,
                        {
                            "schema_version": SCHEMA_VERSION,
                            "record_type": "stage_blocked",
                            "created_at_utc": datetime.now(timezone.utc).isoformat(),
                            "global_step": global_step,
                            "stage": stage.name,
                            "reason": "maximum_stage_budget_without_confirmed_gate",
                            "stage_evaluation": dict(stage_summary),
                        },
                    )
                    write_summary(target_summary, stage_summary, status="stage_blocked")
                    print(f"stage_blocked={stage.name}", flush=True)
                    return 3
                next_eval += args.eval_every
    except KeyboardInterrupt:
        save(args.latest_checkpoint)
        print(f"interrupted; resume={args.latest_checkpoint}", flush=True)
        return 130
    finally:
        close_collectors()

    save(args.latest_checkpoint)
    if last_target_summary is not None and last_stage_summary is not None:
        write_summary(
            last_target_summary,
            last_stage_summary,
            status="max_steps_reached",
        )
    print(f"target not reached by max_env_steps={args.max_env_steps}", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
