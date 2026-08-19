#!/usr/bin/env python3
"""AGENT-21~24 live frozen-snapshot self-play and empirical evaluation."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import uuid

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from blackout_rl import (  # noqa: E402
    ContractBlackOutEnv,
    PPOConfig,
    ParallelRolloutCollector,
    SubmissionPolicy,
    TeamTrainingReward,
    checkpoint_payload,
    load_checkpoint,
    ppo_update,
    save_checkpoint,
    team_agents,
)
from blackout_rl.evaluation_protocol import SeedSplits  # noqa: E402
from blackout_rl.ippo_training import select_training_device  # noqa: E402
from blackout_rl.logging_schema import sha256_file  # noqa: E402
from blackout_rl.policy import DeterministicCheckpointPolicy, PolicyArtifact  # noqa: E402
from blackout_rl.self_play import (  # noqa: E402
    FrozenCheckpointOpponent,
    MatchupResult,
    SideBalancedSampler,
    SnapshotPool,
    StabilityThresholds,
    checkpoint_evaluation_matrix,
    past_opponent_regressions,
    self_play_stable,
    should_expand_to_psro,
    snapshot_from_checkpoint,
)
from eval.evaluator import evaluate_episode, summarize_episodes  # noqa: E402
from scripts.train_ippo_vs_scripted import GAME_COMMIT, PYTHON_API_COMMIT  # noqa: E402


def _save_actor(path: Path, model, *, step: int, seed: int, generation: int) -> None:
    config = dict(model.model_config)
    config.pop("n_actions", None)
    policy = SubmissionPolicy(**config)
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    policy.actor_critic.load_state_dict(state)
    save_checkpoint(
        path,
        checkpoint_payload(
            policy,
            global_step=step,
            training_seed=seed,
            source={"phase": "phase3-self-play", "generation": str(generation)},
        ),
    )


def _checkpoint_match_job(job: dict) -> list[dict]:
    torch.set_num_threads(1)
    build = Path(job["build"])
    model_checkpoint = Path(job["model_checkpoint"])
    opponent_checkpoint = Path(job["opponent_checkpoint"])
    seed = int(job["seed"])
    env = ContractBlackOutEnv(
        env_path=str(build), no_graphics=False, time_scale=float(job["time_scale"])
    )
    model_artifact = PolicyArtifact.from_file(
        job["model_id"], "ippo-checkpoint", model_checkpoint
    )
    opponent_artifact = PolicyArtifact.from_file(
        job["opponent_id"], "ippo-checkpoint", opponent_checkpoint
    )
    episodes = []
    try:
        for pair_index, model_team in enumerate((0, 1)):
            model_policy = DeterministicCheckpointPolicy(
                model_checkpoint, team=model_team, seed=212100 + seed
            )
            opponent_policy = DeterministicCheckpointPolicy(
                opponent_checkpoint, team=1 - model_team, seed=212200 + seed
            )
            score_horizon = job.get("score_horizon_steps")
            if score_horizon is None:
                episodes.append(evaluate_episode(
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
                    pair_id=f"{job['model_id']}-vs-{job['opponent_id']}-seed-{seed}",
                    pair_index=pair_index,
                    max_steps=int(job["max_steps"]),
                ))
                continue

            observations, _ = env.reset(seed=seed)
            last_info = {"score_0": 0.0, "score_1": 0.0}
            terminal_winner = None
            steps = 0
            for steps in range(1, int(score_horizon) + 1):
                actions = model_policy.act(observations, team_agents(model_team))
                actions.update(
                    opponent_policy.act(observations, team_agents(1 - model_team))
                )
                observations, _, terminations, truncations, infos = env.step(actions)
                if any(truncations.values()):
                    raise RuntimeError("unexpected truncation in horizon evaluation")
                info = dict(next(iter(infos.values())))
                if any(terminations.values()):
                    terminal_winner = int(info["winner"])
                    break
                last_info = info
            score_a = float(last_info["score_0"]) * 100.0
            score_b = float(last_info["score_1"]) * 100.0
            winner = (
                terminal_winner
                if terminal_winner is not None
                else 0 if score_a > score_b else 1 if score_b > score_a else -1
            )
            model_score = score_a if model_team == 0 else score_b
            opponent_score = score_b if model_team == 0 else score_a
            episodes.append(
                {
                    "pair_id": f"{job['model_id']}-vs-{job['opponent_id']}-seed-{seed}",
                    "pair_index": pair_index,
                    "seed": seed,
                    "model": model_artifact.to_dict(),
                    "opponent": opponent_artifact.to_dict(),
                    "side_assignment": {
                        "model_team": model_team,
                        "model_side": "A" if model_team == 0 else "B",
                    },
                    "winner": {"team": winner},
                    "model_result": (
                        "draw" if winner == -1 else "win" if winner == model_team else "loss"
                    ),
                    "score": {
                        "team_a": score_a,
                        "team_b": score_b,
                        "model": model_score,
                        "opponent": opponent_score,
                        "model_minus_opponent": model_score - opponent_score,
                    },
                    "evaluation": {
                        "kind": "fixed_score_horizon",
                        "horizon_steps": int(score_horizon),
                        "steps_executed": steps,
                    },
                }
            )
    finally:
        env.close()
    return episodes


def _run_jobs(jobs: list[dict], workers: int) -> list[dict]:
    if workers <= 1:
        groups = [_checkpoint_match_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
            groups = list(executor.map(_checkpoint_match_job, jobs))
    return [episode for group in groups for episode in group]


def _jobs(
    build: Path,
    matchups: list[tuple[str, Path, str, Path]],
    seeds: list[int],
    *,
    time_scale: float,
    max_steps: int,
    score_horizon_steps: int | None = None,
) -> list[dict]:
    return [
        {
            "build": str(build),
            "model_id": model_id,
            "model_checkpoint": str(model_path),
            "opponent_id": opponent_id,
            "opponent_checkpoint": str(opponent_path),
            "seed": seed,
            "time_scale": time_scale,
            "max_steps": max_steps,
            **(
                {"score_horizon_steps": score_horizon_steps}
                if score_horizon_steps is not None
                else {}
            ),
        }
        for model_id, model_path, opponent_id, opponent_path in matchups
        for seed in seeds
    ]


def _directed_result(
    episodes: list[dict], model_id: str, opponent_id: str, seeds: list[int]
) -> MatchupResult:
    selected = [
        row
        for row in episodes
        if row["model"]["policy_id"] == model_id
        and row["opponent"]["policy_id"] == opponent_id
    ]
    summary = summarize_episodes(selected)
    return MatchupResult(
        model_id=model_id,
        opponent_id=opponent_id,
        pair_ids=tuple(f"{model_id}-vs-{opponent_id}-seed-{seed}" for seed in seeds),
        seeds=tuple(seeds),
        win_rate=float(summary["win_rate"]),
        mean_score_diff=float(summary["mean_model_score_diff"]),
    )


def _reciprocal_result(result: MatchupResult, episodes: list[dict]) -> MatchupResult:
    """Reverse attribution for the exact same side-swapped paired games."""
    selected = [
        row
        for row in episodes
        if row["model"]["policy_id"] == result.model_id
        and row["opponent"]["policy_id"] == result.opponent_id
    ]
    reverse_win_rate = sum(row["model_result"] == "loss" for row in selected) / len(selected)
    return MatchupResult(
        model_id=result.opponent_id,
        opponent_id=result.model_id,
        pair_ids=tuple(f"reciprocal-{pair_id}" for pair_id in result.pair_ids),
        seeds=result.seeds,
        win_rate=reverse_win_rate,
        mean_score_diff=-result.mean_score_diff,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--steps-per-generation", type=int, default=128)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        default=ROOT / "checkpoints/phase3_agent14_score_terminal_only.pt",
    )
    args = parser.parse_args()
    if args.generations < 8:
        raise ValueError("AGENT-24 saturation audit requires at least eight generations")
    if args.steps_per_generation <= 0:
        raise ValueError("steps-per-generation must be positive")

    build = ROOT / "builds/BlackOut.app"
    splits = SeedSplits.from_dict(
        json.loads((ROOT / "configs/seed_splits_v1.json").read_text())
    )
    train_seeds = list(splits.seeds_for("train", purpose="training"))
    dev_seed = list(splits.seeds_for("dev", purpose="model_selection"))[:1]
    device = select_training_device(args.device)
    torch.manual_seed(212400)
    np.random.seed(212400)
    if device.resolved == "mps":
        torch.mps.manual_seed(212400)
    policy, _ = load_checkpoint(args.initial_checkpoint, device=device.resolved)
    model = policy.actor_critic
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    ppo_config = PPOConfig(update_epochs=2, minibatch_size=128, target_kl=0.03)
    pool = SnapshotPool(capacity=8, latest_probability=0.5, seed=212401)
    sides = SideBalancedSampler(first_side=0)
    checkpoints: dict[str, Path] = {}

    gen0 = ROOT / "checkpoints/phase3_selfplay_gen_00.pt"
    _save_actor(gen0, model, step=0, seed=212400, generation=0)
    checkpoints["gen_00"] = gen0
    pool.add(snapshot_from_checkpoint(gen0, generation=0, global_step=0))
    history = []
    for generation in range(1, args.generations + 1):
        learner_side = sides.sample()
        opponent_snapshot = pool.sample()
        opponent_snapshot.validate_artifact()
        env = ContractBlackOutEnv(
            env_path=str(build), no_graphics=False, time_scale=50.0
        )
        collector = ParallelRolloutCollector(
            env,
            model,
            FrozenCheckpointOpponent(
                opponent_snapshot, team=1 - learner_side, device="cpu"
            ),
            learning_team=learner_side,
            device=device.resolved,
            reward_transform=TeamTrainingReward(learner_side),
        )
        try:
            buffer = collector.collect(
                args.steps_per_generation,
                seed=train_seeds[(generation - 1) % len(train_seeds)],
            )
            batch = buffer.as_batch(gamma=0.999, gae_lambda=0.95)
            diagnostics = ppo_update(model, optimizer, batch, config=ppo_config)
        finally:
            env.close()
        global_step = generation * args.steps_per_generation
        checkpoint = ROOT / f"checkpoints/phase3_selfplay_gen_{generation:02d}.pt"
        _save_actor(
            checkpoint,
            model,
            step=global_step,
            seed=212400,
            generation=generation,
        )
        checkpoint_id = f"gen_{generation:02d}"
        checkpoints[checkpoint_id] = checkpoint
        new_snapshot = snapshot_from_checkpoint(
            checkpoint, generation=generation, global_step=global_step
        )
        pool.add(new_snapshot)
        opponent_snapshot.validate_artifact()
        row = {
            "generation": generation,
            "global_step": global_step,
            "train_seed": train_seeds[(generation - 1) % len(train_seeds)],
            "learner_side": learner_side,
            "opponent_snapshot": asdict(opponent_snapshot),
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": sha256_file(checkpoint),
            "pool_generations": [snapshot.generation for snapshot in pool.snapshots],
            "mean_agent_reward": float(batch.reward.mean()),
            "episodes_completed": collector.episodes_completed,
            "entropy": diagnostics.entropy,
            "approximate_kl": diagnostics.approximate_kl,
            "value_error": diagnostics.value_loss,
            "diagnostics": diagnostics.to_dict(),
        }
        history.append(row)

    thresholds = StabilityThresholds()
    stable = self_play_stable(history, thresholds)
    major_ids = ("gen_00", "gen_02", "gen_04", f"gen_{args.generations:02d}")
    matrix_matchups = [
        (major_ids[left], checkpoints[major_ids[left]], major_ids[right], checkpoints[major_ids[right]])
        for left in range(len(major_ids))
        for right in range(left + 1, len(major_ids))
    ]
    matrix_episodes = _run_jobs(
        _jobs(
            build,
            matrix_matchups,
            dev_seed,
            time_scale=100.0,
            max_steps=22_000,
            score_horizon_steps=5_000,
        ),
        args.workers,
    )
    forward_rows = [
        _directed_result(matrix_episodes, model_id, opponent_id, dev_seed)
        for model_id, _, opponent_id, _ in matrix_matchups
    ]
    matrix_rows = [
        row
        for forward in forward_rows
        for row in (forward, _reciprocal_result(forward, matrix_episodes))
    ]
    matrix = checkpoint_evaluation_matrix(major_ids, matrix_rows)

    scripted_proxy = ROOT / "checkpoints/win_70_vs_scripted.pt"
    fixed_matchups = [
        (checkpoint_id, checkpoint, "win_70_exploiter", scripted_proxy)
        for checkpoint_id, checkpoint in checkpoints.items()
    ]
    fixed_episodes = _run_jobs(
        _jobs(build, fixed_matchups, dev_seed, time_scale=100.0, max_steps=22_000),
        args.workers,
    )
    fixed_summaries = {
        checkpoint_id: summarize_episodes(
            [row for row in fixed_episodes if row["model"]["policy_id"] == checkpoint_id]
        )
        for checkpoint_id in checkpoints
    }

    latest_id = f"gen_{args.generations:02d}"
    past_ids = ("gen_00", "gen_02")
    candidate_past = {name: matrix[latest_id][name]["win_rate"] for name in past_ids}
    reference_past = {name: matrix["gen_04"][name]["win_rate"] for name in past_ids}
    regressions = past_opponent_regressions(
        candidate_past, reference_past, tolerance=0.05
    )
    reference_exploiter = fixed_summaries["gen_04"]["win_rate"]
    candidate_exploiter = fixed_summaries[latest_id]["win_rate"]
    exploiter_regression = candidate_exploiter + 0.05 < reference_exploiter

    best_score = float("-inf")
    last_improvement = 0
    fixed_curve = []
    for generation in range(args.generations + 1):
        checkpoint_id = f"gen_{generation:02d}"
        score = float(fixed_summaries[checkpoint_id]["mean_model_score_diff"])
        fixed_curve.append(
            {
                "generation": generation,
                "win_rate": fixed_summaries[checkpoint_id]["win_rate"],
                "mean_score_diff": score,
            }
        )
        if score > best_score + 1.0:
            best_score = score
            last_improvement = generation
    plateau_generations = args.generations - last_improvement
    cyclic_regressions = len(regressions) + int(exploiter_regression)
    pool_saturated = len(pool.snapshots) == pool.capacity
    expand_psro = should_expand_to_psro(
        generations=args.generations,
        plateau_generations=plateau_generations,
        cyclic_regressions=cyclic_regressions,
    )
    if expand_psro and not pool_saturated:
        raise RuntimeError("PSRO cannot expand before the snapshot pool saturates")

    payload = {
        "schema_version": "blackout.phase3_agent21_24.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": str(uuid.uuid4()),
        "initial_checkpoint": str(args.initial_checkpoint.relative_to(ROOT)),
        "train_seeds": train_seeds,
        "evaluation_dev_seeds": dev_seed,
        "generations": args.generations,
        "steps_per_generation": args.steps_per_generation,
        "device": asdict(device),
        "ppo_config": asdict(ppo_config),
        "snapshot_pool": {
            "capacity": pool.capacity,
            "latest_probability": pool.latest_probability,
            "saturated": pool_saturated,
            "final_generations": [snapshot.generation for snapshot in pool.snapshots],
            "side_counts": sides.counts,
            "side_balanced": sides.balanced,
        },
        "agent21": {
            "history": history,
            "thresholds": asdict(thresholds),
            "stable": stable,
        },
        "agent22": {
            "major_generations": major_ids,
            "matrix": matrix,
            "episode_count": len(matrix_episodes),
            "paired_side_swap": True,
            "reciprocal_directions_derived_from_same_games": True,
            "evaluation_kind": "fixed_score_horizon",
            "horizon_steps": 5_000,
        },
        "agent23": {
            "reference_generation": "gen_04",
            "candidate_generation": latest_id,
            "past_opponent_candidate": candidate_past,
            "past_opponent_reference": reference_past,
            "past_opponent_regressions": list(regressions),
            "exploiter": "win_70_vs_scripted.pt",
            "reference_exploiter_win_rate": reference_exploiter,
            "candidate_exploiter_win_rate": candidate_exploiter,
            "exploiter_regression": exploiter_regression,
            "passes": not regressions and not exploiter_regression,
        },
        "agent24": {
            "fixed_exploiter_curve": fixed_curve,
            "plateau_generations": plateau_generations,
            "cyclic_regressions": cyclic_regressions,
            "pool_saturated_before_decision": pool_saturated,
            "expand_to_psro": expand_psro,
        },
        "artifacts": {
            "checkpoints": {
                name: str(path.relative_to(ROOT)) for name, path in checkpoints.items()
            },
            "matrix_episodes": matrix_episodes,
            "fixed_exploiter_episodes": fixed_episodes,
        },
    }
    output = ROOT / "logs/phase3_agent21_24_self_play.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "stable": stable,
                "agent23_passes": payload["agent23"]["passes"],
                "expand_to_psro": expand_psro,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
