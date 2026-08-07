"""BlackOut episode evaluator that never derives the winner from shaping reward."""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from blackout_rl import ContractBlackOutEnv, team_agents
from blackout_rl.logging_schema import (
    EPISODE_SCHEMA_VERSION,
    model_result,
    sha256_file,
    validate_episode_log,
    winner_from_scores,
)
from blackout_rl.policy import PolicyArtifact, RandomPolicy
from blackout_rl.reward import ScoreDeltaRewardTracker


def _executable_in(build: Path) -> Path:
    if build.suffix != ".app":
        return build
    candidates = [path for path in (build / "Contents" / "MacOS").iterdir() if path.is_file()]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one executable in app bundle, got {candidates}")
    return candidates[0]


def evaluate_episode(
    env: ContractBlackOutEnv,
    *,
    build: str | Path,
    game_commit: str,
    python_api_commit: str,
    seed: int,
    model_team: int,
    model_policy: RandomPolicy,
    opponent_policy: RandomPolicy,
    model_artifact: PolicyArtifact,
    opponent_artifact: PolicyArtifact,
    pair_id: str,
    pair_index: int,
    max_steps: int = 22_000,
    target_score: int = 100,
) -> dict[str, Any]:
    """Run one episode and return a validated PREP-10 episode record."""
    if model_team not in (0, 1):
        raise ValueError("model_team must be 0 or 1")
    opponent_team = 1 - model_team
    build_path = Path(build).expanduser().resolve()
    executable = _executable_in(build_path)
    obs, _ = env.reset(seed=seed)
    tracker = ScoreDeltaRewardTracker(target_score=target_score, terminal_win_reward=1.0)
    tracker.reset()
    unity_shaping_sum = {agent: 0.0 for agent in env.possible_agents}
    score_events = 0
    last_info: dict[str, float | int] = {"score_0": 0.0, "score_1": 0.0, "time_left": 1.0}
    terminal_info: dict[str, float | int] = {}
    steps = 0
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()

    while env.agents:
        if steps >= max_steps:
            raise RuntimeError(f"episode exceeded max_steps={max_steps}")
        model_agents = team_agents(model_team)
        opponent_agents = team_agents(opponent_team)
        actions = model_policy.act(obs, model_agents)
        actions.update(opponent_policy.act(obs, opponent_agents))
        obs, unity_rewards, terminations, truncations, infos = env.step(actions)
        steps += 1
        if any(truncations.values()):
            raise RuntimeError(f"unexpected truncation at step {steps}")
        for agent, reward in unity_rewards.items():
            unity_shaping_sum[agent] += float(reward)

        info = dict(next(iter(infos.values())))
        if any(terminations.values()):
            if not all(terminations.values()) or env.agents:
                raise RuntimeError("all agents must terminate simultaneously")
            terminal_info = info
            winner = int(terminal_info["winner"])
            tracker.update_normalized(terminated=True, winner=winner)
        else:
            event = tracker.update_normalized(float(info["score_0"]), float(info["score_1"]))
            if event.score_delta != (0, 0):
                score_events += 1
            last_info = info

        if steps % 5_000 == 0:
            print(
                f"pair={pair_id} game={pair_index} step={steps} "
                f"score=({last_info['score_0']:.2f},{last_info['score_1']:.2f}) "
                f"time_left={last_info['time_left']:.3f}",
                flush=True,
            )

    score_a, score_b = tracker.score
    winner = int(terminal_info["winner"])
    result = model_result(winner, model_team)
    model_score = score_a if model_team == 0 else score_b
    opponent_score = score_b if model_team == 0 else score_a
    last_time_left = float(last_info["time_left"])
    likely_early = last_time_left > 0.001
    if not likely_early and winner != winner_from_scores(score_a, score_b):
        raise RuntimeError(
            f"time-limit winner={winner} does not match observable score=({score_a},{score_b})"
        )

    record: dict[str, Any] = {
        "schema_version": EPISODE_SCHEMA_VERSION,
        "episode_id": str(uuid.uuid4()),
        "pair_id": pair_id,
        "pair_index": pair_index,
        "seed": seed,
        "policy_seeds": {
            "model": model_policy.seed,
            "opponent": opponent_policy.seed,
        },
        "side_assignment": {
            "model_team": model_team,
            "model_side": "A" if model_team == 0 else "B",
            "opponent_team": opponent_team,
            "opponent_side": "A" if opponent_team == 0 else "B",
        },
        "model": model_artifact.to_dict(),
        "opponent": opponent_artifact.to_dict(),
        "environment": {
            "build_path": str(build_path),
            "executable_path": str(executable),
            "executable_sha256": sha256_file(executable),
            "game_commit": game_commit,
            "python_api_commit": python_api_commit,
            "adapter": "blackout_rl.env.ContractBlackOutEnv",
        },
        "episode_length": {
            "steps": steps,
            "wall_seconds": round(time.perf_counter() - started, 3),
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "last_time_left_normalized": last_time_left,
        },
        "score": {
            "team_a": score_a,
            "team_b": score_b,
            "model": model_score,
            "opponent": opponent_score,
            "model_minus_opponent": model_score - opponent_score,
            "source": "last_nonterminal_info",
        },
        "winner": {
            "team": winner,
            "side": "draw" if winner == -1 else "A" if winner == 0 else "B",
            "source": "terminal_info.winner",
        },
        "model_result": result,
        "rewards": {
            "winner_source": "terminal_info.winner",
            "unity_shaping_used_for_winner": False,
            "unity_shaping_sum_by_agent": unity_shaping_sum,
            "score_delta_formula": "team=(delta_own-delta_opponent)/100",
            "score_events": score_events,
            "python_score_delta_cumulative": {
                "team_a": tracker.cumulative_competitive_reward[0],
                "team_b": tracker.cumulative_competitive_reward[1],
            },
            "python_terminal_bonus": {
                "team_a": tracker.cumulative_terminal_bonus[0],
                "team_b": tracker.cumulative_terminal_bonus[1],
            },
        },
        "termination": {
            "kind": "target_score_or_time_limit" if likely_early else "time_limit",
            "all_agents_terminated": True,
            "truncated": False,
            "terminal_frame_scores_normalized": {
                "team_a": float(terminal_info.get("score_0", 0.0)),
                "team_b": float(terminal_info.get("score_1", 0.0)),
            },
            "score_is_preterminal": True,
            "score_limitation": (
                "Unity resets score scalars on the terminal frame; a target-score-ending "
                "deposit may not be represented in the last observable score."
            ),
        },
    }
    validate_episode_log(record)
    return record


def summarize_episodes(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    if not episodes:
        raise ValueError("cannot summarize an empty episode list")
    wins = sum(episode["model_result"] == "win" for episode in episodes)
    draws = sum(episode["model_result"] == "draw" for episode in episodes)
    losses = sum(episode["model_result"] == "loss" for episode in episodes)
    score_diffs = [episode["score"]["model_minus_opponent"] for episode in episodes]
    by_side = {}
    for side in ("A", "B"):
        selected = [episode for episode in episodes if episode["side_assignment"]["model_side"] == side]
        by_side[side] = {
            "episodes": len(selected),
            "wins": sum(episode["model_result"] == "win" for episode in selected),
            "draws": sum(episode["model_result"] == "draw" for episode in selected),
            "losses": sum(episode["model_result"] == "loss" for episode in selected),
            "mean_model_score_diff": (
                sum(episode["score"]["model_minus_opponent"] for episode in selected) / len(selected)
                if selected
                else None
            ),
        }
    return {
        "episodes": len(episodes),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / len(episodes),
        "mean_model_score_diff": sum(score_diffs) / len(score_diffs),
        "by_model_side": by_side,
        "winner_derived_from_unity_shaping": False,
    }
