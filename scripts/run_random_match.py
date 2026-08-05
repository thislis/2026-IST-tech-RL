#!/usr/bin/env python3
"""Run one seeded random-vs-random BlackOut match and save a JSON result."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from blackout_env import BlackOutEnv


TARGET_SCORE = 100


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--policy-seed", type=int, default=20260805)
    parser.add_argument("--time-scale", type=float, default=50.0)
    parser.add_argument("--max-steps", type=int, default=30_000)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    build = args.build.expanduser().resolve()
    if not build.exists():
        raise FileNotFoundError(build)

    if build.suffix == ".app":
        macos_dir = build / "Contents" / "MacOS"
        candidates = [path for path in macos_dir.iterdir() if path.is_file()]
        if len(candidates) != 1:
            raise RuntimeError(f"expected one executable in {macos_dir}, got {candidates}")
        executable = candidates[0]
    else:
        executable = build
    if not executable.is_file():
        raise FileNotFoundError(f"Unity executable not found: {executable}")

    rng = np.random.default_rng(args.policy_seed)
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    total_rewards: dict[str, float] = {}
    terminal_info: dict[str, float | int] = {}
    last_preterminal_info: dict[str, float | int] = {}
    steps = 0

    # On macOS, ML-Agents expects the .app bundle path and resolves the
    # Contents/MacOS executable itself. Keep ``executable`` for hashing only.
    env = BlackOutEnv(
        env_path=str(build),
        no_graphics=False,
        time_scale=args.time_scale,
    )
    try:
        obs, _ = env.reset(seed=args.seed)
        total_rewards = {agent: 0.0 for agent in env.possible_agents}

        while env.agents:
            if steps >= args.max_steps:
                raise RuntimeError(f"episode exceeded max_steps={args.max_steps}")

            actions = {
                agent: rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
                for agent in env.agents
            }
            obs, rewards, terminations, truncations, infos = env.step(actions)
            if any(truncations.values()):
                raise RuntimeError("unexpected truncation: BlackOut episodes must terminate")

            for agent, reward in rewards.items():
                total_rewards[agent] += float(reward)
            steps += 1

            if any(terminations.values()):
                terminal_info = dict(next(iter(infos.values())))
            else:
                # The Unity scene starts the next episode immediately after
                # EndEpisode(), so score fields on the terminal frame are reset
                # to zero. Preserve the final pre-terminal score observation.
                last_preterminal_info = dict(next(iter(infos.values())))

            if steps % 5_000 == 0:
                info = next(iter(infos.values()))
                print(
                    f"step={steps} score=({info.get('score_0', 0.0):.2f},"
                    f" {info.get('score_1', 0.0):.2f}) "
                    f"time_left={info.get('time_left', 0.0):.3f}",
                    flush=True,
                )
    finally:
        env.close()

    terminal_scores = (
        float(terminal_info.get("score_0", 0.0)),
        float(terminal_info.get("score_1", 0.0)),
    )
    if terminal_scores == (0.0, 0.0) and last_preterminal_info:
        score_info = last_preterminal_info
        score_source = "last_preterminal_info"
    else:
        score_info = terminal_info
        score_source = "terminal_info"

    score_a_norm = float(score_info["score_0"])
    score_b_norm = float(score_info["score_1"])
    score_a = int(round(score_a_norm * TARGET_SCORE))
    score_b = int(round(score_b_norm * TARGET_SCORE))
    winner = int(terminal_info["winner"])
    expected_winner = 0 if score_a > score_b else 1 if score_b > score_a else -1
    if winner != expected_winner:
        raise AssertionError(
            f"winner/score mismatch: winner={winner}, scores=({score_a}, {score_b})"
        )

    result = {
        "schema_version": 1,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "build": {
            "path": str(build),
            "executable_path": str(executable),
            "executable_sha256": sha256_file(executable),
        },
        "seed": args.seed,
        "policy_seed": args.policy_seed,
        "time_scale": args.time_scale,
        "episode_steps": steps,
        "wall_seconds": round(time.perf_counter() - started, 3),
        "score": {
            "team_a": score_a,
            "team_b": score_b,
            "team_a_normalized": score_a_norm,
            "team_b_normalized": score_b_norm,
            "source": score_source,
            "terminal_frame_normalized": {
                "team_a": terminal_scores[0],
                "team_b": terminal_scores[1],
            },
        },
        "winner": winner,
        "winner_label": {0: "team_a", 1: "team_b", -1: "draw"}[winner],
        "total_reward_by_agent": total_rewards,
        "terminal_verified_against_score": True,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
