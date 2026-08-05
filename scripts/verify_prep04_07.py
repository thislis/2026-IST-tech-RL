#!/usr/bin/env python3
"""Run the live-Unity PREP-04~07 contract, identity, and action experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from blackout_rl import (
    ContractBlackOutEnv,
    GRAPHIC_CHANNEL_NAMES,
    VECTOR_SIZE,
    canonical_agents,
    parse_graphic,
    parse_vector,
    stack_observations,
    team_agents,
)


STATIC_MAP_CHANNELS = (1, 2, 3)
NOOP = np.zeros(2, dtype=np.float32)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def executable_in(build: Path) -> Path:
    if build.suffix != ".app":
        return build
    candidates = [path for path in (build / "Contents" / "MacOS").iterdir() if path.is_file()]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one executable in app bundle, got {candidates}")
    return candidates[0]


def noop_actions(agents: list[str]) -> dict[str, np.ndarray]:
    return {agent: NOOP.copy() for agent in agents}


def actions_for(env: ContractBlackOutEnv, target: str, action: np.ndarray) -> dict[str, np.ndarray]:
    actions = noop_actions(env.agents)
    actions[target] = np.asarray(action, dtype=np.float32)
    return actions


def assert_live_observation(obs: dict[str, dict[str, np.ndarray]]) -> None:
    expected = set(canonical_agents())
    if set(obs) != expected:
        raise AssertionError(f"expected all agents, got {list(obs)}")
    for agent in canonical_agents():
        vector = obs[agent]["vector"]
        graphic = obs[agent]["graphic"]
        parse_vector(vector)
        parse_graphic(graphic)
        if vector.shape != (96,) or vector.dtype != np.float32:
            raise AssertionError(f"bad vector contract for {agent}: {vector.shape}/{vector.dtype}")
        if graphic.shape != (96, 96, 11) or graphic.dtype != np.float32:
            raise AssertionError(f"bad graphic contract for {agent}: {graphic.shape}/{graphic.dtype}")


def step_noop(env: ContractBlackOutEnv):
    return env.step(noop_actions(env.agents))


def reset_with_ready_map(env: ContractBlackOutEnv, seed: int) -> tuple[dict, dict, int]:
    obs, infos = env.reset(seed=seed)
    for wait_steps in range(11):
        graphic = obs["unit_0"]["graphic"]
        if graphic.max() > 0.0 and np.all(graphic.sum(axis=-1) == 1.0):
            return obs, infos, wait_steps
        obs, _, terminations, truncations, _ = step_noop(env)
        if any(terminations.values()) or any(truncations.values()):
            raise AssertionError("episode ended while waiting for semantic map")
    raise AssertionError("semantic map did not become ready within 10 steps")


def decision_order(env: ContractBlackOutEnv) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for behavior in env._resolve_behavior_names():
        decision_steps, _ = env._unity_env.get_steps(behavior)
        rows = []
        for row, agent_id in enumerate(decision_steps.agent_id):
            agent, _ = env._extract_step(decision_steps, agent_id)
            rows.append({"row": row, "ml_agent_id": int(agent_id), "agent": agent})
        result.append({"behavior": behavior, "rows": rows})
    return result


def snapshot(env: ContractBlackOutEnv, seed: int) -> tuple[dict, dict[str, object]]:
    obs, _, wait_steps = reset_with_ready_map(env, seed)
    batch = stack_observations(obs)
    metadata = {
        "seed": seed,
        "map_ready_after_steps": wait_steps,
        "raw_obs_order": list(obs),
        "env_agents_order": list(env.agents),
        "canonical_batch_order": list(batch.agent_names),
        "decision_rows": decision_order(env),
    }
    return obs, metadata


def position(obs: dict[str, dict[str, np.ndarray]], target_index: int) -> np.ndarray:
    return parse_vector(obs["unit_0"]["vector"]).positions[target_index].copy()


def move_once(
    env: ContractBlackOutEnv,
    obs: dict[str, dict[str, np.ndarray]],
    target: str,
    action: np.ndarray,
) -> tuple[dict, dict[str, object]]:
    target_index = int(target.split("_")[1])
    before = position(obs, target_index)
    next_obs, _, terminations, truncations, _ = env.step(actions_for(env, target, action))
    if any(terminations.values()) or any(truncations.values()):
        raise AssertionError("episode ended during displacement measurement")
    after = position(next_obs, target_index)
    delta = after - before
    magnitude = float(np.linalg.norm(delta))
    action_array = np.asarray(action, dtype=np.float32)
    action_norm = float(np.linalg.norm(action_array))
    cosine = None
    if magnitude > 0.0 and action_norm > 0.0:
        cosine = float(np.dot(delta, action_array) / (magnitude * action_norm))
    return next_obs, {
        "action": action_array.tolist(),
        "before": before.tolist(),
        "after": after.tolist(),
        "delta": delta.tolist(),
        "magnitude": magnitude,
        "direction_cosine": cosine,
    }


def return_from_move(
    env: ContractBlackOutEnv,
    obs: dict[str, dict[str, np.ndarray]],
    target: str,
    action: np.ndarray,
) -> dict:
    opposite = -np.asarray(action, dtype=np.float32)
    if np.linalg.norm(opposite) == 0.0:
        return obs
    obs, _ = move_once(env, obs, target, opposite)
    return obs


def action_experiment(env: ContractBlackOutEnv, seed: int) -> dict[str, object]:
    obs, _, _ = reset_with_ready_map(env, seed)
    target = "unit_0"

    # Move away from the corner spawn into an open interior area. Stop as soon
    # as both axes have moved far enough to leave clearance for all 8 probes.
    warmup_steps = 0
    start = position(obs, 0)
    while warmup_steps < 100:
        current = position(obs, 0)
        if current[0] >= start[0] + 0.12 and current[1] <= start[1] - 0.12:
            break
        obs, _ = move_once(env, obs, target, np.array([1.0, -1.0], dtype=np.float32))
        warmup_steps += 1
    anchor = position(obs, 0)
    if warmup_steps == 100:
        raise AssertionError(f"failed to move unit_0 away from spawn; start={start}, anchor={anchor}")

    obs, zero = move_once(env, obs, target, np.array([0.0, 0.0], dtype=np.float32))
    if zero["magnitude"] > 1e-7:
        raise AssertionError(f"zero action moved the unit: {zero}")

    speed_probes: dict[str, dict[str, object]] = {}
    for label, action in (
        ("small", np.array([0.1, 0.0], dtype=np.float32)),
        ("large", np.array([1.0, 0.0], dtype=np.float32)),
        ("oversized", np.array([2.0, 0.0], dtype=np.float32)),
    ):
        obs, measurement = move_once(env, obs, target, action)
        speed_probes[label] = measurement
        obs = return_from_move(env, obs, target, action)

    small_mag = float(speed_probes["small"]["magnitude"])
    large_mag = float(speed_probes["large"]["magnitude"])
    oversized_mag = float(speed_probes["oversized"]["magnitude"])
    if min(small_mag, large_mag, oversized_mag) <= 1e-6:
        raise AssertionError(f"non-zero action failed to move unit: {speed_probes}")
    small_large_equal = bool(np.isclose(small_mag, large_mag, rtol=0.03, atol=1e-7))
    oversized_clipped = bool(np.isclose(oversized_mag, large_mag, rtol=0.03, atol=1e-7))
    if not small_large_equal or not oversized_clipped:
        raise AssertionError(f"action normalization/clipping mismatch: {speed_probes}")

    directions = {
        "east": (1.0, 0.0),
        "north_east": (1.0, 1.0),
        "north": (0.0, 1.0),
        "north_west": (-1.0, 1.0),
        "west": (-1.0, 0.0),
        "south_west": (-1.0, -1.0),
        "south": (0.0, -1.0),
        "south_east": (1.0, -1.0),
    }
    direction_results: dict[str, dict[str, object]] = {}
    for label, values in directions.items():
        action = np.asarray(values, dtype=np.float32)
        obs, measurement = move_once(env, obs, target, action)
        direction_results[label] = measurement
        obs = return_from_move(env, obs, target, action)

    magnitudes = np.asarray([result["magnitude"] for result in direction_results.values()])
    cosines = np.asarray([result["direction_cosine"] for result in direction_results.values()])
    eight_match = bool(
        np.all(cosines >= 0.99)
        and np.allclose(magnitudes, np.median(magnitudes), rtol=0.06, atol=1e-7)
    )
    if not eight_match:
        raise AssertionError(f"8-direction movement mismatch: {direction_results}")

    return {
        "seed": seed,
        "target_agent": target,
        "start_position_normalized": start.tolist(),
        "anchor_position_normalized": anchor.tolist(),
        "warmup_steps": warmup_steps,
        "zero": zero,
        "speed_probes": speed_probes,
        "small_large_equal_speed": small_large_equal,
        "oversized_action_clipped": oversized_clipped,
        "directions": direction_results,
        "eight_directions_match": eight_match,
    }


def run_to_terminal(env: ContractBlackOutEnv, seed: int, policy_seed: int, max_steps: int) -> dict[str, object]:
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(policy_seed)
    expected_agents = set(canonical_agents())
    raw_orders: set[tuple[str, ...]] = set()
    last_preterminal_info: dict[str, float | int] = {}
    steps = 0
    started = time.perf_counter()
    while env.agents:
        if steps >= max_steps:
            raise AssertionError(f"episode exceeded max_steps={max_steps}")
        actions = {
            agent: rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
            for agent in env.agents
        }
        obs, rewards, terminations, truncations, infos = env.step(actions)
        steps += 1
        raw_orders.add(tuple(obs))
        for mapping_name, mapping in (
            ("obs", obs),
            ("rewards", rewards),
            ("terminations", terminations),
            ("truncations", truncations),
            ("infos", infos),
        ):
            if set(mapping) != expected_agents:
                raise AssertionError(f"{mapping_name} lost agents at step {steps}: {list(mapping)}")
        if any(truncations.values()):
            raise AssertionError(f"unexpected truncation at step {steps}")
        if any(terminations.values()):
            if not all(terminations.values()) or env.agents:
                raise AssertionError("agents did not terminate simultaneously")
            terminal_info = dict(next(iter(infos.values())))
            terminal_batch = stack_observations(obs)
            if terminal_batch.agent_names != canonical_agents():
                raise AssertionError("canonical terminal batch order changed")
        else:
            if any(terminations.values()):
                raise AssertionError("partial termination observed")
            last_preterminal_info = dict(next(iter(infos.values())))
        if steps % 5_000 == 0:
            info = next(iter(infos.values()))
            print(
                f"terminal-run step={steps} "
                f"score=({info.get('score_0', 0.0):.2f},{info.get('score_1', 0.0):.2f}) "
                f"time_left={info.get('time_left', 0.0):.3f}",
                flush=True,
            )

    score_a = int(round(float(last_preterminal_info["score_0"]) * 100))
    score_b = int(round(float(last_preterminal_info["score_1"]) * 100))
    winner = int(terminal_info["winner"])
    expected_winner = 0 if score_a > score_b else 1 if score_b > score_a else -1
    winner_matches = winner == expected_winner
    if not winner_matches:
        raise AssertionError(f"winner={winner} does not match score=({score_a},{score_b})")

    return {
        "seed": seed,
        "policy_seed": policy_seed,
        "steps": steps,
        "wall_seconds": round(time.perf_counter() - started, 3),
        "score": {"team_a": score_a, "team_b": score_b},
        "winner": winner,
        "winner_label": {0: "team_a", 1: "team_b", -1: "draw"}[winner],
        "winner_matches_score": winner_matches,
        "terminal_frame_scores_normalized": {
            "team_a": float(terminal_info["score_0"]),
            "team_b": float(terminal_info["score_1"]),
        },
        "raw_obs_orders_seen": [list(order) for order in sorted(raw_orders)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--time-scale", type=float, default=50.0)
    parser.add_argument("--max-steps", type=int, default=22_000)
    args = parser.parse_args()

    build = args.build.expanduser().resolve()
    executable = executable_in(build)
    generated_at = datetime.now(timezone.utc)
    env = ContractBlackOutEnv(env_path=str(build), no_graphics=False, time_scale=args.time_scale)
    try:
        first_obs, first_meta = snapshot(env, 40407)
        assert_live_observation(first_obs)
        first_batch = stack_observations(first_obs)

        # A real parallel step: all 10 action, observation, reward, done, and info keys.
        stepped_obs, rewards, terminations, truncations, infos = step_noop(env)
        assert_live_observation(stepped_obs)
        expected = set(canonical_agents())
        step_ok = all(
            set(mapping) == expected
            for mapping in (stepped_obs, rewards, terminations, truncations, infos)
        ) and not any(terminations.values()) and not any(truncations.values())
        if not step_ok:
            raise AssertionError("PettingZoo parallel step contract failed")

        second_obs, second_meta = snapshot(env, 40407)
        third_obs, third_meta = snapshot(env, 40408)
        first_static = first_obs["unit_0"]["graphic"][..., STATIC_MAP_CHANNELS]
        second_static = second_obs["unit_0"]["graphic"][..., STATIC_MAP_CHANNELS]
        third_static = third_obs["unit_0"]["graphic"][..., STATIC_MAP_CHANNELS]
        same_seed_equal = bool(np.array_equal(first_static, second_static))
        if not same_seed_equal:
            raise AssertionError("same seed did not reproduce static map channels")

        parsed_a = parse_vector(first_obs["unit_0"]["vector"])
        parsed_b = parse_vector(first_obs["unit_5"]["vector"])
        map_a = first_obs["unit_0"]["graphic"]
        map_b = first_obs["unit_5"]["graphic"]
        team_swap = bool(
            np.array_equal(parsed_a.positions, parsed_b.positions)
            and np.array_equal(parsed_a.team_signs, -parsed_b.team_signs)
            and np.array_equal(map_a[..., 2], map_b[..., 3])
            and np.array_equal(map_a[..., 3], map_b[..., 2])
            and np.array_equal(map_a[..., 4], map_b[..., 5])
            and np.array_equal(map_a[..., 5], map_b[..., 4])
        )
        if not team_swap:
            raise AssertionError("Team A/B perspective swap contract failed")

        team_a_identical = all(
            np.array_equal(first_obs["unit_0"]["vector"], first_obs[agent]["vector"])
            and np.array_equal(first_obs["unit_0"]["graphic"], first_obs[agent]["graphic"])
            for agent in team_agents(0)[1:]
        )
        team_b_identical = all(
            np.array_equal(first_obs["unit_5"]["vector"], first_obs[agent]["vector"])
            and np.array_equal(first_obs["unit_5"]["graphic"], first_obs[agent]["graphic"])
            for agent in team_agents(1)[1:]
        )
        canonical_stable = all(
            meta["canonical_batch_order"] == list(canonical_agents())
            and meta["env_agents_order"] == list(canonical_agents())
            for meta in (first_meta, second_meta, third_meta)
        )
        if not canonical_stable:
            raise AssertionError("canonical agent order changed across reset")

        action_results = action_experiment(env, 40707)
        terminal = run_to_terminal(env, seed=20260805, policy_seed=20260805, max_steps=args.max_steps)

        evidence = {
            "schema_version": 1,
            "generated_at_utc": generated_at.isoformat(),
            "build": {
                "path": str(build),
                "executable_path": str(executable),
                "executable_sha256": sha256_file(executable),
            },
            "observation_contract": {
                "vector_shape": list(first_batch.vectors.shape[1:]),
                "vector_dtype": str(first_batch.vectors.dtype),
                "graphic_hwc_shape": list(first_batch.graphics_hwc.shape[1:]),
                "graphic_chw_shape": list(first_batch.graphics_chw.shape[1:]),
                "graphic_dtype": str(first_batch.graphics_hwc.dtype),
                "channel_names": list(GRAPHIC_CHANNEL_NAMES),
                "field_validation_passed": True,
            },
            "pettingzoo_contract": {
                "reset_ok": set(first_obs) == expected and all(info == {} for info in env.reset(seed=40409)[1].values()),
                "step_ok": step_ok,
                "termination_ok": terminal["steps"] > 0,
                "no_truncation": True,
                "terminal_agent_count": 10,
                "observation_space_contains_reset": all(
                    env.observation_space(agent).contains(first_obs[agent])
                    for agent in canonical_agents()
                ),
                "action_shape": [2],
                "action_range": [-1.0, 1.0],
            },
            "seed_reproducibility": {
                "same_seed": 40407,
                "different_seed": 40408,
                "static_channels": ["wall", "ally_storage", "enemy_storage"],
                "same_seed_equal": same_seed_equal,
                "different_seed_equal": bool(np.array_equal(first_static, third_static)),
            },
            "agent_identity_and_order": {
                "canonical_agents": list(canonical_agents()),
                "team_a_agents": list(team_agents(0)),
                "team_b_agents": list(team_agents(1)),
                "team_batch_sizes": [len(team_agents(0)), len(team_agents(1))],
                "canonical_batch_stable": canonical_stable,
                "raw_obs_order_stable_across_resets": len({tuple(meta["raw_obs_order"]) for meta in (first_meta, second_meta, third_meta)}) == 1,
                "reset_samples": [first_meta, second_meta, third_meta],
                "team_perspective_swap_ok": team_swap,
                "same_team_initial_observations_identical": {
                    "team_a": team_a_identical,
                    "team_b": team_b_identical,
                },
                "self_id_present_in_policy_observation": False,
                "self_id_resolution": "Use PettingZoo agent name and team-local canonical slot_id 0..4 from blackout_rl.batching.",
            },
            "action_semantics": action_results,
            "terminal": terminal,
        }
    finally:
        env.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(evidence, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
