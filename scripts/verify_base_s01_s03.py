#!/usr/bin/env python3
"""Independently verify semantic battery decoding and live A* navigation."""

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
    AStarPlanner,
    ContractBlackOutEnv,
    GridCell,
    NoOpPolicy,
    PathNotFound,
    SemanticMapDecoder,
    WaypointFollower,
    canonical_agents,
    cell_center_normalized,
    grid_line,
    normalized_to_cell,
    parse_graphic,
    parse_vector,
    path_cost,
)


GAME_COMMIT = "d2220a7d01be88d413f551efd529f4758833be8b"
PYTHON_API_COMMIT = "6ba7d9993cf1bdefe1ed480c8efbcabcb923f539"


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


def unit_position(obs: dict, agent: str) -> np.ndarray:
    index = int(agent.split("_")[1])
    return parse_vector(obs[agent]["vector"]).positions[index].copy()


def unit_holding_item(obs: dict, agent: str) -> int:
    index = int(agent.split("_")[1])
    return int(parse_vector(obs[agent]["vector"]).holding_item_ids[index])


def step_noop(env: ContractBlackOutEnv, obs: dict, policy: NoOpPolicy):
    return env.step(policy.act(obs, env.agents))


def ready_observation(
    env: ContractBlackOutEnv,
    seed: int,
    policy: NoOpPolicy,
) -> tuple[dict, int]:
    obs, _ = env.reset(seed=seed)
    for wait_steps in range(21):
        graphic = obs["unit_0"]["graphic"]
        if graphic.max() > 0.0:
            parse_graphic(graphic)
            decoded = SemanticMapDecoder().decode(graphic)
            if decoded.cells("battery"):
                return obs, wait_steps
        obs, _, terminations, truncations, _ = step_noop(env, obs, policy)
        if any(terminations.values()) or any(truncations.values()):
            raise RuntimeError("episode ended while waiting for semantic map")
    raise RuntimeError("semantic map with batteries was not ready within 20 steps")


def choose_detour_target(
    start: GridCell,
    battery_cells: tuple[GridCell, ...],
    planner: AStarPlanner,
) -> tuple[GridCell, tuple[GridCell, ...], tuple[GridCell, ...], tuple[GridCell, ...]] | None:
    candidates = []
    for target in battery_cells:
        try:
            path = planner.plan(start, target)
        except PathNotFound:
            continue
        direct = grid_line(start, target)
        direct_walls = tuple(
            cell
            for cell in direct[1:-1]
            if not planner.walkable_grid[cell.y, cell.x]
        )
        if direct_walls:
            candidates.append((path_cost(path), target, path, direct, direct_walls))
    if not candidates:
        return None
    _, target, path, direct, direct_walls = min(
        candidates,
        key=lambda item: (item[0], item[1].x, item[1].y),
    )
    return target, path, direct, direct_walls


def coordinate_calibration(decoded, position: np.ndarray) -> dict:
    unit_points = decoded.points("ally_unit")
    if not unit_points:
        raise AssertionError("semantic map has no ally unit pixels")
    nearest = min(
        unit_points,
        key=lambda point: float(
            np.linalg.norm(np.asarray(point.normalized_xy, dtype=np.float32) - position)
        ),
    )
    error = float(
        np.linalg.norm(np.asarray(nearest.normalized_xy, dtype=np.float32) - position)
    )
    error_pixels = error * max(decoded.width_pixels, decoded.height_pixels)
    if error_pixels > 1.5:
        raise AssertionError(
            f"vector/semantic coordinate mismatch: error={error_pixels:.3f} pixels"
        )
    return {
        "vector_position_normalized": position.tolist(),
        "nearest_ally_unit_pixel": [nearest.row, nearest.column],
        "nearest_ally_unit_position_normalized": list(nearest.normalized_xy),
        "euclidean_error_normalized": error,
        "euclidean_error_pixels": error_pixels,
        "passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=210301)
    parser.add_argument("--seed-search", type=int, default=20)
    parser.add_argument("--agent", default="unit_0", choices=canonical_agents())
    parser.add_argument("--time-scale", type=float, default=50.0)
    parser.add_argument("--max-navigation-steps", type=int, default=3_000)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    build = args.build.expanduser().resolve()
    executable = executable_in(build)
    if not executable.is_file():
        raise FileNotFoundError(executable)
    if args.agent not in canonical_agents()[:5]:
        raise ValueError("live navigation verification currently targets Team A")

    no_op = NoOpPolicy(seed=0)
    decoder = SemanticMapDecoder()
    env = ContractBlackOutEnv(
        env_path=str(build),
        no_graphics=False,
        time_scale=args.time_scale,
    )
    selected = None
    try:
        for offset in range(args.seed_search):
            seed = args.seed + offset
            obs, map_wait_steps = ready_observation(env, seed, no_op)
            decoded = decoder.decode(obs[args.agent]["graphic"])
            start_position = unit_position(obs, args.agent)
            start_cell = normalized_to_cell(
                start_position,
                width_cells=decoded.width_cells,
                height_cells=decoded.height_cells,
            )
            planner = AStarPlanner(decoded.walkable_grid)
            detour = choose_detour_target(start_cell, decoded.cells("battery"), planner)
            if detour is not None:
                selected = (
                    seed,
                    obs,
                    map_wait_steps,
                    decoded,
                    start_position,
                    start_cell,
                    planner,
                    detour,
                )
                break
        if selected is None:
            raise RuntimeError(
                f"no reachable battery behind a wall in seeds {args.seed}.."
                f"{args.seed + args.seed_search - 1}"
            )

        (
            seed,
            obs,
            map_wait_steps,
            decoded,
            start_position,
            start_cell,
            planner,
            detour,
        ) = selected
        target, initial_path, direct_path, direct_walls = detour
        calibration = coordinate_calibration(decoded, start_position)
        initial_positions = {
            agent: unit_position(obs, agent) for agent in canonical_agents()
        }
        follower = WaypointFollower(
            decoded.width_cells,
            decoded.height_cells,
            tolerance=0.004,
            stall_limit=30,
        )
        follower.set_path(initial_path)
        current_path = initial_path
        replans = []
        collision_cells: set[GridCell] = set()
        sampled_trajectory = []
        visited_cells = []
        started_at = datetime.now(timezone.utc)
        started = time.perf_counter()
        pickup_step = None

        for navigation_step in range(1, args.max_navigation_steps + 1):
            position = unit_position(obs, args.agent)
            current_cell = follower.current_cell(position)
            visited_cells.append(current_cell)
            if not decoded.walkable_grid[current_cell.y, current_cell.x]:
                raise AssertionError(f"unit entered decoded wall cell {current_cell}")
            action = follower.action(position)
            actions = no_op.act(obs, env.agents)
            actions[args.agent] = action
            obs, _, terminations, truncations, infos = env.step(actions)
            if any(truncations.values()):
                raise AssertionError("unexpected truncation during navigation")
            if any(terminations.values()):
                raise AssertionError("episode terminated before battery pickup")
            if navigation_step == 1 or navigation_step % 10 == 0:
                sampled_trajectory.append(
                    {
                        "step": navigation_step,
                        "position_normalized": position.tolist(),
                        "cell": [current_cell.x, current_cell.y],
                        "action": action.tolist(),
                        "waypoint": (
                            None
                            if follower.current_waypoint is None
                            else [follower.current_waypoint.x, follower.current_waypoint.y]
                        ),
                    }
                )
            if unit_holding_item(obs, args.agent) == 1:
                pickup_step = navigation_step
                break
            if follower.replan_required:
                position = unit_position(obs, args.agent)
                replan_start = follower.current_cell(position)
                collided_cell = follower.current_waypoint
                if collided_cell is not None and collided_cell != target:
                    collision_cells.add(collided_cell)
                current_path = planner.plan_avoiding(
                    replan_start,
                    target,
                    collision_cells,
                )
                follower.set_path(current_path)
                replans.append(
                    {
                        "step": navigation_step,
                        "start": [replan_start.x, replan_start.y],
                        "path_length": len(current_path),
                        "reason": "no_position_progress",
                        "temporarily_blocked_cells": [
                            [cell.x, cell.y] for cell in sorted(collision_cells)
                        ],
                    }
                )

        if pickup_step is None:
            raise AssertionError(
                f"{args.agent} did not pick up target battery within "
                f"{args.max_navigation_steps} steps"
            )

        end_position = unit_position(obs, args.agent)
        end_cell = follower.current_cell(end_position)
        final_positions = {
            agent: unit_position(obs, agent) for agent in canonical_agents()
        }
        non_target_displacements = {
            agent: float(np.linalg.norm(final_positions[agent] - initial_positions[agent]))
            for agent in canonical_agents()
            if agent != args.agent
        }
        max_non_target_displacement = max(non_target_displacements.values())
        single_unit_motion_passed = max_non_target_displacement <= 1e-7
        if not single_unit_motion_passed:
            raise AssertionError(
                "a NoOp-controlled unit moved: "
                f"max displacement={max_non_target_displacement}"
            )
        path_is_wall_free = all(
            decoded.walkable_grid[cell.y, cell.x] for cell in initial_path
        )
        obstacle_avoidance_passed = bool(
            direct_walls and path_is_wall_free and unit_holding_item(obs, args.agent) == 1
        )
        if not obstacle_avoidance_passed:
            raise AssertionError("obstacle-avoidance verification did not pass")

        result = {
            "schema_version": "blackout.base_s01_s03_navigation.v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "environment": {
                "seed": seed,
                "requested_seed_start": args.seed,
                "build_path": str(build),
                "executable_path": str(executable),
                "executable_sha256": sha256_file(executable),
                "game_commit": GAME_COMMIT,
                "python_api_commit": PYTHON_API_COMMIT,
                "adapter": "blackout_rl.env.ContractBlackOutEnv",
                "time_scale": args.time_scale,
                "map_ready_after_steps": map_wait_steps,
            },
            "semantic_decode": {
                "resolution_scale": decoded.resolution_scale,
                "pixel_shape": [decoded.height_pixels, decoded.width_pixels],
                "grid_shape": [decoded.height_cells, decoded.width_cells],
                "wall_cells": len(decoded.cells("wall")),
                "ally_storage_cells": len(decoded.cells("ally_storage")),
                "enemy_storage_cells": len(decoded.cells("enemy_storage")),
                "battery_cells": len(decoded.cells("battery")),
                "special_item_cells": sum(
                    len(decoded.cells(name))
                    for name in (
                        "buff_speed",
                        "debuff_speed",
                        "buff_size",
                        "debuff_size",
                    )
                ),
                "coordinate_calibration": calibration,
            },
            "plan": {
                "agent": args.agent,
                "start_position_normalized": start_position.tolist(),
                "start_cell": [start_cell.x, start_cell.y],
                "target_battery_cell": [target.x, target.y],
                "target_center_normalized": cell_center_normalized(
                    target,
                    width_cells=decoded.width_cells,
                    height_cells=decoded.height_cells,
                ).tolist(),
                "direct_grid_line": [[cell.x, cell.y] for cell in direct_path],
                "direct_line_blocked_cells": [[cell.x, cell.y] for cell in direct_walls],
                "astar_path": [[cell.x, cell.y] for cell in initial_path],
                "astar_path_cells": len(initial_path),
                "astar_path_cost": path_cost(initial_path),
                "astar_path_wall_free": path_is_wall_free,
                "safe_diagonal_corner_rule": True,
            },
            "execution": {
                "started_at_utc": started_at.isoformat(),
                "wall_seconds": round(time.perf_counter() - started, 3),
                "navigation_steps": pickup_step,
                "pickup_item_id": unit_holding_item(obs, args.agent),
                "battery_pickup_confirmed": True,
                "end_position_normalized": end_position.tolist(),
                "end_cell": [end_cell.x, end_cell.y],
                "unique_visited_cells": len(set(visited_cells)),
                "visited_wall_cells": 0,
                "non_target_policy": "NoOpPolicy",
                "max_non_target_displacement_normalized": max_non_target_displacement,
                "non_target_displacement_by_agent": non_target_displacements,
                "single_unit_motion_passed": single_unit_motion_passed,
                "replans": replans,
                "final_active_path": [[cell.x, cell.y] for cell in current_path],
                "sampled_trajectory": sampled_trajectory,
            },
            "verification": {
                "semantic_coordinate_alignment_passed": calibration["passed"],
                "direct_route_intersects_wall": bool(direct_walls),
                "planned_route_avoids_walls": path_is_wall_free,
                "battery_pickup_confirmed": True,
                "single_unit_motion_passed": single_unit_motion_passed,
                "obstacle_avoidance_passed": obstacle_avoidance_passed,
            },
        }
    finally:
        env.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["verification"], ensure_ascii=False, indent=2))
    print(f"wrote live navigation evidence to {args.output}")


if __name__ == "__main__":
    main()
