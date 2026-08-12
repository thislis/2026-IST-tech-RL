#!/usr/bin/env python3
"""Live absorption-cycle and danger-route verification for BASE-S11~S12."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from blackout_rl import (
    AStarPlanner,
    ContractBlackOutEnv,
    GridCell,
    NoOpPolicy,
    Role,
    ScriptedTeamController,
    SemanticMapDecoder,
    WaypointFollower,
    WeightedAStarPlanner,
    battery_cells_from_graphic,
    build_danger_map,
    normalized_to_cell,
    parse_vector,
    path_cost,
    team_agents,
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


def ready_observation(env, seed, no_op):
    observations, _ = env.reset(seed=seed)
    decoder = SemanticMapDecoder()
    for waited in range(21):
        decoded = decoder.decode(observations["unit_0"]["graphic"])
        if decoded.cells("battery"):
            return observations, decoded, waited
        observations, _, terminations, truncations, _ = env.step(
            no_op.act(observations, env.agents)
        )
        if any(terminations.values()) or any(truncations.values()):
            raise RuntimeError("episode ended while waiting for map")
    raise RuntimeError("semantic map unavailable")


def choose_staging_target(planner: AStarPlanner, start: GridCell) -> tuple[GridCell, tuple[GridCell, ...]]:
    candidates = (
        GridCell(13, 10), GridCell(10, 13), GridCell(14, 12), GridCell(12, 14),
        GridCell(9, 11), GridCell(11, 9),
    )
    routes = []
    for target in candidates:
        try:
            route = planner.plan(start, target)
        except Exception:
            continue
        routes.append((path_cost(route), target.x, target.y, target, route))
    if not routes:
        raise RuntimeError("no central staging target reachable by enemy unit")
    _, _, _, target, route = min(routes)
    return target, route


def danger_route_audit(
    planner: AStarPlanner,
    enemy: GridCell,
) -> dict:
    danger = build_danger_map(planner.walkable_grid, (enemy,), Role.WORKER)
    weighted = WeightedAStarPlanner(planner.walkable_grid, danger.traversal_multiplier)
    height, width = planner.walkable_grid.shape
    ring = [
        GridCell(x, y)
        for y in range(height)
        for x in range(width)
        if planner.walkable_grid[y, x]
        and 4 <= max(abs(x - enemy.x), abs(y - enemy.y)) <= 7
    ]
    for start in ring:
        for goal in reversed(ring):
            first_vector = (start.x - enemy.x, start.y - enemy.y)
            second_vector = (goal.x - enemy.x, goal.y - enemy.y)
            if first_vector[0] * second_vector[0] + first_vector[1] * second_vector[1] >= 0:
                continue
            try:
                geometric = planner.plan(start, goal)
                safer = weighted.plan(start, goal)
            except Exception:
                continue
            geometric_cost = weighted.path_cost(geometric)
            safer_cost = weighted.path_cost(safer)
            if geometric != safer and safer_cost + 1e-6 < geometric_cost:
                return {
                    "enemy_cell": [enemy.x, enemy.y],
                    "start": [start.x, start.y],
                    "goal": [goal.x, goal.y],
                    "geometric_path": [[cell.x, cell.y] for cell in geometric],
                    "worker_weighted_path": [[cell.x, cell.y] for cell in safer],
                    "geometric_cells": len(geometric),
                    "worker_weighted_cells": len(safer),
                    "geometric_path_worker_weighted_cost": geometric_cost,
                    "worker_weighted_cost": safer_cost,
                    "geometric_min_enemy_distance": min(
                        max(abs(cell.x - enemy.x), abs(cell.y - enemy.y))
                        for cell in geometric
                    ),
                    "weighted_min_enemy_distance": min(
                        max(abs(cell.x - enemy.x), abs(cell.y - enemy.y))
                        for cell in safer
                    ),
                }
    raise AssertionError("could not find a live-map route changed by enemy danger cost")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=241112)
    parser.add_argument("--time-scale", type=float, default=50.0)
    parser.add_argument("--max-steps", type=int, default=1_500)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    build = args.build.expanduser().resolve()
    executable = executable_in(build)
    output = args.output.resolve()
    no_op = NoOpPolicy()
    controller = ScriptedTeamController(0, seed=args.seed)
    env = ContractBlackOutEnv(
        env_path=str(build), no_graphics=False, time_scale=args.time_scale
    )
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    try:
        observations, decoded, waited = ready_observation(env, args.seed, no_op)
        planner = AStarPlanner(decoded.walkable_grid)
        parsed = parse_vector(observations["unit_5"]["vector"])
        enemy_start = normalized_to_cell(
            parsed.positions[5],
            width_cells=decoded.width_cells,
            height_cells=decoded.height_cells,
        )
        staging_target, staging_route = choose_staging_target(planner, enemy_start)
        staging_follower = WaypointFollower(
            decoded.width_cells, decoded.height_cells, tolerance=0.004
        )
        staging_follower.set_path(staging_route)
        staged_step = None
        strategy_transitions = []
        last_mode = None
        secure_delivery_events = []
        storage_samples = []
        max_storage_batteries_before_absorption = 0
        storage_batteries_after_absorption = None
        score_before_absorption = 0.0
        score_after_absorption = 0.0
        latest_info = {"score_0": 0.0, "score_1": 0.0, "time_left": 1.0}
        audit = None

        for step in range(args.max_steps):
            controlled = controller.act(observations, team_agents(0))
            assert controller.last_absorption_state is not None
            clock = controller.last_absorption_state
            mode = controller.last_strategy_mode
            if mode is not None and mode.value != last_mode:
                strategy_transitions.append(
                    {
                        "step": step,
                        "mode": mode.value,
                        "phase": clock.phase.value,
                        "cycle_index": clock.cycle_index,
                        "elapsed_seconds": clock.elapsed_seconds,
                        "seconds_until_absorption": clock.seconds_until_absorption,
                    }
                )
                last_mode = mode.value
            secure_delivery_events.extend(
                event for event in controller.last_events if event["kind"] == "secure_delivery"
            )

            actions = no_op.act(observations, env.agents)
            parsed_enemy = parse_vector(observations["unit_5"]["vector"])
            enemy_position = parsed_enemy.positions[5]
            enemy_cell = normalized_to_cell(
                enemy_position,
                width_cells=decoded.width_cells,
                height_cells=decoded.height_cells,
            )
            if staged_step is None:
                actions["unit_5"] = staging_follower.action(enemy_position)
                if enemy_cell == staging_target or staging_follower.complete:
                    staged_step = step
                    audit = danger_route_audit(planner, enemy_cell)
            else:
                # Run one collector and the guard. This exercises collection,
                # secure delivery and guard danger routing without reaching 100 early.
                actions["unit_0"] = controlled["unit_0"]
                actions["unit_3"] = controlled["unit_3"]

            dynamic_batteries = set(
                battery_cells_from_graphic(observations["unit_0"]["graphic"])
            )
            storage_count = len(dynamic_batteries & set(decoded.cells("ally_storage")))
            sample = {
                "step": step,
                "elapsed_seconds": clock.elapsed_seconds,
                "cycle_index": clock.cycle_index,
                "ally_storage_battery_cells": storage_count,
            }
            if (
                step % 25 == 0
                or not storage_samples
                or storage_samples[-1]["ally_storage_battery_cells"] != storage_count
                or any(entry["step"] == step for entry in strategy_transitions)
            ):
                storage_samples.append(sample)
            if clock.cycle_index == 0:
                max_storage_batteries_before_absorption = max(
                    max_storage_batteries_before_absorption, storage_count
                )
                score_before_absorption = float(latest_info["score_0"])
            elif clock.seconds_since_absorption >= 0.5:
                storage_batteries_after_absorption = storage_count
                score_after_absorption = float(latest_info["score_0"])

            observations, _, terminations, truncations, infos = env.step(actions)
            if any(truncations.values()):
                raise AssertionError("unexpected truncation")
            latest_info = dict(next(iter(infos.values())))
            if any(terminations.values()):
                raise AssertionError("episode terminated before first absorption audit")
            if (
                clock.cycle_index >= 1
                and clock.seconds_since_absorption >= 1.0
                and audit is not None
            ):
                completed_step = step + 1
                completed_elapsed = clock.elapsed_seconds
                break
        else:
            raise AssertionError("first absorption cycle not completed")

        modes = [entry["mode"] for entry in strategy_transitions]
        checks = {
            "observed_fresh_collect_secure_fresh_cycle": modes[:4]
            == ["fresh_collection", "collection", "secure", "fresh_collection"],
            "crossed_first_twenty_second_absorption": completed_elapsed >= 20.0,
            "enemy_unit_staged_in_central_field": staged_step is not None,
            "worker_weighted_route_changed_for_live_enemy": audit is not None
            and audit["geometric_path"] != audit["worker_weighted_path"],
            "worker_weighted_route_has_lower_risk_cost": audit is not None
            and audit["worker_weighted_cost"]
            < audit["geometric_path_worker_weighted_cost"],
            "score_was_retained_across_absorption": score_after_absorption
            >= score_before_absorption,
            "controller_remained_nonterminal": True,
        }
        if not all(checks.values()):
            raise AssertionError(f"strategy checks failed: {checks}")
        result = {
            "schema_version": "blackout.base_s11_s12_strategy.v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "environment": {
                "seed": args.seed,
                "build_path": str(build),
                "executable_sha256": sha256_file(executable),
                "game_commit": GAME_COMMIT,
                "python_api_commit": PYTHON_API_COMMIT,
                "time_scale": args.time_scale,
                "map_ready_after_steps": waited,
            },
            "execution": {
                "started_at_utc": started_at.isoformat(),
                "wall_seconds": round(time.perf_counter() - started, 3),
                "completed_step": completed_step,
                "completed_elapsed_seconds": completed_elapsed,
                "enemy_start": [enemy_start.x, enemy_start.y],
                "enemy_staging_target": [staging_target.x, staging_target.y],
                "enemy_staged_step": staged_step,
                "strategy_transitions": strategy_transitions,
                "secure_delivery_events": secure_delivery_events,
                "max_ally_storage_battery_cells_before_absorption": max_storage_batteries_before_absorption,
                "ally_storage_battery_cells_after_absorption": storage_batteries_after_absorption,
                "score_before_absorption": score_before_absorption,
                "score_after_absorption": score_after_absorption,
                "last_score": latest_info,
                "storage_samples": storage_samples,
            },
            "danger_route_audit": audit,
            "verification": checks,
        }
    finally:
        env.close()

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["verification"], ensure_ascii=False, indent=2))
    print(f"wrote strategy evidence to {output}")


if __name__ == "__main__":
    main()
