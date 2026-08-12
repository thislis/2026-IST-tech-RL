#!/usr/bin/env python3
"""Live five-unit coordination smoke test for BASE-S04~S07 and BASE-S13."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
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
    ScriptedTrajectoryRecorder,
    SemanticMapDecoder,
    StuckDetector,
    TeamStateTracker,
    WaypointFollower,
    assignment_targets,
    greedy_path_assignment,
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


def ready_observation(
    env: ContractBlackOutEnv,
    seed: int,
    no_op: NoOpPolicy,
) -> tuple[dict, int]:
    obs, _ = env.reset(seed=seed)
    decoder = SemanticMapDecoder()
    for wait_steps in range(21):
        graphic = obs["unit_0"]["graphic"]
        if graphic.max() > 0.0 and decoder.decode(graphic).cells("battery"):
            return obs, wait_steps
        obs, _, terminations, truncations, _ = env.step(no_op.act(obs, env.agents))
        if any(terminations.values()) or any(truncations.values()):
            raise RuntimeError("episode ended while waiting for semantic map")
    raise RuntimeError("semantic map was not ready within 20 steps")


def event_payload(kind: str, **values) -> dict:
    return {"kind": kind, **values}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=240513)
    parser.add_argument("--time-scale", type=float, default=50.0)
    parser.add_argument("--max-steps", type=int, default=1_500)
    parser.add_argument("--record-every", type=int, default=5)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trajectory", required=True, type=Path)
    args = parser.parse_args()

    build = args.build.expanduser().resolve()
    executable = executable_in(build)
    if not executable.is_file():
        raise FileNotFoundError(executable)
    output_path = args.output.resolve()
    trajectory_path = args.trajectory.resolve()
    agents = team_agents(0)
    no_op = NoOpPolicy()
    decoder = SemanticMapDecoder()
    env = ContractBlackOutEnv(
        env_path=str(build),
        no_graphics=False,
        time_scale=args.time_scale,
    )
    recorder = None
    try:
        obs, map_wait_steps = ready_observation(env, args.seed, no_op)
        decoded = decoder.decode(obs[agents[0]]["graphic"])
        planner = AStarPlanner(decoded.walkable_grid)
        tracker = TeamStateTracker(
            0,
            width_cells=decoded.width_cells,
            height_cells=decoded.height_cells,
        )
        snapshot = tracker.update(obs, step=0)
        initial_positions = {
            unit.agent: np.asarray(unit.position_normalized, dtype=np.float32)
            for unit in snapshot.units
        }

        # This coordination smoke deliberately assigns all five role-labelled
        # units to batteries. Role-specific shrine/FSM behavior starts at S08~S10.
        assignments = greedy_path_assignment(
            snapshot.units,
            decoded.cells("battery"),
            planner,
            eligible_roles=None,
        )
        if len(assignments) != 5:
            raise AssertionError(f"expected five reachable assignments, got {len(assignments)}")
        targets = assignment_targets(assignments)
        if len(set(targets.values())) != 5:
            raise AssertionError("task assignment produced duplicate battery targets")
        tracker.set_targets(targets)
        snapshot = tracker.update(obs, step=0)

        followers = {}
        paths = {}
        collision_cells = {agent: set() for agent in agents}
        for assignment in assignments:
            follower = WaypointFollower(
                decoded.width_cells,
                decoded.height_cells,
                tolerance=0.004,
                stall_limit=30,
            )
            follower.set_path(assignment.path)
            followers[assignment.agent] = follower
            paths[assignment.agent] = assignment.path

        stuck_detector = StuckDetector(
            window_steps=30,
            min_displacement_normalized=0.01,
        )
        recorder = ScriptedTrajectoryRecorder(
            trajectory_path,
            metadata={
                "seed": args.seed,
                "team": 0,
                "policy_id": "base-s04-s07-coordination-smoke",
                "record_every": args.record_every,
                "reward_semantics": "reward returned after applying the recorded action",
                "game_commit": GAME_COMMIT,
                "python_api_commit": PYTHON_API_COMMIT,
                "executable_sha256": sha256_file(executable),
            },
        )

        assignment_events = [
            event_payload(
                "assignment",
                agent=assignment.agent,
                role=assignment.role.value,
                target=[assignment.target.x, assignment.target.y],
                path_cost=assignment.path_cost,
            )
            for assignment in assignments
        ]
        picked: dict[str, int] = {}
        pickup_cells: dict[str, GridCell] = {}
        stuck_events = []
        replan_events = []
        reassignment_events = []
        visited_wall_cells = []
        previous_rewards = {agent: 0.0 for agent in env.possible_agents}
        latest_info = {"score_0": 0.0, "score_1": 0.0, "time_left": 1.0}
        started_at = datetime.now(timezone.utc)
        started = time.perf_counter()

        for step in range(args.max_steps):
            snapshot = tracker.update(obs, step=step)
            state_by_agent = snapshot.by_agent()
            actions = no_op.act(obs, env.agents)
            step_events = assignment_events if step == 0 else []

            for agent in agents:
                state = state_by_agent[agent]
                if state.holding_item_id == 1:
                    if agent not in picked:
                        picked[agent] = step
                        pickup_cells[agent] = state.cell
                        step_events.append(
                            event_payload(
                                "battery_pickup",
                                agent=agent,
                                target=[state.target.x, state.target.y]
                                if state.target is not None
                                else None,
                                cell=[state.cell.x, state.cell.y],
                            )
                        )
                    stuck_detector.reset(agent)
                    continue

                follower = followers[agent]
                action = follower.action(state.position_normalized)
                stuck = stuck_detector.observe(state, action)
                if stuck is not None:
                    stuck_record = event_payload(
                        "stuck",
                        agent=agent,
                        step=step,
                        target=[stuck.target.x, stuck.target.y],
                        displacement_normalized=stuck.displacement_normalized,
                        consecutive_events=stuck.consecutive_events,
                    )
                    stuck_events.append(stuck_record)
                    step_events.append(stuck_record)
                    collided = follower.current_waypoint
                    if collided is not None and collided != state.target:
                        collision_cells[agent].add(collided)
                    try:
                        assert state.target is not None
                        new_path = planner.plan_avoiding(
                            state.cell,
                            state.target,
                            collision_cells[agent],
                        )
                        follower.set_path(new_path)
                        paths[agent] = new_path
                        replan_record = event_payload(
                            "replan",
                            agent=agent,
                            target=[state.target.x, state.target.y],
                            path_cells=len(new_path),
                            temporarily_blocked=[
                                [cell.x, cell.y]
                                for cell in sorted(collision_cells[agent])
                            ],
                        )
                        replan_events.append(replan_record)
                        step_events.append(replan_record)
                    except (PathNotFound, AssertionError):
                        other_targets = {
                            target
                            for other, target in targets.items()
                            if other != agent
                        }
                        replacement = greedy_path_assignment(
                            (state,),
                            decoded.cells("battery"),
                            planner,
                            eligible_roles=None,
                            reserved_targets=other_targets,
                        )
                        if not replacement:
                            raise AssertionError(f"no replacement target for stuck {agent}")
                        selected = replacement[0]
                        old_target = state.target
                        targets[agent] = selected.target
                        tracker.set_target(agent, selected.target)
                        follower.set_path(selected.path)
                        paths[agent] = selected.path
                        collision_cells[agent].clear()
                        stuck_detector.reset(agent)
                        reassignment = event_payload(
                            "target_reassigned",
                            agent=agent,
                            old_target=[old_target.x, old_target.y]
                            if old_target is not None
                            else None,
                            new_target=[selected.target.x, selected.target.y],
                        )
                        reassignment_events.append(reassignment)
                        step_events.append(reassignment)
                    action = follower.action(state.position_normalized)
                actions[agent] = action

                if decoded.wall_grid[state.cell.y, state.cell.x]:
                    visited_wall_cells.append([agent, state.cell.x, state.cell.y, step])

            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            if any(truncations.values()):
                raise AssertionError("unexpected truncation during coordination smoke")
            if any(terminations.values()):
                raise AssertionError("episode ended before coordination smoke completed")
            latest_info = dict(next(iter(infos.values())))
            should_record = (
                step == 0
                or step % args.record_every == 0
                or bool(step_events)
                or len(picked) == 5
            )
            if should_record:
                recorder.record_step(
                    snapshot=snapshot,
                    observations=obs,
                    actions={agent: actions[agent] for agent in agents},
                    rewards={agent: rewards[agent] for agent in agents},
                    score={
                        "team_a_normalized": float(latest_info["score_0"]),
                        "team_b_normalized": float(latest_info["score_1"]),
                        "time_left_normalized": float(latest_info["time_left"]),
                    },
                    events=step_events,
                )
            previous_rewards = rewards
            obs = next_obs
            if len(picked) == 5:
                completed_step = step + 1
                break
        else:
            raise AssertionError(
                f"only {len(picked)}/5 units picked a battery within {args.max_steps} steps"
            )

        final_snapshot = tracker.update(obs, step=completed_step)
        final_by_agent = final_snapshot.by_agent()
        displacement = {
            agent: float(
                np.linalg.norm(
                    np.asarray(final_by_agent[agent].position_normalized, dtype=np.float32)
                    - initial_positions[agent]
                )
            )
            for agent in agents
        }
        pickup_target_distance = {
            agent: max(
                abs(pickup_cells[agent].x - targets[agent].x),
                abs(pickup_cells[agent].y - targets[agent].y),
            )
            for agent in agents
        }
        role_counts = Counter(unit.role.value for unit in final_snapshot.units)
        checks = {
            "five_unique_assignments": len(set(targets.values())) == 5,
            "all_five_units_moved": all(value > 0.02 for value in displacement.values()),
            "all_five_units_hold_battery": all(
                final_by_agent[agent].holding_item_id == 1 for agent in agents
            ),
            "pickup_near_assigned_target": all(
                distance <= 1 for distance in pickup_target_distance.values()
            ),
            "no_decoded_wall_cell_entered": not visited_wall_cells,
            "default_role_split_3_1_1": role_counts
            == {"worker": 3, "guard": 1, "carrier": 1},
            "trajectory_records_written": recorder.records > 0,
        }
        if not all(checks.values()):
            raise AssertionError(f"coordination smoke checks failed: {checks}")

        recorder.close(
            summary={
                "status": "complete",
                "completed_step": completed_step,
                "picked_agents": list(agents),
                "checks": checks,
            }
        )
        trajectory_sha256 = sha256_file(trajectory_path)
        result = {
            "schema_version": "blackout.base_s04_s07_s13_coordination.v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "environment": {
                "seed": args.seed,
                "build_path": str(build),
                "executable_path": str(executable),
                "executable_sha256": sha256_file(executable),
                "game_commit": GAME_COMMIT,
                "python_api_commit": PYTHON_API_COMMIT,
                "time_scale": args.time_scale,
                "map_ready_after_steps": map_wait_steps,
            },
            "roles": {
                unit.agent: {"slot_id": unit.slot_id, "role": unit.role.value}
                for unit in final_snapshot.units
            },
            "initial_assignments": {
                assignment.agent: {
                    "role": assignment.role.value,
                    "target": [assignment.target.x, assignment.target.y],
                    "path_cost": assignment.path_cost,
                    "path_cells": len(assignment.path),
                }
                for assignment in assignments
            },
            "execution": {
                "started_at_utc": started_at.isoformat(),
                "wall_seconds": round(time.perf_counter() - started, 3),
                "completed_step": completed_step,
                "pickup_step_by_agent": picked,
                "pickup_cell_by_agent": {
                    agent: [cell.x, cell.y] for agent, cell in pickup_cells.items()
                },
                "pickup_target_chebyshev_distance": pickup_target_distance,
                "displacement_normalized_by_agent": displacement,
                "stuck_events": stuck_events,
                "replan_events": replan_events,
                "target_reassignment_events": reassignment_events,
                "visited_wall_cells": visited_wall_cells,
                "last_rewards": {
                    agent: float(previous_rewards[agent]) for agent in agents
                },
                "last_score": latest_info,
            },
            "trajectory": {
                "path": str(trajectory_path),
                "sha256": trajectory_sha256,
                "schema_version": "blackout.scripted_trajectory.v1",
                "record_every": args.record_every,
                "step_records": recorder.records,
            },
            "verification": checks,
        }
    except Exception:
        if recorder is not None and not recorder.closed:
            recorder.close(summary={"status": "error"})
        raise
    finally:
        env.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["verification"], ensure_ascii=False, indent=2))
    print(f"wrote coordination evidence to {output_path}")
    print(f"wrote trajectory to {trajectory_path}")


if __name__ == "__main__":
    main()
