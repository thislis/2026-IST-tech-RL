#!/usr/bin/env python3
"""Live 3-worker/1-guard/1-carrier FSM verification for BASE-S08~S10."""

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
    ContractBlackOutEnv,
    NoOpPolicy,
    ScriptedTeamController,
    ScriptedTrajectoryRecorder,
    SemanticMapDecoder,
    parse_vector,
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
    observations, _ = env.reset(seed=seed)
    decoder = SemanticMapDecoder()
    for wait_steps in range(21):
        decoded = decoder.decode(observations["unit_0"]["graphic"])
        if observations["unit_0"]["graphic"].max() > 0.0 and decoded.cells("battery"):
            return observations, wait_steps
        observations, _, terminations, truncations, _ = env.step(
            no_op.act(observations, env.agents)
        )
        if any(terminations.values()) or any(truncations.values()):
            raise RuntimeError("episode ended while waiting for semantic map")
    raise RuntimeError("semantic map was not ready within 20 steps")


def event_step(events: list[dict], agent: str, kind: str) -> int | None:
    for event in events:
        if event["agent"] == agent and event["kind"] == kind:
            return int(event["step"])
    return None


def transition_step(
    events: list[dict],
    agent: str,
    reason: str,
    *,
    after_step: int | None = None,
) -> int | None:
    for event in events:
        if (
            event["agent"] == agent
            and event["kind"] == "fsm_transition"
            and event.get("reason") == reason
            and (after_step is None or int(event["step"]) > after_step)
        ):
            return int(event["step"])
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=240810)
    parser.add_argument("--time-scale", type=float, default=50.0)
    parser.add_argument("--max-steps", type=int, default=2_500)
    parser.add_argument("--record-every", type=int, default=10)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trajectory", required=True, type=Path)
    args = parser.parse_args()

    build = args.build.expanduser().resolve()
    executable = executable_in(build)
    if not executable.is_file():
        raise FileNotFoundError(executable)
    output_path = args.output.resolve()
    trajectory_path = args.trajectory.resolve()
    controlled_agents = team_agents(0)
    workers = controlled_agents[:3]
    guard = controlled_agents[3]
    carrier = controlled_agents[4]
    no_op = NoOpPolicy()
    controller = ScriptedTeamController(0, seed=args.seed)
    decoder = SemanticMapDecoder()
    env = ContractBlackOutEnv(
        env_path=str(build),
        no_graphics=False,
        time_scale=args.time_scale,
    )
    recorder = None
    try:
        observations, map_wait_steps = ready_observation(env, args.seed, no_op)
        initial_map = decoder.decode(observations[controlled_agents[0]]["graphic"])
        initial_batteries = len(initial_map.cells("battery"))
        recorder = ScriptedTrajectoryRecorder(
            trajectory_path,
            metadata={
                "seed": args.seed,
                "team": 0,
                "policy_id": "base-s08-s10-role-fsm",
                "record_every": args.record_every,
                "reward_semantics": "reward returned after applying the recorded action",
                "game_commit": GAME_COMMIT,
                "python_api_commit": PYTHON_API_COMMIT,
                "executable_sha256": sha256_file(executable),
            },
        )
        all_events: list[dict] = []
        visited_wall_cells = []
        latest_info = {"score_0": 0.0, "score_1": 0.0, "time_left": 1.0}
        previous_rewards = {agent: 0.0 for agent in env.possible_agents}
        started_at = datetime.now(timezone.utc)
        started = time.perf_counter()

        for step in range(args.max_steps):
            team_actions = controller.act(observations, controlled_agents)
            assert controller.last_snapshot is not None
            snapshot = controller.last_snapshot
            step_events = list(controller.last_events)
            all_events.extend(step_events)
            actions = no_op.act(observations, env.agents)
            actions.update(team_actions)

            for unit in snapshot.units:
                if initial_map.wall_grid[unit.cell.y, unit.cell.x]:
                    visited_wall_cells.append(
                        [unit.agent, unit.cell.x, unit.cell.y, step]
                    )

            next_observations, rewards, terminations, truncations, infos = env.step(actions)
            if any(truncations.values()):
                raise AssertionError("unexpected truncation during role FSM smoke")
            latest_info = dict(next(iter(infos.values())))
            should_record = (
                step == 0
                or step % args.record_every == 0
                or bool(step_events)
            )
            if should_record:
                recorder.record_step(
                    snapshot=snapshot,
                    observations=observations,
                    actions={agent: actions[agent] for agent in controlled_agents},
                    rewards={agent: rewards[agent] for agent in controlled_agents},
                    score={
                        "team_a_normalized": float(latest_info["score_0"]),
                        "team_b_normalized": float(latest_info["score_1"]),
                        "time_left_normalized": float(latest_info["time_left"]),
                    },
                    events=step_events,
                )
            previous_rewards = rewards
            observations = next_observations

            worker_deposits = {
                agent: event_step(all_events, agent, "battery_deposit")
                for agent in workers
            }
            worker_retargets = {
                agent: transition_step(
                    all_events,
                    agent,
                    "retarget_ready",
                    after_step=worker_deposits[agent],
                )
                if worker_deposits[agent] is not None
                else None
                for agent in workers
            }
            guard_transform = event_step(all_events, guard, "class_transform")
            guard_patrol = event_step(all_events, guard, "patrol_waypoint_reached")
            carrier_transform = event_step(all_events, carrier, "class_transform")
            carrier_pickup = event_step(all_events, carrier, "battery_pickup")
            carrier_deposit = event_step(all_events, carrier, "battery_deposit")
            milestones_complete = (
                all(value is not None for value in worker_deposits.values())
                and all(value is not None for value in worker_retargets.values())
                and guard_transform is not None
                and guard_patrol is not None
                and carrier_transform is not None
                and carrier_pickup is not None
                and carrier_deposit is not None
            )
            if milestones_complete:
                completed_step = step + 1
                break
            if any(terminations.values()):
                raise AssertionError("episode terminated before all role milestones completed")
        else:
            counts = Counter(event["kind"] for event in all_events)
            raise AssertionError(
                f"role milestones incomplete after {args.max_steps} steps; events={dict(counts)}"
            )

        final_classes = {
            agent: parse_vector(observations[agent]["vector"]).self_class_id
            for agent in controlled_agents
        }
        worker_pickups = {
            agent: event_step(all_events, agent, "battery_pickup")
            for agent in workers
        }
        worker_deposits = {
            agent: event_step(all_events, agent, "battery_deposit")
            for agent in workers
        }
        worker_retargets = {
            agent: transition_step(
                all_events,
                agent,
                "retarget_ready",
                after_step=worker_deposits[agent],
            )
            if worker_deposits[agent] is not None
            else None
            for agent in workers
        }
        guard_transform = event_step(all_events, guard, "class_transform")
        guard_patrol = event_step(all_events, guard, "patrol_waypoint_reached")
        carrier_transform = event_step(all_events, carrier, "class_transform")
        carrier_pickup = event_step(all_events, carrier, "battery_pickup")
        carrier_deposit = event_step(all_events, carrier, "battery_deposit")
        carrier_assignment = next(
            event
            for event in all_events
            if event["agent"] == carrier
            and event["kind"] == "battery_assigned"
            and event.get("selection") == "farthest_reachable"
        )
        initial_worker_assignments = {
            agent: next(
                event
                for event in all_events
                if event["agent"] == agent
                and event["kind"] == "battery_assigned"
            )
            for agent in workers
        }
        checks = {
            "three_workers_completed_pickup_deposit_retarget": all(
                worker_pickups[agent] is not None
                and worker_deposits[agent] is not None
                and worker_retargets[agent] is not None
                for agent in workers
            ),
            "workers_started_with_unique_battery_targets": len(
                {tuple(event["target"]) for event in initial_worker_assignments.values()}
            ) == 3,
            "guard_transformed_and_patrolled": (
                guard_transform is not None and guard_patrol is not None
            ),
            "carrier_transformed_and_delivered_far_battery": (
                carrier_transform is not None
                and carrier_pickup is not None
                and carrier_deposit is not None
                and carrier_assignment["selection"] == "farthest_reachable"
            ),
            "role_classes_remain_worker_worker_worker_guard_carrier": final_classes
            == {
                "unit_0": 0,
                "unit_1": 0,
                "unit_2": 0,
                "unit_3": 1,
                "unit_4": 2,
            },
            "team_score_increased": float(latest_info["score_0"]) > 0.0,
            "no_decoded_wall_cell_entered": not visited_wall_cells,
            "trajectory_records_written": recorder.records > 0,
        }
        if not all(checks.values()):
            raise AssertionError(f"role FSM smoke checks failed: {checks}")

        recorder.close(
            summary={
                "status": "complete",
                "completed_step": completed_step,
                "checks": checks,
            }
        )
        event_counts = Counter(event["kind"] for event in all_events)
        result = {
            "schema_version": "blackout.base_s08_s10_role_fsm.v1",
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
                "initial_battery_cells": initial_batteries,
            },
            "execution": {
                "started_at_utc": started_at.isoformat(),
                "wall_seconds": round(time.perf_counter() - started, 3),
                "completed_step": completed_step,
                "worker_pickup_step": worker_pickups,
                "worker_deposit_step": worker_deposits,
                "worker_retarget_step": worker_retargets,
                "initial_worker_assignments": initial_worker_assignments,
                "guard_transform_step": guard_transform,
                "guard_patrol_step": guard_patrol,
                "carrier_transform_step": carrier_transform,
                "carrier_pickup_step": carrier_pickup,
                "carrier_deposit_step": carrier_deposit,
                "carrier_far_assignment": carrier_assignment,
                "final_classes": final_classes,
                "final_phases": controller.phases,
                "event_counts": dict(event_counts),
                "stuck_replan_events": [
                    event for event in all_events if event["kind"] == "stuck_replan"
                ],
                "visited_wall_cells": visited_wall_cells,
                "last_rewards": {
                    agent: float(previous_rewards[agent]) for agent in controlled_agents
                },
                "last_score": latest_info,
            },
            "trajectory": {
                "path": str(trajectory_path),
                "sha256": sha256_file(trajectory_path),
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
    print(f"wrote role FSM evidence to {output_path}")
    print(f"wrote role FSM trajectory to {trajectory_path}")


if __name__ == "__main__":
    main()
