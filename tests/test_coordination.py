"""BASE-S04~S07 and BASE-S13 state, assignment, and recording tests."""

from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from blackout_rl import (
    AStarPlanner,
    GridCell,
    Role,
    ScriptedTrajectoryRecorder,
    StuckDetector,
    TeamStateTracker,
    UnitState,
    decode_graphic,
    default_roles,
    greedy_path_assignment,
    team_agents,
)
from blackout_rl.observation import VECTOR_SIZE


ROOT = Path(__file__).resolve().parents[1]
LIVE_COORDINATION = ROOT / "logs" / "base_s04_s07_s13_coordination.json"


def vector_for(
    observer_index: int,
    *,
    class_id: int = 0,
    holding_by_unit: dict[int, int] | None = None,
) -> np.ndarray:
    vector = np.zeros(VECTOR_SIZE, dtype=np.float32)
    holding_by_unit = holding_by_unit or {}
    observer_team = 0 if observer_index < 5 else 1
    for unit_index in range(10):
        start = unit_index * 9
        vector[start : start + 2] = (
            (unit_index + 1.5) / 24.0,
            (22.5 - unit_index) / 24.0,
        )
        unit_team = 0 if unit_index < 5 else 1
        vector[start + 2] = 1.0 if unit_team == observer_team else -1.0
        vector[start + 3 + holding_by_unit.get(unit_index, 0)] = 1.0
    vector[90 + class_id] = 1.0
    vector[93:] = (0.2, 0.1, 0.8)
    return vector


def graphic() -> np.ndarray:
    result = np.zeros((8, 8, 11), dtype=np.float32)
    result[..., 0] = 1.0
    result[4:8, 4:8, 0] = 0.0
    result[4:8, 4:8, 1] = 1.0
    result[1, 2, 0] = 0.0
    result[1, 2, 6] = 1.0
    return result


def team_observations(team: int) -> dict[str, dict[str, np.ndarray]]:
    class_ids = (0, 0, 0, 1, 2)
    observations = {}
    for slot, agent in enumerate(team_agents(team)):
        unit_index = team * 5 + slot
        observations[agent] = {
            "vector": vector_for(
                unit_index,
                class_id=class_ids[slot],
                holding_by_unit={team * 5 + 1: 1},
            ),
            "graphic": graphic(),
        }
    return observations


def unit_state(
    agent: str,
    slot: int,
    cell: GridCell,
    *,
    role: Role = Role.WORKER,
    step: int = 0,
    position: tuple[float, float] | None = None,
    target: GridCell | None = None,
) -> UnitState:
    normalized = position or ((cell.x + 0.5) / 6.0, (cell.y + 0.5) / 6.0)
    return UnitState(
        agent=agent,
        team=0,
        slot_id=slot,
        unit_index=slot,
        position_normalized=normalized,
        cell=cell,
        holding_item_id=0,
        class_id=0,
        role=role,
        target=target,
        step=step,
    )


class TeamStateTests(unittest.TestCase):
    def test_default_roles_are_stable_three_one_one(self) -> None:
        roles = default_roles(0)
        self.assertEqual(list(roles.values()).count(Role.WORKER), 3)
        self.assertEqual(list(roles.values()).count(Role.GUARD), 1)
        self.assertEqual(list(roles.values()).count(Role.CARRIER), 1)
        self.assertEqual(roles["unit_3"], Role.GUARD)
        self.assertEqual(roles["unit_4"], Role.CARRIER)

    def test_team_tracker_uses_local_slots_and_each_agents_self_class(self) -> None:
        tracker = TeamStateTracker(1)
        tracker.set_target("unit_5", GridCell(4, 4))
        snapshot = tracker.update(team_observations(1), step=7)
        self.assertEqual(tuple(unit.agent for unit in snapshot.units), team_agents(1))
        self.assertEqual(tuple(unit.slot_id for unit in snapshot.units), tuple(range(5)))
        self.assertEqual(tuple(unit.unit_index for unit in snapshot.units), tuple(range(5, 10)))
        self.assertEqual(tuple(unit.class_id for unit in snapshot.units), (0, 0, 0, 1, 2))
        self.assertEqual(snapshot.units[0].target, GridCell(4, 4))
        self.assertTrue(snapshot.units[1].is_carrying)
        self.assertAlmostEqual(snapshot.own_score, 0.2)
        self.assertAlmostEqual(snapshot.opponent_score, 0.1)
        self.assertAlmostEqual(snapshot.time_left, 0.8)

        updated = tracker.update(team_observations(1), step=8)
        self.assertEqual(tuple(unit.role for unit in updated.units), tuple(default_roles(1).values()))
        self.assertEqual(updated.units[0].target, GridCell(4, 4))

    def test_tracker_rejects_missing_team_member(self) -> None:
        observations = team_observations(0)
        observations.pop("unit_2")
        with self.assertRaises(KeyError):
            TeamStateTracker(0).update(observations, step=0)


class StuckDetectorTests(unittest.TestCase):
    def test_stationary_commanded_unit_triggers_and_moving_unit_does_not(self) -> None:
        target = GridCell(5, 5)
        detector = StuckDetector(window_steps=5, min_displacement_normalized=0.01)
        action = np.asarray((1.0, 0.0), dtype=np.float32)
        first = unit_state(
            "unit_0", 0, GridCell(0, 0), step=0, position=(0.1, 0.1), target=target
        )
        stationary = unit_state(
            "unit_0", 0, GridCell(0, 0), step=5, position=(0.1, 0.1), target=target
        )
        self.assertIsNone(detector.observe(first, action))
        event = detector.observe(stationary, action)
        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event.agent, "unit_0")
        self.assertEqual(event.consecutive_events, 1)

        moving_detector = StuckDetector(window_steps=5, min_displacement_normalized=0.01)
        self.assertIsNone(moving_detector.observe(first, action))
        moving = unit_state(
            "unit_0", 0, GridCell(1, 0), step=5, position=(0.2, 0.1), target=target
        )
        self.assertIsNone(moving_detector.observe(moving, action))

    def test_noop_and_target_change_reset_window(self) -> None:
        detector = StuckDetector(window_steps=2)
        first = unit_state(
            "unit_0", 0, GridCell(0, 0), step=0, target=GridCell(5, 5)
        )
        self.assertIsNone(detector.observe(first, np.ones(2, dtype=np.float32)))
        self.assertIsNone(detector.observe(first, np.zeros(2, dtype=np.float32)))
        changed = unit_state(
            "unit_0", 0, GridCell(0, 0), step=3, target=GridCell(4, 4)
        )
        self.assertIsNone(detector.observe(changed, np.ones(2, dtype=np.float32)))

    def test_stuck_event_can_drive_an_alternate_route(self) -> None:
        walkable = np.ones((5, 5), dtype=np.bool_)
        planner = AStarPlanner(walkable)
        target = GridCell(4, 2)
        detector = StuckDetector(window_steps=3, min_displacement_normalized=0.01)
        state = unit_state(
            "unit_0",
            0,
            GridCell(0, 2),
            step=0,
            position=(0.1, 0.5),
            target=target,
        )
        action = np.asarray((1.0, 0.0), dtype=np.float32)
        self.assertIsNone(detector.observe(state, action))
        stalled = unit_state(
            "unit_0",
            0,
            GridCell(0, 2),
            step=3,
            position=(0.1, 0.5),
            target=target,
        )
        self.assertIsNotNone(detector.observe(stalled, action))
        route = planner.plan_avoiding(stalled.cell, target, {GridCell(1, 2)})
        self.assertNotIn(GridCell(1, 2), route)
        self.assertEqual(route[-1], target)


class AssignmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.planner = AStarPlanner(np.ones((6, 6), dtype=np.bool_))
        self.units = (
            unit_state("unit_0", 0, GridCell(0, 0)),
            unit_state("unit_1", 1, GridCell(0, 5)),
            unit_state("unit_2", 2, GridCell(5, 0)),
            unit_state("unit_3", 3, GridCell(5, 5), role=Role.GUARD),
            unit_state("unit_4", 4, GridCell(3, 3), role=Role.CARRIER),
        )
        self.targets = (GridCell(1, 0), GridCell(0, 4), GridCell(4, 0), GridCell(4, 4))

    def test_worker_assignment_is_unique_reachable_and_deterministic(self) -> None:
        first = greedy_path_assignment(
            self.units,
            self.targets,
            self.planner,
            eligible_roles={Role.WORKER},
        )
        second = greedy_path_assignment(
            reversed(self.units),
            reversed(self.targets),
            self.planner,
            eligible_roles={Role.WORKER},
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertEqual(len({assignment.agent for assignment in first}), 3)
        self.assertEqual(len({assignment.target for assignment in first}), 3)
        self.assertTrue(all(assignment.role == Role.WORKER for assignment in first))
        self.assertTrue(all(assignment.path[0] == self.units[assignment.slot_id].cell for assignment in first))
        self.assertTrue(all(assignment.path[-1] == assignment.target for assignment in first))

    def test_assignment_respects_reserved_and_unreachable_targets(self) -> None:
        walkable = np.ones((6, 6), dtype=np.bool_)
        walkable[4, 0] = False
        assignments = greedy_path_assignment(
            self.units,
            self.targets,
            AStarPlanner(walkable),
            eligible_roles={Role.WORKER},
            reserved_targets={GridCell(1, 0)},
        )
        self.assertNotIn(GridCell(1, 0), {assignment.target for assignment in assignments})
        self.assertNotIn(GridCell(0, 4), {assignment.target for assignment in assignments})


class TrajectoryRecorderTests(unittest.TestCase):
    def test_jsonl_round_trip_contains_required_scripted_fields(self) -> None:
        observations = team_observations(0)
        tracker = TeamStateTracker(0, width_cells=2, height_cells=2)
        tracker.set_target("unit_0", GridCell(0, 1))
        snapshot = tracker.update(observations, step=12)
        actions = {
            agent: np.asarray((1.0, 0.0), dtype=np.float32)
            for agent in team_agents(0)
        }
        rewards = {agent: 0.0 for agent in team_agents(0)}

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.jsonl"
            recorder = ScriptedTrajectoryRecorder(
                path,
                metadata={"seed": 77, "team": 0, "policy_id": "scripted-smoke"},
            )
            recorder.record_step(
                snapshot=snapshot,
                observations=observations,
                actions=actions,
                rewards=rewards,
                score={"team_a": 0, "team_b": 0},
                events=[{"kind": "assignment", "agent": "unit_0"}],
            )
            recorder.close(summary={"status": "complete"})
            records = [json.loads(line) for line in path.read_text().splitlines()]

        self.assertEqual([record["record_type"] for record in records], ["header", "step", "footer"])
        self.assertEqual(records[0]["metadata"]["seed"], 77)
        step = records[1]
        self.assertEqual(len(step["observation"]["vectors"]["unit_0"]), 96)
        self.assertEqual(step["unit_state"]["unit_0"]["role"], "worker")
        self.assertEqual(step["unit_state"]["unit_0"]["target"], [0, 1])
        self.assertEqual(step["events"][0]["kind"], "assignment")
        decoded_ids = decode_graphic(step["observation"]["team_graphic"])
        self.assertTrue(np.array_equal(decoded_ids, np.argmax(graphic(), axis=-1)))
        self.assertEqual(records[2]["records"], 1)


@unittest.skipUnless(LIVE_COORDINATION.is_file(), "live coordination evidence not generated")
class LiveCoordinationEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evidence = json.loads(LIVE_COORDINATION.read_text())
        cls.trajectory_path = Path(cls.evidence["trajectory"]["path"])
        cls.records = [
            json.loads(line) for line in cls.trajectory_path.read_text().splitlines()
        ]

    def test_five_units_receive_unique_targets_and_finish_pickup(self) -> None:
        self.assertTrue(all(self.evidence["verification"].values()))
        assignments = self.evidence["initial_assignments"]
        self.assertEqual(len(assignments), 5)
        self.assertEqual(len({tuple(value["target"]) for value in assignments.values()}), 5)
        self.assertEqual(
            set(self.evidence["execution"]["pickup_step_by_agent"]),
            set(assignments),
        )

    def test_live_roles_are_team_local_three_one_one(self) -> None:
        roles = self.evidence["roles"]
        self.assertEqual([roles[f"unit_{slot}"]["slot_id"] for slot in range(5)], list(range(5)))
        self.assertEqual(
            [roles[f"unit_{slot}"]["role"] for slot in range(5)],
            ["worker", "worker", "worker", "guard", "carrier"],
        )

    def test_live_trajectory_hash_and_record_contract(self) -> None:
        digest = hashlib.sha256(self.trajectory_path.read_bytes()).hexdigest()
        self.assertEqual(digest, self.evidence["trajectory"]["sha256"])
        self.assertEqual(self.records[0]["record_type"], "header")
        self.assertEqual(self.records[0]["metadata"]["seed"], 240513)
        steps = [record for record in self.records if record["record_type"] == "step"]
        self.assertEqual(len(steps), self.evidence["trajectory"]["step_records"])
        self.assertEqual(self.records[-1]["record_type"], "footer")
        self.assertEqual(self.records[-1]["records"], len(steps))
        for record in steps:
            self.assertEqual(set(record["observation"]["vectors"]), set(self.evidence["roles"]))
            self.assertTrue(all(len(vector) == 96 for vector in record["observation"]["vectors"].values()))
            self.assertEqual(set(record["actions"]), set(self.evidence["roles"]))
            self.assertEqual(set(record["rewards"]), set(self.evidence["roles"]))
            self.assertIn("score", record)
            self.assertEqual(decode_graphic(record["observation"]["team_graphic"]).shape, (96, 96))


if __name__ == "__main__":
    unittest.main(verbosity=2)
