"""BASE-S08~S10 role FSM unit and generated-evidence tests."""

from __future__ import annotations

import json
import hashlib
import unittest
from pathlib import Path

import numpy as np

from blackout_rl import (
    AStarPlanner,
    CarrierPhase,
    GridCell,
    GuardPhase,
    HUNTER_SHRINE_CELLS,
    ScriptedTeamController,
    WorkerPhase,
    battery_cells_from_graphic,
    carrier_shrine_cell,
    farthest_reachable_cell,
    nearest_reachable_cell,
    select_evasion_target,
    team_agents,
)
from blackout_rl.observation import VECTOR_SIZE


ROOT = Path(__file__).resolve().parents[1]
LIVE_FSM = ROOT / "logs" / "base_s08_s10_role_fsm.json"


def map_graphic(*, batteries: tuple[GridCell, ...] = ()) -> np.ndarray:
    graphic = np.zeros((96, 96, 11), dtype=np.float32)
    graphic[..., 0] = 1.0

    def paint(cell: GridCell, channel: int) -> None:
        row_start = 96 - (cell.y + 1) * 4
        columns = slice(cell.x * 4, (cell.x + 1) * 4)
        rows = slice(row_start, row_start + 4)
        graphic[rows, columns, 0] = 0.0
        graphic[rows, columns, channel] = 1.0

    for cell in (GridCell(3, 21), GridCell(3, 22), GridCell(7, 17)):
        paint(cell, 2)
    for cell in (GridCell(21, 3), GridCell(22, 3)):
        paint(cell, 3)
    for cell in batteries:
        paint(cell, 6)
    return graphic


def observations(
    *,
    batteries: tuple[GridCell, ...],
    positions: dict[int, GridCell] | None = None,
    classes: dict[int, int] | None = None,
    holdings: dict[int, int] | None = None,
    time_left: float = 0.9,
) -> dict[str, dict[str, np.ndarray]]:
    positions = positions or {}
    classes = classes or {}
    holdings = holdings or {}
    defaults = {
        0: GridCell(1, 22),
        1: GridCell(1, 22),
        2: GridCell(1, 22),
        3: GridCell(1, 22),
        4: GridCell(1, 22),
        5: GridCell(22, 1),
        6: GridCell(22, 1),
        7: GridCell(22, 1),
        8: GridCell(22, 1),
        9: GridCell(22, 1),
    }
    defaults.update(positions)
    graphic = map_graphic(batteries=batteries)
    result = {}
    for observer in range(5):
        vector = np.zeros(VECTOR_SIZE, dtype=np.float32)
        for index in range(10):
            start = index * 9
            cell = defaults[index]
            vector[start : start + 2] = (
                (cell.x + 0.5) / 24.0,
                (cell.y + 0.5) / 24.0,
            )
            vector[start + 2] = 1.0 if index < 5 else -1.0
            vector[start + 3 + holdings.get(index, 0)] = 1.0
        vector[90 + classes.get(observer, 0)] = 1.0
        vector[93:] = (0.0, 0.0, time_left)
        result[f"unit_{observer}"] = {"vector": vector, "graphic": graphic.copy()}
    return result


BATTERIES = (
    GridCell(1, 18),
    GridCell(5, 18),
    GridCell(8, 17),
    GridCell(10, 10),
    GridCell(15, 8),
    GridCell(18, 5),
)


class TargetSelectionTests(unittest.TestCase):
    def test_fixed_shrine_contract(self) -> None:
        self.assertEqual(carrier_shrine_cell(0), GridCell(1, 20))
        self.assertEqual(carrier_shrine_cell(1), GridCell(20, 1))
        self.assertEqual(set(HUNTER_SHRINE_CELLS), {
            GridCell(11, 11), GridCell(11, 12), GridCell(12, 11), GridCell(12, 12)
        })

    def test_nearest_and_farthest_reachable_targets(self) -> None:
        planner = AStarPlanner(np.ones((8, 8), dtype=np.bool_))
        targets = (GridCell(1, 0), GridCell(6, 6), GridCell(0, 3))
        nearest, _, nearest_cost = nearest_reachable_cell(GridCell(0, 0), targets, planner)
        farthest, _, farthest_cost = farthest_reachable_cell(GridCell(0, 0), targets, planner)
        self.assertEqual(nearest, GridCell(1, 0))
        self.assertEqual(farthest, GridCell(6, 6))
        self.assertGreater(farthest_cost, nearest_cost)

    def test_evasion_target_increases_enemy_distance(self) -> None:
        planner = AStarPlanner(np.ones((9, 9), dtype=np.bool_))
        start = GridCell(4, 4)
        enemy = GridCell(4, 3)
        target = select_evasion_target(start, (enemy,), planner, search_radius=3)
        before = max(abs(start.x - enemy.x), abs(start.y - enemy.y))
        after = max(abs(target.x - enemy.x), abs(target.y - enemy.y))
        self.assertGreater(after, before)

    def test_vectorized_battery_cells_match_bottom_left_coordinates(self) -> None:
        graphic = map_graphic(batteries=(GridCell(2, 18), GridCell(20, 3)))
        self.assertEqual(
            battery_cells_from_graphic(graphic),
            (GridCell(2, 18), GridCell(20, 3)),
        )


class RoleFSMTests(unittest.TestCase):
    def test_workers_get_unique_batteries_without_crossing_shrines(self) -> None:
        controller = ScriptedTeamController(0)
        actions = controller.act(observations(batteries=BATTERIES), team_agents(0))
        worker_targets = [controller.targets[f"unit_{slot}"] for slot in range(3)]
        self.assertEqual(len(set(worker_targets)), 3)
        self.assertEqual(
            [controller.phases[f"unit_{slot}"] for slot in range(3)],
            [WorkerPhase.PICKUP.value] * 3,
        )
        forbidden = set(HUNTER_SHRINE_CELLS) | {carrier_shrine_cell(0)}
        for slot in range(3):
            self.assertFalse(set(controller.paths[f"unit_{slot}"]) & forbidden)
            self.assertLessEqual(float(np.linalg.norm(actions[f"unit_{slot}"])), 1.000001)

    def test_worker_pickup_deliver_retarget_cycle(self) -> None:
        controller = ScriptedTeamController(0)
        first = observations(batteries=BATTERIES)
        controller.act(first, team_agents(0))
        target = controller.targets["unit_0"]
        assert target is not None

        picked = observations(
            batteries=tuple(cell for cell in BATTERIES if cell != target),
            positions={0: target},
            holdings={0: 1},
            time_left=0.89,
        )
        controller.act(picked, team_agents(0))
        self.assertEqual(controller.phases["unit_0"], WorkerPhase.DELIVER.value)
        self.assertIn("storage_assigned", {event["kind"] for event in controller.last_events})

        storage = controller.targets["unit_0"]
        assert storage is not None
        deposited = observations(
            batteries=tuple(cell for cell in BATTERIES if cell != target),
            positions={0: storage},
            holdings={0: 0},
            time_left=0.88,
        )
        controller.act(deposited, team_agents(0))
        self.assertEqual(controller.phases["unit_0"], WorkerPhase.RETARGET.value)
        self.assertIn("battery_deposit", {event["kind"] for event in controller.last_events})
        controller.act(deposited, team_agents(0))
        self.assertEqual(controller.phases["unit_0"], WorkerPhase.PICKUP.value)

    def test_full_storage_timeout_retargets_a_different_component(self) -> None:
        controller = ScriptedTeamController(0)
        controller.act(observations(batteries=BATTERIES), team_agents(0))
        battery = controller.targets["unit_0"]
        assert battery is not None
        carrying = observations(
            batteries=tuple(cell for cell in BATTERIES if cell != battery),
            positions={0: battery},
            holdings={0: 1},
            time_left=0.89,
        )
        controller.act(carrying, team_agents(0))
        storage = controller.targets["unit_0"]
        assert storage is not None
        stalled = observations(
            batteries=tuple(cell for cell in BATTERIES if cell != battery),
            positions={0: storage},
            holdings={0: 1},
            time_left=0.88,
        )
        events = []
        for _ in range(13):
            controller.act(stalled, team_agents(0))
            events.extend(controller.last_events)
        replacement = controller.targets["unit_0"]
        self.assertIsNotNone(replacement)
        self.assertNotEqual(replacement, storage)
        self.assertIn("storage_retarget", {event["kind"] for event in events})

    def test_guard_transforms_then_chases_near_enemy(self) -> None:
        controller = ScriptedTeamController(0, chase_radius_cells=5)
        near_center = {
            3: GridCell(11, 11),
            5: GridCell(14, 11),
            6: GridCell(22, 1),
            7: GridCell(22, 1),
            8: GridCell(22, 1),
            9: GridCell(22, 1),
        }
        controller.act(
            observations(
                batteries=BATTERIES,
                positions=near_center,
                classes={3: 1},
            ),
            team_agents(0),
        )
        self.assertEqual(controller.phases["unit_3"], GuardPhase.CHASE.value)
        kinds = {event["kind"] for event in controller.last_events if event["agent"] == "unit_3"}
        self.assertIn("class_transform", kinds)
        self.assertIn("enemy_chase", kinds)

    def test_carrier_transforms_selects_far_battery_and_evades(self) -> None:
        controller = ScriptedTeamController(0, evade_radius_cells=3)
        carrier_cell = carrier_shrine_cell(0)
        controller.act(
            observations(
                batteries=BATTERIES,
                positions={4: carrier_cell},
                classes={4: 2},
            ),
            team_agents(0),
        )
        self.assertEqual(controller.phases["unit_4"], CarrierPhase.PICKUP.value)
        assignment = next(
            event
            for event in controller.last_events
            if event["agent"] == "unit_4" and event["kind"] == "battery_assigned"
        )
        self.assertEqual(assignment["selection"], "farthest_reachable")

        close_enemy = observations(
            batteries=BATTERIES,
            positions={4: GridCell(10, 10), 5: GridCell(11, 10)},
            classes={4: 2},
            time_left=0.89,
        )
        controller.act(close_enemy, team_agents(0))
        self.assertEqual(controller.phases["unit_4"], CarrierPhase.EVADE.value)
        self.assertIn(
            "evade_started",
            {event["kind"] for event in controller.last_events if event["agent"] == "unit_4"},
        )


@unittest.skipUnless(LIVE_FSM.is_file(), "live BASE-S08~S10 evidence not generated")
class LiveRoleFSMEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evidence = json.loads(LIVE_FSM.read_text())

    def test_all_role_fsm_live_checks_pass(self) -> None:
        self.assertTrue(all(self.evidence["verification"].values()))

    def test_live_worker_guard_carrier_milestones(self) -> None:
        execution = self.evidence["execution"]
        self.assertEqual(set(execution["worker_deposit_step"]), {"unit_0", "unit_1", "unit_2"})
        for agent in ("unit_0", "unit_1", "unit_2"):
            self.assertLess(execution["worker_pickup_step"][agent], execution["worker_deposit_step"][agent])
            self.assertLess(execution["worker_deposit_step"][agent], execution["worker_retarget_step"][agent])
        self.assertIsNotNone(execution["guard_transform_step"])
        self.assertIsNotNone(execution["guard_patrol_step"])
        self.assertLess(execution["guard_transform_step"], execution["guard_patrol_step"])
        self.assertIsNotNone(execution["carrier_transform_step"])
        self.assertIsNotNone(execution["carrier_deposit_step"])
        self.assertLess(execution["carrier_transform_step"], execution["carrier_pickup_step"])
        self.assertLess(execution["carrier_pickup_step"], execution["carrier_deposit_step"])

    def test_live_trajectory_hash_and_footer(self) -> None:
        path = Path(self.evidence["trajectory"]["path"])
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), self.evidence["trajectory"]["sha256"])
        records = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(records[0]["record_type"], "header")
        self.assertEqual(records[-1]["record_type"], "footer")
        self.assertEqual(records[-1]["records"], self.evidence["trajectory"]["step_records"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
