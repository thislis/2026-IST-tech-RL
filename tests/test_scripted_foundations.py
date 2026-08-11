"""BASE-S01~S03 unit tests for policies, semantic decoding, and navigation."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from blackout_rl import (
    ActionPolicy,
    AStarPlanner,
    FixedDirectionPolicy,
    GridCell,
    NoOpPolicy,
    RandomPolicy,
    SemanticMapDecoder,
    WaypointFollower,
    cell_center_normalized,
    grid_line,
    normalized_to_cell,
    normalized_to_pixel,
    path_cost,
    pixel_to_normalized,
)


ROOT = Path(__file__).resolve().parents[1]
LIVE_EVIDENCE = ROOT / "logs" / "base_s01_s03_live_navigation.json"


def empty_graphic(height: int = 8, width: int = 8) -> np.ndarray:
    graphic = np.zeros((height, width, 11), dtype=np.float32)
    graphic[..., 0] = 1.0
    return graphic


def paint(graphic: np.ndarray, row: int, column: int, channel: int) -> None:
    graphic[row, column, :] = 0.0
    graphic[row, column, channel] = 1.0


class PolicyPrimitiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.observations = {f"unit_{index}": {} for index in range(5)}
        self.agents = tuple(self.observations)

    def test_random_policy_seed_dtype_and_range(self) -> None:
        first = RandomPolicy(101).act(self.observations, self.agents)
        second = RandomPolicy(101).act(self.observations, self.agents)
        self.assertEqual(tuple(first), self.agents)
        for agent in self.agents:
            self.assertTrue(np.array_equal(first[agent], second[agent]))
            self.assertEqual(first[agent].dtype, np.float32)
            self.assertEqual(first[agent].shape, (2,))
            self.assertTrue(np.all((-1.0 <= first[agent]) & (first[agent] <= 1.0)))

    def test_noop_policy_returns_independent_zero_actions(self) -> None:
        actions = NoOpPolicy().act(self.observations, self.agents)
        self.assertTrue(all(np.array_equal(action, np.zeros(2)) for action in actions.values()))
        actions["unit_0"][0] = 1.0
        self.assertEqual(float(actions["unit_1"][0]), 0.0)

    def test_fixed_direction_policy_normalizes_and_copies(self) -> None:
        policy = FixedDirectionPolicy((3.0, 4.0))
        actions = policy.act(self.observations, self.agents)
        for action in actions.values():
            self.assertEqual(action.dtype, np.float32)
            self.assertTrue(np.allclose(action, np.asarray((0.6, 0.8), dtype=np.float32)))
        actions["unit_0"][0] = -1.0
        self.assertAlmostEqual(float(actions["unit_1"][0]), 0.6)

    def test_policy_rejects_missing_observation(self) -> None:
        with self.assertRaises(KeyError):
            NoOpPolicy().act(self.observations, (*self.agents, "unit_5"))

    def test_policy_primitives_satisfy_evaluator_protocol(self) -> None:
        self.assertIsInstance(RandomPolicy(1), ActionPolicy)
        self.assertIsInstance(NoOpPolicy(), ActionPolicy)
        self.assertIsInstance(FixedDirectionPolicy((1.0, 0.0)), ActionPolicy)


class SemanticDecoderTests(unittest.TestCase):
    def test_top_down_pixel_and_bottom_left_world_round_trip(self) -> None:
        row, column = normalized_to_pixel((0.3125, 0.8125), width=8, height=8)
        self.assertEqual((row, column), (1, 2))
        normalized = pixel_to_normalized(row, column, width=8, height=8)
        self.assertEqual(normalized, (0.3125, 0.8125))
        self.assertEqual(
            normalized_to_cell(normalized, width_cells=2, height_cells=2),
            GridCell(0, 1),
        )

    def test_decoder_extracts_walls_storage_units_and_items(self) -> None:
        graphic = empty_graphic()
        # Bottom-right map cell (x=1,y=0) is a full 4x4 wall tile. Because image
        # rows are top-down, bottom cell pixels occupy rows 4..7.
        for row in range(4, 8):
            for column in range(4, 8):
                paint(graphic, row, column, 1)
        paint(graphic, 1, 1, 2)  # ally storage in cell (0,1)
        paint(graphic, 2, 2, 4)  # ally unit in cell (0,1)
        paint(graphic, 1, 3, 6)  # battery in cell (0,1)
        paint(graphic, 3, 3, 7)  # special item in cell (0,1)
        paint(graphic, 2, 3, 8)
        paint(graphic, 3, 2, 9)
        paint(graphic, 2, 1, 10)

        decoded = SemanticMapDecoder(resolution_scale=4).decode(graphic)
        self.assertEqual(decoded.cells("wall"), (GridCell(1, 0),))
        self.assertEqual(decoded.cells("ally_storage"), (GridCell(0, 1),))
        self.assertEqual(decoded.cells("ally_unit"), (GridCell(0, 1),))
        self.assertEqual(decoded.cells("battery"), (GridCell(0, 1),))
        self.assertEqual(decoded.cells("buff_speed"), (GridCell(0, 1),))
        self.assertEqual(decoded.cells("debuff_speed"), (GridCell(0, 1),))
        self.assertEqual(decoded.cells("buff_size"), (GridCell(0, 1),))
        self.assertEqual(decoded.cells("debuff_size"), (GridCell(0, 1),))
        self.assertTrue(decoded.wall_grid[0, 1])
        self.assertFalse(decoded.walkable_grid[0, 1])
        self.assertEqual(decoded.points("battery")[0].normalized_xy, (0.4375, 0.8125))

    def test_storage_connected_components(self) -> None:
        graphic = empty_graphic(12, 12)
        for row, column in ((1, 1), (1, 5), (5, 9)):
            paint(graphic, row, column, 2)
        decoded = SemanticMapDecoder().decode(graphic)
        components = decoded.connected_cell_components("ally_storage")
        self.assertEqual(
            components,
            ((GridCell(0, 2), GridCell(1, 2)), (GridCell(2, 1),)),
        )


class NavigationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.walkable = np.ones((7, 7), dtype=np.bool_)
        self.walkable[0:5, 3] = False
        self.start = GridCell(1, 1)
        self.goal = GridCell(5, 1)

    def test_astar_avoids_wall_and_forbids_corner_cutting(self) -> None:
        planner = AStarPlanner(self.walkable)
        path = planner.plan(self.start, self.goal)
        self.assertEqual(path[0], self.start)
        self.assertEqual(path[-1], self.goal)
        self.assertTrue(all(self.walkable[cell.y, cell.x] for cell in path))
        self.assertGreater(max(cell.y for cell in path), 4)
        self.assertGreater(path_cost(path), 4.0)
        self.assertTrue(any(not self.walkable[cell.y, cell.x] for cell in grid_line(self.start, self.goal)))
        for first, second in zip(path, path[1:]):
            dx = second.x - first.x
            dy = second.y - first.y
            if dx and dy:
                self.assertTrue(self.walkable[first.y, first.x + dx])
                self.assertTrue(self.walkable[first.y + dy, first.x])

    def test_astar_is_deterministic(self) -> None:
        planner = AStarPlanner(self.walkable)
        self.assertEqual(planner.plan(self.start, self.goal), planner.plan(self.start, self.goal))

    def test_waypoint_follower_outputs_bounded_action_and_replan_signal(self) -> None:
        follower = WaypointFollower(4, 4, stall_limit=2)
        path = (GridCell(0, 0), GridCell(1, 0), GridCell(2, 0))
        follower.set_path(path)
        start = cell_center_normalized(path[0], width_cells=4, height_cells=4)
        action = follower.action(start)
        self.assertTrue(np.allclose(action, (1.0, 0.0)))
        self.assertEqual(action.dtype, np.float32)
        follower.action(start)
        follower.action(start)
        self.assertTrue(follower.replan_required)

        middle = cell_center_normalized(path[1], width_cells=4, height_cells=4)
        self.assertTrue(np.allclose(follower.action(middle), (1.0, 0.0)))
        goal = cell_center_normalized(path[2], width_cells=4, height_cells=4)
        self.assertTrue(np.array_equal(follower.action(goal), np.zeros(2, dtype=np.float32)))
        self.assertTrue(follower.complete)

    def test_stall_signal_can_trigger_an_alternate_route(self) -> None:
        walkable = np.ones((3, 4), dtype=np.bool_)
        original = AStarPlanner(walkable).plan(GridCell(0, 1), GridCell(3, 1))
        follower = WaypointFollower(4, 3, stall_limit=1)
        follower.set_path(original)
        position = cell_center_normalized(GridCell(0, 1), width_cells=4, height_cells=3)
        follower.action(position)
        follower.action(position)
        self.assertTrue(follower.replan_required)

        alternate = AStarPlanner(walkable).plan_avoiding(
            GridCell(0, 1),
            GridCell(3, 1),
            (GridCell(1, 1),),
        )
        self.assertNotIn(GridCell(1, 1), alternate)
        self.assertNotEqual(original, alternate)


@unittest.skipUnless(LIVE_EVIDENCE.exists(), "run scripts/verify_base_s01_s03.py first")
class LiveSingleUnitNavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evidence = json.loads(LIVE_EVIDENCE.read_text())

    def test_live_battery_route_avoids_direct_obstacle(self) -> None:
        verification = self.evidence["verification"]
        self.assertTrue(verification["semantic_coordinate_alignment_passed"])
        self.assertTrue(verification["direct_route_intersects_wall"])
        self.assertTrue(verification["planned_route_avoids_walls"])
        self.assertTrue(verification["battery_pickup_confirmed"])
        self.assertTrue(verification["obstacle_avoidance_passed"])
        self.assertEqual(self.evidence["execution"]["pickup_item_id"], 1)

    def test_live_verification_moves_only_target_unit(self) -> None:
        execution = self.evidence["execution"]
        self.assertEqual(self.evidence["plan"]["agent"], "unit_0")
        self.assertEqual(execution["non_target_policy"], "NoOpPolicy")
        self.assertEqual(execution["max_non_target_displacement_normalized"], 0.0)
        self.assertTrue(execution["single_unit_motion_passed"])
        self.assertEqual(execution["visited_wall_cells"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
