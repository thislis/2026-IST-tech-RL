"""BASE-S11 absorption strategy and BASE-S12 danger-map tests."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from blackout_rl import (
    AbsorptionPhase,
    AStarPlanner,
    GridCell,
    Role,
    ScriptedTeamController,
    StrategyMode,
    WeightedAStarPlanner,
    absorption_state,
    build_danger_map,
    choose_strategy_mode,
    team_agents,
)
from tests.test_scripted_fsm import BATTERIES, observations


ROOT = Path(__file__).resolve().parents[1]
LIVE_STRATEGY = ROOT / "logs" / "base_s11_s12_strategy.json"


def normalized_time_left(elapsed_seconds: float) -> float:
    return (420.0 - elapsed_seconds) / 420.0


class AbsorptionClockTests(unittest.TestCase):
    def test_phase_boundaries_repeat_every_twenty_seconds(self) -> None:
        cases = (
            (0.0, 0, 20.0, AbsorptionPhase.POST_ABSORPTION),
            (2.99, 0, 17.01, AbsorptionPhase.POST_ABSORPTION),
            (3.0, 0, 17.0, AbsorptionPhase.COLLECT),
            (15.99, 0, 4.01, AbsorptionPhase.COLLECT),
            (16.0, 0, 4.0, AbsorptionPhase.SECURE),
            (19.99, 0, 0.01, AbsorptionPhase.SECURE),
            (20.0, 1, 20.0, AbsorptionPhase.POST_ABSORPTION),
            (36.0, 1, 4.0, AbsorptionPhase.SECURE),
        )
        for elapsed, cycle, until, phase in cases:
            with self.subTest(elapsed=elapsed):
                state = absorption_state(normalized_time_left(elapsed))
                self.assertEqual(state.cycle_index, cycle)
                self.assertAlmostEqual(state.seconds_until_absorption, until, places=5)
                self.assertEqual(state.phase, phase)

    def test_strategy_mode_prioritizes_secure_and_gates_raids(self) -> None:
        collect = absorption_state(normalized_time_left(8.0))
        secure = absorption_state(normalized_time_left(18.0))
        post = absorption_state(normalized_time_left(21.0))
        self.assertEqual(
            choose_strategy_mode(
                secure,
                own_score=0.0,
                opponent_score=0.8,
                enemy_storage_batteries=5,
            ),
            StrategyMode.SECURE,
        )
        self.assertEqual(
            choose_strategy_mode(
                collect,
                own_score=0.1,
                opponent_score=0.3,
                enemy_storage_batteries=2,
            ),
            StrategyMode.RAID,
        )
        self.assertEqual(
            choose_strategy_mode(
                post,
                own_score=0.1,
                opponent_score=0.1,
                enemy_storage_batteries=0,
            ),
            StrategyMode.FRESH_COLLECTION,
        )

    def test_secure_phase_redirects_a_carrying_worker_to_storage(self) -> None:
        controller = ScriptedTeamController(0)
        obs = observations(
            batteries=BATTERIES,
            holdings={0: 1},
            time_left=normalized_time_left(17.0),
        )
        controller.act(obs, team_agents(0))
        self.assertEqual(controller.last_strategy_mode, StrategyMode.SECURE)
        self.assertEqual(controller.phases["unit_0"], "deliver")
        self.assertIsNotNone(controller.targets["unit_0"])
        kinds = {event["kind"] for event in controller.last_events if event["agent"] == "unit_0"}
        self.assertIn("secure_delivery", kinds)
        self.assertIn("storage_assigned", kinds)


class DangerMapTests(unittest.TestCase):
    def test_worker_and_carrier_avoid_while_guard_is_attracted(self) -> None:
        walkable = np.ones((11, 11), dtype=np.bool_)
        enemy = GridCell(5, 5)
        worker = build_danger_map(walkable, (enemy,), Role.WORKER)
        carrier = build_danger_map(walkable, (enemy,), Role.CARRIER)
        guard = build_danger_map(walkable, (enemy,), Role.GUARD)
        self.assertGreater(worker.at(enemy), 1.0)
        self.assertGreater(carrier.at(enemy), worker.at(enemy))
        self.assertLess(guard.at(enemy), 1.0)
        self.assertEqual(worker.at(GridCell(0, 0)), 1.0)

    def test_weighted_worker_path_avoids_enemy_exposure(self) -> None:
        walkable = np.ones((9, 13), dtype=np.bool_)
        start = GridCell(1, 4)
        goal = GridCell(11, 4)
        enemy = GridCell(6, 4)
        geometric = AStarPlanner(walkable).plan(start, goal)
        danger = build_danger_map(
            walkable,
            (enemy,),
            Role.WORKER,
            worker_radius=2,
            worker_peak=20.0,
        )
        weighted_planner = WeightedAStarPlanner(walkable, danger.traversal_multiplier)
        safer = weighted_planner.plan(start, goal)
        self.assertIn(enemy, geometric)
        self.assertNotIn(enemy, safer)
        self.assertNotEqual(safer, geometric)
        self.assertLess(
            weighted_planner.path_cost(safer),
            weighted_planner.path_cost(geometric),
        )

    def test_guard_weighted_path_prefers_enemy_side(self) -> None:
        walkable = np.ones((9, 13), dtype=np.bool_)
        start = GridCell(1, 4)
        goal = GridCell(11, 4)
        enemy = GridCell(6, 2)
        danger = build_danger_map(walkable, (enemy,), Role.GUARD)
        planner = WeightedAStarPlanner(walkable, danger.traversal_multiplier)
        near_enemy = (
            GridCell(1, 4), GridCell(2, 3), GridCell(3, 2), GridCell(4, 2),
            GridCell(5, 2), GridCell(6, 2), GridCell(7, 2), GridCell(8, 2),
            GridCell(9, 2), GridCell(10, 3), GridCell(11, 4),
        )
        far_from_enemy = (
            GridCell(1, 4), GridCell(2, 5), GridCell(3, 6), GridCell(4, 6),
            GridCell(5, 6), GridCell(6, 6), GridCell(7, 6), GridCell(8, 6),
            GridCell(9, 6), GridCell(10, 5), GridCell(11, 4),
        )
        self.assertLess(planner.path_cost(near_enemy), planner.path_cost(far_from_enemy))


@unittest.skipUnless(LIVE_STRATEGY.is_file(), "live BASE-S11~S12 evidence not generated")
class LiveStrategyEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evidence = json.loads(LIVE_STRATEGY.read_text())

    def test_all_live_checks_pass(self) -> None:
        self.assertTrue(all(self.evidence["verification"].values()))

    def test_live_cycle_crosses_first_absorption_in_order(self) -> None:
        transitions = self.evidence["execution"]["strategy_transitions"]
        modes = [entry["mode"] for entry in transitions]
        self.assertEqual(modes[:4], [
            "fresh_collection", "collection", "secure", "fresh_collection"
        ])
        self.assertEqual(transitions[3]["cycle_index"], 1)
        self.assertGreaterEqual(self.evidence["execution"]["completed_elapsed_seconds"], 20.0)

    def test_live_map_danger_audit_prefers_safer_route(self) -> None:
        audit = self.evidence["danger_route_audit"]
        self.assertNotEqual(audit["geometric_path"], audit["worker_weighted_path"])
        self.assertLess(
            audit["worker_weighted_cost"],
            audit["geometric_path_worker_weighted_cost"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
