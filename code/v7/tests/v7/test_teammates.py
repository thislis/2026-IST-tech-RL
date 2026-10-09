"""Regression for a neural unit carrying an item while on a forbidden shrine."""
import unittest
from unittest.mock import patch
import numpy as np
from blackout_rl.batching import team_agents
from blackout_rl.mappo_v6 import ROLES
from blackout_rl.navigation import PathNotFound
from blackout_rl.scripted_fsm import ScriptedTeamController, WorkerPhase, carrier_shrine_cell
from blackout_rl.v7_2.teammates import ScriptedTeammates
from tests.test_scripted_fsm import observations, BATTERIES
from tests.test_mappo_v6 import all_observations


class TeammateTests(unittest.TestCase):
    def test_original_failure_and_excluded_neural_unit(self):
        obs = observations(batteries=BATTERIES)
        legacy = ScriptedTeamController(0, roles=ROLES, chase_radius_cells=48)
        legacy.act(obs, team_agents(0))
        legacy._runtimes['unit_0'].phase = WorkerPhase.DELIVER
        legacy._clear_target('unit_0')
        on_shrine = observations(batteries=BATTERIES, positions={0:carrier_shrine_cell(0)}, holdings={0:1})
        with self.assertRaisesRegex(PathNotFound, 'no reachable target'):
            legacy.act(on_shrine, team_agents(0))
        fixed = ScriptedTeammates(0, [0])
        fixed.act(obs, team_agents(0))
        # Even a stale externally controlled runtime must never be planned.
        fixed.controller._runtimes['unit_0'].phase = WorkerPhase.DELIVER
        actions = fixed.act(on_shrine, team_agents(0))
        self.assertEqual(set(actions), set(team_agents(0)) - {'unit_0'})
        self.assertNotIn('unit_0', fixed.controller.last_decisions)
        self.assertIsNone(fixed.controller.targets['unit_0'])
        self.assertTrue(any(np.linalg.norm(a) > 0 for a in actions.values()))

    def test_active_scripted_worker_can_leave_forbidden_start(self):
        teammate = ScriptedTeammates(0, [0])
        teammate.act(observations(batteries=BATTERIES), team_agents(0))
        teammate.controller._runtimes['unit_1'].phase = WorkerPhase.DELIVER
        teammate.controller._clear_target('unit_1')
        obs = observations(batteries=BATTERIES, positions={1:carrier_shrine_cell(0)}, holdings={1:1})
        actions = teammate.act(obs, team_agents(0))
        self.assertGreater(np.linalg.norm(actions['unit_1']), 0)
        self.assertEqual(teammate.controller.paths['unit_1'][0], carrier_shrine_cell(0))

    def test_unreachable_unit_waits_without_stopping_other_units(self):
        teammate = ScriptedTeammates(0, [0])
        obs = observations(batteries=BATTERIES, holdings={1:1})
        teammate.act(obs, team_agents(0))
        teammate.controller._runtimes['unit_1'].phase = WorkerPhase.DELIVER
        teammate.controller._clear_target('unit_1')
        with patch.object(teammate.controller, '_assign_delivery', side_effect=PathNotFound('disconnected storage')):
            actions = teammate.act(obs, team_agents(0))
        np.testing.assert_array_equal(actions['unit_1'], [0,0])
        self.assertTrue(any(np.linalg.norm(actions[a])>0 for a in ('unit_2','unit_3','unit_4')))
        self.assertEqual([e['agent'] for e in teammate.last_events if e['kind']=='assignment_unreachable'], ['unit_1'])
        with patch.object(teammate.controller, '_assign_delivery', side_effect=ValueError('invalid observation')):
            with self.assertRaises(ValueError):teammate.act(obs, team_agents(0))

    def test_team_b_multiple_slots_and_full_neural_team(self):
        teammate = ScriptedTeammates(1, [0,3])
        actions = teammate.act(all_observations(), team_agents(1))
        self.assertEqual(set(actions), {'unit_6','unit_7','unit_9'})
        teammate.reset()
        self.assertEqual(teammate.active_agents, ('unit_6','unit_7','unit_9'))
        all_neural = ScriptedTeammates(1, list(range(5)))
        self.assertEqual(all_neural.act(all_observations(), team_agents(1)), {})

    def test_wall_is_not_opened_when_start_is_exempted(self):
        teammate = ScriptedTeammates(0, [0]); c = teammate.controller
        teammate.act(observations(batteries=BATTERIES), team_agents(0))
        start = carrier_shrine_cell(0)
        c._planner.walkable_grid[start.y,start.x] = False
        planner = c._planner_avoiding([start], start=start)
        self.assertFalse(planner.walkable_grid[start.y,start.x])


if __name__ == '__main__':unittest.main()
