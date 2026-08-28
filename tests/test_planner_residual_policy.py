"""Planner-residual checkpoint inference contract."""

from __future__ import annotations

import unittest

import numpy as np

from blackout_rl.batching import team_agents
from blackout_rl.policy import DeterministicCheckpointPolicy


class _Adapter:
    def act(self, observations):
        agents = tuple(observations)
        return {
            agent: np.asarray((0.0, 0.0) if row == 0 else (1.0, 0.0), dtype=np.float32)
            for row, agent in enumerate(agents)
        }


class _Planner:
    def act(self, observations, agents):
        return {agent: np.asarray((0.0, 1.0), dtype=np.float32) for agent in agents}

    def reset(self):
        pass


class PlannerResidualPolicyTests(unittest.TestCase):
    def test_zero_neural_action_falls_back_and_direction_overrides(self) -> None:
        policy = DeterministicCheckpointPolicy.__new__(DeterministicCheckpointPolicy)
        policy.team = 0
        policy._guardrail_mode = "planner_residual_v1"
        policy._adapter = _Adapter()
        policy._safety_controller = _Planner()
        agents = team_agents(0)
        observations = {agent: {} for agent in agents}

        actions = policy.act(observations, agents)

        self.assertTrue(np.array_equal(actions[agents[0]], (0.0, 1.0)))
        self.assertTrue(np.array_equal(actions[agents[1]], (1.0, 0.0)))


if __name__ == "__main__":
    unittest.main()
