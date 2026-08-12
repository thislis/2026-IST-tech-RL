"""BASE-R06~R09 rollout, GAE, PPO update, and diagnostics tests."""

from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from blackout_rl import (
    GRAPHIC_CHANNEL_NAMES,
    IPPOActorCritic,
    NoOpPolicy,
    PPOConfig,
    PPO_DIAGNOSTIC_SCHEMA_VERSION,
    PPODiagnosticLogger,
    ParallelRolloutCollector,
    TeamTrainingReward,
    generalized_advantage_estimate,
    ppo_update,
)
from blackout_rl.batching import canonical_agents, team_agents
from blackout_rl.observation import VECTOR_SIZE


def mock_vector(step: int, agent_index: int) -> np.ndarray:
    vector = np.zeros(VECTOR_SIZE, dtype=np.float32)
    for unit in range(10):
        start = unit * 9
        vector[start : start + 2] = (
            (unit + step) % 10 / 10.0,
            (9 - unit + step) % 10 / 10.0,
        )
        vector[start + 2] = 1.0 if unit < 5 else -1.0
        vector[start + 3 + (unit % 6)] = 1.0
    vector[90 + (agent_index % 3)] = 1.0
    vector[93:] = (step / 20.0, (step + 1) / 20.0, 1.0 - step / 20.0)
    return vector


def mock_graphic(step: int) -> np.ndarray:
    ids = (
        np.arange(16 * 16, dtype=np.int64).reshape(16, 16) + step
    ) % len(GRAPHIC_CHANNEL_NAMES)
    return np.eye(len(GRAPHIC_CHANNEL_NAMES), dtype=np.float32)[ids]


class MockParallelEnv:
    """Two-step episodes alternating true termination and time-limit truncation."""

    def __init__(self) -> None:
        self.agents = list(canonical_agents())
        self.episode = -1
        self.step_in_episode = 0
        self.actions_seen: list[dict[str, np.ndarray]] = []
        self.reset_seeds: list[int | None] = []

    def _observations(self) -> dict[str, dict[str, np.ndarray]]:
        return {
            agent: {
                "vector": mock_vector(self.step_in_episode, index),
                "graphic": mock_graphic(self.step_in_episode),
            }
            for index, agent in enumerate(canonical_agents())
        }

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        del options
        self.episode += 1
        self.step_in_episode = 0
        self.agents = list(canonical_agents())
        self.reset_seeds.append(seed)
        observations = self._observations()
        return observations, {agent: {} for agent in self.agents}

    def step(self, actions: dict[str, np.ndarray]) -> tuple[dict, dict, dict, dict, dict]:
        self.actions_seen.append(actions)
        self.assert_valid_actions(actions)
        self.step_in_episode += 1
        boundary = self.step_in_episode == 2
        terminated = boundary and self.episode % 2 == 0
        truncated = boundary and self.episode % 2 == 1
        observations = self._observations()
        rewards = {
            agent: float((index % 5) + self.step_in_episode)
            for index, agent in enumerate(canonical_agents())
        }
        terminations = {agent: terminated for agent in canonical_agents()}
        truncations = {agent: truncated for agent in canonical_agents()}
        info = {
            "episode": self.episode,
            "score_0": self.step_in_episode / 100.0,
            "score_1": 0.0,
        }
        if terminated:
            info["winner"] = 0
        infos = {agent: dict(info) for agent in canonical_agents()}
        if boundary:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def assert_valid_actions(self, actions: dict[str, np.ndarray]) -> None:
        if set(actions) != set(canonical_agents()):
            raise AssertionError("parallel step did not receive all ten agents")
        for action in actions.values():
            if action.shape != (2,) or action.dtype != np.float32:
                raise AssertionError("invalid action shape or dtype")
            if not np.all((-1.0 <= action) & (action <= 1.0)):
                raise AssertionError("action outside official bounds")


def small_model() -> IPPOActorCritic:
    return IPPOActorCritic(
        hidden_dim=32,
        entity_dim=8,
        context_dim=8,
        graphic_dim=16,
        slot_embedding_dim=6,
    )


class GAEAndBufferTests(unittest.TestCase):
    def test_truncation_bootstraps_but_does_not_leak_to_next_episode(self) -> None:
        reward = torch.tensor([[1.0], [1.0], [1.0]])
        value = torch.zeros_like(reward)
        next_value = torch.tensor([[0.0], [5.0], [100.0]])
        terminated = torch.tensor([[False], [False], [True]])
        truncated = torch.tensor([[False], [True], [False]])
        advantage, returns = generalized_advantage_estimate(
            reward,
            value,
            next_value,
            terminated,
            truncated,
            gamma=0.9,
            gae_lambda=1.0,
        )
        self.assertTrue(torch.allclose(advantage[:, 0], torch.tensor([5.95, 5.5, 1.0])))
        self.assertTrue(torch.equal(advantage, returns))

    def test_termination_ignores_even_nonzero_next_value(self) -> None:
        advantage, _ = generalized_advantage_estimate(
            torch.tensor([[2.0]]),
            torch.tensor([[0.5]]),
            torch.tensor([[999.0]]),
            torch.tensor([[True]]),
            torch.tensor([[False]]),
            gamma=0.99,
            gae_lambda=0.95,
        )
        self.assertAlmostEqual(float(advantage), 1.5)


class ParallelCollectorTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2206)
        self.env = MockParallelEnv()
        self.model = small_model()
        self.collector = ParallelRolloutCollector(
            self.env,
            self.model,
            NoOpPolicy(),
            learning_team=0,
        )

    def test_parallel_collector_records_all_agent_fields_and_episode_masks(self) -> None:
        buffer = self.collector.collect(5, seed=606)
        batch = buffer.as_batch(gamma=0.99, gae_lambda=0.95)

        self.assertEqual(len(buffer), 5)
        self.assertEqual(len(batch), 25)
        self.assertEqual(tuple(batch.vector.shape), (25, 96))
        self.assertEqual(tuple(batch.graphic.shape), (25, 11, 16, 16))
        self.assertEqual(tuple(batch.action_index.shape), (25,))
        self.assertEqual(tuple(batch.old_log_prob.shape), (25,))
        self.assertEqual(tuple(batch.old_value.shape), (25,))
        self.assertEqual(tuple(batch.reward.shape), (25,))
        self.assertEqual(tuple(batch.terminated.shape), (25,))
        self.assertEqual(tuple(batch.truncated.shape), (25,))
        self.assertTrue(torch.equal(batch.slot_id[:5], torch.arange(5)))
        self.assertTrue(torch.isfinite(batch.advantage).all())
        self.assertAlmostEqual(float(batch.advantage.mean()), 0.0, places=5)
        self.assertAlmostEqual(float(batch.advantage.std(unbiased=False)), 1.0, places=5)

        starts = batch.episode_start.reshape(5, 5).all(dim=1)
        self.assertTrue(torch.equal(starts, torch.tensor([True, False, True, False, True])))
        self.assertTrue(batch.terminated.reshape(5, 5)[1].all())
        self.assertTrue(batch.truncated.reshape(5, 5)[3].all())
        self.assertEqual(self.collector.episodes_completed, 2)
        self.assertEqual(self.env.reset_seeds, [606, None, None])
        self.assertEqual(len(self.env.actions_seen), 5)
        for actions in self.env.actions_seen:
            for agent in team_agents(1):
                self.assertTrue(np.array_equal(actions[agent], np.zeros(2, dtype=np.float32)))

    def test_collector_continues_across_calls_without_silent_reseed(self) -> None:
        first = self.collector.collect(1, seed=7)
        second = self.collector.collect(1)
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        self.assertEqual(self.env.reset_seeds, [7, None])
        with self.assertRaisesRegex(ValueError, "only be supplied"):
            self.collector.collect(1, seed=8)

    def test_score_delta_team_reward_is_connected_to_collector(self) -> None:
        env = MockParallelEnv()
        reward_transform = TeamTrainingReward(learning_team=0)
        collector = ParallelRolloutCollector(
            env,
            small_model(),
            NoOpPolicy(),
            learning_team=0,
            reward_transform=reward_transform,
        )
        batch = collector.collect(2, seed=10).as_batch(
            gamma=0.99, gae_lambda=0.95, normalize_advantage=False
        )
        rewards = batch.reward.reshape(2, 5)
        self.assertTrue(torch.allclose(rewards[0], torch.full((5,), 0.01)))
        self.assertTrue(torch.allclose(rewards[1], torch.full((5,), 1.0)))


class PPOUpdateTests(unittest.TestCase):
    def test_clipped_update_changes_parameters_and_reports_all_diagnostics(self) -> None:
        torch.manual_seed(2208)
        env = MockParallelEnv()
        model = small_model()
        collector = ParallelRolloutCollector(env, model, NoOpPolicy(), learning_team=0)
        batch = collector.collect(6, seed=8).as_batch(gamma=0.99, gae_lambda=0.95)
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        diagnostics = ppo_update(
            model,
            optimizer,
            batch,
            config=PPOConfig(
                update_epochs=3,
                minibatch_size=7,
                clip_coef=0.2,
                value_clip_coef=0.2,
                max_grad_norm=0.25,
            ),
        )

        self.assertTrue(any(not torch.equal(before[name], value) for name, value in model.named_parameters()))
        report = diagnostics.to_dict()
        expected = {
            "policy_loss",
            "value_loss",
            "entropy",
            "approximate_kl",
            "clip_fraction",
            "explained_variance",
            "gradient_norm",
            "epochs_completed",
            "minibatches",
            "samples",
            "early_stopped",
        }
        self.assertEqual(set(report), expected)
        for name in expected - {"early_stopped"}:
            self.assertTrue(np.isfinite(report[name]), name)
        self.assertGreaterEqual(diagnostics.approximate_kl, -1e-6)
        self.assertGreaterEqual(diagnostics.clip_fraction, 0.0)
        self.assertLessEqual(diagnostics.clip_fraction, 1.0)
        self.assertGreater(diagnostics.entropy, 0.0)
        self.assertGreater(diagnostics.gradient_norm, 0.0)
        self.assertEqual(diagnostics.epochs_completed, 3)
        self.assertEqual(diagnostics.samples, 30)

        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "ppo.jsonl"
            record = PPODiagnosticLogger(log_path).log(
                diagnostics,
                update=2,
                global_step=30,
                config=PPOConfig(update_epochs=3, minibatch_size=7),
            )
            round_tripped = json.loads(log_path.read_text())
        self.assertEqual(record, round_tripped)
        self.assertEqual(record["schema_version"], PPO_DIAGNOSTIC_SCHEMA_VERSION)
        self.assertEqual(record["global_step"], 30)
        self.assertEqual(set(record["metrics"]), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
