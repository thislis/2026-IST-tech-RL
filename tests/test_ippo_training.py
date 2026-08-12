"""Stopping and device-selection tests for the IPPO-vs-scripted trainer."""

from __future__ import annotations

import unittest

import torch

from blackout_rl import NoOpPolicy, ParallelRolloutCollector
from blackout_rl.ippo_training import (
    online_imitation_update,
    passed_win_rate,
    required_wins,
    select_training_device,
)
from tests.test_ppo_training import MockParallelEnv, small_model


class DeviceSelectionTests(unittest.TestCase):
    def test_explicit_cpu_is_stable(self) -> None:
        selected = select_training_device("cpu")
        self.assertEqual(selected.resolved, "cpu")
        self.assertEqual(selected.requested, "cpu")

    def test_unknown_device_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "auto, cpu, mps"):
            select_training_device("gpu")


class WinRateStoppingTests(unittest.TestCase):
    def test_seven_of_ten_is_required_for_seventy_percent(self) -> None:
        self.assertEqual(required_wins(10, 0.70), 7)
        self.assertFalse(passed_win_rate(wins=6, episodes=10, threshold=0.70))
        self.assertTrue(passed_win_rate(wins=7, episodes=10, threshold=0.70))

    def test_non_integral_threshold_rounds_up(self) -> None:
        self.assertEqual(required_wins(6, 0.70), 5)


class OnlineImitationTests(unittest.TestCase):
    def test_teacher_update_changes_actor_and_reports_accuracy(self) -> None:
        model = small_model()
        collector = ParallelRolloutCollector(
            MockParallelEnv(), model, NoOpPolicy(), learning_team=0, teacher=NoOpPolicy()
        )
        batch = collector.collect(4, seed=9).as_batch(gamma=0.99, gae_lambda=0.95)
        before = model.actor.weight.detach().clone()
        metrics = online_imitation_update(
            model, torch.optim.Adam(model.parameters(), lr=1e-3), batch, minibatch_size=7
        )
        self.assertFalse(torch.equal(before, model.actor.weight))
        self.assertGreater(metrics.loss, 0.0)
        self.assertGreaterEqual(metrics.accuracy, 0.0)
        self.assertLessEqual(metrics.accuracy, 1.0)
        self.assertEqual(metrics.samples, len(batch))


if __name__ == "__main__":
    unittest.main(verbosity=2)
