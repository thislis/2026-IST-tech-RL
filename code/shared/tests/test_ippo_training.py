"""Stopping and device-selection tests for the IPPO-vs-scripted trainer."""

from __future__ import annotations

import unittest

import torch

from blackout_rl import NoOpPolicy, ParallelRolloutCollector
from blackout_rl.ippo_training import (
    TeacherReplayBuffer,
    online_imitation_update,
    passed_win_rate,
    required_wins,
    select_training_device,
    teacher_replay_update,
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

    def test_compressed_teacher_replay_wraps_and_updates(self) -> None:
        model = small_model()
        collector = ParallelRolloutCollector(
            MockParallelEnv(), model, NoOpPolicy(), learning_team=0, teacher=NoOpPolicy()
        )
        batch = collector.collect(4, seed=10).as_batch(gamma=0.99, gae_lambda=0.95)
        replay = TeacherReplayBuffer(capacity=15, seed=3)
        replay.add(batch)
        self.assertEqual(len(replay), 15)
        self.assertEqual(replay.graphic_ids.dtype, torch.uint8)
        metrics = teacher_replay_update(
            model,
            torch.optim.Adam(model.parameters(), lr=1e-3),
            replay,
            minibatches=2,
            minibatch_size=7,
        )
        self.assertEqual(metrics.samples, 14)
        self.assertGreaterEqual(metrics.accuracy, 0.0)

    def test_teacher_replay_ignores_unavailable_labels(self) -> None:
        model = small_model()
        collector = ParallelRolloutCollector(
            MockParallelEnv(), model, NoOpPolicy(), learning_team=0, teacher=NoOpPolicy()
        )
        batch = collector.collect(2, seed=12).as_batch(gamma=0.99, gae_lambda=0.95)
        batch.teacher_action_index.fill_(-100)
        replay = TeacherReplayBuffer(capacity=20)
        replay.add(batch)
        self.assertEqual(len(replay), 0)

    def test_teacher_replay_checkpoint_round_trip_preserves_sampling(self) -> None:
        model = small_model()
        collector = ParallelRolloutCollector(
            MockParallelEnv(), model, NoOpPolicy(), learning_team=0, teacher=NoOpPolicy()
        )
        batch = collector.collect(3, seed=13).as_batch(gamma=0.99, gae_lambda=0.95)
        replay = TeacherReplayBuffer(capacity=20, seed=4)
        replay.add(batch)
        restored = TeacherReplayBuffer(capacity=20, seed=999)
        restored.load_state_dict(replay.state_dict())
        self.assertEqual(len(restored), len(replay))
        expected = replay.sample(7, device="cpu")
        actual = restored.sample(7, device="cpu")
        for expected_tensor, actual_tensor in zip(expected, actual):
            self.assertTrue(torch.equal(expected_tensor, actual_tensor))

    def test_teacher_replay_can_restore_into_larger_capacity(self) -> None:
        model = small_model()
        collector = ParallelRolloutCollector(
            MockParallelEnv(), model, NoOpPolicy(), learning_team=0, teacher=NoOpPolicy()
        )
        batch = collector.collect(3, seed=15).as_batch(gamma=0.99, gae_lambda=0.95)
        replay = TeacherReplayBuffer(capacity=10, seed=5)
        replay.add(batch)
        restored = TeacherReplayBuffer(capacity=20, seed=999)
        restored.load_state_dict(replay.state_dict())
        self.assertEqual(len(restored), 10)
        self.assertEqual(restored.position, 10)
        restored.add(batch)
        self.assertEqual(len(restored), 20)

    def test_teacher_replay_class_balancing_rejects_missing_classes(self) -> None:
        model = small_model()
        collector = ParallelRolloutCollector(
            MockParallelEnv(), model, NoOpPolicy(), learning_team=0, teacher=NoOpPolicy()
        )
        batch = collector.collect(3, seed=18).as_batch(gamma=0.99, gae_lambda=0.95)
        replay = TeacherReplayBuffer(32, seed=18)
        replay.add(batch)
        with self.assertRaisesRegex(ValueError, "missing an action class"):
            teacher_replay_update(
                model,
                torch.optim.Adam(model.parameters()),
                replay,
                minibatches=1,
                class_balance_power=0.5,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
