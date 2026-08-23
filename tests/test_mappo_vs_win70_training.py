from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from blackout_rl import IPPOActorCritic, load_checkpoint
from blackout_rl.mappo import initialize_mappo_from_ippo
from scripts.train_mappo_vs_win70 import (
    CyclicSeedStream,
    load_mappo_training_checkpoint,
    promotion_evaluation_eligible,
    required_target_wins,
    save_mappo_training_checkpoint,
    target_reached,
)


ROOT = Path(__file__).resolve().parents[1]


class MAPPOVsWin70TrainingTests(unittest.TestCase):
    def test_episode_seed_stream_is_cyclic_and_resumable(self) -> None:
        stream = CyclicSeedStream((3001, 3002), cursor=1)
        self.assertEqual(stream.next_seed(), 3002)
        restored = CyclicSeedStream.from_state(
            stream.state_dict(), expected_seeds=(3001, 3002)
        )
        self.assertEqual(restored.next_seed(), 3001)
        with self.assertRaisesRegex(ValueError, "differs"):
            CyclicSeedStream.from_state(
                stream.state_dict(), expected_seeds=(3001, 3003)
            )

    def test_eighty_five_percent_requires_nine_of_ten_side_swapped_games(self) -> None:
        self.assertEqual(required_target_wins(10, 0.85), 9)
        self.assertFalse(target_reached({"episodes": 10, "wins": 8}, 0.85))
        self.assertTrue(target_reached({"episodes": 10, "wins": 9}, 0.85))
        committed = [3101, 3102, 3103, 3104, 3105]
        self.assertTrue(promotion_evaluation_eligible(committed, committed))
        self.assertFalse(promotion_evaluation_eligible([3101], committed))

    def test_training_checkpoint_restores_actor_critic_optimizer_and_metadata(self) -> None:
        torch.manual_seed(8500)
        actor = IPPOActorCritic(
            hidden_dim=16,
            entity_dim=4,
            context_dim=4,
            graphic_dim=8,
            slot_embedding_dim=4,
        )
        model = initialize_mappo_from_ippo(actor)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        config = {"learning_rate": 3e-4, "target_win_rate": 0.85}
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "resume.pt"
            save_mappo_training_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                global_step=512,
                update=1,
                training_seed=8500,
                run_id="test-run",
                opponent_checkpoint=ROOT / "checkpoints/win_70_vs_scripted.pt",
                initial_checkpoint=ROOT / "checkpoints/phase3_selfplay_gen_08.pt",
                config=config,
                best_evaluation={"win_rate": 0.2, "mean_model_score_diff": -10.0},
                collector_state={"version": "persistent_two_side_v2"},
            )
            ordinary_policy, ordinary_payload = load_checkpoint(checkpoint)
            restored, restored_optimizer, payload = load_mappo_training_checkpoint(
                checkpoint, device="cpu"
            )

        self.assertEqual(ordinary_payload["training"]["global_step"], 512)
        self.assertEqual(payload["mappo_training"]["update"], 1)
        self.assertEqual(
            payload["mappo_training"]["collector_state"]["version"],
            "persistent_two_side_v2",
        )
        self.assertEqual(restored_optimizer.param_groups[0]["lr"], 3e-4)
        for expected, actual in zip(
            model.actor_model.parameters(), ordinary_policy.actor_critic.parameters()
        ):
            self.assertTrue(torch.equal(expected, actual))
        for expected, actual in zip(model.parameters(), restored.parameters()):
            self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main(verbosity=2)
