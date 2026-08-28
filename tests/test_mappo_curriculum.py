from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

import torch

from blackout_rl import TeacherReplayBuffer
from blackout_rl.mappo import initialize_mappo_from_ippo
from blackout_rl.mappo_curriculum import (
    DEFAULT_MAPPO_CURRICULUM,
    EpisodeOpponentMixture,
    MAPPOCurriculumController,
)
from scripts.train_mappo_teacher_curriculum_v3 import (
    load_v4_checkpoint,
    save_v4_checkpoint,
)
from tests.test_ppo_training import small_model


ROOT = Path(__file__).resolve().parents[1]


class _Policy:
    def __init__(self, value: str) -> None:
        self.value = value
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def act(self, observations, agents):
        del observations
        return {agent: self.value for agent in agents}


class MAPPOCurriculumTests(unittest.TestCase):
    def test_dagger_and_bc_schedules_decay(self) -> None:
        stage = DEFAULT_MAPPO_CURRICULUM[0]
        self.assertEqual(stage.name, "planner_residual_warmup")
        self.assertEqual(stage.teacher_forcing_probability(0), 0.5)
        self.assertEqual(stage.teacher_forcing_probability(100_000), 0.0)
        self.assertEqual(stage.bc_coefficient(0), 0.2)
        self.assertAlmostEqual(stage.bc_coefficient(100_000), 0.05)

    def test_promotion_requires_both_sides_and_max_budget_is_fail_closed(self) -> None:
        controller = MAPPOCurriculumController(stage_index=1, stage_start_step=100_000)
        one_sided = {
            "win_rate": 0.8,
            "mean_model_score_diff": 20.0,
            "by_model_side": {"A": {"wins": 4}, "B": {"wins": 0}},
        }
        self.assertIsNone(controller.promotion_reason(one_sided, global_step=160_000))
        self.assertIsNone(controller.promotion_reason(one_sided, global_step=350_000))
        self.assertTrue(controller.budget_exhausted(global_step=400_000))

    def test_episode_mixture_keeps_one_policy_until_reset_and_restores_rng(self) -> None:
        policies = {"weak": _Policy("weak"), "full": _Policy("full")}
        mixture = EpisodeOpponentMixture(
            policies, {"weak": 0.75, "full": 0.25}, seed=17
        )
        mixture.reset()
        first = mixture.act({}, ("unit_0",))["unit_0"]
        self.assertEqual(mixture.act({}, ("unit_0",))["unit_0"], first)
        state = mixture.state_dict()

        restored = EpisodeOpponentMixture(
            {"weak": _Policy("weak"), "full": _Policy("full")},
            {"weak": 0.75, "full": 0.25},
            seed=999,
        )
        restored.load_state_dict(state)
        mixture.reset()
        restored.reset()
        self.assertEqual(restored.selected_id, mixture.selected_id)
        self.assertEqual(restored.selection_counts, mixture.selection_counts)

    def test_v3_checkpoint_restores_model_optimizer_replay_and_runtime(self) -> None:
        model = initialize_mappo_from_ippo(small_model())
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
        replay = TeacherReplayBuffer(16, seed=5)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "v3.pt"
            save_v4_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                replay=replay,
                global_step=2048,
                update=1,
                run_id="v3-test",
                seed=9500,
                initial_checkpoint=ROOT / "checkpoints/win_70_vs_scripted.pt",
                opponent_checkpoint=ROOT / "checkpoints/win_70_vs_scripted.pt",
                config={"learning_rate": 5e-5, "seed": 9500},
                runtime_state={"curriculum": {"stage_index": 0}},
                best_evaluation=None,
                planner_residual=True,
            )
            restored, restored_optimizer, restored_replay, payload = load_v4_checkpoint(
                checkpoint, device="cpu", replay_capacity=16
            )

        self.assertEqual(payload["mappo_v4_training"]["global_step"], 2048)
        self.assertEqual(
            payload["inference_guardrail"]["mode"], "planner_residual_v1"
        )
        self.assertEqual(
            payload["mappo_v4_training"]["runtime_state"]["curriculum"]["stage_index"],
            0,
        )
        self.assertEqual(len(restored_replay), 0)
        self.assertEqual(restored_optimizer.param_groups[0]["lr"], 5e-5)
        for expected, actual in zip(model.parameters(), restored.parameters()):
            self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main(verbosity=2)
