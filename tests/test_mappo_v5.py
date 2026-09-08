from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from blackout_rl.mappo import (
    initialize_mappo_from_ippo,
    initialize_planner_conditioned_residual,
)
from blackout_rl.mappo_curriculum_v5 import (
    DEFAULT_MAPPO_V5_CURRICULUM,
    MAPPOV5CurriculumController,
)
from blackout_rl.policy import DeterministicCheckpointPolicy
from scripts.train_mappo_planner_residual_v5 import (
    load_v5_checkpoint,
    save_v5_checkpoint,
)
from tests.test_ppo_training import small_model


ROOT = Path(__file__).resolve().parents[1]


class MAPPOV5Tests(unittest.TestCase):
    def test_preservation_gate_is_baseline_calibrated_and_short(self) -> None:
        stage = DEFAULT_MAPPO_V5_CURRICULUM[0]
        self.assertEqual(stage.name, "planner_preservation")
        self.assertEqual(stage.maximum_steps, 16_384)
        self.assertFalse(stage.use_ppo)
        self.assertEqual(stage.planner_forcing, 1.0)
        controller = MAPPOV5CurriculumController()
        summary = {
            "win_rate": 0.8,
            "mean_model_score_diff": 21.5,
            "by_model_side": {
                "A": {"wins": 5, "win_rate": 1.0},
                "B": {"wins": 3, "win_rate": 0.6},
            },
        }
        self.assertFalse(controller.gate_passed(summary, global_step=10_000))
        self.assertTrue(controller.gate_passed(summary, global_step=16_384))

    def test_post_warmup_uses_ppo_and_override_regularization_not_bc(self) -> None:
        stage = DEFAULT_MAPPO_V5_CURRICULUM[1]
        self.assertTrue(stage.use_ppo)
        self.assertEqual(stage.planner_forcing, 0.0)
        self.assertGreater(stage.override_penalty(0), stage.override_penalty(stage.maximum_steps))

    def test_v5_checkpoint_restores_head_and_inference_contract(self) -> None:
        model = initialize_mappo_from_ippo(small_model())
        initialize_planner_conditioned_residual(model, fallback_logit_bias=6.5)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "v5.pt"
            save_v5_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                global_step=2048,
                update=1,
                run_id="v5-test",
                seed=10500,
                initial_checkpoint=ROOT / "checkpoints/win_70_vs_scripted.pt",
                opponent_checkpoint=ROOT / "checkpoints/win_70_vs_scripted.pt",
                config={
                    "learning_rate": 5e-5,
                    "fallback_logit_bias": 6.5,
                },
                runtime_state={"curriculum": {"stage_index": 0}},
                best_target_evaluation=None,
                best_stage_evaluation=None,
                fallback_logit_bias=6.5,
            )
            restored, restored_optimizer, payload = load_v5_checkpoint(
                checkpoint, device="cpu"
            )
            inference = DeterministicCheckpointPolicy(checkpoint, team=0)

        self.assertIsNotNone(restored.planner_residual_head)
        self.assertEqual(payload["inference_guardrail"]["mode"], "planner_residual_v2")
        self.assertEqual(payload["inference_guardrail"]["max_overrides_per_step"], 1)
        self.assertIsNotNone(inference._planner_residual_v2_head)
        self.assertEqual(restored_optimizer.param_groups[0]["lr"], 5e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
