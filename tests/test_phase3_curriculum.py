from __future__ import annotations

import json
import numpy as np
from pathlib import Path
import unittest

from blackout_rl.curriculum import (
    DEFAULT_CURRICULUM, CurriculumStage, LinearShapingAnnealer, PotentialNavigationShaping,
    NavigationShapedTeamReward, PromotionComparison, StageEvaluation, combined_reward, evaluate_stage_gate,
    promote_special_item_curriculum, separate_stage_results, validate_curriculum,
)
from blackout_rl.training_reward import RewardMode, TrainingRewardConfig


def _observation(x: float, target_column: int) -> dict[str, np.ndarray]:
    vector = np.zeros(96, dtype=np.float32)
    vector[0] = x
    vector[1] = 0.5
    vector[3] = 1.0
    graphic = np.zeros((5, 5, 11), dtype=np.float32)
    graphic[2, target_column, 6] = 1.0
    return {"vector": vector, "graphic": graphic}


class CurriculumTests(unittest.TestCase):
    def test_declared_stages_are_ordered_and_evaluated_separately(self) -> None:
        validate_curriculum(DEFAULT_CURRICULUM)
        result = StageEvaluation(CurriculumStage.RANDOM, "random-v1", .9, 12, .1, 10)
        self.assertTrue(evaluate_stage_gate(DEFAULT_CURRICULUM[0], result))
        separated = separate_stage_results([result])
        self.assertEqual(set(separated), {"random"})

    def test_potential_shaping_rewards_progress_and_terminal_has_no_residual(self) -> None:
        shaping = PotentialNavigationShaping(gamma=.99)
        self.assertGreater(shaping.reward(10, 8), 0)
        self.assertEqual(shaping.reward(0, 0, terminal=True), 0)

    def test_annealing_leaves_score_and_terminal_as_final_reward(self) -> None:
        schedule = LinearShapingAnnealer(1, 0, 100)
        self.assertEqual(combined_reward(score_reward=2, terminal_reward=1, navigation_reward=10,
                                         environment_step=100, annealer=schedule), 3)
        self.assertGreater(schedule.weight(0), schedule.weight(50))

    def test_live_navigation_reward_tracks_transition_and_anneals_to_zero(self) -> None:
        shaped = NavigationShapedTeamReward(
            0,
            base_config=TrainingRewardConfig(
                mode=RewardMode.SCORE_DELTA,
                terminal_win_reward=0.0,
            ),
            gamma=1.0,
            initial_weight=1.0,
            anneal_steps=1,
        )
        agents = tuple(f"agent_{index}" for index in range(5))
        current = {agent: _observation(0.0, 4) for agent in agents}
        nearer = {agent: _observation(0.5, 4) for agent in agents}
        boundaries = {agent: False for agent in agents}
        shaped.observe_transition(current, nearer, boundaries, boundaries, agents)
        self.assertGreater(shaped.last_navigation_reward, 0.0)

        base_inputs = {agent: 0.0 for agent in agents}
        infos = {agent: {"score_0": 0.0, "score_1": 0.0} for agent in agents}
        reward = shaped(base_inputs, boundaries, boundaries, infos, agents)
        self.assertTrue(all(value > 0.0 for value in reward.values()))

        shaped.observe_transition(nearer, current, boundaries, boundaries, agents)
        self.assertEqual(shaped.last_navigation_reward, 0.0)

    def test_special_items_require_no_collection_regression(self) -> None:
        degraded = PromotionComparison(.6,.6,10,9,True,True)
        self.assertFalse(promote_special_item_curriculum(degraded))
        with self.assertRaisesRegex(ValueError, "common seeds"):
            promote_special_item_curriculum(PromotionComparison(.5,.6,0,1,False,True))

    def test_agent14_live_evidence_is_equal_budget_and_anneals_to_zero(self) -> None:
        root = Path(__file__).resolve().parents[1]
        payload = json.loads((root / "logs/phase3_agent14_navigation.json").read_text())
        self.assertTrue(payload["same_environment_steps"])
        self.assertEqual(payload["environment_steps_per_arm"], 512)
        arms = payload["arms"]
        self.assertEqual(set(arms), {"score_terminal_only", "navigation_annealed"})
        self.assertEqual(
            arms["navigation_annealed"]["training_history"][-1]["navigation_weight"],
            0.0,
        )
        self.assertFalse(payload["decision"]["promote_navigation_shaping"])


if __name__ == "__main__": unittest.main(verbosity=2)
