from __future__ import annotations

import unittest

from blackout_rl.curriculum import (
    DEFAULT_CURRICULUM, CurriculumStage, LinearShapingAnnealer, PotentialNavigationShaping,
    PromotionComparison, StageEvaluation, combined_reward, evaluate_stage_gate,
    promote_special_item_curriculum, separate_stage_results, validate_curriculum,
)


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

    def test_special_items_require_no_collection_regression(self) -> None:
        degraded = PromotionComparison(.6,.6,10,9,True,True)
        self.assertFalse(promote_special_item_curriculum(degraded))
        with self.assertRaisesRegex(ValueError, "common seeds"):
            promote_special_item_curriculum(PromotionComparison(.5,.6,0,1,False,True))


if __name__ == "__main__": unittest.main(verbosity=2)
