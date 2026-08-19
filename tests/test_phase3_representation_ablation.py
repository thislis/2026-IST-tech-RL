"""Evidence-contract tests for the AGENT-06/08/11 live ablation."""

from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "logs/phase3_representation_ablation.json"


@unittest.skipUnless(EVIDENCE.is_file(), "representation ablation not generated")
class RepresentationAblationEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = json.loads(EVIDENCE.read_text())

    def test_all_arms_share_fixed_dev_seeds_and_training_budget(self) -> None:
        self.assertEqual(self.payload["dev_seeds"], [3101, 3102, 3103, 3104, 3105])
        self.assertEqual(
            self.payload["training_budget"],
            {
                "gradient_steps_per_arm": 1500,
                "batch_size": 256,
                "learning_rate": 0.0001,
                "auxiliary_coef": 0.1,
            },
        )
        for arm in self.payload["arms"].values():
            summary = arm["summary"]
            self.assertEqual(summary["episodes"], 10)
            self.assertTrue(summary["side_bias_diagnostics"]["balanced_model_side_exposure"])
            self.assertTrue(summary["side_bias_diagnostics"]["evaluator_side_attribution_passed"])
            self.assertFalse(summary["winner_derived_from_unity_shaping"])

    def test_attention_and_global_local_are_not_promoted_on_score_regression(self) -> None:
        attention = self.payload["agent06_attention_minus_flatten"]
        global_local = self.payload["agent08_global_local_minus_legacy"]
        self.assertEqual(attention["win_rate_delta"], 0.0)
        self.assertLess(attention["score_diff_delta"], 0.0)
        self.assertEqual(global_local["win_rate_delta"], 0.0)
        self.assertLess(global_local["score_diff_delta"], 0.0)

    def test_no_auxiliary_target_passes_win_and_score_gate(self) -> None:
        self.assertEqual(self.payload["agent11_selected_targets"], [])
        self.assertEqual(
            set(self.payload["agent11_auxiliary_results"]),
            {"role", "holding_item", "seconds_to_absorption", "score_delta"},
        )
        for result in self.payload["agent11_auxiliary_results"].values():
            self.assertFalse(
                result["win_rate_delta"] > 0.0
                and result["score_diff_delta"] >= 0.0
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
