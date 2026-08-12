"""Generated evidence checks for BASE-S14 and BASE-S15."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from blackout_rl.logging_schema import validate_series_log


ROOT = Path(__file__).resolve().parents[1]
BASE14 = ROOT / "logs" / "base_s14_scripted_vs_random.json"
BASE15 = ROOT / "logs" / "base_s15_special_item_ablation.json"


@unittest.skipUnless(BASE14.is_file(), "BASE-S14 evaluation not generated")
class ScriptedVsRandomEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.series = json.loads(BASE14.read_text())

    def test_paired_held_out_series_is_valid_and_balanced(self) -> None:
        validate_series_log(self.series)
        config = self.series["series_config"]
        self.assertTrue(config["held_out"])
        self.assertTrue(config["side_swap"])
        self.assertEqual(self.series["summary"]["episodes"], 2 * len(config["seeds"]))
        self.assertTrue(self.series["summary"]["side_bias_diagnostics"]["evaluator_side_attribution_passed"])

    def test_scripted_agent_has_consistent_random_advantage(self) -> None:
        summary = self.series["summary"]
        self.assertGreater(summary["win_rate"], 0.5)
        self.assertGreater(summary["mean_model_score_diff"], 0.0)
        self.assertGreater(summary["by_model_side"]["A"]["win_rate"], 0.5)
        self.assertGreater(summary["by_model_side"]["B"]["win_rate"], 0.5)

    def test_winner_is_terminal_and_every_episode_finishes_cleanly(self) -> None:
        for episode in self.series["episodes"]:
            self.assertEqual(episode["winner"]["source"], "terminal_info.winner")
            self.assertFalse(episode["rewards"]["unity_shaping_used_for_winner"])
            self.assertTrue(episode["termination"]["all_agents_terminated"])
            self.assertFalse(episode["termination"]["truncated"])


@unittest.skipUnless(BASE15.is_file(), "BASE-S15 ablation not generated")
class SpecialItemAblationEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ablation = json.loads(BASE15.read_text())

    def test_ablation_uses_same_seeds_and_side_swap(self) -> None:
        battery = self.ablation["battery_only"]
        special = self.ablation["special_items"]
        self.assertEqual(battery["seeds"], special["seeds"])
        self.assertTrue(battery["side_swap"] and special["side_swap"])
        self.assertGreater(special["special_item_event_totals"].get("special_item_pickup", 0), 0)

    def test_special_item_candidate_was_not_promoted_without_improvement(self) -> None:
        delta = self.ablation["delta_special_minus_battery"]
        self.assertEqual(delta["win_rate"], 0.0)
        self.assertLess(delta["mean_model_score_diff"], 0.0)
        self.assertGreater(delta["mean_episode_steps"], 0.0)
        self.assertFalse(self.ablation["decision"]["promote_special_items"])

    def test_default_policy_matches_recorded_promotion_decision(self) -> None:
        decision = self.ablation["decision"]
        config = json.loads((ROOT / decision["default_policy_config"]).read_text())
        self.assertEqual(config["special_items"], decision["promote_special_items"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
