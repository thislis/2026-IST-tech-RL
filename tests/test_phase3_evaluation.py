"""Phase-3 seed isolation and paired-bootstrap tests."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from blackout_rl.evaluation_protocol import SeedSplits, paired_seed_bootstrap_ci


ROOT = Path(__file__).resolve().parents[1]


def episode(pair: str, seed: int, team: int, result: str, score: float) -> dict:
    return {
        "pair_id": pair,
        "seed": seed,
        "side_assignment": {"model_team": team},
        "model_result": result,
        "score": {"model_minus_opponent": score},
    }


class SeedSplitTests(unittest.TestCase):
    def test_committed_splits_are_disjoint_and_test_is_final_only(self) -> None:
        splits = SeedSplits.from_dict(json.loads((ROOT / "configs/seed_splits_v1.json").read_text()))
        self.assertEqual(splits.seeds_for("dev", purpose="model_selection"), splits.dev)
        with self.assertRaises(PermissionError):
            splits.seeds_for("test", purpose="model_selection")

    def test_overlap_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "leakage"):
            SeedSplits((1, 2), (2, 3), (4,)).validate()


class PairedBootstrapTests(unittest.TestCase):
    def test_resamples_pairs_instead_of_individual_games(self) -> None:
        episodes = [
            episode("s1", 1, 0, "win", 10), episode("s1", 1, 1, "loss", -2),
            episode("s2", 2, 0, "win", 6), episode("s2", 2, 1, "win", 2),
        ]
        result = paired_seed_bootstrap_ci(episodes, metric="win_rate", resamples=500, seed=7)
        self.assertEqual(result.pairs, 2)
        self.assertEqual(result.estimate, 0.75)
        self.assertGreaterEqual(result.lower, 0.5)
        self.assertLessEqual(result.upper, 1.0)

    def test_incomplete_side_pair_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "one game on each"):
            paired_seed_bootstrap_ci([episode("s1", 1, 0, "win", 1)], metric="score_diff")


if __name__ == "__main__":
    unittest.main(verbosity=2)
