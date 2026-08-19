from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from blackout_rl.self_play import (
    MatchupResult, OpponentSuite, SideBalancedSampler, SnapshotPool, StabilityThresholds,
    checkpoint_evaluation_matrix, past_opponent_regressions, self_play_stable,
    should_expand_to_psro, snapshot_from_checkpoint,
)


class SelfPlayTests(unittest.TestCase):
    def test_pool_is_bounded_immutable_and_can_mix_latest_with_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths=[]
            for i in range(3):
                path=Path(tmp)/f"{i}.pt"; path.write_bytes(f"checkpoint-{i}".encode()); paths.append(path)
            pool=SnapshotPool(capacity=2, latest_probability=1, seed=4)
            for i,path in enumerate(paths): pool.add(snapshot_from_checkpoint(path,generation=i,global_step=i*10))
            self.assertEqual(len(pool.snapshots),2); self.assertEqual(pool.sample().generation,2)
            paths[-1].write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError,"changed"): pool.snapshots[-1].validate_artifact()

    def test_side_sampling_is_balanced_at_every_prefix(self) -> None:
        sampler=SideBalancedSampler(first_side=1)
        sides=[sampler.sample() for _ in range(11)]
        self.assertTrue(sampler.balanced); self.assertEqual(sides[:4],[1,0,1,0])

    def test_stability_thresholds_and_regression_suite(self) -> None:
        self.assertTrue(self_play_stable([{"entropy":.5,"approximate_kl":.01,"value_error":1}], StabilityThresholds()))
        self.assertEqual(past_opponent_regressions({"old":.4},{"old":.6},tolerance=.1),("old",))
        suite=OpponentSuite("random","scripted","ippo",("m1","m2"),"latest")
        self.assertEqual(len(suite.all_ids()),6)

    def test_matrix_requires_every_directed_paired_matchup(self) -> None:
        rows=[MatchupResult("a","b",("p1",),(1,),.5,0), MatchupResult("b","a",("p1",),(1,),.5,0)]
        matrix=checkpoint_evaluation_matrix(("a","b"),rows)
        self.assertEqual(matrix["a"]["b"]["win_rate"],.5)

    def test_psro_is_deferred_until_snapshot_pool_saturates(self) -> None:
        self.assertFalse(should_expand_to_psro(generations=7,plateau_generations=7,cyclic_regressions=7))
        self.assertTrue(should_expand_to_psro(generations=8,plateau_generations=4,cyclic_regressions=2))

    def test_frozen_checkpoint_rejects_live_learner_artifact(self) -> None:
        from blackout_rl.self_play import FrozenCheckpointOpponent
        checkpoint=Path(__file__).resolve().parents[1]/"checkpoints/base_r13_bc_warm_start.pt"
        snapshot=snapshot_from_checkpoint(checkpoint,generation=0,global_step=0)
        with self.assertRaisesRegex(ValueError,"live learner"):
            FrozenCheckpointOpponent(snapshot,team=1,learner_sha256=snapshot.checkpoint_sha256)

    def test_agent21_24_live_evidence_satisfies_declared_gates(self) -> None:
        root = Path(__file__).resolve().parents[1]
        payload = json.loads((root / "logs/phase3_agent21_24_self_play.json").read_text())
        self.assertEqual(payload["generations"], 8)
        self.assertEqual(len(payload["agent21"]["history"]), 8)
        self.assertTrue(payload["agent21"]["stable"])
        self.assertTrue(payload["snapshot_pool"]["saturated"])
        self.assertEqual(payload["snapshot_pool"]["side_counts"], [4, 4])
        ids = payload["agent22"]["major_generations"]
        self.assertEqual(len(ids), 4)
        for model in ids:
            self.assertEqual(set(payload["agent22"]["matrix"][model]), set(ids) - {model})
        self.assertTrue(payload["agent23"]["passes"])
        self.assertTrue(payload["agent24"]["pool_saturated_before_decision"])
        self.assertFalse(payload["agent24"]["expand_to_psro"])


if __name__ == "__main__": unittest.main(verbosity=2)
