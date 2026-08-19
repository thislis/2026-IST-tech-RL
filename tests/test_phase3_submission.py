from __future__ import annotations

from pathlib import Path
import json
import unittest
import torch

from blackout_rl.submission_validation import (
    PromotionEvidence, assert_deterministic_inference, benchmark_inference, clean_room_load,
    promote_final_model, select_lightweight_candidate,
)
from submission.policy import load_policy


ROOT=Path(__file__).resolve().parents[1]
CHECKPOINT=ROOT/"checkpoints/base_r13_bc_warm_start.pt"


class SubmissionTests(unittest.TestCase):
    def test_standalone_policy_is_deterministic_and_batched_contract_is_strict(self) -> None:
        model=load_policy(CHECKPOINT)
        inputs=(torch.zeros(5,96),torch.zeros(5,11,96,96))
        assert_deterministic_inference(model,inputs)
        self.assertEqual(model(*inputs).shape,(5,2))
        with self.assertRaisesRegex(ValueError,"5,96"): model(torch.zeros(4,96),torch.zeros(4,11,96,96))
        with self.assertRaisesRegex(TypeError,"float32"): model(torch.zeros(5,96,dtype=torch.float64),inputs[1])

    def test_clean_room_has_only_policy_and_checkpoint_artifacts(self) -> None:
        result=clean_room_load(ROOT/"submission/policy.py",CHECKPOINT)
        self.assertTrue(result["passed"]); self.assertEqual(result["files"],["checkpoint.pt","policy.py"])

    def test_latency_and_lightweight_selection_preserve_performance(self) -> None:
        model=load_policy(CHECKPOINT); result=benchmark_inference(model,(torch.zeros(5,96),torch.zeros(5,11,32,32)),runs=2)
        self.assertGreater(result.parameters,0); self.assertGreater(result.median_ms,0)
        selected=select_lightweight_candidate([
            {"candidate":"baseline","win_rate":.6,"latency_ms":4,"parameters":100},
            {"candidate":"small","win_rate":.595,"latency_ms":2,"parameters":50},
            {"candidate":"bad","win_rate":.4,"latency_ms":1,"parameters":10},])
        self.assertEqual(selected,"small")

    def test_lightweight_selection_can_preserve_score_as_well_as_win_rate(self) -> None:
        selected=select_lightweight_candidate([
            {"candidate":"baseline","win_rate":.6,"mean_score_diff":3,"latency_ms":4,"parameters":100},
            {"candidate":"small","win_rate":.6,"mean_score_diff":2.5,"latency_ms":2,"parameters":50},
            {"candidate":"score_regression","win_rate":.6,"mean_score_diff":-10,"latency_ms":1,"parameters":20},
        ],max_score_drop=1)
        self.assertEqual(selected,"small")

    def test_final_promotion_requires_gpu_and_no_regression(self) -> None:
        evidence=PromotionEvidence("candidate",.6,2,.05,(),True,True,("cpu","mps"))
        self.assertTrue(promote_final_model(evidence))
        self.assertFalse(promote_final_model(PromotionEvidence("candidate",.6,2,.05,("old",),True,True,("cpu","mps"))))

    def test_agent29_32_empirical_evidence_is_fail_closed(self) -> None:
        payload=json.loads((ROOT/"logs/phase3_agent29_32_submission.json").read_text())
        agent29=payload["agent29"]
        self.assertEqual(agent29["selected"],"legacy_no_local_encoder")
        baseline=agent29["candidates"]["baseline"]
        selected=agent29["candidates"][agent29["selected"]]
        self.assertLess(selected["parameters"],baseline["parameters"])
        self.assertLess(selected["latency_ms"],baseline["latency_ms"])
        self.assertGreaterEqual(selected["mean_score_diff"],baseline["mean_score_diff"])
        self.assertEqual(set(payload["agent31"]["devices_passed"]),{"cpu","mps"})
        self.assertTrue(payload["agent31"]["passed"])
        self.assertEqual(payload["agent32"]["head_to_head"]["episodes"],10)
        self.assertFalse(payload["agent32"]["candidate_gate_passed"])
        self.assertFalse(payload["agent32"]["decision"]["submission_ready"])


if __name__ == "__main__": unittest.main(verbosity=2)
