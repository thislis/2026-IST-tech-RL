from __future__ import annotations

from pathlib import Path
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

    def test_final_promotion_requires_gpu_and_no_regression(self) -> None:
        evidence=PromotionEvidence("candidate",.6,2,.05,(),True,True,("cpu","mps"))
        self.assertTrue(promote_final_model(evidence))
        self.assertFalse(promote_final_model(PromotionEvidence("candidate",.6,2,.05,("old",),True,True,("cpu","mps"))))


if __name__ == "__main__": unittest.main(verbosity=2)
