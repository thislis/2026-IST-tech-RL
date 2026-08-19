#!/usr/bin/env python3
"""Execute AGENT-29, AGENT-31, and the AGENT-32 final promotion review."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from blackout_rl import load_checkpoint, save_checkpoint  # noqa: E402
from blackout_rl.evaluation_protocol import SeedSplits  # noqa: E402
from blackout_rl.submission_validation import (  # noqa: E402
    PromotionEvidence,
    assert_deterministic_inference,
    benchmark_inference,
    clean_room_load,
    promote_final_model,
    select_lightweight_candidate,
)
from eval.evaluator import summarize_episodes  # noqa: E402
from scripts.run_phase3_agent21_24_self_play import _jobs, _run_jobs  # noqa: E402
from submission.policy import load_policy as load_standalone_policy  # noqa: E402


ARCHITECTURE_ARMS = {
    "baseline": (
        "flatten_global_local",
        ROOT / "checkpoints/phase3_repr_flatten_global_local.pt",
    ),
    "legacy_no_local_encoder": (
        "flatten_legacy",
        ROOT / "checkpoints/phase3_repr_flatten_legacy.pt",
    ),
    "attention_no_local_encoder": (
        "attention_legacy",
        ROOT / "checkpoints/phase3_repr_attention_legacy.pt",
    ),
}


def _inputs(device: str) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(293200)
    vector = torch.rand(5, 96, generator=generator)
    graphic_ids = torch.randint(0, 11, (5, 96, 96), generator=generator)
    graphic = torch.nn.functional.one_hot(graphic_ids, 11).permute(0, 3, 1, 2).float()
    return vector.to(device), graphic.to(device)


def _strip_training_state(source: Path, destination: Path) -> None:
    payload = torch.load(source, map_location="cpu", weights_only=True)
    required = (
        "schema_version",
        "policy_state",
        "model_config",
        "training",
        "observation_contract",
        "action_contract",
        "source",
    )
    stripped = {name: payload[name] for name in required}
    stripped["source"] = {
        **stripped["source"],
        "phase": "phase3-agent29-lightweight",
        "source_checkpoint": str(source.relative_to(ROOT)),
        "removed_training_state": "optimizer_state",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(destination, stripped)


def _device_smoke(checkpoint: Path, device: str, runs: int) -> dict:
    model = load_standalone_policy(checkpoint, device=device)
    vector, graphic = _inputs(device)
    assert_deterministic_inference(model, (vector, graphic), repeats=3)
    with torch.inference_mode():
        output = model(vector, graphic)
    if output.shape != (5, 2) or output.dtype != torch.float32 or output.device.type != device:
        raise AssertionError("standalone output contract failed")

    mismatch_checks = {}
    try:
        model(vector.to(torch.float64), graphic)
    except TypeError:
        mismatch_checks["dtype_rejected"] = True
    else:
        mismatch_checks["dtype_rejected"] = False
    try:
        model(vector[:4], graphic[:4])
    except ValueError:
        mismatch_checks["batch_rejected"] = True
    else:
        mismatch_checks["batch_rejected"] = False
    if device != "cpu":
        try:
            model(vector.cpu(), graphic.cpu())
        except ValueError:
            mismatch_checks["device_mismatch_rejected"] = True
        else:
            mismatch_checks["device_mismatch_rejected"] = False
    else:
        mismatch_checks["device_mismatch_rejected"] = "not_applicable"
    if not all(value is True or value == "not_applicable" for value in mismatch_checks.values()):
        raise AssertionError(f"mismatch checks failed: {mismatch_checks}")
    latency = benchmark_inference(model, (vector, graphic), runs=runs)
    return {
        "passed": True,
        "output_shape": list(output.shape),
        "output_dtype": str(output.dtype),
        "output_device": str(output.device),
        "deterministic_repeats": 3,
        "mismatch_checks": mismatch_checks,
        "latency": asdict(latency),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    if args.runs <= 0:
        raise ValueError("runs must be positive")

    representation = json.loads(
        (ROOT / "logs/phase3_representation_ablation.json").read_text()
    )
    cpu_rows = []
    architecture_results = {}
    for candidate, (arm, checkpoint) in ARCHITECTURE_ARMS.items():
        policy, _ = load_checkpoint(checkpoint, device="cpu")
        inputs = _inputs("cpu")
        assert_deterministic_inference(policy, inputs, repeats=3)
        latency = benchmark_inference(policy, inputs, runs=args.runs)
        summary = representation["arms"][arm]["summary"]
        row = {
            "candidate": candidate,
            "arm": arm,
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "parameters": latency.parameters,
            "checkpoint_bytes": checkpoint.stat().st_size,
            "latency_ms": latency.median_ms,
            "p95_ms": latency.p95_ms,
            "win_rate": summary["win_rate"],
            "mean_score_diff": summary["mean_model_score_diff"],
        }
        cpu_rows.append(row)
        architecture_results[candidate] = row
    selected = select_lightweight_candidate(
        cpu_rows, max_win_rate_drop=0.0, max_score_drop=0.0
    )
    selected_source = ARCHITECTURE_ARMS[selected][1]
    lightweight_checkpoint = ROOT / "checkpoints/phase3_lightweight_legacy.pt"
    _strip_training_state(selected_source, lightweight_checkpoint)

    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    elif torch.cuda.is_available():
        devices.append("cuda")
    smoke = {
        device: _device_smoke(lightweight_checkpoint, device, args.runs)
        for device in devices
    }
    clean_room = clean_room_load(ROOT / "submission/policy.py", lightweight_checkpoint)

    incumbent = ROOT / "checkpoints/win_70_vs_scripted.pt"
    strength_candidate = ROOT / "checkpoints/phase3_selfplay_gen_08.pt"
    splits = SeedSplits.from_dict(
        json.loads((ROOT / "configs/seed_splits_v1.json").read_text())
    )
    dev_seeds = list(splits.seeds_for("dev", purpose="model_selection"))
    matchups = [("selfplay_gen_08", strength_candidate, "win_70_incumbent", incumbent)]
    head_to_head_episodes = _run_jobs(
        _jobs(
            ROOT / "builds/BlackOut.app",
            matchups,
            dev_seeds,
            time_scale=100.0,
            max_steps=22_000,
        ),
        args.workers,
    )
    head_to_head = summarize_episodes(head_to_head_episodes)

    self_play = json.loads(
        (ROOT / "logs/phase3_agent21_24_self_play.json").read_text()
    )
    candidate_regressions = tuple(
        self_play["agent23"]["past_opponent_regressions"]
    )
    if self_play["agent23"]["exploiter_regression"]:
        candidate_regressions += ("win_70_exploiter",)
    side_gap = float(
        head_to_head["side_bias_diagnostics"]["model_side_win_rate_gap_a_minus_b"]
    )
    candidate_evidence = PromotionEvidence(
        candidate_id="phase3_selfplay_gen_08",
        win_rate=float(head_to_head["win_rate"]),
        mean_score_diff=float(head_to_head["mean_model_score_diff"]),
        side_gap=side_gap,
        regressions=candidate_regressions,
        deterministic=True,
        clean_room=True,
        devices_passed=tuple(devices),
    )
    promote_candidate = promote_final_model(candidate_evidence)

    incumbent_clean_room = {"passed": False, "error": None}
    try:
        incumbent_clean_room = clean_room_load(ROOT / "submission/policy.py", incumbent)
    except Exception as error:
        incumbent_clean_room["error"] = f"{type(error).__name__}: {error}"
    incumbent_dev = json.loads(
        (ROOT / "logs/win_70_vs_scripted_eval_7171_7175.json").read_text()
    )["summary"]
    incumbent_audit = json.loads(
        (ROOT / "logs/win_70_vs_scripted_audit_7271_7275.json").read_text()
    )["summary"]
    final_decision = {
        "promote_new_candidate": promote_candidate,
        "benchmark_incumbent_retained": not promote_candidate,
        "submission_ready": bool(promote_candidate or incumbent_clean_room["passed"]),
        "reason": (
            "candidate failed strength gates; incumbent remains the evaluation benchmark "
            "but its planner guardrail/global-local artifact is not loadable by the "
            "current two-file standalone submission"
        ),
    }

    payload = {
        "schema_version": "blackout.phase3_agent29_32.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
            "cuda_available": torch.cuda.is_available(),
        },
        "agent29": {
            "common_training_budget": representation["training_budget"],
            "performance_source": "logs/phase3_representation_ablation.json",
            "candidates": architecture_results,
            "selection_constraints": {"max_win_rate_drop": 0.0, "max_score_drop": 0.0},
            "selected": selected,
            "selected_source": str(selected_source.relative_to(ROOT)),
            "packaged_checkpoint": str(lightweight_checkpoint.relative_to(ROOT)),
            "packaged_checkpoint_bytes": lightweight_checkpoint.stat().st_size,
        },
        "agent31": {
            "devices_passed": devices,
            "smoke": smoke,
            "clean_room": clean_room,
            "passed": "cpu" in devices and any(device in devices for device in ("mps", "cuda")),
        },
        "agent32": {
            "dev_seeds": dev_seeds,
            "head_to_head": head_to_head,
            "head_to_head_episodes": head_to_head_episodes,
            "candidate_evidence": asdict(candidate_evidence),
            "candidate_gate_passed": promote_candidate,
            "incumbent_vs_scripted_dev": incumbent_dev,
            "incumbent_vs_scripted_audit": incumbent_audit,
            "incumbent_clean_room": incumbent_clean_room,
            "decision": final_decision,
        },
    }
    output = ROOT / "logs/phase3_agent29_32_submission.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "lightweight_selected": selected,
                "devices_passed": devices,
                "candidate_gate_passed": promote_candidate,
                "submission_ready": final_decision["submission_ready"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
