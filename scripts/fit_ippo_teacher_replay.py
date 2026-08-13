#!/usr/bin/env python3
"""Fit an IPPO actor repeatedly on the teacher replay stored in a checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from blackout_rl import (  # noqa: E402
    TeacherReplayBuffer,
    load_checkpoint,
    save_checkpoint,
    teacher_replay_update,
)
from blackout_rl.ippo_training import select_training_device  # noqa: E402


def _evaluate_replay(
    model: torch.nn.Module,
    replay: TeacherReplayBuffer,
    *,
    indices: torch.Tensor,
    batch_size: int,
    device: str,
) -> dict[str, object]:
    assert replay.vector is not None
    assert replay.graphic_ids is not None
    assert replay.slot_id is not None
    assert replay.action_index is not None
    confusion = torch.zeros((9, 9), dtype=torch.int64)
    loss_sum = 0.0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            vector = replay.vector[selected].to(device)
            graphic = F.one_hot(
                replay.graphic_ids[selected].to(torch.int64), num_classes=11
            ).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
            slot_id = replay.slot_id[selected].to(device)
            labels = replay.action_index[selected].to(device)
            logits = model(vector, graphic, slot_id).action_logits
            loss_sum += float(F.cross_entropy(logits, labels, reduction="sum").cpu())
            predicted = torch.argmax(logits, dim=-1).cpu()
            flat = labels.cpu() * 9 + predicted
            confusion += torch.bincount(flat, minlength=81).reshape(9, 9)
    total = int(confusion.sum())
    correct = int(torch.diagonal(confusion).sum())
    class_total = confusion.sum(dim=1)
    return {
        "loss": loss_sum / total,
        "accuracy": correct / total,
        "samples": total,
        "recall_by_action": [
            (int(confusion[index, index]) / int(class_total[index]))
            if int(class_total[index])
            else None
            for index in range(9)
        ],
        "confusion": confusion.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--gradient-steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--validation-samples", type=int, default=10_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--class-balance-power", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=8070)
    args = parser.parse_args()
    if args.gradient_steps <= 0 or args.batch_size <= 0 or args.validation_samples <= 0:
        parser.error("gradient steps, batch size, and validation samples must be positive")
    if not 0.0 <= args.class_balance_power <= 1.0:
        parser.error("--class-balance-power must be in [0,1]")

    device = select_training_device(args.device)
    torch.manual_seed(args.seed)
    if device.resolved == "mps":
        torch.mps.manual_seed(args.seed)
    policy, payload = load_checkpoint(args.input, device=device.resolved)
    if "teacher_replay_state" not in payload:
        raise ValueError("input checkpoint has no teacher replay")
    replay_capacity = max(
        int(payload["teacher_replay_state"]["capacity"]),
        int(payload["teacher_replay_state"]["size"]),
    )
    replay = TeacherReplayBuffer(replay_capacity, seed=args.seed + 1)
    replay.load_state_dict(payload["teacher_replay_state"])
    validation_generator = torch.Generator().manual_seed(args.seed + 2)
    validation_indices = torch.randperm(
        len(replay), generator=validation_generator
    )[: min(args.validation_samples, len(replay))]
    model = policy.actor_critic
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    before = _evaluate_replay(
        model,
        replay,
        indices=validation_indices,
        batch_size=args.batch_size,
        device=device.resolved,
    )
    print(
        f"device={device.resolved} replay={len(replay)} "
        f"before_accuracy={before['accuracy']:.4f}",
        flush=True,
    )
    chunks = []
    remaining = args.gradient_steps
    while remaining:
        count = min(100, remaining)
        metrics = teacher_replay_update(
            model,
            optimizer,
            replay,
            minibatches=count,
            minibatch_size=args.batch_size,
            class_balance_power=args.class_balance_power,
        )
        remaining -= count
        completed = args.gradient_steps - remaining
        chunks.append({"completed_gradient_steps": completed, **asdict(metrics)})
        print(
            f"gradient_steps={completed}/{args.gradient_steps} "
            f"loss={metrics.loss:.4f} accuracy={metrics.accuracy:.4f}",
            flush=True,
        )
    after = _evaluate_replay(
        model,
        replay,
        indices=validation_indices,
        batch_size=args.batch_size,
        device=device.resolved,
    )
    payload["policy_state"] = policy.state_dict()
    payload["optimizer_state"] = optimizer.state_dict()
    payload["offline_teacher_fit"] = {
        "source_checkpoint": str(args.input.resolve()),
        "gradient_steps": args.gradient_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "class_balance_power": args.class_balance_power,
        "seed": args.seed,
        "validation": {"before": before, "after": after},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(args.output, payload)
    result = {
        "device": asdict(device),
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "replay_size": len(replay),
        "before": before,
        "after": after,
        "training_chunks": chunks,
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"after_accuracy={after['accuracy']:.4f} saved={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
