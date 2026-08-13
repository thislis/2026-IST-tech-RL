#!/usr/bin/env python3
"""Recover exact scripted actions and fit a bounded continuous IPPO actor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from blackout_rl import TeacherReplayBuffer, load_checkpoint, save_checkpoint  # noqa: E402
from blackout_rl.action_distribution import categorical_action  # noqa: E402
from blackout_rl.ippo_training import select_training_device  # noqa: E402


def _recover_actions(replay: TeacherReplayBuffer, cache: Path) -> torch.Tensor:
    if cache.exists():
        actions = torch.load(cache, map_location="cpu", weights_only=True)
        if actions.shape != (len(replay), 2):
            raise ValueError("cached continuous teacher actions have the wrong shape")
        return actions
    assert replay.vector is not None
    assert replay.graphic_ids is not None
    assert replay.slot_id is not None
    assert replay.action_index is not None
    if len(replay) % 5:
        raise ValueError("teacher replay rows are not grouped into five-agent frames")
    actions = categorical_action(replay.action_index[: len(replay)]).clone()
    blocks = replay.vector[: len(replay), :90].reshape(-1, 10, 9)
    team_offset = (blocks[:, 0, 2] < 0.0).to(torch.int64) * 5
    self_index = team_offset + replay.slot_id[: len(replay)]
    positions = blocks[
        torch.arange(len(replay)), self_index, :2
    ]
    delta = positions[5:] - positions[:-5]
    magnitude = torch.linalg.vector_norm(delta, dim=-1)
    same_slot = replay.slot_id[5 : len(replay)] == replay.slot_id[: len(replay) - 5]
    same_team = team_offset[5:] == team_offset[:-5]
    forward_time = replay.vector[5 : len(replay), -1] <= (
        replay.vector[: len(replay) - 5, -1] + 1e-5
    )
    # Exclude stationary/collision frames and large respawn teleports. Their
    # intended direction is more faithfully represented by the categorical label.
    valid = same_slot & same_team & forward_time & (magnitude > 1e-7) & (magnitude < 0.02)
    actions[:-5][valid] = delta[valid] / magnitude[valid, None]
    print(
        f"recovered_motion_actions={int(valid.sum())}/{len(replay)} "
        f"fallback_categorical={len(replay) - int(valid.sum())}",
        flush=True,
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(actions, cache)
    return actions


def _metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    error = torch.linalg.vector_norm(predicted - target, dim=-1)
    moving = torch.linalg.vector_norm(target, dim=-1) > 1e-5
    cosine = F.cosine_similarity(predicted[moving], target[moving], dim=-1)
    return {
        "mse": float(F.mse_loss(predicted, target)),
        "mean_l2": float(error.mean()),
        "p95_l2": float(torch.quantile(error, 0.95)),
        "moving_cosine": float(cosine.mean()) if bool(moving.any()) else 1.0,
    }


def _evaluate(model, replay, actions, indices, batch_size, device):
    assert replay.vector is not None
    assert replay.graphic_ids is not None
    assert replay.slot_id is not None
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            vector = replay.vector[selected].to(device)
            graphic = F.one_hot(
                replay.graphic_ids[selected].long(), num_classes=11
            ).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
            slot = replay.slot_id[selected].to(device)
            predictions.append(torch.tanh(model(vector, graphic, slot).action_logits).cpu())
    return _metrics(torch.cat(predictions), actions[indices])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action-cache", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--gradient-steps", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--validation-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=9070)
    args = parser.parse_args()
    device = select_training_device(args.device)
    torch.manual_seed(args.seed)
    policy, payload = load_checkpoint(args.input, device=device.resolved)
    if policy.actor_critic.model_config["action_head_version"] != "continuous_tanh_v1":
        raise ValueError("input checkpoint does not use a continuous actor head")
    replay_state = payload.get("teacher_replay_state")
    if replay_state is None:
        raise ValueError("input checkpoint has no teacher replay")
    replay = TeacherReplayBuffer(int(replay_state["capacity"]), seed=args.seed + 1)
    replay.load_state_dict(replay_state)
    actions = _recover_actions(replay, args.action_cache)
    generator = torch.Generator().manual_seed(args.seed + 2)
    validation = torch.randperm(len(replay), generator=generator)[
        : min(args.validation_samples, len(replay))
    ]
    model = policy.actor_critic
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    before = _evaluate(
        model, replay, actions, validation, args.batch_size, device.resolved
    )
    print(f"device={device.resolved} replay={len(replay)} before={before}", flush=True)
    chunks = []
    for step in range(1, args.gradient_steps + 1):
        indices = torch.randint(
            len(replay), (args.batch_size,), generator=generator
        )
        assert replay.vector is not None
        assert replay.graphic_ids is not None
        assert replay.slot_id is not None
        vector = replay.vector[indices].to(device.resolved)
        graphic = F.one_hot(
            replay.graphic_ids[indices].long(), num_classes=11
        ).permute(0, 3, 1, 2).to(device=device.resolved, dtype=torch.float32)
        slot = replay.slot_id[indices].to(device.resolved)
        target = actions[indices].to(device.resolved)
        predicted = torch.tanh(model(vector, graphic, slot).action_logits)
        loss = F.smooth_l1_loss(predicted, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        if step % 100 == 0:
            batch_metrics = _metrics(predicted.detach().cpu(), target.cpu())
            chunks.append({"gradient_steps": step, **batch_metrics})
            print(f"gradient_steps={step}/{args.gradient_steps} {batch_metrics}", flush=True)
    after = _evaluate(model, replay, actions, validation, args.batch_size, device.resolved)
    payload["policy_state"] = policy.state_dict()
    payload["optimizer_state"] = optimizer.state_dict()
    payload["continuous_teacher_fit"] = {
        "source_checkpoint": str(args.input.resolve()),
        "gradient_steps": args.gradient_steps,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "before": before,
        "after": after,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(args.output, payload)
    report = {"before": before, "after": after, "chunks": chunks}
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"after={after} saved={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
