#!/usr/bin/env python3
"""Fit the recurrent state of a continuous IPPO actor on ordered teacher replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from blackout_rl import TeacherReplayBuffer, load_checkpoint, save_checkpoint  # noqa: E402
from blackout_rl.ippo_training import select_training_device  # noqa: E402


def _segments(replay: TeacherReplayBuffer) -> list[tuple[int, int]]:
    assert replay.vector is not None
    assert replay.slot_id is not None
    if len(replay) % 5:
        raise ValueError("teacher replay is not grouped into five-agent frames")
    frames = replay.vector[: len(replay)].reshape(-1, 5, 96)
    slots = replay.slot_id[: len(replay)].reshape(-1, 5)
    if not bool(torch.all(slots == torch.arange(5))):
        raise ValueError("teacher replay frame slot order is invalid")
    team = frames[:, 0, 2] < 0.0
    time_left = frames[:, 0, -1]
    starts = [0]
    for index in range(1, len(frames)):
        if bool(team[index] != team[index - 1]) or float(time_left[index]) > float(
            time_left[index - 1]
        ) + 1e-5:
            starts.append(index)
    starts.append(len(frames))
    segments = [
        (start, stop)
        for start, stop in zip(starts, starts[1:])
        if stop - start >= 8 and float(time_left[start]) >= 0.98
    ]
    if not segments:
        raise ValueError("teacher replay contains no complete recurrent segments")
    return segments


def _precompute_latent(model, replay, batch_size, device):
    assert replay.vector is not None
    assert replay.graphic_ids is not None
    assert replay.slot_id is not None
    encoded = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(replay), batch_size):
            stop = min(start + batch_size, len(replay))
            vector = replay.vector[start:stop].to(device)
            graphic = F.one_hot(
                replay.graphic_ids[start:stop].long(), num_classes=11
            ).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
            slot = replay.slot_id[start:stop].to(device)
            encoded.append(model.encode(vector, graphic, slot).cpu())
    return torch.cat(encoded).reshape(-1, 5, model.model_config["hidden_dim"])


def _evaluate(model, latent, actions, segments, device):
    squared_error = 0.0
    absolute_error = 0.0
    cosine_sum = 0.0
    moving_count = 0
    count = 0
    model.eval()
    with torch.inference_mode():
        for start, stop in segments:
            hidden = torch.zeros((5, latent.shape[-1]), device=device)
            for frame in range(start, stop):
                hidden = model.recurrent(latent[frame].to(device), hidden)
                predicted = torch.tanh(model.actor(hidden)).cpu()
                target = actions[frame]
                error = predicted - target
                squared_error += float(error.square().sum())
                absolute_error += float(torch.linalg.vector_norm(error, dim=-1).sum())
                moving = torch.linalg.vector_norm(target, dim=-1) > 1e-5
                if bool(moving.any()):
                    cosine_sum += float(
                        F.cosine_similarity(predicted[moving], target[moving], dim=-1).sum()
                    )
                    moving_count += int(moving.sum())
                count += 5
    return {
        "mse": squared_error / (count * 2),
        "mean_l2": absolute_error / count,
        "moving_cosine": cosine_sum / moving_count,
        "samples": count,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action-cache", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--encode-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=10070)
    args = parser.parse_args()
    device = select_training_device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    policy, payload = load_checkpoint(args.input, device=device.resolved)
    model = policy.actor_critic
    if model.recurrent is None:
        raise ValueError("input checkpoint has no recurrent module")
    replay_state = payload.get("teacher_replay_state")
    if replay_state is None:
        raise ValueError("input checkpoint has no teacher replay")
    replay = TeacherReplayBuffer(int(replay_state["capacity"]), seed=args.seed)
    replay.load_state_dict(replay_state)
    actions = torch.load(args.action_cache, map_location="cpu", weights_only=True)
    actions = actions.reshape(-1, 5, 2)
    segments = _segments(replay)
    latent = _precompute_latent(
        model, replay, args.encode_batch_size, device.resolved
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.recurrent.parameters():
        parameter.requires_grad_(True)
    for parameter in model.actor.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(
        [*model.recurrent.parameters(), *model.actor.parameters()],
        lr=args.learning_rate,
    )
    before = _evaluate(model, latent, actions, segments, device.resolved)
    print(f"device={device.resolved} segments={len(segments)} before={before}", flush=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        order = list(segments)
        random.shuffle(order)
        loss_sum = 0.0
        chunks = 0
        model.train()
        for start, stop in order:
            hidden = torch.zeros((5, latent.shape[-1]), device=device.resolved)
            for chunk_start in range(start, stop, args.sequence_length):
                chunk_stop = min(chunk_start + args.sequence_length, stop)
                predictions = []
                for frame in range(chunk_start, chunk_stop):
                    hidden = model.recurrent(latent[frame].to(device.resolved), hidden)
                    predictions.append(torch.tanh(model.actor(hidden)))
                predicted = torch.stack(predictions)
                target = actions[chunk_start:chunk_stop].to(device.resolved)
                loss = F.smooth_l1_loss(predicted, target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [*model.recurrent.parameters(), *model.actor.parameters()], 0.5
                )
                optimizer.step()
                hidden = hidden.detach()
                loss_sum += float(loss.detach())
                chunks += 1
        metrics = _evaluate(model, latent, actions, segments, device.resolved)
        history.append({"epoch": epoch, "training_loss": loss_sum / chunks, **metrics})
        print(f"epoch={epoch}/{args.epochs} {history[-1]}", flush=True)
    payload["policy_state"] = policy.state_dict()
    payload.pop("optimizer_state", None)
    payload["recurrent_teacher_fit"] = {
        "source_checkpoint": str(args.input.resolve()),
        "epochs": args.epochs,
        "sequence_length": args.sequence_length,
        "learning_rate": args.learning_rate,
        "segments": len(segments),
        "before": before,
        "after": history[-1],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(args.output, payload)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(
        json.dumps({"before": before, "history": history}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"saved={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
