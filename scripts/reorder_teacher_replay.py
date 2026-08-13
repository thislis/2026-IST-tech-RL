#!/usr/bin/env python3
"""Reorder interleaved two-side teacher replay into contiguous team trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from blackout_rl import TeacherReplayBuffer, save_checkpoint  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--drop-prefix-frames",
        type=int,
        default=0,
        help="discard replay frames collected before the current uninterrupted run",
    )
    args = parser.parse_args()
    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    state = payload.get("teacher_replay_state")
    if state is None:
        raise ValueError("checkpoint has no teacher replay")
    replay = TeacherReplayBuffer(int(state["capacity"]))
    replay.load_state_dict(state)
    if len(replay) % 5:
        raise ValueError("teacher replay is not grouped into five-agent frames")
    assert replay.vector is not None
    assert replay.graphic_ids is not None
    assert replay.slot_id is not None
    assert replay.action_index is not None
    frames = len(replay) // 5
    if not 0 <= args.drop_prefix_frames < frames:
        raise ValueError("drop-prefix-frames must leave at least one frame")
    vector = replay.vector[: len(replay)].reshape(frames, 5, 96)
    graphic_shape = replay.graphic_ids.shape[1:]
    graphic = replay.graphic_ids[: len(replay)].reshape(frames, 5, *graphic_shape)
    slot = replay.slot_id[: len(replay)].reshape(frames, 5)
    action = replay.action_index[: len(replay)].reshape(frames, 5)
    vector = vector[args.drop_prefix_frames :]
    graphic = graphic[args.drop_prefix_frames :]
    slot = slot[args.drop_prefix_frames :]
    action = action[args.drop_prefix_frames :]
    team_one = vector[:, 0, 2] < 0.0
    order = torch.cat(
        (torch.nonzero(~team_one).flatten(), torch.nonzero(team_one).flatten())
    )
    replay.vector[: len(order) * 5].copy_(vector[order].reshape(-1, 96))
    replay.graphic_ids[: len(order) * 5].copy_(
        graphic[order].reshape(-1, *graphic_shape)
    )
    replay.slot_id[: len(order) * 5].copy_(slot[order].reshape(-1))
    replay.action_index[: len(order) * 5].copy_(action[order].reshape(-1))
    replay.size = len(order) * 5
    replay.position = replay.size % replay.capacity
    payload["teacher_replay_state"] = replay.state_dict()
    payload["teacher_replay_reorder"] = {
        "source": str(args.input.resolve()),
        "dropped_prefix_frames": args.drop_prefix_frames,
        "frames": len(order),
        "ordering": "team-0-chronological-then-team-1-chronological",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(args.output, payload)
    print(
        f"saved={args.output} frames={len(order)} "
        f"team0={int((~team_one).sum())} team1={int(team_one.sum())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
