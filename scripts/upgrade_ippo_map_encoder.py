#!/usr/bin/env python3
"""Upgrade an IPPO checkpoint to the global/local semantic map encoder."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from blackout_rl import SubmissionPolicy, load_checkpoint, save_checkpoint  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--map-encoder-version",
        choices=(
            "global_local_v1",
            "global_local_spatial_v2",
            "global_local_multitarget_v3",
        ),
        default="global_local_v1",
    )
    parser.add_argument(
        "--preserve-teacher-replay",
        action="store_true",
        help="retain the observation-only teacher replay for post-upgrade fitting",
    )
    parser.add_argument(
        "--entity-order-version",
        choices=("absolute", "team_relative_v1"),
    )
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument(
        "--action-head-version",
        choices=("categorical9", "continuous_tanh_v1"),
    )
    parser.add_argument("--recurrent-version", choices=("none", "gru_v1"))
    args = parser.parse_args()
    old_policy, payload = load_checkpoint(args.source, device="cpu")
    config = dict(payload["model_config"])
    config.pop("n_actions", None)
    config["map_encoder_version"] = args.map_encoder_version
    if args.entity_order_version is not None:
        config["entity_order_version"] = args.entity_order_version
    if args.hidden_dim is not None:
        if args.hidden_dim <= 0:
            parser.error("--hidden-dim must be positive")
        config["hidden_dim"] = args.hidden_dim
    if args.action_head_version is not None:
        config["action_head_version"] = args.action_head_version
    if args.recurrent_version is not None:
        config["recurrent_version"] = args.recurrent_version
    new_policy = SubmissionPolicy(**config)
    old_state = old_policy.actor_critic.state_dict()
    new_state = new_policy.actor_critic.state_dict()
    copied = {
        name: value
        for name, value in old_state.items()
        if name in new_state
        and new_state[name].shape == value.shape
    }
    new_state.update(copied)
    new_policy.actor_critic.load_state_dict(new_state)
    payload["policy_state"] = new_policy.state_dict()
    payload["model_config"] = dict(new_policy.actor_critic.model_config)
    if args.action_head_version is not None:
        continuous = args.action_head_version == "continuous_tanh_v1"
        payload["action_contract"] = {
            **payload["action_contract"],
            "distribution": "continuous_tanh_v1" if continuous else "categorical_9",
            "inference": (
                "deterministic_tanh_mean" if continuous else "deterministic_argmax"
            ),
        }
    payload.pop("optimizer_state", None)
    if not args.preserve_teacher_replay:
        payload.pop("teacher_replay_state", None)
    payload["source"] = {
        **payload.get("source", {}),
        "map_encoder_upgrade_from": str(args.source.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(args.output, payload)
    print(f"copied_tensors={len(copied)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
