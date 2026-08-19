#!/usr/bin/env python3
"""Train and evaluate AGENT-06/08/11 representation ablation arms."""

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
    SubmissionPolicy,
    TeacherReplayBuffer,
    checkpoint_payload,
    save_checkpoint,
)
from blackout_rl.evaluation_protocol import (  # noqa: E402
    SeedSplits,
    paired_seed_bootstrap_ci,
)
from blackout_rl.ippo_training import select_training_device  # noqa: E402
from blackout_rl.representation import select_auxiliary_targets  # noqa: E402
from scripts.train_ippo_vs_scripted import evaluate_checkpoint  # noqa: E402


ARMS = {
    "flatten_legacy": {
        "entity_encoder_version": "flatten_v1",
        "map_encoder_version": "legacy",
        "auxiliary_targets": (),
    },
    "attention_legacy": {
        "entity_encoder_version": "self_attention_v1",
        "map_encoder_version": "legacy",
        "auxiliary_targets": (),
    },
    "flatten_global_local": {
        "entity_encoder_version": "flatten_v1",
        "map_encoder_version": "global_local_v1",
        "auxiliary_targets": (),
    },
    "global_local_aux_role": {
        "entity_encoder_version": "flatten_v1",
        "map_encoder_version": "global_local_v1",
        "auxiliary_targets": ("role",),
    },
    "global_local_aux_holding_item": {
        "entity_encoder_version": "flatten_v1",
        "map_encoder_version": "global_local_v1",
        "auxiliary_targets": ("holding_item",),
    },
    "global_local_aux_seconds_to_absorption": {
        "entity_encoder_version": "flatten_v1",
        "map_encoder_version": "global_local_v1",
        "auxiliary_targets": ("seconds_to_absorption",),
    },
    "global_local_aux_score_delta": {
        "entity_encoder_version": "flatten_v1",
        "map_encoder_version": "global_local_v1",
        "auxiliary_targets": ("score_delta",),
    },
}


def _load_replay(path: Path, seed: int) -> tuple[TeacherReplayBuffer, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("teacher_replay_state")
    if state is None:
        raise ValueError("replay checkpoint has no teacher replay")
    replay = TeacherReplayBuffer(int(state["capacity"]), seed=seed)
    replay.load_state_dict(state)
    if len(replay) % 5:
        raise ValueError("teacher replay must contain complete five-agent frames")
    return replay, payload


def _self_indices(vector: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
    blocks = vector[:, :90].reshape(-1, 10, 9)
    allies = torch.topk(blocks[:, :, 2], 5, dim=1).indices
    allies = torch.sort(allies, dim=1).values
    return allies[torch.arange(len(vector)), slot]


def _auxiliary_targets(replay: TeacherReplayBuffer) -> dict[str, torch.Tensor]:
    assert replay.vector is not None
    assert replay.slot_id is not None
    vector = replay.vector[: len(replay)]
    slot = replay.slot_id[: len(replay)]
    blocks = vector[:, :90].reshape(-1, 10, 9)
    self_index = _self_indices(vector, slot)
    self_blocks = blocks[torch.arange(len(vector)), self_index]
    role_table = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int64)
    role = role_table[slot]
    holding_item = torch.argmax(self_blocks[:, 3:9], dim=1)
    time_left = vector[:, -1]
    seconds = torch.remainder(time_left * 300.0, 20.0) / 20.0

    frames = vector.reshape(-1, 5, 96)
    frame_delta = torch.zeros(len(frames), dtype=torch.float32)
    same_episode = frames[1:, 0, -1] <= frames[:-1, 0, -1] + 1e-5
    frame_delta[:-1][same_episode] = (
        frames[1:, 0, 93] - frames[:-1, 0, 93]
    )[same_episode]
    score_delta = frame_delta[:, None].expand(-1, 5).reshape(-1)
    return {
        "role": role,
        "holding_item": holding_item,
        "seconds_to_absorption": seconds,
        "score_delta": score_delta,
    }


def _class_weights(labels: torch.Tensor, power: float) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=9).to(torch.float32)
    if bool((counts == 0).any()):
        raise ValueError("teacher replay is missing an action class")
    weights = counts.pow(-power)
    return weights / weights.mean()


def _batch(
    replay: TeacherReplayBuffer, indices: torch.Tensor, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert replay.vector is not None
    assert replay.graphic_ids is not None
    assert replay.slot_id is not None
    assert replay.action_index is not None
    vector = replay.vector[indices].to(device)
    graphic = F.one_hot(
        replay.graphic_ids[indices].long(), num_classes=11
    ).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
    return (
        vector,
        graphic,
        replay.slot_id[indices].to(device),
        replay.action_index[indices].to(device),
    )


def _measure(
    model,
    replay: TeacherReplayBuffer,
    aux_cache: dict[str, torch.Tensor],
    indices: torch.Tensor,
    batch_size: int,
    device: str,
) -> dict:
    correct = 0
    count = 0
    auxiliary_sum = {name: 0.0 for name in model.model_config["auxiliary_targets"]}
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            vector, graphic, slot, action = _batch(replay, selected, device)
            latent = model.encode(vector, graphic, slot)
            predicted = model.actor(latent)
            correct += int((predicted.argmax(dim=1) == action).sum())
            count += len(selected)
            predictions = model.predict_auxiliary(latent)
            for name, value in predictions.items():
                target = aux_cache[name][selected].to(device)
                if name in {"role", "holding_item"}:
                    loss = F.cross_entropy(value, target.long(), reduction="sum")
                else:
                    loss = F.smooth_l1_loss(value, target, reduction="sum")
                auxiliary_sum[name] += float(loss)
    return {
        "action_accuracy": correct / count,
        "samples": count,
        "auxiliary_loss": {
            name: total / count for name, total in auxiliary_sum.items()
        },
    }


def train(args: argparse.Namespace) -> None:
    device = select_training_device(args.device)
    replay, source = _load_replay(args.replay_checkpoint, args.seed + 1)
    aux_cache = _auxiliary_targets(replay)
    assert replay.action_index is not None
    weights = _class_weights(replay.action_index[: len(replay)], args.class_balance_power)
    validation_generator = torch.Generator().manual_seed(args.seed + 2)
    validation = torch.randperm(len(replay), generator=validation_generator)[
        : min(args.validation_samples, len(replay))
    ]
    base_config = dict(source["model_config"])
    base_config.pop("n_actions", None)
    base_config.update(
        {
            "entity_order_version": "absolute",
            "action_head_version": "categorical9",
            "recurrent_version": "none",
        }
    )
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.training_dir.mkdir(parents=True, exist_ok=True)
    for arm_name, arm in ARMS.items():
        torch.manual_seed(args.seed)
        if device.resolved == "mps":
            torch.mps.manual_seed(args.seed)
        config = {**base_config, **arm}
        policy = SubmissionPolicy(**config).to(device.resolved)
        model = policy.actor_critic
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        generator = torch.Generator().manual_seed(args.seed + 3)
        before = _measure(
            model, replay, aux_cache, validation, args.batch_size, device.resolved
        )
        history = []
        for step in range(1, args.gradient_steps + 1):
            indices = torch.randint(
                len(replay), (args.batch_size,), generator=generator
            )
            vector, graphic, slot, action = _batch(replay, indices, device.resolved)
            latent = model.encode(vector, graphic, slot)
            logits = model.actor(latent)
            action_loss = F.cross_entropy(
                logits, action, weight=weights.to(device.resolved)
            )
            predictions = model.predict_auxiliary(latent)
            aux_losses = []
            for name, prediction in predictions.items():
                target = aux_cache[name][indices].to(device.resolved)
                if name in {"role", "holding_item"}:
                    aux_losses.append(F.cross_entropy(prediction, target.long()))
                else:
                    aux_losses.append(F.smooth_l1_loss(prediction, target))
            auxiliary_loss = (
                torch.stack(aux_losses).sum() if aux_losses else action_loss.new_zeros(())
            )
            loss = action_loss + args.auxiliary_coef * auxiliary_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            if step % args.report_every == 0 or step == args.gradient_steps:
                row = {
                    "gradient_step": step,
                    "action_loss": float(action_loss.detach()),
                    "auxiliary_loss": float(auxiliary_loss.detach()),
                    "gradient_norm": float(gradient_norm.detach()),
                }
                history.append(row)
                print(f"arm={arm_name} {row}", flush=True)
        after = _measure(
            model, replay, aux_cache, validation, args.batch_size, device.resolved
        )
        checkpoint = args.checkpoint_dir / f"phase3_repr_{arm_name}.pt"
        payload = checkpoint_payload(
            policy,
            global_step=args.gradient_steps,
            training_seed=args.seed,
            optimizer=optimizer,
            source={
                "phase": "agent06-agent08-agent11-ablation",
                "teacher_replay": str(args.replay_checkpoint.resolve()),
            },
        )
        payload["representation_ablation"] = {
            "arm": arm_name,
            "config": arm,
            "gradient_steps": args.gradient_steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "class_balance_power": args.class_balance_power,
            "auxiliary_coef": args.auxiliary_coef,
            "replay_rows": len(replay),
            "before": before,
            "after": after,
        }
        save_checkpoint(checkpoint, payload)
        metrics = {
            "schema_version": "blackout.phase3_representation_training.v1",
            "arm": arm_name,
            "device": asdict(device),
            "config": arm,
            "before": before,
            "after": after,
            "history": history,
        }
        (args.training_dir / f"phase3_repr_{arm_name}.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )
        print(f"saved={checkpoint} after={after}", flush=True)


def evaluate(args: argparse.Namespace) -> None:
    splits = SeedSplits.from_dict(
        json.loads((ROOT / "configs/seed_splits_v1.json").read_text())
    )
    dev = list(splits.seeds_for("dev", purpose="model_selection"))
    arms = {}
    for arm_name, config in ARMS.items():
        checkpoint = args.checkpoint_dir / f"phase3_repr_{arm_name}.pt"
        output = args.evaluation_dir / f"phase3_repr_{arm_name}_dev.json"
        series = evaluate_checkpoint(
            build=(ROOT / "builds/BlackOut.app").resolve(),
            checkpoint=checkpoint.resolve(),
            opponent_config=(
                ROOT / "configs/policies/scripted_battery_v1.json"
            ).resolve(),
            seeds=dev,
            workers=args.workers,
            time_scale=args.time_scale,
            max_episode_steps=22_000,
            global_step=args.gradient_steps,
            output=output.resolve(),
        )
        arms[arm_name] = {
            "config": config,
            "checkpoint": str(checkpoint.resolve()),
            "evaluation_log": str(output.resolve()),
            "summary": series["summary"],
            "win_rate_ci": paired_seed_bootstrap_ci(
                series["episodes"], metric="win_rate", seed=args.seed
            ).to_dict(),
            "score_diff_ci": paired_seed_bootstrap_ci(
                series["episodes"], metric="score_diff", seed=args.seed + 1
            ).to_dict(),
        }
        print(f"evaluated={arm_name} summary={series['summary']}", flush=True)

    baseline = arms["flatten_legacy"]["summary"]
    global_local = arms["flatten_global_local"]["summary"]
    auxiliary_results = {}
    for target in (
        "role",
        "holding_item",
        "seconds_to_absorption",
        "score_delta",
    ):
        summary = arms[f"global_local_aux_{target}"]["summary"]
        auxiliary_results[target] = {
            "win_rate_delta": summary["win_rate"] - global_local["win_rate"],
            "score_diff_delta": summary["mean_model_score_diff"]
            - global_local["mean_model_score_diff"],
        }
    payload = {
        "schema_version": "blackout.phase3_representation_ablation.v1",
        "train_seed": args.seed,
        "dev_seeds": dev,
        "teacher_replay": str(args.replay_checkpoint.resolve()),
        "training_budget": {
            "gradient_steps_per_arm": args.gradient_steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "auxiliary_coef": args.auxiliary_coef,
        },
        "arms": arms,
        "agent06_attention_minus_flatten": {
            "win_rate_delta": arms["attention_legacy"]["summary"]["win_rate"]
            - baseline["win_rate"],
            "score_diff_delta": arms["attention_legacy"]["summary"][
                "mean_model_score_diff"
            ]
            - baseline["mean_model_score_diff"],
        },
        "agent08_global_local_minus_legacy": {
            "win_rate_delta": global_local["win_rate"] - baseline["win_rate"],
            "score_diff_delta": global_local["mean_model_score_diff"]
            - baseline["mean_model_score_diff"],
        },
        "agent11_auxiliary_results": auxiliary_results,
        "agent11_selected_targets": list(
            select_auxiliary_targets(auxiliary_results)
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"saved={args.output}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("train", "evaluate"))
    parser.add_argument(
        "--replay-checkpoint",
        type=Path,
        default=ROOT / "checkpoints/ippo_counter_teacher_ordered.pt",
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=ROOT / "checkpoints"
    )
    parser.add_argument("--training-dir", type=Path, default=ROOT / "logs")
    parser.add_argument("--evaluation-dir", type=Path, default=ROOT / "logs")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "logs/phase3_representation_ablation.json",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--gradient-steps", type=int, default=1_500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--validation-samples", type=int, default=20_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--class-balance-power", type=float, default=0.5)
    parser.add_argument("--auxiliary-coef", type=float, default=0.1)
    parser.add_argument("--report-every", type=int, default=250)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--time-scale", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=36001)
    args = parser.parse_args()
    args.replay_checkpoint = args.replay_checkpoint.resolve()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    args.training_dir = args.training_dir.resolve()
    args.evaluation_dir = args.evaluation_dir.resolve()
    args.output = args.output.resolve()
    if args.command == "train":
        train(args)
    else:
        evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
