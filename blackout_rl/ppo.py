"""Clipped PPO update and diagnostics for BlackOut IPPO rollouts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import torch
from torch import nn

from .action_distribution import evaluate_categorical_action
from .ippo_model import IPPOActorCritic
from .rollout import RolloutBatch


PPO_DIAGNOSTIC_SCHEMA_VERSION = "blackout.ppo_diagnostic.v1"


@dataclass(frozen=True)
class PPOConfig:
    update_epochs: int = 4
    minibatch_size: int = 256
    clip_coef: float = 0.2
    value_clip_coef: float | None = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float | None = None

    def validate(self) -> None:
        if self.update_epochs <= 0 or self.minibatch_size <= 0:
            raise ValueError("update_epochs and minibatch_size must be positive")
        if self.clip_coef < 0.0:
            raise ValueError("clip_coef must be non-negative")
        if self.value_clip_coef is not None and self.value_clip_coef < 0.0:
            raise ValueError("value_clip_coef must be non-negative")
        if self.value_loss_coef < 0.0 or self.entropy_coef < 0.0:
            raise ValueError("loss coefficients must be non-negative")
        if self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")
        if self.target_kl is not None and self.target_kl <= 0.0:
            raise ValueError("target_kl must be positive")


@dataclass(frozen=True)
class PPODiagnostics:
    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float
    explained_variance: float | None
    gradient_norm: float
    epochs_completed: int
    minibatches: int
    samples: int
    early_stopped: bool

    def to_dict(self) -> dict[str, float | int | bool | None]:
        return asdict(self)


class PPODiagnosticLogger:
    """Append strict JSONL update diagnostics for experiment monitoring."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def log(
        self,
        diagnostics: PPODiagnostics,
        *,
        update: int,
        global_step: int,
        config: PPOConfig,
    ) -> dict[str, object]:
        if update < 0 or global_step < 0:
            raise ValueError("update and global_step must be non-negative")
        metrics = diagnostics.to_dict()
        for name, value in metrics.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"diagnostic {name} must be finite or null")
        record: dict[str, object] = {
            "schema_version": PPO_DIAGNOSTIC_SCHEMA_VERSION,
            "update": update,
            "global_step": global_step,
            "config": asdict(config),
            "metrics": metrics,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        return record


def explained_variance(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes differ")
    target_variance = torch.var(target, unbiased=False)
    if not bool(torch.isfinite(target_variance)) or float(target_variance) <= 1e-12:
        return torch.tensor(float("nan"), dtype=target.dtype, device=target.device)
    return 1.0 - torch.var(target - prediction, unbiased=False) / target_variance


def ppo_update(
    model: IPPOActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: RolloutBatch,
    *,
    config: PPOConfig | None = None,
) -> PPODiagnostics:
    """Run one multi-epoch clipped PPO update and return aggregate diagnostics."""
    config = config or PPOConfig()
    config.validate()
    if len(batch) == 0:
        raise ValueError("PPO batch must not be empty")
    device = next(model.parameters()).device
    batch = batch.to(device)
    model.train()
    sums = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approximate_kl": 0.0,
        "clip_fraction": 0.0,
        "gradient_norm": 0.0,
    }
    weighted_samples = 0
    minibatches = 0
    epochs_completed = 0
    early_stopped = False

    for epoch in range(config.update_epochs):
        permutation = torch.randperm(len(batch), device=device)
        epoch_kl_sum = 0.0
        epoch_samples = 0
        for start in range(0, len(batch), config.minibatch_size):
            indices = permutation[start : start + config.minibatch_size]
            output = model(
                batch.vector[indices], batch.graphic[indices], batch.slot_id[indices]
            )
            evaluated = evaluate_categorical_action(
                output.action_logits, batch.action_index[indices]
            )
            log_ratio = evaluated.log_prob - batch.old_log_prob[indices]
            ratio = log_ratio.exp()
            advantage = batch.advantage[indices]
            unclipped_policy_loss = -advantage * ratio
            clipped_policy_loss = -advantage * torch.clamp(
                ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef
            )
            policy_loss = torch.maximum(unclipped_policy_loss, clipped_policy_loss).mean()

            if config.value_clip_coef is None:
                value_loss = 0.5 * (output.value - batch.return_[indices]).square().mean()
            else:
                value_delta = output.value - batch.old_value[indices]
                clipped_value = batch.old_value[indices] + value_delta.clamp(
                    -config.value_clip_coef, config.value_clip_coef
                )
                value_loss = 0.5 * torch.maximum(
                    (output.value - batch.return_[indices]).square(),
                    (clipped_value - batch.return_[indices]).square(),
                ).mean()
            entropy = evaluated.entropy.mean()
            loss = (
                policy_loss
                + config.value_loss_coef * value_loss
                - config.entropy_coef * entropy
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError("non-finite PPO gradient norm")
            optimizer.step()

            with torch.no_grad():
                approximate_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = (
                    (torch.abs(ratio - 1.0) > config.clip_coef).to(torch.float32).mean()
                )
            count = int(indices.numel())
            values = {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy,
                "approximate_kl": approximate_kl,
                "clip_fraction": clip_fraction,
                "gradient_norm": gradient_norm,
            }
            for name, value in values.items():
                sums[name] += float(value.detach()) * count
            epoch_kl_sum += float(approximate_kl) * count
            epoch_samples += count
            weighted_samples += count
            minibatches += 1
        epochs_completed = epoch + 1
        if config.target_kl is not None and epoch_kl_sum / epoch_samples > config.target_kl:
            early_stopped = True
            break

    with torch.inference_mode():
        updated_values: list[torch.Tensor] = []
        for start in range(0, len(batch), config.minibatch_size):
            stop = min(start + config.minibatch_size, len(batch))
            updated_values.append(
                model(
                    batch.vector[start:stop],
                    batch.graphic[start:stop],
                    batch.slot_id[start:stop],
                ).value
            )
        variance = explained_variance(torch.cat(updated_values), batch.return_)
    explained_value = float(variance)
    explained = None if math.isnan(explained_value) else explained_value
    return PPODiagnostics(
        policy_loss=sums["policy_loss"] / weighted_samples,
        value_loss=sums["value_loss"] / weighted_samples,
        entropy=sums["entropy"] / weighted_samples,
        approximate_kl=sums["approximate_kl"] / weighted_samples,
        clip_fraction=sums["clip_fraction"] / weighted_samples,
        explained_variance=explained,
        gradient_norm=sums["gradient_norm"] / weighted_samples,
        epochs_completed=epochs_completed,
        minibatches=minibatches,
        samples=len(batch),
        early_stopped=early_stopped,
    )
