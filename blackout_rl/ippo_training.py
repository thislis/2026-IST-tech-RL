"""Reusable device and stopping rules for long-running IPPO experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .ippo_model import IPPOActorCritic
from .rollout import RolloutBatch


@dataclass(frozen=True)
class DeviceSelection:
    """Resolved torch device together with an audit-friendly explanation."""

    requested: str
    resolved: str
    reason: str


@dataclass(frozen=True)
class OnlineImitationMetrics:
    """Aggregate teacher-label metrics for one rollout."""

    loss: float
    accuracy: float
    samples: int
    gradient_norm: float


def select_training_device(requested: str = "auto") -> DeviceSelection:
    """Prefer MPS in auto mode only when this PyTorch process can use it."""

    normalized = requested.lower()
    if normalized not in {"auto", "cpu", "mps"}:
        raise ValueError("device must be one of: auto, cpu, mps")
    mps = getattr(torch.backends, "mps", None)
    mps_built = bool(mps and mps.is_built())
    mps_available = bool(mps and mps.is_available())
    if normalized == "mps":
        if not mps_available:
            raise RuntimeError(
                "MPS was requested but is unavailable "
                f"(built={mps_built}, available={mps_available})"
            )
        return DeviceSelection("mps", "mps", "explicitly requested and available")
    if normalized == "cpu":
        return DeviceSelection("cpu", "cpu", "explicitly requested")
    if mps_available:
        return DeviceSelection("auto", "mps", "MPS is available in this PyTorch process")
    return DeviceSelection(
        "auto",
        "cpu",
        f"MPS unavailable (built={mps_built}, available={mps_available})",
    )


def required_wins(episodes: int, threshold: float) -> int:
    """Return the smallest integer win count satisfying a strict win-rate target."""

    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0,1]")
    # Avoid importing math for a single exact ceiling operation.
    raw = episodes * threshold
    rounded = int(raw)
    return rounded if raw == rounded else rounded + 1


def passed_win_rate(*, wins: int, episodes: int, threshold: float) -> bool:
    """Use wins/all episodes; draws do not count as wins."""

    if not 0 <= wins <= episodes:
        raise ValueError("wins must be in [0, episodes]")
    return wins >= required_wins(episodes, threshold)


def online_imitation_update(
    model: IPPOActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: RolloutBatch,
    *,
    epochs: int = 1,
    minibatch_size: int = 128,
    loss_coef: float = 1.0,
    max_grad_norm: float = 0.5,
) -> OnlineImitationMetrics:
    """Fit the actor to scripted teacher decisions observed on current-policy states."""

    if batch.teacher_action_index is None:
        raise ValueError("rollout batch has no teacher action labels")
    if epochs <= 0 or minibatch_size <= 0 or loss_coef <= 0.0 or max_grad_norm <= 0.0:
        raise ValueError("online imitation hyperparameters must be positive")
    device = next(model.parameters()).device
    batch = batch.to(device)
    model.train()
    loss_sum = 0.0
    correct = 0
    samples = 0
    gradient_sum = 0.0
    minibatches = 0
    for _ in range(epochs):
        for start in range(0, len(batch), minibatch_size):
            stop = min(start + minibatch_size, len(batch))
            logits = model(
                batch.vector[start:stop],
                batch.graphic[start:stop],
                batch.slot_id[start:stop],
            ).action_logits
            labels = batch.teacher_action_index[start:stop]
            unscaled_loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            (loss_coef * unscaled_loss).backward()
            gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError("non-finite online imitation gradient norm")
            optimizer.step()
            count = int(labels.numel())
            loss_sum += float(unscaled_loss.detach()) * count
            correct += int((torch.argmax(logits.detach(), dim=-1) == labels).sum())
            samples += count
            gradient_sum += float(gradient_norm.detach())
            minibatches += 1
    return OnlineImitationMetrics(
        loss=loss_sum / samples,
        accuracy=correct / samples,
        samples=samples,
        gradient_norm=gradient_sum / minibatches,
    )
