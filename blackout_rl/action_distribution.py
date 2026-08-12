"""Nine-way categorical action distribution used by IPPO training and inference."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch.distributions import Categorical


ACTION_DIRECTIONS = torch.tensor(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 0.0],
        [-1.0, -1.0],
        [0.0, -1.0],
        [1.0, -1.0],
    ],
    dtype=torch.float32,
)
N_ACTIONS = len(ACTION_DIRECTIONS)


class ActionSelection(NamedTuple):
    """One sampled or deterministic categorical decision and PPO statistics."""

    index: torch.Tensor
    action: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor


def _validate_logits(logits: torch.Tensor) -> None:
    if not torch.is_floating_point(logits):
        raise TypeError(f"logits must be floating point, got {logits.dtype}")
    if logits.ndim != 2 or logits.shape[1] != N_ACTIONS:
        raise ValueError(f"logits must have shape (B,{N_ACTIONS}), got {tuple(logits.shape)}")
    if not bool(torch.all(torch.isfinite(logits))):
        raise ValueError("logits contain NaN or infinity")


def categorical_action(index: torch.Tensor) -> torch.Tensor:
    """Convert NoOp+8-direction indices to normalized continuous ``(dx,dy)``."""
    if index.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TypeError(f"action index must be integer, got {index.dtype}")
    if not bool(torch.all((index >= 0) & (index < N_ACTIONS))):
        raise ValueError(f"action index must be in [0,{N_ACTIONS - 1}]")
    table = ACTION_DIRECTIONS.to(device=index.device)
    action = table[index.to(torch.int64)]
    norm = torch.linalg.vector_norm(action, dim=-1, keepdim=True).clamp_min(1.0)
    return action / norm


def select_categorical_action(
    logits: torch.Tensor,
    *,
    deterministic: bool = False,
) -> ActionSelection:
    """Sample during training or use argmax during deterministic evaluation."""
    _validate_logits(logits)
    distribution = Categorical(logits=logits)
    index = torch.argmax(logits, dim=-1) if deterministic else distribution.sample()
    return ActionSelection(
        index=index,
        action=categorical_action(index),
        log_prob=distribution.log_prob(index),
        entropy=distribution.entropy(),
    )


def evaluate_categorical_action(
    logits: torch.Tensor,
    index: torch.Tensor,
) -> ActionSelection:
    """Re-evaluate stored indices under current logits for a PPO update."""
    _validate_logits(logits)
    if index.shape != logits.shape[:1]:
        raise ValueError(f"index must have shape ({logits.shape[0]},), got {tuple(index.shape)}")
    if index.dtype != torch.int64:
        raise TypeError(f"index must have dtype int64, got {index.dtype}")
    distribution = Categorical(logits=logits)
    return ActionSelection(
        index=index,
        action=categorical_action(index),
        log_prob=distribution.log_prob(index),
        entropy=distribution.entropy(),
    )


def deterministic_action(logits: torch.Tensor) -> torch.Tensor:
    """Choose categorical argmax and adapt it to the official continuous space."""
    return select_categorical_action(logits, deterministic=True).action
