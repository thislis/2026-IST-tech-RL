"""Entity-aware IPPO actor/critic encoders and shared policy network."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from .action_distribution import N_ACTIONS
from .batching import N_TEAM_AGENTS
from .observation import (
    GRAPHIC_CHANNEL_NAMES,
    N_CLASSES,
    N_UNITS,
    UNIT_BLOCK_SIZE,
    VECTOR_SIZE,
)


VECTOR_CONTEXT_SIZE = N_CLASSES + 3  # self class, own score, opponent score, time


class VectorEncoding(NamedTuple):
    """Structured vector latents retained for future attention/auxiliary heads."""

    entities: torch.Tensor
    context: torch.Tensor
    latent: torch.Tensor


class ActorCriticOutput(NamedTuple):
    """Training output: categorical actor logits and one local value per row."""

    action_logits: torch.Tensor
    value: torch.Tensor


def _validate_floating_batch(tensor: torch.Tensor, *, name: str) -> None:
    if not torch.is_floating_point(tensor):
        raise TypeError(f"{name} must be floating point, got {tensor.dtype}")
    if not bool(torch.all(torch.isfinite(tensor))):
        raise ValueError(f"{name} contains NaN or infinity")


class EntityVectorEncoder(nn.Module):
    """Encode ten unit blocks separately from class/score/time context."""

    def __init__(
        self,
        *,
        vector_size: int = VECTOR_SIZE,
        entity_dim: int = 32,
        context_dim: int = 32,
    ) -> None:
        super().__init__()
        expected_size = N_UNITS * UNIT_BLOCK_SIZE + VECTOR_CONTEXT_SIZE
        if vector_size != expected_size:
            raise ValueError(f"entity vector layout requires vector_size={expected_size}")
        self.vector_size = vector_size
        self.entity_dim = entity_dim
        self.context_dim = context_dim
        self.entity_encoder = nn.Sequential(
            nn.Linear(UNIT_BLOCK_SIZE, entity_dim),
            nn.ReLU(),
            nn.Linear(entity_dim, entity_dim),
            nn.ReLU(),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(VECTOR_CONTEXT_SIZE, context_dim),
            nn.ReLU(),
        )
        self.output_dim = N_UNITS * entity_dim + context_dim

    def forward(self, vector: torch.Tensor) -> VectorEncoding:
        if vector.ndim != 2 or vector.shape[1] != self.vector_size:
            raise ValueError(f"vector must have shape (B,{self.vector_size})")
        _validate_floating_batch(vector, name="vector")
        entity_end = N_UNITS * UNIT_BLOCK_SIZE
        entity_blocks = vector[:, :entity_end].reshape(-1, N_UNITS, UNIT_BLOCK_SIZE)
        entities = self.entity_encoder(entity_blocks)
        context = self.context_encoder(vector[:, entity_end:])
        latent = torch.cat((entities.flatten(start_dim=1), context), dim=-1)
        return VectorEncoding(entities=entities, context=context, latent=latent)


class SemanticCNNEncoder(nn.Module):
    """Encode an ``11xHxW`` semantic map into a fixed-length latent."""

    def __init__(
        self,
        *,
        n_channels: int = len(GRAPHIC_CHANNEL_NAMES),
        graphic_dim: int = 128,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.output_dim = graphic_dim
        self.network = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=5, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, graphic_dim),
            nn.ReLU(),
        )

    def forward(self, graphic: torch.Tensor) -> torch.Tensor:
        if graphic.ndim != 4 or graphic.shape[1] != self.n_channels:
            raise ValueError(f"graphic must have shape (B,{self.n_channels},H,W)")
        if graphic.shape[2] < 8 or graphic.shape[3] < 8:
            raise ValueError("graphic height and width must both be at least 8")
        _validate_floating_batch(graphic, name="graphic")
        return self.network(graphic)


class IPPOActorCritic(nn.Module):
    """Slot-aware shared actor and agent-local critic for one observation row."""

    def __init__(
        self,
        vector_size: int = VECTOR_SIZE,
        n_channels: int = len(GRAPHIC_CHANNEL_NAMES),
        slot_embedding_dim: int = 16,
        hidden_dim: int = 128,
        entity_dim: int = 32,
        context_dim: int = 32,
        graphic_dim: int = 128,
    ) -> None:
        super().__init__()
        self.model_config = {
            "vector_size": vector_size,
            "n_channels": n_channels,
            "slot_embedding_dim": slot_embedding_dim,
            "hidden_dim": hidden_dim,
            "entity_dim": entity_dim,
            "context_dim": context_dim,
            "graphic_dim": graphic_dim,
            "n_actions": N_ACTIONS,
        }
        self.vector_encoder = EntityVectorEncoder(
            vector_size=vector_size,
            entity_dim=entity_dim,
            context_dim=context_dim,
        )
        self.graphic_encoder = SemanticCNNEncoder(
            n_channels=n_channels,
            graphic_dim=graphic_dim,
        )
        self.slot_embedding = nn.Embedding(N_TEAM_AGENTS, slot_embedding_dim)
        self.fusion = nn.Sequential(
            nn.Linear(
                self.vector_encoder.output_dim + graphic_dim + slot_embedding_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # These are single shared heads applied to every team-local slot row.
        self.actor = nn.Linear(hidden_dim, N_ACTIONS)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        vector: torch.Tensor,
        graphic: torch.Tensor,
        slot_id: torch.Tensor,
    ) -> ActorCriticOutput:
        if vector.ndim != 2 or vector.shape[1] != self.model_config["vector_size"]:
            raise ValueError(f"vector must have shape (B,{self.model_config['vector_size']})")
        if graphic.ndim != 4 or graphic.shape[1] != self.model_config["n_channels"]:
            raise ValueError(
                f"graphic must have shape (B,{self.model_config['n_channels']},H,W)"
            )
        if vector.shape[0] != graphic.shape[0]:
            raise ValueError("vector and graphic batch sizes differ")
        if slot_id.shape != (vector.shape[0],) or slot_id.dtype != torch.int64:
            raise ValueError("slot_id must be int64 with shape (B,)")
        if slot_id.device != vector.device or graphic.device != vector.device:
            raise ValueError("vector, graphic, and slot_id must be on the same device")
        if not bool(torch.all((slot_id >= 0) & (slot_id < N_TEAM_AGENTS))):
            raise ValueError(f"slot_id must be in [0,{N_TEAM_AGENTS - 1}]")

        vector_encoding = self.vector_encoder(vector)
        latent = self.fusion(
            torch.cat(
                (
                    vector_encoding.latent,
                    self.graphic_encoder(graphic),
                    self.slot_embedding(slot_id),
                ),
                dim=-1,
            )
        )
        return ActorCriticOutput(
            action_logits=self.actor(latent),
            value=self.critic(latent).squeeze(-1),
        )


# Backward-compatible name used by PREP-12 and existing checkpoints/tests.
ReferenceActorCritic = IPPOActorCritic
