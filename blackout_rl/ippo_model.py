"""Entity-aware IPPO actor/critic encoders and shared policy network."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

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


class GlobalLocalSemanticEncoder(nn.Module):
    """Preserve global layout and a high-resolution crop around each acting unit."""

    def __init__(
        self, *, n_channels: int, graphic_dim: int, crop_size: int = 32
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.crop_size = crop_size
        self.global_network = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=5, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.local_network = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.projection = nn.Sequential(nn.Linear(1024, graphic_dim), nn.ReLU())

    def forward(self, graphic: torch.Tensor, self_position: torch.Tensor) -> torch.Tensor:
        if graphic.ndim != 4 or graphic.shape[1] != self.n_channels:
            raise ValueError("graphic has the wrong global/local encoder shape")
        if self_position.shape != (len(graphic), 2):
            raise ValueError("self_position must have shape (B,2)")
        _validate_floating_batch(graphic, name="graphic")
        height, width = graphic.shape[-2:]
        scale_x = (self.crop_size - 1) / max(width - 1, 1)
        scale_y = (self.crop_size - 1) / max(height - 1, 1)
        theta = graphic.new_zeros((len(graphic), 2, 3))
        theta[:, 0, 0] = scale_x
        theta[:, 1, 1] = scale_y
        theta[:, 0, 2] = self_position[:, 0].clamp(0, 1) * 2 - 1
        theta[:, 1, 2] = 1 - self_position[:, 1].clamp(0, 1) * 2
        grid = F.affine_grid(
            theta,
            (len(graphic), self.n_channels, self.crop_size, self.crop_size),
            align_corners=True,
        )
        local = F.grid_sample(
            graphic, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        return self.projection(
            torch.cat((self.global_network(graphic), self.local_network(local)), dim=-1)
        )


class SpatialTargetResidual(nn.Module):
    """Encode nearest task-relevant semantic targets relative to the acting unit."""

    TARGET_CHANNELS = (6, 2, 3, 5, 7)  # battery, own/enemy storage, enemy, speed shrine

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(2 * len(self.TARGET_CHANNELS), 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, graphic: torch.Tensor, self_position: torch.Tensor) -> torch.Tensor:
        height, width = graphic.shape[-2:]
        y, x = torch.meshgrid(
            torch.linspace(1.0, 0.0, height, device=graphic.device),
            torch.linspace(0.0, 1.0, width, device=graphic.device),
            indexing="ij",
        )
        coordinates = torch.stack((x, y), dim=-1).reshape(1, height * width, 2)
        relative = coordinates - self_position[:, None, :]
        distance = relative.square().sum(dim=-1)
        features = []
        for channel in self.TARGET_CHANNELS:
            mask = graphic[:, channel].reshape(len(graphic), -1) > 0.5
            nearest = distance.masked_fill(~mask, float("inf")).argmin(dim=1)
            selected = relative[
                torch.arange(len(graphic), device=graphic.device), nearest
            ]
            selected = torch.where(mask.any(dim=1, keepdim=True), selected, 0.0)
            features.append(selected)
        return self.projection(torch.cat(features, dim=-1))


class MultiTargetResidual(nn.Module):
    """Expose multiple nearby batteries and relative unit geometry explicitly."""

    TARGET_COUNTS = ((6, 3), (2, 1), (3, 1), (5, 1), (7, 1), (9, 1))

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        input_dim = 2 * sum(count for _, count in self.TARGET_COUNTS) + 3 * N_UNITS
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(
        self,
        graphic: torch.Tensor,
        self_position: torch.Tensor,
        entity_blocks: torch.Tensor,
    ) -> torch.Tensor:
        height, width = graphic.shape[-2:]
        y, x = torch.meshgrid(
            torch.linspace(1.0, 0.0, height, device=graphic.device),
            torch.linspace(0.0, 1.0, width, device=graphic.device),
            indexing="ij",
        )
        coordinates = torch.stack((x, y), dim=-1).reshape(1, height * width, 2)
        relative = coordinates - self_position[:, None, :]
        distance = relative.square().sum(dim=-1)
        target_features = []
        for channel, count in self.TARGET_COUNTS:
            mask = graphic[:, channel].reshape(len(graphic), -1) > 0.5
            available = mask.sum(dim=1).clamp(max=count)
            selected_indices = torch.topk(
                distance.masked_fill(~mask, float("inf")),
                count,
                dim=1,
                largest=False,
            ).indices
            selected = relative.gather(
                1, selected_indices.unsqueeze(-1).expand(-1, -1, 2)
            )
            valid = torch.arange(count, device=graphic.device)[None] < available[:, None]
            target_features.append(torch.where(valid.unsqueeze(-1), selected, 0.0))
        entity_relative = entity_blocks[:, :, :2] - self_position[:, None, :]
        entity_features = torch.cat(
            (entity_relative, entity_blocks[:, :, 2:3]), dim=-1
        ).flatten(start_dim=1)
        targets = torch.cat(target_features, dim=1).flatten(start_dim=1)
        return self.projection(torch.cat((targets, entity_features), dim=-1))


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
        map_encoder_version: str = "legacy",
        entity_order_version: str = "absolute",
        action_head_version: str = "categorical9",
        recurrent_version: str = "none",
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
            "map_encoder_version": map_encoder_version,
            "entity_order_version": entity_order_version,
            "action_head_version": action_head_version,
            "recurrent_version": recurrent_version,
            "n_actions": N_ACTIONS,
        }
        self.vector_encoder = EntityVectorEncoder(
            vector_size=vector_size,
            entity_dim=entity_dim,
            context_dim=context_dim,
        )
        if entity_order_version not in {"absolute", "team_relative_v1"}:
            raise ValueError(f"unknown entity order version: {entity_order_version}")
        if action_head_version not in {"categorical9", "continuous_tanh_v1"}:
            raise ValueError(f"unknown action head version: {action_head_version}")
        if recurrent_version not in {"none", "gru_v1"}:
            raise ValueError(f"unknown recurrent version: {recurrent_version}")
        if map_encoder_version == "legacy":
            self.graphic_encoder = SemanticCNNEncoder(
                n_channels=n_channels, graphic_dim=graphic_dim
            )
        elif map_encoder_version in {
            "global_local_v1",
            "global_local_spatial_v2",
            "global_local_multitarget_v3",
        }:
            self.graphic_encoder = GlobalLocalSemanticEncoder(
                n_channels=n_channels, graphic_dim=graphic_dim
            )
        else:
            raise ValueError(f"unknown map encoder version: {map_encoder_version}")
        self.spatial_target_residual = (
            SpatialTargetResidual(graphic_dim)
            if map_encoder_version == "global_local_spatial_v2"
            else None
        )
        self.multi_target_residual = (
            MultiTargetResidual(graphic_dim)
            if map_encoder_version == "global_local_multitarget_v3"
            else None
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
        self.recurrent = (
            nn.GRUCell(hidden_dim, hidden_dim)
            if recurrent_version == "gru_v1"
            else None
        )
        # These are single shared heads applied to every team-local slot row.
        actor_output_dim = N_ACTIONS if action_head_version == "categorical9" else 2
        self.actor = nn.Linear(hidden_dim, actor_output_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def encode(
        self,
        vector: torch.Tensor,
        graphic: torch.Tensor,
        slot_id: torch.Tensor,
    ) -> torch.Tensor:
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

        entity_blocks = vector[:, : N_UNITS * UNIT_BLOCK_SIZE].reshape(
            -1, N_UNITS, UNIT_BLOCK_SIZE
        )
        if self.model_config["entity_order_version"] == "team_relative_v1":
            # The environment's vector is absolute unit_0..unit_9 order while all
            # other fields are observer-relative. Canonicalize allies first so one
            # shared policy sees the same entity layout on both physical sides.
            order = torch.argsort(entity_blocks[:, :, 2], dim=1, descending=True)
            entity_blocks = entity_blocks.gather(
                1, order.unsqueeze(-1).expand(-1, -1, UNIT_BLOCK_SIZE)
            )
            encoded_vector = torch.cat(
                (
                    entity_blocks.flatten(start_dim=1),
                    vector[:, N_UNITS * UNIT_BLOCK_SIZE :],
                ),
                dim=1,
            )
            self_position = entity_blocks[
                torch.arange(len(vector), device=vector.device), slot_id, :2
            ]
        else:
            encoded_vector = vector
            ally_indices = torch.topk(
                entity_blocks[:, :, 2], N_TEAM_AGENTS, dim=1
            ).indices
            ally_indices = torch.sort(ally_indices, dim=1).values
            self_indices = ally_indices[
                torch.arange(len(vector), device=vector.device), slot_id
            ]
            self_position = entity_blocks[
                torch.arange(len(vector), device=vector.device), self_indices, :2
            ]
        vector_encoding = self.vector_encoder(encoded_vector)
        graphic_latent = (
            self.graphic_encoder(graphic)
            if self.model_config["map_encoder_version"] == "legacy"
            else self.graphic_encoder(graphic, self_position)
        )
        if self.spatial_target_residual is not None:
            graphic_latent = graphic_latent + self.spatial_target_residual(
                graphic, self_position
            )
        if self.multi_target_residual is not None:
            graphic_latent = graphic_latent + self.multi_target_residual(
                graphic, self_position, entity_blocks
            )
        return self.fusion(
            torch.cat(
                (
                    vector_encoding.latent,
                    graphic_latent,
                    self.slot_embedding(slot_id),
                ),
                dim=-1,
            )
        )

    def forward_recurrent(
        self,
        vector: torch.Tensor,
        graphic: torch.Tensor,
        slot_id: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[ActorCriticOutput, torch.Tensor | None]:
        latent = self.encode(vector, graphic, slot_id)
        next_hidden = None
        if self.recurrent is not None:
            if hidden is None:
                hidden = latent.new_zeros(latent.shape)
            if hidden.shape != latent.shape:
                raise ValueError("recurrent hidden state must match fused latent shape")
            next_hidden = self.recurrent(latent, hidden)
            latent = next_hidden
        return (
            ActorCriticOutput(
                action_logits=self.actor(latent),
                value=self.critic(latent).squeeze(-1),
            ),
            next_hidden,
        )

    def forward(
        self,
        vector: torch.Tensor,
        graphic: torch.Tensor,
        slot_id: torch.Tensor,
    ) -> ActorCriticOutput:
        output, _ = self.forward_recurrent(vector, graphic, slot_id)
        return output


# Backward-compatible name used by PREP-12 and existing checkpoints/tests.
ReferenceActorCritic = IPPOActorCritic
