"""Phase-3 observation features, representation experiments, and role analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import html
import math
from pathlib import Path
import json
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .batching import N_TEAM_AGENTS
from .observation import N_UNITS, UNIT_BLOCK_SIZE, VECTOR_SIZE


EPISODE_SECONDS = 300.0
ABSORPTION_SECONDS = 20.0


def self_relative_positions(vector: torch.Tensor, slot_id: torch.Tensor, team: torch.Tensor) -> torch.Tensor:
    """Return all ten positions relative to the row's canonical self unit."""
    if vector.ndim != 2 or vector.shape[1] != VECTOR_SIZE:
        raise ValueError(f"vector must have shape (B,{VECTOR_SIZE})")
    if slot_id.shape != (len(vector),) or team.shape != (len(vector),):
        raise ValueError("slot_id and team must have shape (B,)")
    if not bool(torch.all((slot_id >= 0) & (slot_id < N_TEAM_AGENTS))):
        raise ValueError("slot_id outside team-local range")
    if not bool(torch.all((team == 0) | (team == 1))):
        raise ValueError("team must be 0 or 1")
    blocks = vector[:, : N_UNITS * UNIT_BLOCK_SIZE].reshape(-1, N_UNITS, UNIT_BLOCK_SIZE)
    indices = team.to(torch.int64) * N_TEAM_AGENTS + slot_id.to(torch.int64)
    own = blocks[torch.arange(len(vector), device=vector.device), indices, :2]
    return blocks[:, :, :2] - own[:, None, :]


def absorption_phase_features(time_left: torch.Tensor) -> torch.Tensor:
    """Encode the 20-second cycle from normalized 300-second time remaining."""
    if torch.any((time_left < 0) | (time_left > 1)):
        raise ValueError("time_left must be normalized to [0,1]")
    elapsed = (1.0 - time_left) * EPISODE_SECONDS
    angle = 2.0 * math.pi * torch.remainder(elapsed, ABSORPTION_SECONDS) / ABSORPTION_SECONDS
    return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)


class UnitEntityAttention(nn.Module):
    """Self-conditioned multi-head attention over the ten unit embeddings."""

    def __init__(self, entity_dim: int, *, heads: int = 4) -> None:
        super().__init__()
        if entity_dim % heads:
            raise ValueError("entity_dim must be divisible by heads")
        self.attention = nn.MultiheadAttention(entity_dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(entity_dim)

    def forward(self, entities: torch.Tensor, self_index: torch.Tensor) -> torch.Tensor:
        if entities.ndim != 3 or self_index.shape != (len(entities),):
            raise ValueError("entities must be (B,N,D) and self_index (B,)")
        query = entities[torch.arange(len(entities), device=entities.device), self_index][:, None]
        attended, _ = self.attention(query, entities, entities, need_weights=False)
        return self.norm(query + attended).squeeze(1)


class GlobalLocalMapEncoder(nn.Module):
    """Fuse a global map summary with a self-centred high-resolution crop."""

    def __init__(self, channels: int = 11, *, crop_size: int = 17, output_dim: int = 128) -> None:
        super().__init__()
        if crop_size <= 0 or crop_size % 2 == 0:
            raise ValueError("crop_size must be a positive odd number")
        self.crop_size = crop_size
        self.global_net = nn.Sequential(
            nn.Conv2d(channels, 16, 5, stride=4, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2,2)), nn.Flatten(),
        )
        self.local_net = nn.Sequential(
            nn.Conv2d(channels, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2,2)), nn.Flatten(),
        )
        self.projection = nn.Sequential(nn.Linear(128, output_dim), nn.ReLU())

    def local_crop(self, graphic: torch.Tensor, self_position: torch.Tensor) -> torch.Tensor:
        if graphic.ndim != 4 or self_position.shape != (len(graphic), 2):
            raise ValueError("graphic must be BCHW and self_position Bx2")
        radius = self.crop_size // 2
        padded = F.pad(graphic, (radius, radius, radius, radius))
        height, width = graphic.shape[-2:]
        xy = torch.round(self_position.clamp(0, 1) * torch.tensor(
            [width - 1, height - 1], device=graphic.device
        )).to(torch.int64)
        crops = [
            padded[row, :, int(xy[row,1]):int(xy[row,1])+self.crop_size,
                   int(xy[row,0]):int(xy[row,0])+self.crop_size]
            for row in range(len(graphic))
        ]
        return torch.stack(crops)

    def forward(self, graphic: torch.Tensor, self_position: torch.Tensor) -> torch.Tensor:
        return self.projection(torch.cat((
            self.global_net(graphic), self.local_net(self.local_crop(graphic, self_position))
        ), dim=-1))


AUXILIARY_TARGETS = ("role", "holding_item", "seconds_to_absorption", "score_delta")


class AuxiliaryTargetLogger:
    def __init__(self, path: str | Path) -> None: self.path=Path(path)
    def log(self, *, step: int, targets: Mapping[str, torch.Tensor],
            predictions: Mapping[str, torch.Tensor]) -> None:
        if set(targets) != set(AUXILIARY_TARGETS) or set(predictions) != set(AUXILIARY_TARGETS):
            raise ValueError("auxiliary logger requires all declared targets")
        record={"schema_version":"blackout.auxiliary_targets.v1","step":step,"targets":{},"predictions":{}}
        for group,values in (("targets",targets),("predictions",predictions)):
            for name,tensor in values.items(): record[group][name]=tensor.detach().cpu().tolist()
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.open("a",encoding="utf-8") as output:
            output.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+"\n")


class AuxiliaryHeads(nn.Module):
    def __init__(self, latent_dim: int, *, roles: int = 3, items: int = 6) -> None:
        super().__init__()
        self.role = nn.Linear(latent_dim, roles)
        self.holding_item = nn.Linear(latent_dim, items)
        self.seconds_to_absorption = nn.Linear(latent_dim, 1)
        self.score_delta = nn.Linear(latent_dim, 1)

    def forward(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "role": self.role(latent), "holding_item": self.holding_item(latent),
            "seconds_to_absorption": self.seconds_to_absorption(latent).squeeze(-1),
            "score_delta": self.score_delta(latent).squeeze(-1),
        }


def auxiliary_losses(
    predictions: Mapping[str, torch.Tensor], targets: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    if set(predictions) != set(AUXILIARY_TARGETS) or set(targets) != set(AUXILIARY_TARGETS):
        raise ValueError("auxiliary predictions/targets must contain the four declared targets")
    return {
        "role": F.cross_entropy(predictions["role"], targets["role"].long()),
        "holding_item": F.cross_entropy(predictions["holding_item"], targets["holding_item"].long()),
        "seconds_to_absorption": F.smooth_l1_loss(predictions["seconds_to_absorption"], targets["seconds_to_absorption"]),
        "score_delta": F.smooth_l1_loss(predictions["score_delta"], targets["score_delta"]),
    }


def select_auxiliary_targets(results: Mapping[str, Mapping[str, float]]) -> tuple[str, ...]:
    """Retain only independently tested targets that improve held-out wins without score regression."""
    unknown = set(results) - set(AUXILIARY_TARGETS)
    if unknown:
        raise ValueError(f"unknown auxiliary targets: {sorted(unknown)}")
    return tuple(sorted(name for name, row in results.items()
                        if row["win_rate_delta"] > 0 and row["score_diff_delta"] >= 0))


@dataclass(frozen=True)
class SlotRoleMetrics:
    slot: int
    path: tuple[tuple[float, float], ...]
    collected: int
    deaths: int
    transformations: int
    storage_visits: int


def summarize_roles(records: Sequence[Mapping[str, Any]]) -> tuple[SlotRoleMetrics, ...]:
    results: list[SlotRoleMetrics] = []
    for slot in range(N_TEAM_AGENTS):
        rows = [row for row in records if int(row["slot"]) == slot]
        results.append(SlotRoleMetrics(
            slot=slot, path=tuple((float(row["x"]), float(row["y"])) for row in rows),
            collected=sum(bool(row.get("collected")) for row in rows),
            deaths=sum(bool(row.get("died")) for row in rows),
            transformations=sum(bool(row.get("transformed")) for row in rows),
            storage_visits=sum(bool(row.get("storage_visit")) for row in rows),
        ))
    return tuple(results)


def render_role_svg(metrics: Sequence[SlotRoleMetrics], path: str | Path) -> None:
    colors = ("#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ea580c")
    lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="800" height="520" viewBox="0 0 800 520">',
             '<rect width="800" height="520" fill="white"/>']
    for row in metrics:
        points = " ".join(f"{20+x*460:.1f},{500-y*460:.1f}" for x,y in row.path)
        if points: lines.append(f'<polyline points="{points}" fill="none" stroke="{colors[row.slot]}" stroke-width="2"/>')
        label = html.escape(f"slot {row.slot}: collect={row.collected} death={row.deaths} transform={row.transformations} storage={row.storage_visits}")
        lines.append(f'<text x="510" y="{40+row.slot*45}" fill="{colors[row.slot]}" font-size="14">{label}</text>')
    lines.append("</svg>")
    output = Path(path); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
