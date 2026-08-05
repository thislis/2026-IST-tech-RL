"""Canonical observation batching that never relies on dictionary iteration order."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .observation import GRAPHIC_CHANNEL_NAMES, VECTOR_SIZE, parse_graphic, parse_vector


N_AGENTS = 10
N_TEAM_AGENTS = 5


def canonical_agents() -> tuple[str, ...]:
    return tuple(f"unit_{index}" for index in range(N_AGENTS))


def team_agents(team: int) -> tuple[str, ...]:
    if team not in (0, 1):
        raise ValueError(f"team must be 0 or 1, got {team}")
    start = team * N_TEAM_AGENTS
    return tuple(f"unit_{index}" for index in range(start, start + N_TEAM_AGENTS))


@dataclass(frozen=True)
class ObservationBatch:
    agent_names: tuple[str, ...]
    slot_ids: np.ndarray
    vectors: np.ndarray
    graphics_hwc: np.ndarray

    @property
    def graphics_chw(self) -> np.ndarray:
        return np.ascontiguousarray(np.transpose(self.graphics_hwc, (0, 3, 1, 2)))


def stack_observations(
    observations: Mapping[str, Mapping[str, np.ndarray]],
    agents: Sequence[str] | None = None,
) -> ObservationBatch:
    """Stack observations in explicit unit-index order and attach team-local slots.

    `slot_ids` are `0..4` within each team. They are the required self-identity
    signal because the current 96-vector intentionally strips Unity's routing
    `unitIndex` and otherwise gives same-class teammates identical observations.
    """
    ordered = tuple(agents) if agents is not None else canonical_agents()
    unknown = set(ordered) - set(canonical_agents())
    if unknown:
        raise ValueError(f"unknown agent names: {sorted(unknown)}")
    missing = [agent for agent in ordered if agent not in observations]
    if missing:
        raise KeyError(f"missing observations for: {missing}")

    vectors: list[np.ndarray] = []
    graphics: list[np.ndarray] = []
    slots: list[int] = []
    graphic_shape: tuple[int, int, int] | None = None
    for agent in ordered:
        vector = observations[agent]["vector"]
        graphic = observations[agent]["graphic"]
        parse_vector(vector)
        parse_graphic(graphic)
        if vector.shape != (VECTOR_SIZE,):
            raise ValueError(f"unexpected vector shape for {agent}: {vector.shape}")
        if graphic_shape is None:
            graphic_shape = graphic.shape
        elif graphic.shape != graphic_shape:
            raise ValueError("all graphic observations must have the same shape")
        if graphic.shape[-1] != len(GRAPHIC_CHANNEL_NAMES):
            raise ValueError(f"unexpected graphic channels for {agent}: {graphic.shape}")
        vectors.append(vector)
        graphics.append(graphic)
        slots.append(int(agent.split("_")[1]) % N_TEAM_AGENTS)

    return ObservationBatch(
        agent_names=ordered,
        slot_ids=np.asarray(slots, dtype=np.int64),
        vectors=np.stack(vectors).astype(np.float32, copy=False),
        graphics_hwc=np.stack(graphics).astype(np.float32, copy=False),
    )
