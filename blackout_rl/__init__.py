"""Experiment-side utilities for the BlackOut reinforcement-learning project."""

from .batching import ObservationBatch, canonical_agents, stack_observations, team_agents
from .env import ContractBlackOutEnv
from .reward import ScoreDeltaEvent, ScoreDeltaRewardTracker
from .observation import (
    GRAPHIC_CHANNEL_NAMES,
    VECTOR_SIZE,
    GraphicObservation,
    VectorObservation,
    hwc_to_chw,
    parse_graphic,
    parse_vector,
)

__all__ = [
    "GRAPHIC_CHANNEL_NAMES",
    "VECTOR_SIZE",
    "GraphicObservation",
    "ObservationBatch",
    "ContractBlackOutEnv",
    "ScoreDeltaEvent",
    "ScoreDeltaRewardTracker",
    "VectorObservation",
    "canonical_agents",
    "hwc_to_chw",
    "parse_graphic",
    "parse_vector",
    "stack_observations",
    "team_agents",
]
