"""Experiment-side utilities for the BlackOut reinforcement-learning project."""

from .batching import ObservationBatch, canonical_agents, stack_observations, team_agents
from .env import ContractBlackOutEnv
from .reward import ScoreDeltaEvent, ScoreDeltaRewardTracker
from .model_contract import (
    ACTION_DIRECTIONS,
    CHECKPOINT_SCHEMA_VERSION,
    ActorCriticOutput,
    CanonicalTeamModel,
    ReferenceActorCritic,
    SubmissionPolicy,
    TeamModelInput,
    categorical_action,
    checkpoint_payload,
    deterministic_action,
    load_checkpoint,
    save_checkpoint,
    team_model_input,
    validate_checkpoint,
)
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
    "ACTION_DIRECTIONS",
    "CHECKPOINT_SCHEMA_VERSION",
    "ActorCriticOutput",
    "CanonicalTeamModel",
    "VECTOR_SIZE",
    "GraphicObservation",
    "ObservationBatch",
    "ContractBlackOutEnv",
    "ScoreDeltaEvent",
    "ScoreDeltaRewardTracker",
    "ReferenceActorCritic",
    "SubmissionPolicy",
    "TeamModelInput",
    "VectorObservation",
    "canonical_agents",
    "categorical_action",
    "checkpoint_payload",
    "deterministic_action",
    "hwc_to_chw",
    "parse_graphic",
    "parse_vector",
    "load_checkpoint",
    "save_checkpoint",
    "stack_observations",
    "team_agents",
    "team_model_input",
    "validate_checkpoint",
]
