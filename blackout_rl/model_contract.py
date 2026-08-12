"""PREP-12 actor/critic, action, batching, and checkpoint contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from blackout_env import BaseModel
from torch import nn

from .action_distribution import (
    ACTION_DIRECTIONS,
    categorical_action,
    deterministic_action,
)
from .batching import N_TEAM_AGENTS, ObservationBatch, stack_observations, team_agents
from .ippo_model import ActorCriticOutput, IPPOActorCritic, ReferenceActorCritic
from .observation import GRAPHIC_CHANNEL_NAMES, VECTOR_SIZE


CHECKPOINT_SCHEMA_VERSION = "blackout.checkpoint.v1"


@dataclass(frozen=True)
class TeamModelInput:
    """One canonical team batch ready for actor/critic inference."""

    agent_names: tuple[str, ...]
    vector: torch.Tensor
    graphic: torch.Tensor
    slot_id: torch.Tensor


def team_model_input(
    observations: Mapping[str, Mapping[str, np.ndarray]],
    team: int,
    *,
    device: str | torch.device = "cpu",
) -> TeamModelInput:
    """Stack one team as unit-index order with explicit team-local slot IDs."""
    batch: ObservationBatch = stack_observations(observations, agents=team_agents(team))
    return TeamModelInput(
        agent_names=batch.agent_names,
        vector=torch.as_tensor(batch.vectors, dtype=torch.float32, device=device),
        graphic=torch.as_tensor(batch.graphics_chw, dtype=torch.float32, device=device),
        slot_id=torch.as_tensor(batch.slot_ids, dtype=torch.int64, device=device),
    )


class SubmissionPolicy(nn.Module):
    """Official two-input policy facade around the actor/critic.

    The fallback `forward` assigns slots by batch row solely for compatibility
    with blackout-env's current `load_checkpoint` helper. Production project
    evaluation uses `CanonicalTeamModel`, which derives slots from agent names.
    The official evaluator must guarantee team batch order before this fallback
    can be relied on for submission.
    """

    def __init__(self, **model_kwargs: Any) -> None:
        super().__init__()
        self.actor_critic = IPPOActorCritic(**model_kwargs)

    def forward_with_slots(
        self,
        vector: torch.Tensor,
        graphic: torch.Tensor,
        slot_id: torch.Tensor,
    ) -> torch.Tensor:
        return deterministic_action(self.actor_critic(vector, graphic, slot_id).action_logits)

    def forward(self, vector: torch.Tensor, graphic: torch.Tensor) -> torch.Tensor:
        if vector.shape[0] != N_TEAM_AGENTS:
            raise ValueError("official team inference batch must contain exactly 5 agents")
        slot_id = torch.arange(N_TEAM_AGENTS, dtype=torch.int64, device=vector.device)
        return self.forward_with_slots(vector, graphic, slot_id)


class CanonicalTeamModel(BaseModel):
    """BaseModel adapter that never assigns slots from dictionary iteration order."""

    def __init__(
        self,
        policy: SubmissionPolicy,
        team: int,
        device: str | torch.device = "cpu",
    ) -> None:
        self._policy = policy.to(device).eval()
        self._team = team
        self._device = torch.device(device)

    def act(
        self,
        obs: dict[str, dict[str, np.ndarray]],
    ) -> dict[str, np.ndarray]:
        expected = set(team_agents(self._team))
        if set(obs) != expected:
            raise ValueError(f"expected observations for {sorted(expected)}, got {sorted(obs)}")
        batch = team_model_input(obs, self._team, device=self._device)
        with torch.inference_mode():
            action = self._policy.forward_with_slots(
                batch.vector, batch.graphic, batch.slot_id
            ).cpu().numpy().astype(np.float32, copy=False)
        return {agent: action[row] for row, agent in enumerate(batch.agent_names)}


def checkpoint_payload(
    model: SubmissionPolicy,
    *,
    global_step: int,
    training_seed: int,
    optimizer: torch.optim.Optimizer | None = None,
    source: Mapping[str, str] | None = None,
    experiment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an upstream-loader-compatible, versioned training checkpoint."""
    payload: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "policy_state": model.state_dict(),
        "model_config": dict(model.actor_critic.model_config),
        "training": {"global_step": int(global_step), "seed": int(training_seed)},
        "observation_contract": {
            "vector_shape": [VECTOR_SIZE],
            "graphic_hwc_shape": [96, 96, len(GRAPHIC_CHANNEL_NAMES)],
            "model_graphic_chw_shape": [len(GRAPHIC_CHANNEL_NAMES), 96, 96],
            "team_batch_size": N_TEAM_AGENTS,
            "slot_ids": list(range(N_TEAM_AGENTS)),
        },
        "action_contract": {
            "distribution": "categorical_9",
            "output_shape": [2],
            "range": [-1.0, 1.0],
            "inference": "deterministic_argmax",
        },
        "source": dict(source or {}),
    }
    if experiment is not None:
        payload["experiment"] = dict(experiment)
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    validate_checkpoint(payload)
    return payload


def validate_checkpoint(payload: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "policy_state",
        "model_config",
        "training",
        "observation_contract",
        "action_contract",
        "source",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"checkpoint missing keys: {sorted(missing)}")
    if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"unsupported checkpoint schema: {payload['schema_version']}")
    if not isinstance(payload["policy_state"], Mapping):
        raise TypeError("policy_state must be a state-dict mapping")
    if payload["observation_contract"].get("team_batch_size") != N_TEAM_AGENTS:
        raise ValueError("checkpoint team batch size does not match runtime contract")
    if payload["action_contract"].get("distribution") != "categorical_9":
        raise ValueError("checkpoint action distribution does not match runtime contract")
    if "experiment" in payload:
        experiment = payload["experiment"]
        required_experiment = {"run_id", "config", "seed", "git_sha", "opponent_id"}
        if not isinstance(experiment, Mapping):
            raise TypeError("checkpoint experiment must be a mapping")
        missing_experiment = required_experiment - set(experiment)
        if missing_experiment:
            raise ValueError(
                f"checkpoint experiment missing keys: {sorted(missing_experiment)}"
            )
        git_sha = experiment["git_sha"]
        if not isinstance(git_sha, str) or len(git_sha) != 40:
            raise ValueError("checkpoint experiment git_sha must have 40 characters")
        if int(experiment["seed"]) != int(payload["training"]["seed"]):
            raise ValueError("checkpoint experiment and training seeds differ")


def save_checkpoint(path: str | Path, payload: Mapping[str, Any]) -> None:
    validate_checkpoint(payload)
    torch.save(dict(payload), Path(path))


def load_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[SubmissionPolicy, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("project checkpoint must be a dictionary")
    validate_checkpoint(payload)
    config = dict(payload["model_config"])
    config.pop("n_actions", None)
    model = SubmissionPolicy(**config)
    model.load_state_dict(payload["policy_state"])
    model.to(device).eval()
    return model, payload
