"""MAPPO/CTDE models, centralized state construction, and joint PPO updates."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .action_distribution import (
    ACTION_DIRECTIONS,
    N_ACTIONS,
    evaluate_categorical_action,
    select_categorical_action,
)
from .batching import N_TEAM_AGENTS, canonical_agents, team_agents
from .ippo_model import IPPOActorCritic, SemanticCNNEncoder
from .observation import GRAPHIC_CHANNEL_NAMES, N_CLASSES, N_UNITS, UNIT_BLOCK_SIZE, parse_vector
from .ppo import PPOConfig, PPODiagnostics, explained_variance
from .rollout import generalized_advantage_estimate
from .model_contract import load_checkpoint, team_model_input


CENTRAL_UNIT_SIZE = UNIT_BLOCK_SIZE + N_CLASSES
CENTRAL_CONTEXT_SIZE = 3  # absolute team A score, team B score, time
CENTRAL_VECTOR_SIZE = N_UNITS * CENTRAL_UNIT_SIZE + CENTRAL_CONTEXT_SIZE
PLANNER_RESIDUAL_CONTEXT_SIZE = N_ACTIONS + 2 + 1 + 2 + 4


@dataclass(frozen=True)
class CentralState:
    vector: torch.Tensor
    graphic: torch.Tensor


class CentralizedStateBuilder:
    """Build a team-perspective critic state from all ten training observations."""

    def build(
        self,
        observations: Mapping[str, Mapping[str, np.ndarray]],
        *,
        learning_team: int,
        device: str | torch.device = "cpu",
    ) -> CentralState:
        if learning_team not in (0, 1):
            raise ValueError("learning_team must be 0 or 1")
        missing = set(canonical_agents()) - set(observations)
        if missing:
            raise KeyError(f"centralized state requires all ten agents: {sorted(missing)}")
        reference_name = team_agents(learning_team)[0]
        reference = parse_vector(observations[reference_name]["vector"])
        blocks = observations[reference_name]["vector"][: N_UNITS * UNIT_BLOCK_SIZE].reshape(
            N_UNITS, UNIT_BLOCK_SIZE
        )
        classes = np.stack(
            [parse_vector(observations[name]["vector"]).self_class_one_hot for name in canonical_agents()]
        )
        unit_features = np.concatenate((blocks, classes), axis=-1).reshape(-1)
        if learning_team == 0:
            scores = (reference.own_score, reference.opponent_score)
        else:
            scores = (reference.opponent_score, reference.own_score)
        central_vector = np.concatenate(
            (unit_features, np.asarray((*scores, reference.time_left), dtype=np.float32))
        ).astype(np.float32, copy=False)
        if central_vector.shape != (CENTRAL_VECTOR_SIZE,):
            raise RuntimeError("centralized vector layout mismatch")
        graphic = observations[reference_name]["graphic"]
        if graphic.ndim != 3 or graphic.shape[-1] != len(GRAPHIC_CHANNEL_NAMES):
            raise ValueError("central graphic must be HWC with 11 channels")
        return CentralState(
            vector=torch.as_tensor(central_vector[None], device=device),
            graphic=torch.as_tensor(
                np.ascontiguousarray(graphic.transpose(2, 0, 1))[None], device=device
            ),
        )


class CentralizedCritic(nn.Module):
    def __init__(self, *, hidden_dim: int = 128, graphic_dim: int = 128) -> None:
        super().__init__()
        self.vector_encoder = nn.Sequential(
            nn.Linear(CENTRAL_VECTOR_SIZE, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.graphic_encoder = SemanticCNNEncoder(graphic_dim=graphic_dim)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim + graphic_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, vector: torch.Tensor, graphic: torch.Tensor) -> torch.Tensor:
        if vector.ndim != 2 or vector.shape[1] != CENTRAL_VECTOR_SIZE:
            raise ValueError(f"central vector must have shape (B,{CENTRAL_VECTOR_SIZE})")
        if vector.shape[0] != graphic.shape[0]:
            raise ValueError("central vector and graphic batch sizes differ")
        return self.value_head(
            torch.cat((self.vector_encoder(vector), self.graphic_encoder(graphic)), dim=-1)
        ).squeeze(-1)


@dataclass(frozen=True)
class MAPPOOutput:
    action_logits: torch.Tensor
    value: torch.Tensor


class MAPPOActorCritic(nn.Module):
    """Decentralized IPPO-compatible actor plus training-only centralized critic."""

    def __init__(self, actor: IPPOActorCritic | None = None) -> None:
        super().__init__()
        self.actor_model = actor or IPPOActorCritic()
        self.planner_residual_head: PlannerConditionedResidualHead | None = None
        self.centralized_critic = CentralizedCritic(
            hidden_dim=int(self.actor_model.model_config["hidden_dim"]),
            graphic_dim=int(self.actor_model.model_config["graphic_dim"]),
        )

    def actor_logits(
        self,
        vector: torch.Tensor,
        graphic: torch.Tensor,
        slot_id: torch.Tensor,
        planner_action_index: torch.Tensor | None = None,
        planner_path_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.planner_residual_head is None:
            return self.actor_model(vector, graphic, slot_id).action_logits
        if planner_action_index is None:
            raise ValueError("planner-conditioned residual actor requires planner actions")
        latent = self.actor_model.encode(vector, graphic, slot_id)
        context = planner_residual_context(
            vector,
            graphic,
            slot_id,
            planner_action_index,
            planner_path_valid=planner_path_valid,
        )
        return self.planner_residual_head(latent, context)

    def forward(
        self,
        vector: torch.Tensor,
        graphic: torch.Tensor,
        slot_id: torch.Tensor,
        central_vector: torch.Tensor,
        central_graphic: torch.Tensor,
        planner_action_index: torch.Tensor | None = None,
        planner_path_valid: torch.Tensor | None = None,
    ) -> MAPPOOutput:
        return MAPPOOutput(
            action_logits=self.actor_logits(
                vector,
                graphic,
                slot_id,
                planner_action_index,
                planner_path_valid,
            ),
            value=self.centralized_critic(central_vector, central_graphic),
        )

    def decentralized_action_logits(
        self, vector: torch.Tensor, graphic: torch.Tensor, slot_id: torch.Tensor
    ) -> torch.Tensor:
        """Submission path: deliberately accepts no centralized state."""
        return self.actor_logits(vector, graphic, slot_id)


def initialize_mappo_from_ippo(ippo: IPPOActorCritic) -> MAPPOActorCritic:
    """Copy every actor-side parameter while creating a fresh centralized critic."""
    return MAPPOActorCritic(actor=deepcopy(ippo))


def initialize_mappo_from_checkpoint(
    checkpoint: str, *, device: str | torch.device = "cpu"
) -> tuple[MAPPOActorCritic, dict[str, Any]]:
    policy, payload = load_checkpoint(checkpoint, device=device)
    model = initialize_mappo_from_ippo(policy.actor_critic).to(device)
    return model, payload


def initialize_planner_residual_head(
    model: MAPPOActorCritic, *, fallback_logit_bias: float = 4.0
) -> None:
    """Start a residual policy at the scripted planner with safe exploration.

    Residual action index 0 means "use the planner action" while indices 1..8
    replace it with a learned direction. Resetting only the final actor head
    retains the pretrained representation and prevents the weak neural core from
    replacing the audited planner before PPO has learned useful deviations.
    """

    actor = model.actor_model
    if actor.model_config["action_head_version"] != "categorical9":
        raise ValueError("planner residual training requires categorical9 actions")
    if not math.isfinite(fallback_logit_bias) or fallback_logit_bias <= 0.0:
        raise ValueError("fallback logit bias must be finite and positive")
    with torch.no_grad():
        actor.actor.weight.zero_()
        actor.actor.bias.zero_()
        actor.actor.bias[0] = fallback_logit_bias


class PlannerConditionedResidualHead(nn.Module):
    """Small v5 head conditioned on the planner decision and task context."""

    def __init__(self, hidden_dim: int, *, fallback_logit_bias: float = 6.5) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if not math.isfinite(fallback_logit_bias) or fallback_logit_bias <= 0.0:
            raise ValueError("fallback logit bias must be finite and positive")
        self.hidden_dim = int(hidden_dim)
        self.context_size = PLANNER_RESIDUAL_CONTEXT_SIZE
        self.network = nn.Sequential(
            nn.Linear(self.hidden_dim + self.context_size, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, N_ACTIONS),
        )
        with torch.no_grad():
            final = self.network[-1]
            assert isinstance(final, nn.Linear)
            final.weight.zero_()
            final.bias.zero_()
            final.bias[0] = fallback_logit_bias

    def forward(self, latent: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.hidden_dim:
            raise ValueError(f"latent must have shape (B,{self.hidden_dim})")
        if context.shape != (len(latent), self.context_size):
            raise ValueError(
                f"planner context must have shape (B,{self.context_size})"
            )
        return self.network(torch.cat((latent, context), dim=-1))


def planner_residual_context(
    vector: torch.Tensor,
    graphic: torch.Tensor,
    slot_id: torch.Tensor,
    planner_action_index: torch.Tensor,
    *,
    planner_path_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build explicit planner, role, path, and nearest-target residual features."""

    batch = len(vector)
    if planner_action_index.shape != (batch,) or planner_action_index.dtype != torch.int64:
        raise ValueError("planner_action_index must be int64 with shape (B,)")
    if not bool(torch.all((planner_action_index >= 0) & (planner_action_index < N_ACTIONS))):
        raise ValueError("planner_action_index is outside the categorical action range")
    if planner_path_valid is None:
        planner_path_valid = torch.ones(batch, dtype=torch.bool, device=vector.device)
    if planner_path_valid.shape != (batch,) or planner_path_valid.dtype != torch.bool:
        raise ValueError("planner_path_valid must be bool with shape (B,)")
    if planner_action_index.device != vector.device or planner_path_valid.device != vector.device:
        raise ValueError("planner residual inputs must share a device")

    blocks = vector[:, : N_UNITS * UNIT_BLOCK_SIZE].reshape(
        batch, N_UNITS, UNIT_BLOCK_SIZE
    )
    ally_indices = torch.topk(blocks[:, :, 2], N_TEAM_AGENTS, dim=1).indices
    ally_indices = torch.sort(ally_indices, dim=1).values
    self_indices = ally_indices[torch.arange(batch, device=vector.device), slot_id]
    self_blocks = blocks[torch.arange(batch, device=vector.device), self_indices]
    self_position = self_blocks[:, :2]
    holding_item = torch.argmax(self_blocks[:, 3:], dim=-1) != 0
    guard = slot_id >= 3

    # Workers seek batteries until carrying, then seek own storage. Guards seek enemies.
    target_channel = torch.where(
        guard,
        torch.full_like(slot_id, 5),
        torch.where(holding_item, torch.full_like(slot_id, 2), torch.full_like(slot_id, 6)),
    )
    height, width = graphic.shape[-2:]
    y, x = torch.meshgrid(
        torch.linspace(1.0, 0.0, height, device=graphic.device),
        torch.linspace(0.0, 1.0, width, device=graphic.device),
        indexing="ij",
    )
    coordinates = torch.stack((x, y), dim=-1).reshape(1, height * width, 2)
    relative = coordinates - self_position[:, None, :]
    distance2 = relative.square().sum(dim=-1)
    flat = graphic.reshape(batch, graphic.shape[1], height * width)
    target_mask = flat[
        torch.arange(batch, device=graphic.device), target_channel
    ] > 0.5
    nearest = distance2.masked_fill(~target_mask, float("inf")).argmin(dim=1)
    nearest_relative = relative[torch.arange(batch, device=vector.device), nearest]
    target_present = target_mask.any(dim=1)
    nearest_relative = torch.where(
        target_present[:, None], nearest_relative, torch.zeros_like(nearest_relative)
    )
    nearest_distance = torch.linalg.vector_norm(nearest_relative, dim=-1, keepdim=True)

    planner_one_hot = F.one_hot(planner_action_index, num_classes=N_ACTIONS).to(vector.dtype)
    role_one_hot = F.one_hot(guard.to(torch.int64), num_classes=2).to(vector.dtype)
    directions = ACTION_DIRECTIONS.to(device=vector.device, dtype=vector.dtype)
    planner_direction = directions[planner_action_index]
    planner_direction = planner_direction / torch.linalg.vector_norm(
        planner_direction, dim=-1, keepdim=True
    ).clamp_min(1.0)
    return torch.cat(
        (
            planner_one_hot,
            role_one_hot,
            planner_path_valid[:, None].to(vector.dtype),
            planner_direction,
            nearest_relative,
            nearest_distance,
            target_present[:, None].to(vector.dtype),
        ),
        dim=-1,
    )


def initialize_planner_conditioned_residual(
    model: MAPPOActorCritic, *, fallback_logit_bias: float = 6.5
) -> None:
    if model.actor_model.model_config["action_head_version"] != "categorical9":
        raise ValueError("planner residual training requires categorical9 actions")
    if model.actor_model.model_config["recurrent_version"] != "none":
        raise ValueError("planner-conditioned residual v2 does not support recurrent actors")
    model.planner_residual_head = PlannerConditionedResidualHead(
        int(model.actor_model.model_config["hidden_dim"]),
        fallback_logit_bias=fallback_logit_bias,
    ).to(next(model.parameters()).device)


def mask_residual_override_logits(
    logits: torch.Tensor, override_eligible: torch.Tensor | None
) -> torch.Tensor:
    """Make fallback the only possible action for non-selected exploration rows."""

    if override_eligible is None:
        return logits
    if override_eligible.shape != logits.shape[:1] or override_eligible.dtype != torch.bool:
        raise ValueError("override_eligible must be bool with shape (B,)")
    masked = logits.clone()
    masked[~override_eligible, 1:] = torch.finfo(masked.dtype).min
    return masked


def planner_relative_residual_action(
    residual_index: torch.Tensor, planner_action_index: torch.Tensor
) -> torch.Tensor:
    """Convert v5 relative corrections to normalized environment actions.

    Indices are keep, left/right 45, left/right 90, stop, reverse, and
    left/right 135 degrees. If the planner itself stops, non-zero directions
    fall back to the ordinary absolute categorical directions.
    """

    if residual_index.shape != planner_action_index.shape:
        raise ValueError("residual and planner action indices must have the same shape")
    if residual_index.dtype != torch.int64 or planner_action_index.dtype != torch.int64:
        raise TypeError("residual and planner action indices must be int64")
    if not bool(torch.all((residual_index >= 0) & (residual_index < N_ACTIONS))):
        raise ValueError("residual action index is outside [0,8]")
    if not bool(torch.all((planner_action_index >= 0) & (planner_action_index < N_ACTIONS))):
        raise ValueError("planner action index is outside [0,8]")
    offsets = torch.tensor(
        (0, 1, -1, 2, -2, 0, 4, 3, -3),
        dtype=torch.int64,
        device=residual_index.device,
    )
    rotated = ((planner_action_index - 1 + offsets[residual_index]) % 8) + 1
    absolute_when_stopped = residual_index
    action_index = torch.where(
        residual_index == 0,
        planner_action_index,
        torch.where(
            residual_index == 5,
            torch.zeros_like(residual_index),
            torch.where(planner_action_index == 0, absolute_when_stopped, rotated),
        ),
    )
    directions = ACTION_DIRECTIONS.to(
        device=residual_index.device, dtype=torch.float32
    )[action_index]
    return directions / torch.linalg.vector_norm(
        directions, dim=-1, keepdim=True
    ).clamp_min(1.0)


def team_alive_mask(active_agents: Sequence[str], learning_team: int) -> torch.Tensor:
    active = set(active_agents)
    return torch.tensor(
        [agent in active for agent in team_agents(learning_team)], dtype=torch.bool
    )


@dataclass(frozen=True)
class MAPPORolloutBatch:
    vector: torch.Tensor
    graphic: torch.Tensor
    slot_id: torch.Tensor
    central_vector: torch.Tensor
    central_graphic: torch.Tensor
    team_mask: torch.Tensor
    action_index: torch.Tensor
    old_log_prob: torch.Tensor
    old_value: torch.Tensor
    advantage: torch.Tensor
    return_: torch.Tensor
    teacher_action_index: torch.Tensor | None = None
    planner_action_index: torch.Tensor | None = None
    planner_path_valid: torch.Tensor | None = None
    residual_override_eligible: torch.Tensor | None = None

    def __len__(self) -> int:
        return int(self.vector.shape[0])

    def to(self, device: str | torch.device) -> "MAPPORolloutBatch":
        fields: dict[str, torch.Tensor | None] = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            fields[name] = value.to(device) if isinstance(value, torch.Tensor) else value
        return MAPPORolloutBatch(**fields)


class JointRolloutBuffer:
    """Time-major team buffer whose critic target survives death and respawn."""

    def __init__(self, n_agents: int = N_TEAM_AGENTS) -> None:
        self.n_agents = n_agents
        self.rows: list[dict[str, torch.Tensor]] = []

    def __len__(self) -> int:
        return len(self.rows)

    def add(self, **fields: torch.Tensor) -> None:
        required = {
            "vector", "graphic", "slot_id", "central_vector", "central_graphic",
            "team_mask", "action_index", "log_prob", "value", "reward",
            "next_value", "terminated", "truncated",
        }
        optional = {
            "teacher_action_index",
            "planner_action_index",
            "planner_path_valid",
            "residual_override_eligible",
        }
        if not required <= set(fields) or set(fields) - required - optional:
            raise ValueError(f"joint rollout fields differ: {sorted(required ^ set(fields))}")
        if self.rows and set(fields) != set(self.rows[0]):
            raise ValueError("teacher labels must be present in every joint transition or none")
        for name in (
            "vector", "graphic", "slot_id", "team_mask", "action_index", "log_prob",
            "reward", "teacher_action_index", "planner_action_index",
            "planner_path_valid", "residual_override_eligible",
        ):
            if name not in fields:
                continue
            if fields[name].shape[0] != self.n_agents:
                raise ValueError(f"{name} must have leading team dimension")
        if fields["team_mask"].dtype != torch.bool:
            raise TypeError("team_mask must be boolean")
        if "teacher_action_index" in fields and fields["teacher_action_index"].dtype != torch.int64:
            raise TypeError("teacher_action_index must be int64")
        if "planner_action_index" in fields and fields["planner_action_index"].dtype != torch.int64:
            raise TypeError("planner_action_index must be int64")
        for name in ("planner_path_valid", "residual_override_eligible"):
            if name in fields and fields[name].dtype != torch.bool:
                raise TypeError(f"{name} must be boolean")
        for name in ("value", "next_value", "terminated", "truncated"):
            if fields[name].numel() != 1:
                raise ValueError(f"{name} must be one team-level scalar")
        self.rows.append({name: value.detach().cpu().clone() for name, value in fields.items()})

    def as_batch(self, *, gamma: float, gae_lambda: float) -> MAPPORolloutBatch:
        if not self.rows:
            raise ValueError("cannot finalize an empty joint rollout")
        stacked = {name: torch.stack([row[name] for row in self.rows]) for name in self.rows[0]}
        alive_count = stacked["team_mask"].sum(dim=1).clamp_min(1)
        team_reward = (stacked["reward"] * stacked["team_mask"]).sum(dim=1) / alive_count
        value = stacked["value"].reshape(-1, 1)
        advantage, returns = generalized_advantage_estimate(
            team_reward.reshape(-1, 1), value, stacked["next_value"].reshape(-1, 1),
            stacked["terminated"].reshape(-1, 1).bool(),
            stacked["truncated"].reshape(-1, 1).bool(), gamma=gamma, gae_lambda=gae_lambda,
        )
        repeat = lambda tensor: tensor.repeat_interleave(self.n_agents, dim=0)
        flat_adv = repeat(advantage).reshape(-1)
        valid = stacked["team_mask"].reshape(-1)
        if int(valid.sum()) > 1:
            selected = flat_adv[valid]
            flat_adv = flat_adv.clone()
            flat_adv[valid] = (selected - selected.mean()) / (selected.std(unbiased=False) + 1e-8)
        return MAPPORolloutBatch(
            vector=stacked["vector"].flatten(0, 1), graphic=stacked["graphic"].flatten(0, 1),
            slot_id=stacked["slot_id"].flatten(),
            central_vector=repeat(stacked["central_vector"]).reshape(-1, CENTRAL_VECTOR_SIZE),
            central_graphic=repeat(stacked["central_graphic"]).flatten(0, 1),
            team_mask=valid, action_index=stacked["action_index"].flatten(),
            old_log_prob=stacked["log_prob"].flatten(), old_value=repeat(value).reshape(-1),
            advantage=flat_adv, return_=repeat(returns).reshape(-1),
            teacher_action_index=(
                stacked["teacher_action_index"].flatten()
                if "teacher_action_index" in stacked
                else None
            ),
            planner_action_index=(
                stacked["planner_action_index"].flatten()
                if "planner_action_index" in stacked
                else None
            ),
            planner_path_valid=(
                stacked["planner_path_valid"].flatten()
                if "planner_path_valid" in stacked
                else None
            ),
            residual_override_eligible=(
                stacked["residual_override_eligible"].flatten()
                if "residual_override_eligible" in stacked
                else None
            ),
        )


class MAPPOParallelRolloutCollector:
    """Collect joint team transitions while keeping actor inputs decentralized."""

    def __init__(self, env: Any, model: MAPPOActorCritic, opponent: Any, *,
                 learning_team: int, device: str | torch.device = "cpu",
                 reward_transform: Any | None = None,
                 episode_seed_provider: Callable[[], int | None] | None = None,
                 teacher: Any | None = None,
                 residual_base: Any | None = None,
                 teacher_forcing_probability: float = 0.0,
                 teacher_seed: int = 0,
                 max_residual_overrides_per_step: int | None = None) -> None:
        if not 0.0 <= teacher_forcing_probability <= 1.0:
            raise ValueError("teacher_forcing_probability must be in [0,1]")
        if teacher is not None and residual_base is not None:
            raise ValueError("teacher and residual base are mutually exclusive")
        if teacher_forcing_probability > 0.0 and teacher is None and residual_base is None:
            raise ValueError("teacher forcing requires a teacher or residual base policy")
        if max_residual_overrides_per_step not in (None, 1):
            raise ValueError("max_residual_overrides_per_step must be None or 1")
        self.env=env; self.model=model.to(device); self.opponent=opponent
        self.learning_team=learning_team; self.device=torch.device(device)
        self.controlled_agents=team_agents(learning_team); self.opponent_agents=team_agents(1-learning_team)
        self.reward_transform=reward_transform
        self.episode_seed_provider=episode_seed_provider
        self.teacher=teacher; self.teacher_forcing_probability=teacher_forcing_probability
        self.residual_base=residual_base
        self.max_residual_overrides_per_step=max_residual_overrides_per_step
        self._teacher_rng=np.random.default_rng(teacher_seed)
        self.teacher_failures=0; self.teacher_forced_steps=0; self.teacher_labeled_steps=0
        self.builder=CentralizedStateBuilder(); self._observations=None
        self.episodes_completed=0; self.terminal_episodes=0; self.truncated_episodes=0
        self.wins=0; self.draws=0; self.losses=0; self.environment_steps=0
        self.residual_fallback_actions=0; self.residual_override_actions=0

    @staticmethod
    def _reset_component(component: Any) -> None:
        reset=getattr(component,"reset",None)
        if callable(reset): reset()

    def reset(self, *, seed: int | None=None) -> None:
        if seed is None and self.episode_seed_provider is not None:
            seed = self.episode_seed_provider()
        self._observations,_=self.env.reset(seed=seed)
        if set(self._observations) != set(canonical_agents()):
            raise ValueError("MAPPO collector requires all ten observations")
        self._reset_component(self.opponent)
        if self.teacher is not None: self._reset_component(self.teacher)
        if self.residual_base is not None: self._reset_component(self.residual_base)
        if self.reward_transform is not None: self._reset_component(self.reward_transform)

    def collect(self, time_steps: int, *, seed: int | None=None) -> JointRolloutBuffer:
        if time_steps <= 0: raise ValueError("time_steps must be positive")
        if self._observations is None: self.reset(seed=seed)
        elif seed is not None: raise ValueError("seed only initializes a collector")
        buffer=JointRolloutBuffer(); self.model.eval()
        for _ in range(time_steps):
            observations=self._observations
            local=team_model_input(observations,self.learning_team,device=self.device)
            central=self.builder.build(observations,learning_team=self.learning_team,device=self.device)
            teacher_action_index=None
            planner_action_index=None
            planner_path_valid=None
            override_eligible=None
            planner_actions=None
            if self.residual_base is not None:
                from .behavior_cloning import action_vector_to_index
                from .navigation import PathNotFound

                planner_path_valid = torch.ones(
                    len(self.controlled_agents), dtype=torch.bool, device=self.device
                )
                try:
                    planner_actions = self.residual_base.act(
                        observations, self.controlled_agents
                    )
                except PathNotFound:
                    self.teacher_failures += 1
                    self._reset_component(self.residual_base)
                    planner_actions = {
                        agent: np.zeros(2, dtype=np.float32)
                        for agent in self.controlled_agents
                    }
                    planner_path_valid.zero_()
                    teacher_action_index = torch.full(
                        (len(self.controlled_agents),),
                        -100,
                        dtype=torch.int64,
                        device=self.device,
                    )
                else:
                    if set(planner_actions) != set(self.controlled_agents):
                        raise ValueError(
                            "residual base policy must return exactly its five agent actions"
                        )
                    teacher_action_index = torch.zeros(
                        len(self.controlled_agents), dtype=torch.int64, device=self.device
                    )
                    self.teacher_labeled_steps += 1
                planner_action_index = torch.tensor(
                    [action_vector_to_index(planner_actions[a]) for a in self.controlled_agents],
                    dtype=torch.int64,
                    device=self.device,
                )
                override_eligible = torch.ones(
                    len(self.controlled_agents), dtype=torch.bool, device=self.device
                )
                if self.max_residual_overrides_per_step == 1:
                    override_eligible.zero_()
                    selected_row = int(
                        self._teacher_rng.integers(0, len(self.controlled_agents))
                    )
                    override_eligible[selected_row] = True
            with torch.inference_mode():
                logits=self.model.actor_logits(
                    local.vector,
                    local.graphic,
                    local.slot_id,
                    planner_action_index,
                    planner_path_valid,
                )
                logits=mask_residual_override_logits(logits,override_eligible)
                selection=select_categorical_action(logits)
                value=self.model.centralized_critic(central.vector,central.graphic)
            selected_action = selection.action
            if (
                self.model.planner_residual_head is not None
                and planner_action_index is not None
            ):
                selected_action = planner_relative_residual_action(
                    selection.index, planner_action_index
                )
            learning_actions={agent:selected_action[row].cpu().numpy().astype(np.float32,copy=False)
                              for row,agent in enumerate(local.agent_names)}
            if self.residual_base is not None:
                assert planner_actions is not None
                force_planner = (
                    self._teacher_rng.random() < self.teacher_forcing_probability
                )
                fallback = selection.index == 0
                if force_planner:
                    fallback = torch.ones_like(fallback)
                    forced_index = torch.zeros_like(selection.index)
                    selection = evaluate_categorical_action(logits, forced_index)
                    self.teacher_forced_steps += 1
                learning_actions = {
                    agent: (
                        planner_actions[agent]
                        if bool(fallback[row])
                        else learning_actions[agent]
                    )
                    for row, agent in enumerate(local.agent_names)
                }
                fallback_count = int(fallback.sum())
                self.residual_fallback_actions += fallback_count
                self.residual_override_actions += len(fallback) - fallback_count
            elif self.teacher is not None:
                from .behavior_cloning import action_vector_to_index
                from .navigation import PathNotFound

                try:
                    teacher_actions=self.teacher.act(observations,self.controlled_agents)
                except PathNotFound:
                    self.teacher_failures += 1
                    self._reset_component(self.teacher)
                    teacher_action_index=torch.full(
                        (len(self.controlled_agents),),-100,dtype=torch.int64,device=self.device
                    )
                else:
                    if set(teacher_actions) != set(self.controlled_agents):
                        raise ValueError("teacher policy must return exactly its five agent actions")
                    teacher_action_index=torch.tensor(
                        [action_vector_to_index(teacher_actions[a]) for a in self.controlled_agents],
                        dtype=torch.int64,device=self.device,
                    )
                    self.teacher_labeled_steps += 1
                    if self._teacher_rng.random() < self.teacher_forcing_probability:
                        learning_actions=teacher_actions
                        self.teacher_forced_steps += 1
            opponent_actions=self.opponent.act(observations,self.opponent_agents)
            if set(opponent_actions) != set(self.opponent_agents):
                raise ValueError("opponent must return exactly its team actions")
            next_obs,rewards,terminations,truncations,infos=self.env.step({**learning_actions,**opponent_actions})
            self.environment_steps += 1
            terminated=all(bool(terminations[a]) for a in self.controlled_agents)
            truncated=all(bool(truncations[a]) for a in self.controlled_agents)
            if any(bool(terminations[a]) for a in self.controlled_agents) != terminated:
                raise RuntimeError("partial team termination is unsupported")
            if any(bool(truncations[a]) for a in self.controlled_agents) != truncated:
                raise RuntimeError("partial team truncation is unsupported")
            if terminated and truncated: raise RuntimeError("transition cannot terminate and truncate")
            next_value=torch.zeros(1,device=self.device)
            if not terminated:
                next_central=self.builder.build(next_obs,learning_team=self.learning_team,device=self.device)
                with torch.inference_mode(): next_value=self.model.centralized_critic(next_central.vector,next_central.graphic)
            transformed=({a:float(rewards[a]) for a in self.controlled_agents}
                         if self.reward_transform is None else
                         self.reward_transform(rewards,terminations,truncations,infos,self.controlled_agents))
            mask=torch.tensor([bool(infos[a].get("alive",True)) for a in self.controlled_agents],dtype=torch.bool)
            fields={
                "vector":local.vector,"graphic":local.graphic,"slot_id":local.slot_id,
                "central_vector":central.vector,"central_graphic":central.graphic,"team_mask":mask,
                "action_index":selection.index,"log_prob":selection.log_prob,"value":value,
                "reward":torch.tensor([float(transformed[a]) for a in self.controlled_agents]),
                "next_value":next_value,"terminated":torch.tensor([terminated]),
                "truncated":torch.tensor([truncated]),
            }
            if teacher_action_index is not None:
                fields["teacher_action_index"]=teacher_action_index
            if planner_action_index is not None:
                fields["planner_action_index"]=planner_action_index
                assert planner_path_valid is not None
                fields["planner_path_valid"]=planner_path_valid
            if override_eligible is not None:
                fields["residual_override_eligible"]=override_eligible
            buffer.add(**fields)
            if terminated or truncated:
                self.episodes_completed += 1
                if terminated:
                    self.terminal_episodes += 1
                    reference_info = infos[self.controlled_agents[0]]
                    if "winner" not in reference_info:
                        raise KeyError("terminal info is missing winner")
                    winner = int(reference_info["winner"])
                    if winner == -1:
                        self.draws += 1
                    elif winner == self.learning_team:
                        self.wins += 1
                    elif winner == 1 - self.learning_team:
                        self.losses += 1
                    else:
                        raise ValueError(f"terminal winner must be -1, 0, or 1; got {winner}")
                else:
                    self.truncated_episodes += 1
                self.reset()
            else: self._observations=next_obs
        return buffer


def mappo_update(
    model: MAPPOActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: MAPPORolloutBatch,
    *,
    config: PPOConfig | None = None,
    override_penalty_coef: float = 0.0,
) -> PPODiagnostics:
    config = config or PPOConfig()
    config.validate()
    if not math.isfinite(override_penalty_coef) or override_penalty_coef < 0.0:
        raise ValueError("override_penalty_coef must be finite and non-negative")
    device = next(model.parameters()).device
    batch = batch.to(device)
    valid_indices = torch.nonzero(batch.team_mask, as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        raise ValueError("MAPPO batch has no live-agent actions")
    totals = dict(policy_loss=0.0, value_loss=0.0, entropy=0.0, approximate_kl=0.0,
                  clip_fraction=0.0, gradient_norm=0.0)
    count = 0
    early = False
    epochs = 0
    for epoch in range(config.update_epochs):
        order = valid_indices[torch.randperm(len(valid_indices), device=device)]
        epoch_kls: list[float] = []
        for start in range(0, len(order), config.minibatch_size):
            ix = order[start:start + config.minibatch_size]
            output = model(
                batch.vector[ix],
                batch.graphic[ix],
                batch.slot_id[ix],
                batch.central_vector[ix],
                batch.central_graphic[ix],
                (
                    batch.planner_action_index[ix]
                    if batch.planner_action_index is not None
                    else None
                ),
                (
                    batch.planner_path_valid[ix]
                    if batch.planner_path_valid is not None
                    else None
                ),
            )
            action_logits = mask_residual_override_logits(
                output.action_logits,
                (
                    batch.residual_override_eligible[ix]
                    if batch.residual_override_eligible is not None
                    else None
                ),
            )
            evaluated = evaluate_categorical_action(action_logits, batch.action_index[ix])
            log_ratio = evaluated.log_prob - batch.old_log_prob[ix]
            ratio = log_ratio.exp()
            policy_loss = torch.maximum(
                -batch.advantage[ix] * ratio,
                -batch.advantage[ix] * ratio.clamp(1-config.clip_coef, 1+config.clip_coef),
            ).mean()
            value_loss = 0.5 * (output.value - batch.return_[ix]).square().mean()
            override_probability = 1.0 - torch.softmax(action_logits, dim=-1)[:, 0]
            loss = (
                policy_loss
                + config.value_loss_coef * value_loss
                - config.entropy_coef * evaluated.entropy.mean()
                + override_penalty_coef * override_probability.mean()
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                metrics = {
                    "policy_loss": policy_loss, "value_loss": value_loss,
                    "entropy": evaluated.entropy.mean(), "approximate_kl": approx_kl,
                    "clip_fraction": ((ratio - 1).abs() > config.clip_coef).float().mean(),
                    "gradient_norm": grad,
                }
                for name, value in metrics.items(): totals[name] += float(value)
                epoch_kls.append(float(approx_kl)); count += 1
        epochs = epoch + 1
        if config.target_kl is not None and epoch_kls and np.mean(epoch_kls) > config.target_kl:
            early = True; break
    with torch.no_grad():
        prediction = model.centralized_critic(batch.central_vector[valid_indices], batch.central_graphic[valid_indices])
        ev = explained_variance(prediction, batch.return_[valid_indices])
    return PPODiagnostics(
        **{name: value/count for name, value in totals.items()},
        explained_variance=None if not torch.isfinite(ev) else float(ev),
        epochs_completed=epochs, minibatches=count, samples=int(valid_indices.numel()), early_stopped=early,
    )


@dataclass(frozen=True)
class AblationArm:
    algorithm: str
    seeds: tuple[int, ...]
    opponent_id: str
    environment_steps: int
    win_rate: float
    mean_score_diff: float


def compare_ippo_mappo(ippo: AblationArm, mappo: AblationArm) -> dict[str, Any]:
    if ippo.algorithm != "ippo" or mappo.algorithm != "mappo":
        raise ValueError("ablation requires ippo and mappo arms")
    if (ippo.seeds, ippo.opponent_id, ippo.environment_steps) != (
        mappo.seeds, mappo.opponent_id, mappo.environment_steps
    ):
        raise ValueError("IPPO/MAPPO ablation must share seeds, opponent, and step budget")
    return {
        "contract_passed": True,
        "win_rate_delta": mappo.win_rate - ippo.win_rate,
        "score_diff_delta": mappo.mean_score_diff - ippo.mean_score_diff,
    }
