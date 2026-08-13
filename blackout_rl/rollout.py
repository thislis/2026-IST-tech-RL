"""PettingZoo parallel rollout collection and episodic GAE buffers for IPPO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .action_distribution import select_categorical_action
from .batching import N_TEAM_AGENTS, team_agents
from .ippo_model import IPPOActorCritic
from .model_contract import team_model_input
from .navigation import PathNotFound


class ParallelEnv(Protocol):
    """Subset of the PettingZoo parallel API required by the collector."""

    agents: list[str]

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]: ...

    def step(
        self, actions: Mapping[str, np.ndarray]
    ) -> tuple[dict, dict, dict, dict, dict]: ...


class OpponentPolicy(Protocol):
    def act(
        self,
        observations: Mapping[str, Mapping[str, np.ndarray]],
        agents: Sequence[str],
    ) -> dict[str, np.ndarray]: ...


RewardTransform = Callable[
    [
        Mapping[str, float],
        Mapping[str, bool],
        Mapping[str, bool],
        Mapping[str, Mapping[str, Any]],
        tuple[str, ...],
    ],
    Mapping[str, float],
]


def _reset_if_supported(component: object) -> None:
    reset = getattr(component, "reset", None)
    if callable(reset):
        reset()


@dataclass(frozen=True)
class RolloutBatch:
    """Flattened PPO batch with old-policy statistics and episode masks."""

    vector: torch.Tensor
    graphic: torch.Tensor
    slot_id: torch.Tensor
    action_index: torch.Tensor
    old_log_prob: torch.Tensor
    old_value: torch.Tensor
    reward: torch.Tensor
    next_value: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    episode_start: torch.Tensor
    advantage: torch.Tensor
    return_: torch.Tensor
    time_steps: int
    n_agents: int
    teacher_action_index: torch.Tensor | None = None

    def __len__(self) -> int:
        return int(self.vector.shape[0])

    def to(self, device: str | torch.device) -> RolloutBatch:
        values: dict[str, Any] = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            values[name] = value.to(device) if isinstance(value, torch.Tensor) else value
        return RolloutBatch(**values)


def decode_rollout_graphic(graphic: torch.Tensor) -> torch.Tensor:
    """Expand compact semantic IDs, while accepting legacy one-hot batches."""

    if graphic.ndim == 4:
        return graphic.to(torch.float32)
    if graphic.ndim != 3 or graphic.dtype != torch.uint8:
        raise ValueError("rollout graphic must be uint8 (B,H,W) IDs or (B,C,H,W)")
    return F.one_hot(graphic.to(torch.int64), num_classes=11).permute(0, 3, 1, 2).to(
        torch.float32
    )


def concatenate_rollout_batches(batches: Sequence[RolloutBatch]) -> RolloutBatch:
    """Concatenate same-schema batches so both physical sides share every update."""

    if not batches:
        raise ValueError("at least one rollout batch is required")
    n_agents = batches[0].n_agents
    if any(batch.n_agents != n_agents for batch in batches):
        raise ValueError("rollout batches have different agent counts")
    optional_presence = [batch.teacher_action_index is not None for batch in batches]
    if any(optional_presence) and not all(optional_presence):
        raise ValueError("teacher labels must be present in every batch or none")
    tensor_names = (
        "vector",
        "graphic",
        "slot_id",
        "action_index",
        "old_log_prob",
        "old_value",
        "reward",
        "next_value",
        "terminated",
        "truncated",
        "episode_start",
        "advantage",
        "return_",
    )
    values = {
        name: torch.cat([getattr(batch, name) for batch in batches], dim=0)
        for name in tensor_names
    }
    teacher = (
        torch.cat([batch.teacher_action_index for batch in batches], dim=0)
        if all(optional_presence)
        else None
    )
    return RolloutBatch(
        **values,
        time_steps=sum(batch.time_steps for batch in batches),
        n_agents=n_agents,
        teacher_action_index=teacher,
    )


def generalized_advantage_estimate(
    reward: torch.Tensor,
    value: torch.Tensor,
    next_value: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE without leaking advantages across episode boundaries.

    A truncation bootstraps from its final observation but stops the recursive
    GAE chain. A true termination neither bootstraps nor continues the chain.
    Inputs use shape ``(T,N)``.
    """
    tensors = (reward, value, next_value, terminated, truncated)
    if reward.ndim != 2 or any(tensor.shape != reward.shape for tensor in tensors[1:]):
        raise ValueError("GAE inputs must all have the same (T,N) shape")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0,1]")
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be in [0,1]")
    if terminated.dtype != torch.bool or truncated.dtype != torch.bool:
        raise TypeError("terminated and truncated must be boolean")
    if bool(torch.any(terminated & truncated)):
        raise ValueError("a transition cannot be both terminated and truncated")

    bootstrap_mask = (~terminated).to(value.dtype)
    continuation_mask = (~(terminated | truncated)).to(value.dtype)
    delta = reward + gamma * next_value * bootstrap_mask - value
    advantage = torch.zeros_like(delta)
    running = torch.zeros_like(delta[0])
    for step in range(delta.shape[0] - 1, -1, -1):
        running = delta[step] + gamma * gae_lambda * continuation_mask[step] * running
        advantage[step] = running
    return advantage, advantage + value


class EpisodicRolloutBuffer:
    """Time-major rollout storage retaining termination and truncation semantics."""

    def __init__(self, *, n_agents: int = N_TEAM_AGENTS) -> None:
        if n_agents <= 0:
            raise ValueError("n_agents must be positive")
        self.n_agents = n_agents
        self._rows: list[dict[str, torch.Tensor]] = []

    def __len__(self) -> int:
        return len(self._rows)

    def add(
        self,
        *,
        vector: torch.Tensor,
        graphic: torch.Tensor,
        slot_id: torch.Tensor,
        action_index: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: torch.Tensor,
        next_value: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        episode_start: torch.Tensor,
        teacher_action_index: torch.Tensor | None = None,
    ) -> None:
        fields = {
            "vector": vector,
            "graphic": graphic,
            "slot_id": slot_id,
            "action_index": action_index,
            "old_log_prob": log_prob,
            "old_value": value,
            "reward": reward,
            "next_value": next_value,
            "terminated": terminated,
            "truncated": truncated,
            "episode_start": episode_start,
        }
        if teacher_action_index is not None:
            fields["teacher_action_index"] = teacher_action_index
        for name, tensor in fields.items():
            if tensor.ndim == 0 or tensor.shape[0] != self.n_agents:
                raise ValueError(f"{name} must have leading shape ({self.n_agents},)")
        if slot_id.dtype != torch.int64 or action_index.dtype != torch.int64:
            raise TypeError("slot_id and action_index must be int64")
        if teacher_action_index is not None and teacher_action_index.dtype != torch.int64:
            raise TypeError("teacher_action_index must be int64")
        if any(fields[name].dtype != torch.bool for name in ("terminated", "truncated", "episode_start")):
            raise TypeError("termination, truncation, and episode_start masks must be boolean")
        if bool(torch.any(terminated & truncated)):
            raise ValueError("a transition cannot be both terminated and truncated")
        self._rows.append({name: tensor.detach().cpu().clone() for name, tensor in fields.items()})

    def as_batch(
        self,
        *,
        gamma: float,
        gae_lambda: float,
        normalize_advantage: bool = True,
        epsilon: float = 1e-8,
    ) -> RolloutBatch:
        if not self._rows:
            raise ValueError("cannot finalize an empty rollout buffer")
        stacked = {
            name: torch.stack([row[name] for row in self._rows])
            for name in self._rows[0]
        }
        advantage, return_ = generalized_advantage_estimate(
            stacked["reward"],
            stacked["old_value"],
            stacked["next_value"],
            stacked["terminated"],
            stacked["truncated"],
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        flat_advantage = advantage.flatten()
        if normalize_advantage and flat_advantage.numel() > 1:
            flat_advantage = (flat_advantage - flat_advantage.mean()) / (
                flat_advantage.std(unbiased=False) + epsilon
            )
        time_steps = len(self._rows)
        return RolloutBatch(
            vector=stacked["vector"].flatten(0, 1),
            graphic=stacked["graphic"].flatten(0, 1),
            slot_id=stacked["slot_id"].flatten(),
            action_index=stacked["action_index"].flatten(),
            old_log_prob=stacked["old_log_prob"].flatten(),
            old_value=stacked["old_value"].flatten(),
            reward=stacked["reward"].flatten(),
            next_value=stacked["next_value"].flatten(),
            terminated=stacked["terminated"].flatten(),
            truncated=stacked["truncated"].flatten(),
            episode_start=stacked["episode_start"].flatten(),
            advantage=flat_advantage,
            return_=return_.flatten(),
            time_steps=time_steps,
            n_agents=self.n_agents,
            teacher_action_index=(
                stacked["teacher_action_index"].flatten()
                if "teacher_action_index" in stacked
                else None
            ),
        )


def _unity_reward_transform(
    rewards: Mapping[str, float],
    _terminations: Mapping[str, bool],
    _truncations: Mapping[str, bool],
    _infos: Mapping[str, Mapping[str, Any]],
    controlled_agents: tuple[str, ...],
) -> Mapping[str, float]:
    return {agent: float(rewards[agent]) for agent in controlled_agents}


class ParallelRolloutCollector:
    """Collect stochastic IPPO transitions from one PettingZoo parallel env."""

    def __init__(
        self,
        env: ParallelEnv,
        model: IPPOActorCritic,
        opponent: OpponentPolicy,
        *,
        learning_team: int,
        device: str | torch.device = "cpu",
        reward_transform: RewardTransform | None = None,
        teacher: OpponentPolicy | None = None,
        teacher_forcing: bool = False,
        teacher_only_collection: bool = False,
    ) -> None:
        self.env = env
        self.model = model.to(device)
        self.opponent = opponent
        self.learning_team = learning_team
        self.device = torch.device(device)
        self.controlled_agents = team_agents(learning_team)
        self.opponent_agents = team_agents(1 - learning_team)
        self.reward_transform = reward_transform or _unity_reward_transform
        self.teacher = teacher
        if teacher_forcing and teacher is None:
            raise ValueError("teacher forcing requires a teacher policy")
        if teacher_only_collection and not teacher_forcing:
            raise ValueError("teacher-only collection requires teacher forcing")
        self.teacher_forcing = teacher_forcing
        self.teacher_only_collection = teacher_only_collection
        self._observations: dict[str, dict[str, np.ndarray]] | None = None
        self._episode_start = True
        self.episodes_completed = 0
        self.teacher_failures = 0

    def reset(self, *, seed: int | None = None) -> None:
        observations, _ = self.env.reset(seed=seed)
        self._require_live_agents(observations)
        _reset_if_supported(self.opponent)
        _reset_if_supported(self.reward_transform)
        if self.teacher is not None:
            _reset_if_supported(self.teacher)
        self._observations = observations
        self._episode_start = True

    def collect(self, time_steps: int, *, seed: int | None = None) -> EpisodicRolloutBuffer:
        if time_steps <= 0:
            raise ValueError("time_steps must be positive")
        if self._observations is None:
            self.reset(seed=seed)
        elif seed is not None:
            raise ValueError("seed can only be supplied when initializing the collector")

        buffer = EpisodicRolloutBuffer(n_agents=len(self.controlled_agents))
        self.model.eval()
        for _ in range(time_steps):
            assert self._observations is not None
            observations = self._observations
            input_device = "cpu" if self.teacher_only_collection else self.device
            model_input = team_model_input(
                observations, self.learning_team, device=input_device
            )
            step_device = model_input.vector.device
            if self.teacher_only_collection:
                # During pure teacher-forced replay collection, PPO statistics and
                # current-policy actions are discarded. Avoid two neural-network
                # inferences per Unity step and populate the unused fields with
                # neutral values instead.
                action_index = torch.zeros(
                    len(self.controlled_agents), dtype=torch.int64, device=step_device
                )
                log_prob = torch.zeros(
                    len(self.controlled_agents), dtype=torch.float32, device=step_device
                )
                value = torch.zeros_like(log_prob)
                learning_actions = {
                    agent: np.zeros(2, dtype=np.float32)
                    for agent in model_input.agent_names
                }
            else:
                with torch.inference_mode():
                    output = self.model(
                        model_input.vector, model_input.graphic, model_input.slot_id
                    )
                    selection = select_categorical_action(output.action_logits)
                action_index = selection.index
                log_prob = selection.log_prob
                value = output.value
                learning_actions = {
                    agent: selection.action[row].cpu().numpy().astype(
                        np.float32, copy=False
                    )
                    for row, agent in enumerate(model_input.agent_names)
                }
            teacher_action_index = None
            if self.teacher is not None:
                from .behavior_cloning import action_vector_to_index

                try:
                    teacher_actions = self.teacher.act(observations, self.controlled_agents)
                except PathNotFound:
                    # A current-policy learner can visit cells outside the scripted
                    # controller's reachable-state assumptions. Such rows are not
                    # valid demonstrations and use the standard ignore label.
                    self.teacher_failures += 1
                    _reset_if_supported(self.teacher)
                    teacher_action_index = torch.full(
                        (len(self.controlled_agents),),
                        -100,
                        dtype=torch.int64,
                        device=step_device,
                    )
                else:
                    if set(teacher_actions) != set(self.controlled_agents):
                        raise ValueError("teacher policy must return exactly its five agent actions")
                    teacher_action_index = torch.tensor(
                        [
                            action_vector_to_index(teacher_actions[agent])
                            for agent in self.controlled_agents
                        ],
                        dtype=torch.int64,
                        device=step_device,
                    )
                    if self.teacher_forcing:
                        learning_actions = teacher_actions
            opponent_actions = self.opponent.act(observations, self.opponent_agents)
            if set(opponent_actions) != set(self.opponent_agents):
                raise ValueError("opponent policy must return exactly its five agent actions")
            actions = {**learning_actions, **opponent_actions}
            next_observations, rewards, terminations, truncations, infos = self.env.step(actions)

            terminated = torch.tensor(
                [bool(terminations[agent]) for agent in self.controlled_agents],
                dtype=torch.bool,
                device=step_device,
            )
            truncated = torch.tensor(
                [bool(truncations[agent]) for agent in self.controlled_agents],
                dtype=torch.bool,
                device=step_device,
            )
            boundary = terminated | truncated
            if bool(torch.any(boundary)) != bool(torch.all(boundary)):
                raise RuntimeError("partial team episode boundaries are not supported")
            if bool(torch.any(terminated & truncated)):
                raise RuntimeError("environment marked a transition terminated and truncated")

            next_value = torch.zeros_like(value)
            if not self.teacher_only_collection and not bool(torch.all(terminated)):
                # Both ordinary transitions and time-limit truncations bootstrap
                # from the observation returned by the step.
                next_input = team_model_input(
                    next_observations, self.learning_team, device=self.device
                )
                with torch.inference_mode():
                    next_value = self.model(
                        next_input.vector, next_input.graphic, next_input.slot_id
                    ).value

            transformed_rewards = self.reward_transform(
                rewards, terminations, truncations, infos, self.controlled_agents
            )
            if set(transformed_rewards) != set(self.controlled_agents):
                raise ValueError("reward transform must return exactly the controlled agents")
            reward = torch.tensor(
                [float(transformed_rewards[agent]) for agent in self.controlled_agents],
                dtype=torch.float32,
                device=step_device,
            )
            buffer.add(
                vector=model_input.vector,
                graphic=torch.argmax(model_input.graphic, dim=1).to(torch.uint8),
                slot_id=model_input.slot_id,
                action_index=action_index,
                log_prob=log_prob,
                value=value,
                reward=reward,
                next_value=next_value,
                terminated=terminated,
                truncated=truncated,
                episode_start=torch.full_like(terminated, self._episode_start),
                teacher_action_index=teacher_action_index,
            )

            if bool(torch.all(boundary)):
                self.episodes_completed += 1
                observations_after_reset, _ = self.env.reset()
                self._require_live_agents(observations_after_reset)
                _reset_if_supported(self.opponent)
                _reset_if_supported(self.reward_transform)
                if self.teacher is not None:
                    _reset_if_supported(self.teacher)
                self._observations = observations_after_reset
                self._episode_start = True
            else:
                self._observations = next_observations
                self._episode_start = False
        return buffer

    def _require_live_agents(
        self, observations: Mapping[str, Mapping[str, np.ndarray]]
    ) -> None:
        expected = set(self.controlled_agents + self.opponent_agents)
        if set(observations) != expected:
            raise ValueError(f"parallel environment must expose all ten agents, got {sorted(observations)}")
