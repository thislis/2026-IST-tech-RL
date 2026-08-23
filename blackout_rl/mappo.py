"""MAPPO/CTDE models, centralized state construction, and joint PPO updates."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .action_distribution import evaluate_categorical_action, select_categorical_action
from .batching import N_TEAM_AGENTS, canonical_agents, team_agents
from .ippo_model import IPPOActorCritic, SemanticCNNEncoder
from .observation import GRAPHIC_CHANNEL_NAMES, N_CLASSES, N_UNITS, UNIT_BLOCK_SIZE, parse_vector
from .ppo import PPOConfig, PPODiagnostics, explained_variance
from .rollout import generalized_advantage_estimate
from .model_contract import load_checkpoint, team_model_input


CENTRAL_UNIT_SIZE = UNIT_BLOCK_SIZE + N_CLASSES
CENTRAL_CONTEXT_SIZE = 3  # absolute team A score, team B score, time
CENTRAL_VECTOR_SIZE = N_UNITS * CENTRAL_UNIT_SIZE + CENTRAL_CONTEXT_SIZE


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
        self.centralized_critic = CentralizedCritic(
            hidden_dim=int(self.actor_model.model_config["hidden_dim"]),
            graphic_dim=int(self.actor_model.model_config["graphic_dim"]),
        )

    def actor_logits(
        self, vector: torch.Tensor, graphic: torch.Tensor, slot_id: torch.Tensor
    ) -> torch.Tensor:
        return self.actor_model(vector, graphic, slot_id).action_logits

    def forward(
        self,
        vector: torch.Tensor,
        graphic: torch.Tensor,
        slot_id: torch.Tensor,
        central_vector: torch.Tensor,
        central_graphic: torch.Tensor,
    ) -> MAPPOOutput:
        return MAPPOOutput(
            action_logits=self.actor_logits(vector, graphic, slot_id),
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

    def __len__(self) -> int:
        return int(self.vector.shape[0])

    def to(self, device: str | torch.device) -> "MAPPORolloutBatch":
        return MAPPORolloutBatch(**{name: getattr(self, name).to(device) for name in self.__dataclass_fields__})


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
        if set(fields) != required:
            raise ValueError(f"joint rollout fields differ: {sorted(required ^ set(fields))}")
        for name in ("vector", "graphic", "slot_id", "team_mask", "action_index", "log_prob", "reward"):
            if fields[name].shape[0] != self.n_agents:
                raise ValueError(f"{name} must have leading team dimension")
        if fields["team_mask"].dtype != torch.bool:
            raise TypeError("team_mask must be boolean")
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
        )


class MAPPOParallelRolloutCollector:
    """Collect joint team transitions while keeping actor inputs decentralized."""

    def __init__(self, env: Any, model: MAPPOActorCritic, opponent: Any, *,
                 learning_team: int, device: str | torch.device = "cpu",
                 reward_transform: Any | None = None,
                 episode_seed_provider: Callable[[], int | None] | None = None) -> None:
        self.env=env; self.model=model.to(device); self.opponent=opponent
        self.learning_team=learning_team; self.device=torch.device(device)
        self.controlled_agents=team_agents(learning_team); self.opponent_agents=team_agents(1-learning_team)
        self.reward_transform=reward_transform
        self.episode_seed_provider=episode_seed_provider
        self.builder=CentralizedStateBuilder(); self._observations=None
        self.episodes_completed=0; self.terminal_episodes=0; self.truncated_episodes=0
        self.wins=0; self.draws=0; self.losses=0; self.environment_steps=0

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
            with torch.inference_mode():
                logits=self.model.actor_logits(local.vector,local.graphic,local.slot_id)
                selection=select_categorical_action(logits)
                value=self.model.centralized_critic(central.vector,central.graphic)
            learning_actions={agent:selection.action[row].cpu().numpy().astype(np.float32,copy=False)
                              for row,agent in enumerate(local.agent_names)}
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
            buffer.add(vector=local.vector,graphic=local.graphic,slot_id=local.slot_id,
                       central_vector=central.vector,central_graphic=central.graphic,team_mask=mask,
                       action_index=selection.index,log_prob=selection.log_prob,value=value,
                       reward=torch.tensor([float(transformed[a]) for a in self.controlled_agents]),
                       next_value=next_value,terminated=torch.tensor([terminated]),truncated=torch.tensor([truncated]))
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
) -> PPODiagnostics:
    config = config or PPOConfig()
    config.validate()
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
            output = model(batch.vector[ix], batch.graphic[ix], batch.slot_id[ix],
                           batch.central_vector[ix], batch.central_graphic[ix])
            evaluated = evaluate_categorical_action(output.action_logits, batch.action_index[ix])
            log_ratio = evaluated.log_prob - batch.old_log_prob[ix]
            ratio = log_ratio.exp()
            policy_loss = torch.maximum(
                -batch.advantage[ix] * ratio,
                -batch.advantage[ix] * ratio.clamp(1-config.clip_coef, 1+config.clip_coef),
            ).mean()
            value_loss = 0.5 * (output.value - batch.return_[ix]).square().mean()
            loss = policy_loss + config.value_loss_coef * value_loss - config.entropy_coef * evaluated.entropy.mean()
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
