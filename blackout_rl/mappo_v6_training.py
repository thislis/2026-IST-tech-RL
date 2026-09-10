"""On-policy v6 collection/PPO and training-only failure seed prioritization."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .batching import canonical_agents, team_agents
from .mappo import CentralizedStateBuilder
from .mappo_v6 import V6Model, PlannerFeatures, joint_distribution, apply_joint_action
from .ppo import PPOConfig
from .rollout import generalized_advantage_estimate
from .training_reward import TeamTrainingReward, TrainingRewardConfig


class FailureSeedSampler:
    """Replay *environments*, never stale PPO transitions; retain 40% uniform mass."""
    def __init__(self, seeds, seed: int, state: dict | None = None):
        self.seeds = tuple(int(x) for x in seeds)
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("unique training seeds required")
        self.rng = np.random.default_rng(seed)
        self.failure = np.full(len(self.seeds), .5)
        self.visits = np.zeros(len(self.seeds), dtype=np.int64)
        if state is not None:
            if tuple(state["seeds"]) != self.seeds:
                raise ValueError("resume training seed distribution differs")
            self.failure = np.asarray(state["failure"], dtype=float)
            self.visits = np.asarray(state["visits"], dtype=np.int64)
            self.rng.bit_generator.state = state["rng"]

    def probabilities(self):
        priority = .1 + self.failure
        return .4/len(self.seeds) + .6 * priority/priority.sum()

    def next_seed(self):
        index = int(self.rng.choice(len(self.seeds), p=self.probabilities()))
        self.visits[index] += 1
        return self.seeds[index]

    def observe(self, seed: int, result: str):
        if result not in ("win", "draw", "loss"):
            raise ValueError("invalid episode result")
        index = self.seeds.index(seed)
        self.failure[index] = .8*self.failure[index] + .2*{"win": 0, "draw": .5, "loss": 1}[result]

    def state_dict(self):
        return {"seeds": list(self.seeds), "failure": self.failure.tolist(),
                "visits": self.visits.tolist(), "rng": self.rng.bit_generator.state}


def central_features(observations, team, device):
    central = CentralizedStateBuilder().build(observations, learning_team=team, device=device)
    return torch.cat((central.vector, F.adaptive_avg_pool2d(central.graphic, (4,4)).flatten(1)), -1)[0]


class V6Collector:
    def __init__(self, env, model: V6Model, opponent, *, team: int,
                 sampler: FailureSeedSampler, reward_config: TrainingRewardConfig,
                 max_episode_steps: int = 22000, episode_callback=None):
        self.env, self.model, self.opponent, self.team = env, model, opponent, team
        self.sampler, self.max_episode_steps = sampler, max_episode_steps
        self.agents, self.other = team_agents(team), team_agents(1-team)
        self.context = PlannerFeatures(team)
        self.reward = TeamTrainingReward(team, reward_config)
        self.device = next(model.parameters()).device
        self.observations = None
        self.seed = None
        self.episode_steps = 0
        self.episode_callback = episode_callback
        self.counts = dict(environment_steps=0, episodes_completed=0, terminal_episodes=0,
                           truncated_episodes=0, wins=0, draws=0, losses=0,
                           residual_override_actions=0, residual_fallback_actions=0,
                           discarded_partial_episodes=0)

    def reset(self):
        self.seed = self.sampler.next_seed()
        self.observations, _ = self.env.reset(seed=self.seed)
        if set(self.observations) != set(canonical_agents()):
            raise ValueError("all ten observations required")
        self.context.reset()
        self.reward.reset()
        self.opponent.reset()
        self.episode_steps = 0

    def collect(self, steps: int, *, exploration: float, gamma: float, gae_lambda: float,
                force_planner: bool = False):
        if steps <= 0:
            raise ValueError("positive rollout length required")
        if self.observations is None:
            self.reset()
        rows = []
        expected_overrides = 0.0
        actual_overrides = 0
        legal_steps = 0
        self.model.eval()
        for _ in range(steps):
            obs = self.observations
            with torch.no_grad():
                features, valid, planner, alternatives = self.context.prepare(obs, self.model)
                central = central_features(obs, self.team, self.device)
                distribution = joint_distribution(self.model.logits(features), valid, exploration)
                index = torch.zeros((), dtype=torch.int64, device=self.device) if force_planner else distribution.sample()
                value = self.model.critic(central).squeeze(-1)
                log_prob = distribution.log_prob(index)
            legal_steps += int(bool(valid[1:].any()))
            expected_overrides += 0. if force_planner else float(1-distribution.probs[0])
            actual_overrides += int(index != 0)
            actions = apply_joint_action(int(index), planner, alternatives)
            joint = {a: actions[i] for i, a in enumerate(self.agents)}
            joint.update(self.opponent.act(obs, self.other))
            if set(joint) != set(canonical_agents()):
                raise ValueError("joint step requires exactly ten actions")
            next_obs, rewards, terms, truncs, infos = self.env.step(joint)
            self.episode_steps += 1
            self.counts["environment_steps"] += 1
            terminated, truncated = all(terms.values()), all(truncs.values())
            if any(terms.values()) != terminated or any(truncs.values()) != truncated or (terminated and truncated):
                raise RuntimeError("invalid simultaneous episode boundary")
            transformed = self.reward(rewards, terms, truncs, infos, self.agents)
            reward = float(np.mean([transformed[a] for a in self.agents]))
            if not math.isfinite(reward):
                raise RuntimeError("non-finite team reward")
            with torch.no_grad():
                next_value = torch.zeros_like(value) if terminated else self.model.critic(
                    central_features(next_obs, self.team, self.device)).squeeze(-1)
            rows.append(dict(features=features.detach().cpu(), valid=valid.cpu(), central=central.cpu(),
                             action=index.cpu(), old_log_prob=log_prob.cpu(), old_value=value.cpu(),
                             reward=torch.tensor(reward), next_value=next_value.cpu(),
                             terminated=torch.tensor(terminated), truncated=torch.tensor(truncated)))
            if terminated or truncated:
                self.counts["episodes_completed"] += 1
                self.counts["terminal_episodes" if terminated else "truncated_episodes"] += 1
                if terminated:
                    winner = int(infos[self.agents[0]]["winner"])
                    if winner not in (-1,0,1):
                        raise ValueError("invalid terminal winner")
                    result = "draw" if winner == -1 else "win" if winner == self.team else "loss"
                    self.counts[{"win":"wins", "draw":"draws", "loss":"losses"}[result]] += 1
                    self.sampler.observe(self.seed, result)
                    if self.episode_callback:
                        self.episode_callback(dict(seed=self.seed, learner_team=self.team, model_result=result,
                                                   terminal_winner=winner, episode_steps=self.episode_steps,
                                                   score=list(self.reward.tracker.score)))
                self.reset()
            else:
                self.observations = next_obs
                if self.episode_steps >= self.max_episode_steps:
                    raise RuntimeError("episode_continuity_watchdog: no terminal within max_episode_steps")
        self.counts["residual_override_actions"] += actual_overrides
        self.counts["residual_fallback_actions"] += steps*5-actual_overrides
        batch = {key: torch.stack([row[key] for row in rows]) for key in rows[0]}
        advantage, returns = generalized_advantage_estimate(
            batch["reward"][:, None], batch["old_value"][:, None], batch["next_value"][:, None],
            batch["terminated"][:, None], batch["truncated"][:, None], gamma=gamma, gae_lambda=gae_lambda)
        batch["advantage"], batch["return_"] = advantage[:,0], returns[:,0]
        batch["exploration"] = exploration
        diagnostics = {"override_rate": actual_overrides/(steps*5),
                       "team_override_rate": actual_overrides/steps,
                       "expected_team_override_rate": expected_overrides/steps,
                       "legal_correction_step_fraction": legal_steps/steps,
                       "teacher_failures": self.context.failures}
        if exploration > 0 and legal_steps > 100 and actual_overrides == 0:
            raise RuntimeError("exploration_watchdog: no sampled corrections on legal states")
        return batch, diagnostics


def update_v6(model: V6Model, optimizer, batch: dict[str, Any], config: PPOConfig):
    """Each PPO sample is a complete team decision, never an isolated slot."""
    config.validate()
    device = next(model.parameters()).device
    data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
    advantages = data["advantage"]
    advantages = (advantages-advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
    count = len(advantages)
    totals = dict(policy_loss=0., value_loss=0., entropy=0., approximate_kl=0., clip_fraction=0., gradient_norm=0.)
    updates, early = 0, False
    for epoch in range(config.update_epochs):
        order = torch.randperm(count, device=device)
        for ix in order.split(config.minibatch_size):
            distribution = joint_distribution(model.logits(data["features"][ix]), data["valid"][ix], data["exploration"])
            log_ratio = distribution.log_prob(data["action"][ix])-data["old_log_prob"][ix]
            ratio = log_ratio.exp()
            kl = ((ratio-1)-log_ratio).mean()
            if not bool(torch.isfinite(kl)):
                raise RuntimeError("non-finite PPO ratio")
            if config.target_kl is not None and float(kl.detach()) > config.target_kl:
                early = True
                break
            policy_loss = torch.maximum(-advantages[ix]*ratio,
                -advantages[ix]*ratio.clamp(1-config.clip_coef, 1+config.clip_coef)).mean()
            prediction = model.critic(data["central"][ix]).squeeze(-1)
            error = (prediction-data["return_"][ix]).square()
            if config.value_clip_coef is not None:
                clipped = data["old_value"][ix]+(prediction-data["old_value"][ix]).clamp(-config.value_clip_coef, config.value_clip_coef)
                error = torch.maximum(error, (clipped-data["return_"][ix]).square())
            value_loss = .5*error.mean()
            entropy = distribution.entropy().mean()
            loss = policy_loss+config.value_loss_coef*value_loss-config.entropy_coef*entropy
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("non-finite PPO loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                          config.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            metrics = dict(policy_loss=policy_loss, value_loss=value_loss, entropy=entropy,
                           approximate_kl=kl, clip_fraction=((ratio-1).abs()>config.clip_coef).float().mean(), gradient_norm=grad)
            for key, value in metrics.items():
                totals[key] += float(value.detach())
            updates += 1
        if early:
            break
    with torch.no_grad():
        prediction = model.critic(data["central"]).squeeze(-1)
        variance = data["return_"].var(unbiased=False)
        explained = None if variance < 1e-12 else float(1-(data["return_"]-prediction).var(unbiased=False)/variance)
    return {**{k: v/max(updates,1) for k,v in totals.items()}, "explained_variance": explained,
            "epochs_completed": epoch+1, "minibatches": updates, "samples": count, "early_stopped": early}
