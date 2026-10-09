"""Opponent curriculum, potential-based shaping, annealing, and promotion gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np

from .training_reward import TeamTrainingReward, TrainingRewardConfig


class CurriculumStage(str, Enum):
    RANDOM = "random"
    WEAK_SCRIPTED = "weak_scripted"
    FULL_SCRIPTED = "full_scripted"
    FROZEN_RL = "frozen_rl"


@dataclass(frozen=True)
class StageConfig:
    stage: CurriculumStage
    opponent_id: str
    minimum_win_rate: float
    minimum_score_diff: float
    max_side_gap: float

    def validate(self) -> None:
        if not self.opponent_id or not 0 <= self.minimum_win_rate <= 1 or self.max_side_gap < 0:
            raise ValueError("invalid curriculum stage")


DEFAULT_CURRICULUM = (
    StageConfig(CurriculumStage.RANDOM, "random-v1", .80, 10.0, .25),
    StageConfig(CurriculumStage.WEAK_SCRIPTED, "scripted-weak-v1", .65, 0.0, .25),
    StageConfig(CurriculumStage.FULL_SCRIPTED, "scripted-battery-v1", .55, 0.0, .20),
    StageConfig(CurriculumStage.FROZEN_RL, "frozen-rl", .50, 0.0, .15),
)


def validate_curriculum(stages: Sequence[StageConfig]) -> None:
    if tuple(stage.stage for stage in stages) != tuple(CurriculumStage):
        raise ValueError("curriculum must use random -> weak scripted -> full scripted -> frozen RL")
    for stage in stages: stage.validate()


@dataclass(frozen=True)
class StageEvaluation:
    stage: CurriculumStage
    opponent_id: str
    win_rate: float
    mean_score_diff: float
    side_gap: float
    episodes: int


def evaluate_stage_gate(config: StageConfig, result: StageEvaluation) -> bool:
    if result.stage != config.stage or result.opponent_id != config.opponent_id:
        raise ValueError("evaluation does not match curriculum stage")
    if result.episodes <= 0:
        raise ValueError("stage evaluation must contain episodes")
    return (result.win_rate >= config.minimum_win_rate
            and result.mean_score_diff >= config.minimum_score_diff
            and abs(result.side_gap) <= config.max_side_gap)


def separate_stage_results(results: Sequence[StageEvaluation]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in results:
        if row.stage.value in output:
            raise ValueError(f"duplicate stage evaluation: {row.stage.value}")
        output[row.stage.value] = asdict(row)
    return output


class PotentialNavigationShaping:
    """Potential-based navigation reward F=gamma*Phi(s')-Phi(s)."""

    def __init__(self, *, gamma: float, scale: float = 1.0) -> None:
        if not 0 <= gamma <= 1 or scale < 0:
            raise ValueError("invalid navigation shaping configuration")
        self.gamma = gamma; self.scale = scale

    def reward(self, distance: float, next_distance: float, *, terminal: bool = False) -> float:
        if distance < 0 or next_distance < 0:
            raise ValueError("navigation distance cannot be negative")
        current_phi = -distance
        next_phi = 0.0 if terminal else -next_distance
        return self.scale * (self.gamma * next_phi - current_phi)


@dataclass(frozen=True)
class LinearShapingAnnealer:
    initial_weight: float
    final_weight: float
    anneal_steps: int

    def __post_init__(self) -> None:
        if self.initial_weight < 0 or self.final_weight < 0 or self.anneal_steps <= 0:
            raise ValueError("invalid shaping annealing schedule")

    def weight(self, environment_step: int) -> float:
        if environment_step < 0:
            raise ValueError("environment_step must be non-negative")
        progress = min(environment_step / self.anneal_steps, 1.0)
        return self.initial_weight + progress * (self.final_weight - self.initial_weight)


def combined_reward(
    *, score_reward: float, terminal_reward: float, navigation_reward: float,
    environment_step: int, annealer: LinearShapingAnnealer,
) -> float:
    return score_reward + terminal_reward + annealer.weight(environment_step) * navigation_reward


class NavigationShapedTeamReward:
    """Add annealed team-average navigation potential to score/terminal reward."""

    def __init__(
        self,
        learning_team: int,
        *,
        base_config: TrainingRewardConfig,
        gamma: float,
        initial_weight: float,
        anneal_steps: int,
    ) -> None:
        self.learning_team = learning_team
        self.base = TeamTrainingReward(learning_team, base_config)
        self.potential = PotentialNavigationShaping(gamma=gamma)
        self.annealer = LinearShapingAnnealer(initial_weight, 0.0, anneal_steps)
        self.environment_step = 0
        self.last_navigation_reward = 0.0
        self.navigation_reward_sum = 0.0

    @staticmethod
    def _agent_distance(observation: Mapping[str, np.ndarray], agent: str) -> float:
        vector = np.asarray(observation["vector"], dtype=np.float32)
        graphic = np.asarray(observation["graphic"], dtype=np.float32)
        unit_index = int(agent.rsplit("_", 1)[1])
        block = vector[:90].reshape(10, 9)[unit_index]
        self_position = block[:2]
        holding_battery = int(np.argmax(block[3:9])) == 1
        channel = 2 if holding_battery else 6
        rows, columns = np.nonzero(graphic[:, :, channel] > 0.5)
        if not len(rows):
            return 0.0
        height, width = graphic.shape[:2]
        targets = np.stack(
            (
                columns / max(width - 1, 1),
                1.0 - rows / max(height - 1, 1),
            ),
            axis=1,
        )
        return float(np.linalg.norm(targets - self_position, axis=1).min())

    def observe_transition(
        self,
        observations: Mapping[str, Mapping[str, np.ndarray]],
        next_observations: Mapping[str, Mapping[str, np.ndarray]],
        terminations: Mapping[str, bool],
        truncations: Mapping[str, bool],
        controlled_agents: Sequence[str],
    ) -> None:
        current = np.mean(
            [self._agent_distance(observations[agent], agent) for agent in controlled_agents]
        )
        terminal = all(
            bool(terminations[agent] or truncations[agent]) for agent in controlled_agents
        )
        next_distance = (
            0.0
            if terminal
            else np.mean(
                [
                    self._agent_distance(next_observations[agent], agent)
                    for agent in controlled_agents
                ]
            )
        )
        raw = self.potential.reward(float(current), float(next_distance), terminal=terminal)
        weight = self.annealer.weight(self.environment_step)
        self.last_navigation_reward = weight * raw
        self.navigation_reward_sum += self.last_navigation_reward
        self.environment_step += 1

    def __call__(self, rewards, terminations, truncations, infos, controlled_agents):
        base = self.base(rewards, terminations, truncations, infos, controlled_agents)
        return {
            agent: float(base[agent]) + self.last_navigation_reward
            for agent in controlled_agents
        }

    def reset(self) -> None:
        self.base.reset()
        self.last_navigation_reward = 0.0


@dataclass(frozen=True)
class PromotionComparison:
    baseline_win_rate: float
    candidate_win_rate: float
    baseline_score_diff: float
    candidate_score_diff: float
    same_seeds: bool
    same_budget: bool


def promote_special_item_curriculum(result: PromotionComparison) -> bool:
    """Promote only when collection performance is not degraded."""
    if not result.same_seeds or not result.same_budget:
        raise ValueError("promotion comparison must use common seeds and budget")
    return (result.candidate_win_rate >= result.baseline_win_rate
            and result.candidate_score_diff >= result.baseline_score_diff)
