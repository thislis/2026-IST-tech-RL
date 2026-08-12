"""Opponent curriculum, potential-based shaping, annealing, and promotion gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping, Sequence


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
