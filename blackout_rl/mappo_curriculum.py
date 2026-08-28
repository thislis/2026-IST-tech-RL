"""Teacher-assisted MAPPO curriculum and episode-level opponent mixtures."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class MAPPOCurriculumStage:
    name: str
    minimum_steps: int
    maximum_steps: int
    promotion_win_rate: float
    promotion_score_diff: float
    require_both_side_wins: bool
    opponent_weights: Mapping[str, float]
    evaluation_opponent: str
    use_ppo: bool
    bc_coef_start: float
    bc_coef_end: float
    teacher_forcing_start: float = 0.0
    teacher_forcing_end: float = 0.0

    def validate(self) -> None:
        if not self.name or self.minimum_steps < 0 or self.maximum_steps <= 0:
            raise ValueError("invalid curriculum stage identity or budget")
        if self.minimum_steps > self.maximum_steps:
            raise ValueError("minimum stage steps cannot exceed maximum stage steps")
        if not 0.0 <= self.promotion_win_rate <= 1.0:
            raise ValueError("stage promotion win rate must be in [0,1]")
        if self.bc_coef_start < 0.0 or self.bc_coef_end < 0.0:
            raise ValueError("BC coefficients must be non-negative")
        if not 0.0 <= self.teacher_forcing_start <= 1.0 or not 0.0 <= self.teacher_forcing_end <= 1.0:
            raise ValueError("teacher-forcing probabilities must be in [0,1]")
        if not self.opponent_weights or any(weight < 0.0 for weight in self.opponent_weights.values()):
            raise ValueError("opponent weights must be non-negative and non-empty")
        if sum(self.opponent_weights.values()) <= 0.0:
            raise ValueError("opponent weights must have positive mass")
        if self.evaluation_opponent not in self.opponent_weights and self.evaluation_opponent != "historical":
            raise ValueError("evaluation opponent is not declared by the stage")

    def progress(self, stage_steps: int) -> float:
        if stage_steps < 0:
            raise ValueError("stage steps must be non-negative")
        return min(stage_steps / self.maximum_steps, 1.0)

    def bc_coefficient(self, stage_steps: int) -> float:
        progress = self.progress(stage_steps)
        return self.bc_coef_start + progress * (self.bc_coef_end - self.bc_coef_start)

    def teacher_forcing_probability(self, stage_steps: int) -> float:
        progress = self.progress(stage_steps)
        return self.teacher_forcing_start + progress * (
            self.teacher_forcing_end - self.teacher_forcing_start
        )


DEFAULT_MAPPO_CURRICULUM = (
    MAPPOCurriculumStage(
        "planner_residual_warmup", 50_000, 100_000, 0.90, 10.0, True,
        {"base_scripted": 1.0}, "base_scripted", False, 0.20, 0.05, 0.50, 0.0,
    ),
    MAPPOCurriculumStage(
        "base_scripted", 50_000, 300_000, 0.90, 10.0, True,
        {"base_scripted": 1.0}, "base_scripted", True, 0.7, 0.3,
    ),
    MAPPOCurriculumStage(
        "weak_win70", 100_000, 400_000, 0.70, 0.0, True,
        {"weak_win70": 1.0}, "weak_win70", True, 0.5, 0.2,
    ),
    MAPPOCurriculumStage(
        "win70_mix25", 100_000, 300_000, 0.30, -50.0, True,
        {"weak_win70": 0.75, "full_win70": 0.25}, "full_win70", True, 0.3, 0.15,
    ),
    MAPPOCurriculumStage(
        "win70_mix50", 150_000, 400_000, 0.50, -25.0, True,
        {"weak_win70": 0.50, "full_win70": 0.50}, "full_win70", True, 0.2, 0.08,
    ),
    MAPPOCurriculumStage(
        "full_win70", 200_000, 1_000_000, 0.70, -5.0, True,
        {"full_win70": 1.0}, "full_win70", True, 0.10, 0.02,
    ),
    MAPPOCurriculumStage(
        "robust_historical_mix", 0, 2_000_000, 0.85, 0.0, True,
        {"full_win70": 0.80, "historical": 0.20}, "full_win70", True, 0.05, 0.0,
    ),
)


def validate_mappo_curriculum(stages: Sequence[MAPPOCurriculumStage]) -> None:
    expected = tuple(stage.name for stage in DEFAULT_MAPPO_CURRICULUM)
    if tuple(stage.name for stage in stages) != expected:
        raise ValueError(f"MAPPO curriculum order must be {expected}")
    for stage in stages:
        stage.validate()


class MAPPOCurriculumController:
    def __init__(
        self,
        stages: Sequence[MAPPOCurriculumStage] = DEFAULT_MAPPO_CURRICULUM,
        *,
        stage_index: int = 0,
        stage_start_step: int = 0,
    ) -> None:
        validate_mappo_curriculum(stages)
        if not 0 <= stage_index < len(stages) or stage_start_step < 0:
            raise ValueError("invalid curriculum resume state")
        self.stages = tuple(stages)
        self.stage_index = stage_index
        self.stage_start_step = stage_start_step

    @property
    def stage(self) -> MAPPOCurriculumStage:
        return self.stages[self.stage_index]

    def stage_steps(self, global_step: int) -> int:
        if global_step < self.stage_start_step:
            raise ValueError("global step precedes current stage")
        return global_step - self.stage_start_step

    @staticmethod
    def _both_sides_won(summary: Mapping[str, Any]) -> bool:
        sides = summary.get("by_model_side", {})
        return all(int(sides.get(side, {}).get("wins", 0)) >= 1 for side in ("A", "B"))

    def promotion_reason(
        self, summary: Mapping[str, Any], *, global_step: int
    ) -> str | None:
        if self.stage_index == len(self.stages) - 1:
            return None
        elapsed = self.stage_steps(global_step)
        if elapsed < self.stage.minimum_steps:
            return None
        passed = (
            float(summary["win_rate"]) >= self.stage.promotion_win_rate
            and float(summary["mean_model_score_diff"]) >= self.stage.promotion_score_diff
            and (
                not self.stage.require_both_side_wins
                or self._both_sides_won(summary)
            )
        )
        if passed:
            return "evaluation_gate"
        return None

    def budget_exhausted(self, *, global_step: int) -> bool:
        """Report a failed stage budget without silently increasing difficulty."""

        return self.stage_steps(global_step) >= self.stage.maximum_steps

    def promote(self, *, global_step: int) -> tuple[str, str]:
        if self.stage_index == len(self.stages) - 1:
            raise ValueError("final curriculum stage cannot be promoted")
        previous = self.stage.name
        self.stage_index += 1
        self.stage_start_step = global_step
        return previous, self.stage.name

    def state_dict(self) -> dict[str, Any]:
        return {
            "stage_index": self.stage_index,
            "stage_start_step": self.stage_start_step,
            "stage": self.stage.name,
            "stages": [asdict(stage) for stage in self.stages],
        }


class EpisodeOpponentMixture:
    """Choose one frozen opponent per episode and keep it fixed until reset."""

    def __init__(
        self,
        policies: Mapping[str, Any],
        weights: Mapping[str, float],
        *,
        seed: int,
    ) -> None:
        if set(policies) != set(weights) or not policies:
            raise ValueError("opponent policies and weights must have identical IDs")
        if any(weight < 0.0 for weight in weights.values()) or sum(weights.values()) <= 0.0:
            raise ValueError("opponent mixture weights must have positive mass")
        self.policies = dict(policies)
        self.ids = tuple(policies)
        values = np.asarray([weights[policy_id] for policy_id in self.ids], dtype=np.float64)
        self.probabilities = values / values.sum()
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.selected_id: str | None = None
        self.selection_counts = {policy_id: 0 for policy_id in self.ids}

    def reset(self) -> None:
        index = int(self._rng.choice(len(self.ids), p=self.probabilities))
        self.selected_id = self.ids[index]
        self.selection_counts[self.selected_id] += 1
        reset = getattr(self.policies[self.selected_id], "reset", None)
        if callable(reset):
            reset()

    def act(self, observations: Mapping[str, Any], agents: Sequence[str]) -> dict[str, Any]:
        if self.selected_id is None:
            raise RuntimeError("opponent mixture must be reset before acting")
        return self.policies[self.selected_id].act(observations, agents)

    def state_dict(self) -> dict[str, Any]:
        return {
            "selected_id": self.selected_id,
            "selection_counts": dict(self.selection_counts),
            "rng_state": self._rng.bit_generator.state,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        counts = state.get("selection_counts", {})
        if set(counts) != set(self.selection_counts):
            raise ValueError("resume opponent mixture IDs differ")
        self.selection_counts = {key: int(value) for key, value in counts.items()}
        self._rng.bit_generator.state = dict(state["rng_state"])
        # Environment state is not checkpointed, so select a fresh opponent on reset.
        self.selected_id = None
