"""Baseline-calibrated curriculum for planner-conditioned MAPPO residual v5."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class MAPPOV5Stage:
    name: str
    minimum_steps: int
    maximum_steps: int
    promotion_win_rate: float
    promotion_score_diff: float
    require_both_side_wins: bool
    minimum_side_win_rate: float
    opponent_weights: Mapping[str, float]
    evaluation_opponent: str
    use_ppo: bool
    planner_forcing: float
    override_penalty_start: float
    override_penalty_end: float

    def validate(self) -> None:
        if self.minimum_steps < 0 or self.maximum_steps < self.minimum_steps:
            raise ValueError(f"invalid step budget for stage {self.name}")
        if not 0.0 <= self.promotion_win_rate <= 1.0:
            raise ValueError(f"invalid promotion win rate for stage {self.name}")
        if not 0.0 <= self.minimum_side_win_rate <= 1.0:
            raise ValueError(f"invalid side win rate for stage {self.name}")
        if not 0.0 <= self.planner_forcing <= 1.0:
            raise ValueError(f"invalid planner forcing for stage {self.name}")
        if self.override_penalty_start < 0.0 or self.override_penalty_end < 0.0:
            raise ValueError(f"override penalty must be non-negative for {self.name}")
        if not self.opponent_weights or sum(self.opponent_weights.values()) <= 0.0:
            raise ValueError(f"stage {self.name} has no opponent probability mass")
        if any(weight < 0.0 for weight in self.opponent_weights.values()):
            raise ValueError(f"stage {self.name} has a negative opponent weight")

    def override_penalty(self, stage_steps: int) -> float:
        if self.maximum_steps <= 0:
            return self.override_penalty_end
        progress = min(max(stage_steps / self.maximum_steps, 0.0), 1.0)
        return self.override_penalty_start + progress * (
            self.override_penalty_end - self.override_penalty_start
        )


DEFAULT_MAPPO_V5_CURRICULUM = (
    MAPPOV5Stage(
        "planner_preservation", 16_384, 16_384, 0.70, 10.0, True, 0.50,
        {"base_scripted": 1.0}, "base_scripted", False, 1.0, 0.02, 0.02,
    ),
    MAPPOV5Stage(
        "scripted_residual_ppo", 50_000, 300_000, 0.90, 10.0, True, 0.70,
        {"base_scripted": 1.0}, "base_scripted", True, 0.0, 0.010, 0.002,
    ),
    MAPPOV5Stage(
        "weak_win70_mix", 100_000, 400_000, 0.70, 0.0, True, 0.50,
        {"base_scripted": 0.50, "weak_win70": 0.50}, "weak_win70", True, 0.0,
        0.006, 0.001,
    ),
    MAPPOV5Stage(
        "full_win70_mix10", 100_000, 250_000, 0.30, -50.0, True, 0.10,
        {"weak_win70": 0.90, "full_win70": 0.10}, "full_win70", True, 0.0,
        0.004, 0.0008,
    ),
    MAPPOV5Stage(
        "full_win70_mix25", 100_000, 300_000, 0.40, -35.0, True, 0.20,
        {"weak_win70": 0.75, "full_win70": 0.25}, "full_win70", True, 0.0,
        0.003, 0.0005,
    ),
    MAPPOV5Stage(
        "full_win70_mix50", 150_000, 400_000, 0.50, -20.0, True, 0.30,
        {"weak_win70": 0.50, "full_win70": 0.50}, "full_win70", True, 0.0,
        0.002, 0.0003,
    ),
    MAPPOV5Stage(
        "full_win70", 200_000, 1_000_000, 0.70, -5.0, True, 0.50,
        {"full_win70": 1.0}, "full_win70", True, 0.0, 0.001, 0.0001,
    ),
    MAPPOV5Stage(
        "robust_historical_mix", 0, 2_000_000, 0.85, 0.0, True, 0.70,
        {"full_win70": 0.80, "historical": 0.20}, "full_win70", True, 0.0,
        0.0005, 0.0,
    ),
)


class MAPPOV5CurriculumController:
    def __init__(
        self,
        stages: Sequence[MAPPOV5Stage] = DEFAULT_MAPPO_V5_CURRICULUM,
        *,
        stage_index: int = 0,
        stage_start_step: int = 0,
    ) -> None:
        if not stages:
            raise ValueError("v5 curriculum cannot be empty")
        for stage in stages:
            stage.validate()
        if not 0 <= stage_index < len(stages) or stage_start_step < 0:
            raise ValueError("invalid v5 curriculum resume state")
        self.stages = tuple(stages)
        self.stage_index = stage_index
        self.stage_start_step = stage_start_step

    @property
    def stage(self) -> MAPPOV5Stage:
        return self.stages[self.stage_index]

    def stage_steps(self, global_step: int) -> int:
        if global_step < self.stage_start_step:
            raise ValueError("global step precedes current stage")
        return global_step - self.stage_start_step

    @staticmethod
    def both_sides_won(summary: Mapping[str, Any]) -> bool:
        sides = summary.get("by_model_side", {})
        return all(int(sides.get(side, {}).get("wins", 0)) >= 1 for side in ("A", "B"))

    def gate_passed(self, summary: Mapping[str, Any], *, global_step: int) -> bool:
        if self.stage_steps(global_step) < self.stage.minimum_steps:
            return False
        sides = summary.get("by_model_side", {})
        side_rates_pass = all(
            float(sides.get(side, {}).get("win_rate", 0.0))
            >= self.stage.minimum_side_win_rate
            for side in ("A", "B")
        )
        return (
            float(summary["win_rate"]) >= self.stage.promotion_win_rate
            and float(summary["mean_model_score_diff"]) >= self.stage.promotion_score_diff
            and (
                not self.stage.require_both_side_wins
                or self.both_sides_won(summary)
            )
            and side_rates_pass
        )

    def can_promote(self) -> bool:
        return self.stage_index < len(self.stages) - 1

    def budget_exhausted(self, *, global_step: int) -> bool:
        return self.stage_steps(global_step) >= self.stage.maximum_steps

    def promote(self, *, global_step: int) -> tuple[str, str]:
        if not self.can_promote():
            raise ValueError("final v5 stage cannot be promoted")
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
