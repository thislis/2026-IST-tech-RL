"""Selectable Unity-shaping and score-delta team rewards for IPPO rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from .reward import ScoreDeltaEvent, ScoreDeltaRewardTracker


class RewardMode(str, Enum):
    SCORE_DELTA = "score_delta"
    UNITY_SHAPING = "unity_shaping"
    COMBINED = "combined"


@dataclass(frozen=True)
class TrainingRewardConfig:
    mode: RewardMode = RewardMode.SCORE_DELTA
    target_score: int = 100
    terminal_win_reward: float = 1.0
    score_delta_weight: float = 1.0
    unity_shaping_weight: float = 1.0

    def validate(self) -> None:
        if self.target_score <= 0:
            raise ValueError("target_score must be positive")
        if self.terminal_win_reward < 0.0:
            raise ValueError("terminal_win_reward must be non-negative")
        if self.score_delta_weight < 0.0 or self.unity_shaping_weight < 0.0:
            raise ValueError("reward weights must be non-negative")


class TeamTrainingReward:
    """Stateful collector reward transform shared by all five learning agents."""

    def __init__(self, learning_team: int, config: TrainingRewardConfig | None = None) -> None:
        if learning_team not in (0, 1):
            raise ValueError("learning_team must be 0 or 1")
        self.learning_team = learning_team
        self.config = config or TrainingRewardConfig()
        self.config.validate()
        self.tracker = ScoreDeltaRewardTracker(
            target_score=self.config.target_score,
            terminal_win_reward=self.config.terminal_win_reward,
        )
        self.last_event: ScoreDeltaEvent | None = None
        self.reset()

    def reset(self) -> None:
        self.tracker.reset()
        self.last_event = None

    def __call__(
        self,
        unity_rewards: Mapping[str, float],
        terminations: Mapping[str, bool],
        truncations: Mapping[str, bool],
        infos: Mapping[str, Mapping[str, Any]],
        controlled_agents: tuple[str, ...],
    ) -> Mapping[str, float]:
        if not controlled_agents:
            raise ValueError("controlled_agents must not be empty")
        missing = [agent for agent in controlled_agents if agent not in unity_rewards or agent not in infos]
        if missing:
            raise KeyError(f"reward inputs missing controlled agents: {missing}")
        team_terminated = all(bool(terminations[agent]) for agent in controlled_agents)
        team_truncated = all(bool(truncations[agent]) for agent in controlled_agents)
        if team_terminated and team_truncated:
            raise ValueError("transition cannot be both terminated and truncated")
        reference_info = infos[controlled_agents[0]]
        if team_terminated:
            if "winner" not in reference_info:
                raise KeyError("terminal info is missing winner")
            event = self.tracker.update_normalized(
                terminated=True, winner=int(reference_info["winner"])
            )
        else:
            try:
                score_a = float(reference_info["score_0"])
                score_b = float(reference_info["score_1"])
            except KeyError as exc:
                raise KeyError("non-terminal info must contain score_0 and score_1") from exc
            event = self.tracker.update_normalized(score_a, score_b)
        self.last_event = event

        team_score_reward = event.total_reward[self.learning_team]
        rewards: dict[str, float] = {}
        for agent in controlled_agents:
            unity = float(unity_rewards[agent])
            if self.config.mode == RewardMode.SCORE_DELTA:
                reward = self.config.score_delta_weight * team_score_reward
            elif self.config.mode == RewardMode.UNITY_SHAPING:
                reward = self.config.unity_shaping_weight * unity
            else:
                reward = (
                    self.config.score_delta_weight * team_score_reward
                    + self.config.unity_shaping_weight * unity
                )
            rewards[agent] = reward
        return rewards
