"""Python-side score-delta team rewards independent of Unity shaping rewards."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ScoreDeltaEvent:
    """One score transition and the symmetric team rewards derived from it."""

    previous_score: tuple[int, int]
    current_score: tuple[int, int]
    score_delta: tuple[int, int]
    competitive_reward: tuple[float, float]
    terminal_bonus: tuple[float, float]
    total_reward: tuple[float, float]
    terminated: bool
    winner: int | None

    def to_dict(self) -> dict:
        return asdict(self)


class ScoreDeltaRewardTracker:
    """Track observable scores and emit a shared reward for each five-agent team.

    The dense component is the normalized change in score advantage:

    `team_a = (delta_a - delta_b) / target_score`
    `team_b = -team_a`

    A score decrease is therefore negative for the victim and positive for its
    opponent. At termination, Unity exposes reset scores `(0,0)`; those values
    are deliberately ignored and only the winner-based terminal bonus is added.
    """

    def __init__(self, target_score: int = 100, terminal_win_reward: float = 1.0):
        if target_score <= 0:
            raise ValueError("target_score must be positive")
        if terminal_win_reward < 0:
            raise ValueError("terminal_win_reward must be non-negative")
        self.target_score = target_score
        self.terminal_win_reward = float(terminal_win_reward)
        self._score = (0, 0)
        self._cumulative_competitive_reward = [0.0, 0.0]
        self._cumulative_terminal_bonus = [0.0, 0.0]

    @property
    def score(self) -> tuple[int, int]:
        return self._score

    @property
    def cumulative_competitive_reward(self) -> tuple[float, float]:
        return tuple(self._cumulative_competitive_reward)

    @property
    def cumulative_terminal_bonus(self) -> tuple[float, float]:
        return tuple(self._cumulative_terminal_bonus)

    def reset(self, score: tuple[int, int] = (0, 0)) -> None:
        self._validate_score(score)
        self._score = score
        self._cumulative_competitive_reward = [0.0, 0.0]
        self._cumulative_terminal_bonus = [0.0, 0.0]

    def update_points(
        self,
        score: tuple[int, int] | None = None,
        *,
        terminated: bool = False,
        winner: int | None = None,
    ) -> ScoreDeltaEvent:
        if terminated:
            if winner not in (-1, 0, 1):
                raise ValueError("terminal winner must be -1, 0, or 1")
            current = self._score
            delta = (0, 0)
            competitive = (0.0, 0.0)
            if winner == -1:
                terminal_bonus = (0.0, 0.0)
            elif winner == 0:
                terminal_bonus = (self.terminal_win_reward, -self.terminal_win_reward)
            else:
                terminal_bonus = (-self.terminal_win_reward, self.terminal_win_reward)
        else:
            if winner is not None:
                raise ValueError("winner is only valid for a terminal update")
            if score is None:
                raise ValueError("score is required for a non-terminal update")
            self._validate_score(score)
            current = score
            delta = (current[0] - self._score[0], current[1] - self._score[1])
            advantage_delta = (delta[0] - delta[1]) / self.target_score
            competitive = (advantage_delta, -advantage_delta)
            terminal_bonus = (0.0, 0.0)

        previous = self._score
        total = (
            competitive[0] + terminal_bonus[0],
            competitive[1] + terminal_bonus[1],
        )
        self._score = current
        for team in (0, 1):
            self._cumulative_competitive_reward[team] += competitive[team]
            self._cumulative_terminal_bonus[team] += terminal_bonus[team]
        return ScoreDeltaEvent(
            previous_score=previous,
            current_score=current,
            score_delta=delta,
            competitive_reward=competitive,
            terminal_bonus=terminal_bonus,
            total_reward=total,
            terminated=terminated,
            winner=winner,
        )

    def update_normalized(
        self,
        score_a: float | None = None,
        score_b: float | None = None,
        *,
        terminated: bool = False,
        winner: int | None = None,
    ) -> ScoreDeltaEvent:
        score = None
        if not terminated:
            if score_a is None or score_b is None:
                raise ValueError("both normalized scores are required")
            score = (
                int(round(float(score_a) * self.target_score)),
                int(round(float(score_b) * self.target_score)),
            )
        return self.update_points(score, terminated=terminated, winner=winner)

    @staticmethod
    def _validate_score(score: tuple[int, int]) -> None:
        if len(score) != 2 or any(not isinstance(value, int) for value in score):
            raise TypeError("score must be a tuple of two integers")
        if any(value < 0 for value in score):
            raise ValueError("scores must be non-negative")
