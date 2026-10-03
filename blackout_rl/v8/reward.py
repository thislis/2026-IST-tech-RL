"""Idempotent score ledger. No Unity shaping enters the task objective."""
from .contracts import ContractError, OutcomeEvent

class RewardLedger:
    def __init__(self, team, target_score=100, terminal_bonus=1.):
        if team not in (0, 1) or target_score <= 0 or terminal_bonus < 0:
            raise ContractError('invalid reward configuration')
        self.team, self.target, self.bonus = team, target_score, terminal_bonus
        self.run = self.episode = None
        self.events = {}

    def reset(self, run_id, episode_id, points=(0, 0)):
        self.run, self.episode = run_id, episode_id
        self.points = self._points(points)
        self.ended = False
        self.revision = -1

    @staticmethod
    def _points(points):
        if len(points) != 2 or any(type(x) is not int or x < 0 for x in points):
            raise ContractError('raw integer scores required')
        return tuple(points)

    def _delta(self, points):
        points = self._points(points)
        delta = tuple(a-b for a, b in zip(points, self.points))
        self.points = points
        return (delta[self.team]-delta[1-self.team])/self.target

    def observe(self, points, revision):
        points = self._points(points)
        if self.ended or revision < self.revision:
            raise ContractError('late score update')
        if revision == self.revision:
            if points != self.points:
                raise ContractError('conflicting score revision')
            return 0.
        self.revision = revision
        return self._delta(points)

    def finish(self, event: OutcomeEvent):
        if (event.run_id, event.episode_id) != (self.run, self.episode):
            raise ContractError('outcome belongs to another episode')
        if event.key in self.events:
            if self.events[event.key] != event:
                raise ContractError('conflicting duplicate outcome')
            return dict(score_delta=0., terminal_bonus=0., reward=0., duplicate=True)
        if self.ended:
            raise ContractError('second distinct outcome for same episode')
        if event.final_score_points is None:
            raise ContractError('reward needs final score even on truncation')
        delta = self._delta(event.final_score_points)
        bonus = 0. if not event.terminated or event.winner_team == -1 else self.bonus * (1 if event.winner_team == self.team else -1)
        self.events[event.key] = event
        self.ended = True
        return dict(score_delta=delta, terminal_bonus=bonus, reward=delta+bonus, duplicate=False)
