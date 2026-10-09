"""Versioned learner-only isolation; the historical opponent is never monkeypatched."""
import copy
import numpy as np
from blackout_rl.scripted_fsm import ScriptedTeamController
from blackout_rl.navigation import PathNotFound
from blackout_rl.mappo_v6 import PlannerFeatures, ROLES
from .contracts import ContractError

REVISION = 'slot_local_v8_1'

class IsolatedPlanner(ScriptedTeamController):
    def __init__(self, team):
        super().__init__(team, roles=ROLES, chase_radius_cells=48,
                         active_agents=tuple(f'unit_{team*5+s}' for s in range(5)))
        self.unreachable = set()

    def _navigate(self, state):
        try:
            return super()._navigate(state)
        except PathNotFound as exc:
            self.unreachable.add(state.agent)
            self._clear_target(state.agent)
            self._emit(state.agent, 'navigation_unreachable', reason=str(exc))
            return np.zeros(2, np.float32)

    def act(self, observations, agents):
        self.unreachable = set()
        actions = super().act(observations, agents)
        for event in self.last_events:
            if event['kind'] in ('assignment_unreachable', 'navigation_unreachable', 'replan_failed'):
                self.unreachable.add(event['agent'])
        for agent in self.unreachable:
            actions[agent] = np.zeros(2, np.float32)
        return actions

class PlannerAdapter(PlannerFeatures):
    def __init__(self, team):
        super().__init__(team)
        self.planner = IsolatedPlanner(team)

    def prepare(self, observations, model):
        failures = self.failures
        result = super().prepare(observations, model)
        # Never disguise an unexpected team-level exception as a valid policy action.
        if self.failures != failures:
            raise ContractError('unisolated planner failure; invalidate attempt')
        self.teacher_valid = {a: a not in self.planner.unreachable for a in self.agents}
        return result

    def state_digest(self):
        from .contracts import digest
        return digest(dict(step=self.planner._step, phases=self.planner.phases,
                           targets={a: None if t is None else [t.x,t.y] for a,t in self.planner.targets.items()}))
