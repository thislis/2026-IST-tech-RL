"""Versioned mixed-team adapter; plan only the units not controlled by the brain."""
from blackout_rl.batching import team_agents
from blackout_rl.frozen_opponent import FrozenScriptedOpponent
from blackout_rl.mappo_v6 import ROLES
from blackout_rl.scripted_fsm import ScriptedTeamController

LEGACY = 'fixed_scripted_3worker_2guard_radius48'
REVISION = 'fixed_scripted_active_slots_v2'


class ScriptedTeammates:
    def __init__(self, team, controlled_slots):
        if (not controlled_slots or len(set(controlled_slots)) != len(controlled_slots)
                or any(s not in range(5) for s in controlled_slots)):
            raise ValueError('invalid neural slots')
        self.team = team
        self.agents = team_agents(team)
        self.active_agents = tuple(a for i, a in enumerate(self.agents) if i not in controlled_slots)
        self.reset()

    def reset(self):
        self.controller = (ScriptedTeamController(self.team, roles=ROLES, chase_radius_cells=48,
                           active_agents=self.active_agents) if self.active_agents else None)

    def act(self, observations, agents):
        if set(agents) != set(self.agents):
            raise ValueError('mixed-team adapter requires its full team agent list')
        if self.controller is None:
            return {}
        return self.controller.act(observations, self.active_agents)

    @property
    def last_events(self):
        return self.controller.last_events if self.controller else []


def make_teammates(team, config):
    revision = config['controller']['remaining_team_policy']
    if revision == REVISION:
        return ScriptedTeammates(team, config['controller']['controlled_slots'])
    if revision == LEGACY:
        return FrozenScriptedOpponent(team, roles=ROLES, chase_radius_cells=48)
    raise ValueError(f'unsupported teammate policy: {revision}')
