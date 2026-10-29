"""Explicit task/FSM state; each policy instance owns one team stream."""
from dataclasses import dataclass, field, asdict
import numpy as np
from .contracts import STATE_SCHEMA
from .navigation import cells, center, components


@dataclass
class UnitState:
    task_id: str = ''
    kind: str = 'hold'
    source: int = -1
    destination: int = -1
    phase: str = 'hold'
    age: int = 0
    commitment: int = 0
    stuck: int = 0
    delivery_wait: int = 0
    waypoint: int = -1
    blocked: dict = field(default_factory=dict)
    failed_stores: dict = field(default_factory=dict)
    speed: float = 2.5  # cells / second; replaced by public displacement EMA
    service_time: float = .5
    respawn: bool = False
    holding_changed: bool = False
    previous_action: np.ndarray = field(default_factory=lambda: np.zeros(2, np.float32))
    no_progress: int = 0
    no_progress_seconds: float = 0.
    progress_goal: int = -1
    progress_distance: float = -1.


class TeamState:
    def __init__(self):
        self.units = [UnitState() for _ in range(5)]
        self.positions = np.zeros((5,2), np.float32)
        self.previous_positions = self.positions.copy()
        self.previous_holding = np.zeros(5, np.int64)
        self.previous_classes = np.zeros(5, np.int64)
        self.previous_enemy = np.zeros((5,2), np.float32)
        self.enemy_speed = np.full(5, np.nan, np.float32)
        self.scores = np.zeros(2, np.float32)
        self.score_delta = 0.
        self.walls = np.zeros((24,24), bool)
        self.ally_storage = self.walls.copy()
        self.enemy_storage = self.walls.copy()
        self.enemy_base = self.walls.copy()
        self.item_age = np.full(576, 1000, np.int32)
        self.side = -1
        self.time_left = -1.
        self.step_count = 0
        self.last_event_step = -1000
        self.last_event_time = -1.
        self.previous_threat = np.zeros(5, bool)
        self.event_reasons = []

    def update(self, obs, config):
        learned = config.get('policy_mode') == 'learned'
        reasons = []
        reset = self.side != obs.side or obs.time_left > self.time_left+1e-4 or self.step_count == 0
        if reset:
            self.__init__()
            self.side = obs.side
            corner = obs.enemy_positions.mean(0) > .5
            yy, xx = np.indices((24,24))
            self.enemy_base = ((xx >= 21) if corner[0] else (xx < 3)) & ((yy >= 21) if corner[1] else (yy < 3))
            reasons.append('reset')
        elapsed = max(1e-4, (self.time_left-obs.time_left)*420) if not reset else .02
        self.positions = obs.positions.copy()
        if not reset and (np.any(obs.semantic[1]&~self.walls) or
                          np.any(obs.semantic[2]&~self.ally_storage) or
                          np.any(obs.semantic[3]&~self.enemy_storage)):
            reasons.append('map_change')
        self.walls |= obs.semantic[1]
        self.ally_storage |= obs.semantic[2]
        self.enemy_storage |= obs.semantic[3]
        self.item_age = np.minimum(self.item_age+1, 1000)
        self.item_age[obs.semantic[6].reshape(-1)] = 0
        self.score_delta = 0. if reset else float((obs.scores-self.scores) @ np.array([1.,-1.]))
        threat = np.linalg.norm(obs.positions[:,None]-obs.enemy_positions[None], axis=-1).min(-1) < .075
        for i,u in enumerate(self.units):
            displacement = float(np.linalg.norm(obs.positions[i]-self.previous_positions[i]))
            u.respawn = not reset and (displacement > .15 or (obs.classes[i] == 0 and self.previous_classes[i] != 0))
            if u.respawn:
                self.units[i] = u = UnitState(respawn=True)
                reasons.append('respawn')
            u.holding_changed = not reset and obs.holding[i] != self.previous_holding[i]
            if not reset and not u.respawn and displacement > 1e-5:
                observed = displacement*24/elapsed
                u.speed = float(.9*u.speed+.1*np.clip(observed, .15, 15.))
            u.age += 1
            u.commitment = max(0,u.commitment-1)
            u.blocked = {c:t-1 for c,t in u.blocked.items() if t>1}
            u.failed_stores = {c:t-1 for c,t in u.failed_stores.items() if t>1}
            u.stuck = u.stuck+1 if not reset and displacement < 1e-4 and np.linalg.norm(u.previous_action)>.1 else 0
            if not learned and u.stuck >= int(config.get('stuck_steps',30)):
                if u.waypoint >= 0:
                    u.blocked[u.waypoint] = 30
                u.commitment = 0
                u.stuck = 0
                reasons.append('stuck')
            if u.holding_changed:
                reasons.append('holding_change')
                u.commitment = 0
                if obs.holding[i] > 0:
                    # Deterministic pickup transition releases the source
                    # reservation before the next stochastic assignment.
                    u.kind, u.phase = 'deliver', 'return'
                    u.source = -1
                    u.task_id = ''
                else:
                    u.task_id, u.kind, u.phase = '', 'hold', 'hold'
                    u.source = u.destination = -1
                    u.failed_stores.clear()
            if not reset and obs.classes[i] != self.previous_classes[i]:
                u.commitment = 0
                u.task_id, u.kind, u.phase = '', 'hold', 'hold'
                reasons.append('class_change')
            if not learned and u.kind in ('collect-trip','raid-trip') and u.source >= 0 and obs.holding[i] == 0 and self.item_age[u.source] >= 12:
                u.commitment = 0
                u.task_id, u.kind, u.phase = '', 'hold', 'hold'
                reasons.append('source_vanished')
            if not learned and u.kind == 'special-trip' and u.source >= 0 and obs.holding[i] == 0 and not obs.semantic[7:11].any(0).reshape(-1)[u.source]:
                u.commitment = 0
                u.task_id, u.kind, u.phase = '', 'hold', 'hold'
                reasons.append('source_vanished')
            near_store = u.destination >= 0 and np.linalg.norm(center(u.destination)-obs.positions[i]) < .026
            u.delivery_wait = u.delivery_wait+1 if obs.holding[i] and near_store else 0
            if not learned and u.delivery_wait >= int(config.get('delivery_wait_steps',25)):
                for group in components(self.ally_storage):
                    if u.destination in group:
                        u.failed_stores.update({c:200 for c in group})
                u.task_id, u.commitment, u.destination, u.delivery_wait = '',0,-1,0
                reasons.append('storage_failed')
            if threat[i] and not self.previous_threat[i] and obs.classes[i] != 1:
                u.commitment = 0
                if not learned and u.kind.startswith('transform'):
                    u.task_id,u.kind,u.phase = '', 'hold','hold'
                reasons.append('threat')
            if learned:
                goal = u.destination if u.phase == 'return' else u.source
                distance = float(np.linalg.norm(center(goal)-obs.positions[i])*24) if goal >= 0 else -1.
                fresh = (reset or u.respawn or u.holding_changed or goal != u.progress_goal
                         or goal < 0 or u.progress_distance < 0.)
                # A running best distance observes loops as well as a stationary
                # unit. The epsilon is numerical tolerance in cells, not a
                # behavior threshold; no lock, goal or event is changed here.
                if fresh or distance < u.progress_distance-1e-3:
                    u.no_progress, u.no_progress_seconds = 0, 0.
                    u.progress_distance = distance
                else:
                    u.no_progress += 1
                    u.no_progress_seconds += elapsed
                u.progress_goal = goal
        if not reset:
            enemy_delta = np.linalg.norm(obs.enemy_positions-self.previous_enemy, axis=-1)*24/elapsed
            moving = (enemy_delta>.05)&(enemy_delta<15)
            old = np.nan_to_num(self.enemy_speed, nan=2.5)
            self.enemy_speed[moving] = .8*old[moving]+.2*enemy_delta[moving]
        self.previous_positions = obs.positions.copy()
        self.previous_holding = obs.holding.copy()
        self.previous_classes = obs.classes.copy()
        self.previous_enemy = obs.enemy_positions.copy()
        self.scores, self.time_left = obs.scores.copy(),obs.time_left
        self.previous_threat = threat
        self.step_count += 1
        period = int(config.get('event_period',config.get('goal_period',25)))
        if self.step_count-self.last_event_step >= period or (self.last_event_time-obs.time_left)*420 >= period/50:
            reasons.append('period')
        self.event_reasons = sorted(set(reasons))
        return bool(reasons), reset

    def snapshot(self):
        return {'schema':STATE_SCHEMA, 'units':[asdict(u) for u in self.units],
                **{k:v.copy() if isinstance(v,np.ndarray) else v for k,v in vars(self).items() if k!='units'}}
