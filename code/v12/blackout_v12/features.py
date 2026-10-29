"""Versioned public unit-task features; sentinel -1 distinguishes unknowns.

Distance is in cells / 48, time in seconds / 60, counters in steps / 100.
Values are clipped to [-1, 1]. These fixed scales are recorded in registration;
no test observation or hidden stack quantity is used to fit normalization.
"""
import math
import numpy as np
from .contracts import SCHEMA

KINDS = ('collect-trip','deliver','special-trip','intercept','defend','raid-trip',
         'transform-Hunter','transform-Carrier','hold')
FEATURE_NAMES = (
    'class_collector','class_hunter','class_carrier',
    'holding_empty','holding_battery','holding_speed_up','holding_speed_down','holding_size_up','holding_size_down',
    'speed','stuck','task_age',
    *('task_'+kind for kind in KINDS), 'visible','memory_age','protected_storage',
    'distance_go','distance_back','eta_trip','switch_cost','clearance','fallback','alternative_gap',
    'enemy_eta_lower','enemy_eta_upper','arrival_margin_lower','arrival_margin_upper','risk','failed_storage',
    'cost_stale','previous_goal_distance','reachable',
    'own_score','enemy_score','time_left','absorption_sin','absorption_cos',
    'collector_count','hunter_count','carrier_count','holding_count','battery_tiles','reachable_tasks',
    'special_speed_up','special_speed_down','special_size_up','special_size_down','score_delta',
    'same_target','same_intent','respawn','holding_changed','other_best_eta','other_second_eta','uncertainty','target_demand')
assert len(FEATURE_NAMES) == 64
FEATURE_SCHEMA = {'id':SCHEMA, 'names':FEATURE_NAMES, 'unknown':-1., 'range':[-1.,1.],
                  'distance_cells':48., 'time_seconds':60., 'counter_steps':100.}
LEARNED_FEATURE_NAMES = list(FEATURE_NAMES)
LEARNED_FEATURE_NAMES[10:12] = ['no_progress_log', 'task_age_log']
LEARNED_FEATURE_NAMES[37] = 'no_progress_time_log'
LEARNED_FEATURE_NAMES[39] = 'is_evade'
LEARNED_FEATURE_SCHEMA = {**FEATURE_SCHEMA, 'id':'blackout.v12.features.learned.1',
                         'names':tuple(LEARNED_FEATURE_NAMES),
                         'log_counter_steps':21000., 'log_time_seconds':420.}


def build_features(obs, state, tasks, metrics, legal, config=None):
    learned = dict(config or {}).get('policy_mode') == 'learned'
    m = len(tasks)
    result = np.zeros((5,m,64), np.float32)
    phase = (1.-obs.time_left)*420/20*2*math.pi
    team = [*obs.scores,obs.time_left,math.sin(phase),math.cos(phase),
            *(np.bincount(obs.classes,minlength=3)/5),np.count_nonzero(obs.holding)/5,
            np.count_nonzero(obs.semantic[6])/64, 0.,
            *(obs.semantic[7:11].reshape(4,-1).any(-1).astype(float)),state.score_delta]
    for i,u in enumerate(state.units):
        for j,t in enumerate(tasks):
            x = result[i,j]
            x[int(obs.classes[i])] = 1.
            x[3+int(obs.holding[i])] = 1.
            x[9:12] = u.speed/10, u.stuck/100, u.age/100
            if learned:
                # Observation history changes scores only through the network;
                # neither counter triggers an action or a task timeout.
                x[10:12] = (math.log1p(u.no_progress)/math.log1p(21000),
                            math.log1p(u.age)/math.log1p(21000))
            if t.kind in KINDS:
                x[12+KINDS.index(t.kind)] = 1.
            r = metrics[i][j]
            x[21:24] = float(t.visible), t.memory_age/12, float(t.protected)
            x[24:40] = (r['go']/48, r['back']/48, r['eta']/60, r['switch'], r['clearance']/4,
                        r['fallback'],r['alternative']/60,r['enemy_low']/60,
                        r['enemy_high']/60 if r['enemy_high']>=0 else -1,
                        (r['enemy_low']-r['arrival'])/60,
                        (r['enemy_high']-r['arrival'])/60 if r['enemy_high']>=0 else -1,
                        r['risk'],float(r['failed']),0.,r['old_distance']/48,float(legal[i,j]))
            if learned:
                x[37] = math.log1p(u.no_progress_seconds)/math.log1p(420)
                x[39] = float(t.kind == 'evade')
            x[40:56] = team
            x[50] = np.count_nonzero(legal[i])/max(1,m)
            other = sorted(metrics[k][j]['eta'] for k in range(5) if k!=i and legal[k,j])
            x[56:64] = (float(u.task_id==t.id),float(u.kind==t.kind),float(u.respawn),float(u.holding_changed),
                         other[0]/60 if other else -1.,other[1]/60 if len(other)>1 else -1.,
                         float(r['enemy_high']<0),sum(1 for k in range(5) if legal[k,j])/5.)
    return np.clip(result,-1,1)


def fixed_prior(kind, r, obs, unit, same, config):
    """Finite fixed surrogate, never a new reward or a counterfactual label."""
    base = {'collect-trip':8.,'deliver':20.,'special-trip':3.,'intercept':6.,
            'defend':-1.,'raid-trip':4.,'transform-Hunter':2.,'transform-Carrier':2.,'hold':-6.}.get(kind,-20.)
    score = base - .20*r['eta'] - 2.*r['risk'] - r['switch']
    if kind=='intercept':
        score -= max(0.,r['go']-10)*.5
    if kind=='defend':
        score += 5.*r['risk']
    if kind.startswith('transform'):
        score += 6.*r['risk'] - 2.*obs.time_left
    if same:
        score += float(config.get('continue_bonus',.75))
    return float(np.clip(score,-30,30))
