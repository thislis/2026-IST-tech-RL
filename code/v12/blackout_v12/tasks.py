"""Canonical resource tasks, reservation-aware capacities and diverse pruning."""
from dataclasses import dataclass
import numpy as np
from .navigation import INF, HUNTER_SHRINE, CARRIER_SHRINE, cells, center, components
from .features import build_features, fixed_prior


@dataclass(frozen=True)
class Task:
    id: str
    kind: str
    source: int
    capacity: int
    targets: tuple = ()
    private_slot: int = -1
    visible: bool = True
    memory_age: int = 0
    protected: bool = False


@dataclass
class CandidateSet:
    features: np.ndarray
    prior: np.ndarray
    legal: np.ndarray
    capacities: np.ndarray
    tasks: list
    destinations: np.ndarray
    targets: np.ndarray
    locked: np.ndarray
    temperature: float
    diagnostics: dict


def all_tasks(obs, state, config):
    learned = config.get('policy_mode') == 'learned'
    tasks = {}
    stores = components(state.ally_storage)
    store_mask = state.ally_storage | state.enemy_storage
    neutral = (state.item_age.reshape(24,24)<12)&~store_mask
    neutral.reshape(-1)[list(HUNTER_SHRINE)+[CARRIER_SHRINE]] = False
    for cell in np.flatnonzero(neutral):
        t = Task('collect:'+str(cell),'collect-trip',int(cell),1,
                 visible=bool(obs.semantic[6].reshape(-1)[cell]),memory_age=int(state.item_age[cell]))
        tasks[t.id] = t
    for group in stores:
        t = Task('deliver:'+str(group[0]),'deliver',group[0],5,group)
        tasks[t.id] = t
        # Legacy capacity two is a tactical restriction, absent in learned mode.
        t = Task('defend:'+str(group[0]),'defend',group[0],5 if learned else 2,group)
        tasks[t.id] = t
    for cell in np.flatnonzero(obs.semantic[7:11].any(0)):
        t = Task('special:'+str(cell),'special-trip',int(cell),1)
        tasks[t.id] = t
    for k,cell in enumerate(cells(obs.enemy_positions)):
        t = Task('intercept:'+str(k),'intercept',int(cell),5 if learned else 2)
        tasks[t.id] = t
    for cell in np.flatnonzero(state.enemy_storage & obs.semantic[6] & ~state.enemy_base):
        t = Task('raid:'+str(cell),'raid-trip',int(cell),1)
        tasks[t.id] = t
    hunters = max(0,(5 if learned else int(config.get('hunter_cap',2)))-int(np.count_nonzero(obs.classes==1)))
    carriers = max(0,1-int(np.count_nonzero(obs.classes==2)))
    # Capacity includes already committed transformations; never subtract
    # their reservations a second time before passing this to the DP.
    tasks['transform-Hunter'] = Task('transform-Hunter','transform-Hunter',HUNTER_SHRINE[0],hunters,HUNTER_SHRINE)
    tasks['transform-Carrier'] = Task('transform-Carrier','transform-Carrier',CARRIER_SHRINE,carriers)
    return sorted(tasks.values(),key=lambda t:t.id)


def eligible(task, slot, obs, arm, learned=False):
    cls,holding = int(obs.classes[slot]),int(obs.holding[slot])
    fixed_guard = not learned and arm != 'F' and slot >= 3
    if task.kind in ('hold', 'evade'):
        return slot==task.private_slot
    if task.kind in ('collect-trip','special-trip','raid-trip'):
        return cls!=1 and holding==0 and not fixed_guard
    if task.kind=='deliver':
        return cls!=1 and holding>0
    if task.kind=='intercept':
        return learned or cls==1
    if task.kind=='defend':
        return learned or cls==1 or (holding==0 and not fixed_guard)
    if task.kind=='transform-Hunter':
        return cls==0 and (learned or holding==0 and (arm=='F' or fixed_guard))
    if task.kind=='transform-Carrier':
        return cls==0 and (learned or holding==0 and arm=='F')
    return False


def generate_candidates(obs, state, navigation, config, arm='F'):
    learned = config.get('policy_mode') == 'learned'
    maximum = int(config.get('max_tasks',config.get('task_limit',24)))
    if not learned and maximum < 10:
        raise ValueError('max_tasks must reserve five shared and five private slots')
    tasks = all_tasks(obs,state,config)
    if learned:
        # Evading is available to the network even when no enemy is nearby.
        # Its private capacity expresses ownership, never a safety override.
        for i in range(5):
            tasks.append(Task('evade:'+str(int(obs.unit_ids[i])),'evade',int(cells(obs.positions[i])),1,private_slot=i))
    for i in range(5):
        tasks.append(Task('hold:'+str(int(obs.unit_ids[i])),'hold',int(cells(obs.positions[i])),1,private_slot=i))
    m = len(tasks)
    legal = np.zeros((5,m),bool)
    destinations = np.full((5,m),-1,np.int64)
    targets = np.full((5,m),-1,np.int64)
    prior = np.zeros((5,m),np.float32)
    metrics = [[{} for _ in tasks] for _ in range(5)]
    for i,u in enumerate(state.units):
        current = int(cells(obs.positions[i]))
        base_walk = navigation.walkability(state,i,obs.classes[i],obs.holding[i])
        profile = (int(obs.classes[i]),int(obs.holding[i]),1)
        distance,_ = navigation.field(base_walk,[current],profile)
        stores = np.flatnonzero(state.ally_storage).tolist()
        stores = [s for s in stores if learned or s not in u.failed_stores]
        returns,owner = navigation.field(base_walk,stores,profile)
        for j,t in enumerate(tasks):
            walk,d = base_walk,distance
            permitted = eligible(t,i,obs,arm,learned=learned) and t.capacity>0
            if not learned and t.kind.startswith('transform') and np.linalg.norm(obs.enemy_positions-obs.positions[i],axis=-1).min()<.075:
                # A threat interrupt must not recreate the released shrine
                # reservation while the same threat is still present.
                permitted=False
            if t.kind.startswith('transform') and permitted:
                walk = navigation.walkability(state,i,obs.classes[i],obs.holding[i],t.kind,
                                              allow_loaded_transform=learned)
                d,_ = navigation.field(walk,[current],(*profile,t.kind))
            choices = t.targets or (t.source,)
            choices = [c for c in choices if 0<=c<576 and (learned or t.kind!='deliver' or c not in u.failed_stores)]
            goal = min(choices,key=lambda c:(float(d[c]),c)) if choices else -1
            go = float(d[goal]) if goal>=0 else INF
            back,destination = 0.,-1
            if t.kind in ('collect-trip','special-trip','raid-trip') and goal>=0:
                back,destination = float(returns[goal]),int(owner[goal])
            elif t.kind=='deliver':
                destination=goal
            if t.kind in ('hold','evade'):
                go=0.
            legal[i,j] = permitted and go<INF/2 and back<INF/2
            targets[i,j],destinations[i,j] = goal,destination
            target_position = center(goal) if goal>=0 else obs.positions[i]
            enemy_dist = np.linalg.norm(obs.enemy_positions-target_position,axis=-1)*24
            risk = float(np.exp(-enemy_dist.min()/3))
            enemy_low = float(enemy_dist.min()/10.)
            known = np.isfinite(state.enemy_speed)&(state.enemy_speed>.1)
            enemy_high = float((enemy_dist/np.maximum(np.nan_to_num(state.enemy_speed,nan=.1)*.5,.1)).min()) if known.all() else -1.
            eta = (go+back)/max(.15,u.speed)+u.service_time
            same = u.task_id==t.id
            switch = 0. if same or not u.task_id else float(config.get('switch_cost',.75))
            y,x = goal//24,goal%24
            clearance = sum(int(0<=y+dy<24 and 0<=x+dx<24 and base_walk[y+dy,x+dx]) for dx,dy in ((1,0),(-1,0),(0,1),(0,-1))) if goal>=0 else 0.
            r = dict(go=go,back=back,eta=eta,arrival=go/max(.15,u.speed),switch=switch,clearance=clearance,
                     fallback=0.,alternative=0.,enemy_low=enemy_low,enemy_high=enemy_high,risk=risk,
                     failed=goal in u.failed_stores,old_distance=0. if u.source<0 else float(distance[u.source]))
            metrics[i][j] = r
            if not learned:
                prior[i,j] = fixed_prior(t.kind,r,obs,u,same,config)
            if arm!='F' and i>=3 and t.kind=='transform-Hunter':
                prior[i,j] += 20.
        best = min((metrics[i][j]['eta'] for j in range(m) if legal[i,j] and tasks[j].kind!='hold'),default=0.)
        for j in range(m):
            metrics[i][j]['alternative'] = metrics[i][j]['eta']-best
    if learned:
        # Keep the entire legal support. A shortlist whose membership changes
        # with actor weights would invalidate the frozen-support PPO ratio.
        # Task axes are variable in this mode; replay pads with illegal edges.
        selected = [j for j in range(m) if legal[:,j].any()]
        features = build_features(obs,state,tasks,metrics,legal,config)
        return CandidateSet(features[:,selected],prior[:,selected],legal[:,selected],
                            np.array([tasks[j].capacity for j in selected],np.int64),
                            [tasks[j] for j in selected],destinations[:,selected],targets[:,selected],
                            np.zeros(5,bool),float(config.get('temperature',1.)),
                            {'full_tasks':m,'selected_tasks':len(selected),'omitted_best_prior':-30.,
                             'candidate_support':'all_legal','legal_edges':int(legal[:,selected].sum())})
    # Preserve live commitments and at least one reachable delivery edge per
    # carrying unit before optional ranking. No silent mandatory truncation.
    mandatory = set()
    for i,u in enumerate(state.units):
        mandatory.update(j for j,t in enumerate(tasks) if t.id==u.task_id and legal[i,j])
        if obs.holding[i]:
            choices = [j for j,t in enumerate(tasks) if t.kind=='deliver' and legal[i,j]]
            if choices:
                mandatory.add(max(choices,key=lambda j:float(prior[i,j])))
    holds = set(range(m-5,m))
    if len(mandatory|holds)>maximum:
        raise ValueError('required tasks exceed max_tasks: candidate contract failure')
    votes = np.zeros(m,np.float32)
    # Union per-unit best round-trip, contest and safe-return alternatives;
    # aggregate before tie breaking to avoid first-slot ownership of the list.
    for i in range(5):
        choices = [j for j in range(m-5) if legal[i,j]]
        criteria = [lambda j:metrics[i][j]['go']] if arm=='Q0' else [
            lambda j:metrics[i][j]['eta'],
            lambda j:metrics[i][j]['arrival']-metrics[i][j]['enemy_low'],
            lambda j:metrics[i][j]['back']+6*metrics[i][j]['risk']]
        for criterion in criteria:
            ranked = sorted(choices,key=lambda j:(criterion(j),tasks[j].id))[:3]
            for rank,j in enumerate(ranked):
                votes[j] += 3-rank
    tactical = {j for j,t in enumerate(tasks[:-5]) if t.kind.startswith('transform') or t.kind=='special-trip' or (t.kind=='defend' and max(r[j]['risk'] for r in metrics)>.25)}
    optional = [j for j in range(m-5) if j not in mandatory and legal[:,j].any()]
    optional.sort(key=lambda j:(-int(j in tactical),-votes[j],-float(prior[:,j][legal[:,j]].max()),tasks[j].id))
    selected = sorted(mandatory,key=lambda j:tasks[j].id)+optional[:maximum-5-len(mandatory)]
    selected = sorted(selected,key=lambda j:tasks[j].id)+sorted(holds)
    features = build_features(obs,state,tasks,metrics,legal)
    omitted = [j for j in range(m-5) if j not in selected]
    omitted_best = max((float(prior[i,j]) for j in omitted for i in range(5) if legal[i,j]),default=-30.)
    tasks=[tasks[j] for j in selected]
    features,prior,legal = features[:,selected],prior[:,selected],legal[:,selected]
    destinations,targets=destinations[:,selected],targets[:,selected]
    locked=np.zeros(5,bool)
    for i,u in enumerate(state.units):
        match = next((j for j,t in enumerate(tasks) if t.id==u.task_id and legal[i,j]),None)
        if match is not None and (u.commitment>0 or u.kind.startswith('transform')):
            legal[i,:]=False
            legal[i,match]=True
            locked[i]=True
    while len(tasks)<maximum:
        tasks.append(Task('padding:'+str(len(tasks)),'padding',-1,0,visible=False))
    padding=maximum-len(selected)
    if padding:
        features=np.pad(features,((0,0),(0,padding),(0,0)))
        prior=np.pad(prior,((0,0),(0,padding)))
        legal=np.pad(legal,((0,0),(0,padding)))
        destinations=np.pad(destinations,((0,0),(0,padding)),constant_values=-1)
        targets=np.pad(targets,((0,0),(0,padding)),constant_values=-1)
    # This is a surrogate omission diagnostic, not an optimal-game oracle.
    return CandidateSet(features.astype(np.float32),np.clip(prior,-20,20).astype(np.float32),legal,
                        np.array([t.capacity for t in tasks],np.int64),tasks,destinations,targets,locked,
                        float(config.get('temperature',1.)),
                        {'full_tasks':m,'selected_tasks':len(selected),'omitted_best_prior':omitted_best})
