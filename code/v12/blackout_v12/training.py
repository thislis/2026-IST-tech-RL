"""Independent CPU episode waves and one batched, optional GPU learner."""
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
import multiprocessing
from pathlib import Path
import numpy as np
import torch
from .collector import run_episode
from .critic import Critic
from .execution import compatible_manifest, semantic_config
from .io import ROOT, append, atomic_json, atomic_torch, cpu_state, digest, file_hash, read_json, state_hash
from .rewards import gae_episode
from .ppo import update


TENSOR_FIELDS = ('features','prior','legal','capacities','assignment','temperature',
                 'old_log_prob','critic_features')


def run_wave(tasks, workers):
    """No optimizer can run until every submitted episode has finished."""
    def stopped():
        return any(task['config'].get('stop_file') and Path(task['config']['stop_file']).exists()
                   for task in tasks)
    def check_stop():
        if stopped():
            raise InterruptedError('collection stopped before new episode')
    if workers == 1:
        # Inline CPU actors own their RNG just as spawned workers do. Reusing
        # an already-complete cached episode must not alter learner shuffles.
        with torch.random.fork_rng(devices=[]):
            results=[]
            for task in tasks:
                check_stop()
                results.append(run_episode(task))
            return results
    results = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        pending=iter(tasks); jobs={}
        try:
            while True:
                check_stop()
                while len(jobs)<workers:
                    check_stop()
                    task=next(pending,None)
                    if task is None: break
                    jobs[pool.submit(run_episode,task)]=task
                if not jobs: break
                done,_=wait(jobs,timeout=.25,return_when=FIRST_COMPLETED)
                for future in done:
                    jobs.pop(future)
                    results.append(future.result())
        except BaseException:
            for future in jobs: future.cancel()
            raise
    return sorted(results,key=lambda r:r['episode_id'])


def make_batch(episodes, critic, gamma=1., lam=.9975):
    records, advantages, returns = [], [], []
    if not episodes:
        raise ValueError('empty PPO episode wave')
    device = next(critic.parameters()).device
    for episode in episodes:
        if not episode.get('natural_terminal') or not episode.get('on_policy'):
            raise ValueError('PPO accepts only complete on-policy episodes')
        events = episode['events']
        if not events or any(not e.get('on_policy') for e in events):
            raise ValueError('PPO accepts only nonempty on-policy event sequences')
        event_steps=[int(e['step']) for e in events]
        durations=[int(e['duration']) for e in events]
        if (event_steps[0] != 0 or any(b<=a for a,b in zip(event_steps,event_steps[1:]))
                or durations != np.diff(event_steps+[int(episode['env_steps'])]).tolist()
                or any(k<1 for k in durations)):
            raise ValueError('event intervals must cover exactly one complete natural episode')
        with torch.no_grad():
            features = torch.stack([e['critic_features'] for e in events]).to(device)
            values = torch.cat([critic(chunk) for chunk in features.split(256)]).cpu().numpy()
        adv, ret = gae_episode([e['reward'] for e in events], values,
                              [e['duration'] for e in events], gamma, lam)
        advantages.extend(adv); returns.extend(ret); records.extend(events)
    # Learned support contains every legal task, so only the task axis can vary
    # between immutable events. Padding never adds an eligible assignment.
    task_counts = [int(torch.as_tensor(e['features']).shape[-2]) for e in records]
    tasks = max(task_counts)
    task_axes = {'features': 1, 'prior': 1, 'legal': 1, 'capacities': 0}
    batch = {}
    for key in TENSOR_FIELDS:
        values = [torch.as_tensor(e[key]) for e in records]
        if key not in task_axes or min(task_counts) == tasks:
            batch[key] = torch.stack(values)
            continue
        axis = task_axes[key]
        shape = list(values[0].shape)
        shape[axis] = tasks
        padded = values[0].new_zeros((len(records), *shape))
        for row, (value, count) in enumerate(zip(values, task_counts)):
            if value.shape[axis] != count:
                raise ValueError('inconsistent immutable event task axes')
            target = [slice(None)] * value.ndim
            target[axis] = slice(0, count)
            padded[(row, *target)] = value
        batch[key] = padded
    batch.update(advantages=torch.tensor(np.asarray(advantages),dtype=torch.float32),
                 returns=torch.tensor(np.asarray(returns),dtype=torch.float32),
                 on_policy=torch.tensor([e['on_policy'] for e in records],dtype=torch.bool))
    return batch


def _families(cfg):
    from .opponents import frozen_catalog
    rows = cfg.get('opponents') or frozen_catalog()
    if isinstance(rows,dict):
        rows = rows.get('catalog',list(rows.values()))
    catalog = {'planner':[], 'scripted':[], 'direct':[]}
    for row in rows:
        name = row['id']
        family = 'direct' if name.startswith(('v8-','v9-','v10-')) else 'scripted' if name.startswith('scripted-') else 'planner'
        catalog[family].append(row)
    return catalog


def train(output_dir, cfg, steps, seed, arm, initial_checkpoint=None, workers=1, learner_device='cpu'):
    """A budget is a lower stop boundary; the last natural wave may exceed it."""
    from .policy import MyPolicy
    from .runtime import learner_lease
    output_dir = Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    cfg = dict(cfg)
    learner_device=str(learner_device)
    data_collection=arm=='Q2' and cfg.get('data_split')=='data'
    wave_size=(min(int(workers),int(cfg.get('data_collection_workers',workers))) if data_collection
               else int(semantic_config(cfg)['rollout_episodes_per_wave']))
    if int(steps)<1 or int(workers)<1 or wave_size<1:
        raise ValueError('positive episode budget, worker count and rollout wave size required')
    nested = cfg.get('ppo', {})
    for key,value in nested.items():
        cfg.setdefault({'clip':'ppo_clip','epochs':'ppo_epochs'}.get(key,key),value)
    torch.set_num_threads(int(cfg.get('learner_threads',1)))
    torch.manual_seed(int(seed))
    policy = MyPolicy(cfg, arm=arm)
    scorer = policy.scorer.to(learner_device)
    critic = Critic().to(learner_device)
    actor_opt = torch.optim.Adam(scorer.parameters(), lr=float(cfg.get('actor_lr',1e-4)))
    critic_opt = torch.optim.Adam(critic.parameters(), lr=float(cfg.get('critic_lr',3e-4)))
    counters = dict(env_steps=0,discarded_env_steps=0,events=0,wave=0,episodes=0,
                    family_steps={f:0 for f in ('planner','scripted','self_play','direct')},
                    family_assigned={f:0 for f in ('planner','scripted','self_play','direct')},side_steps=[0,0])
    latest = output_dir/'latest.pt'
    identity=dict(seed=int(seed),arm=arm,budget=int(steps),
                  initial=None if initial_checkpoint is None else file_hash(initial_checkpoint))
    signature = digest(dict(identity,config=semantic_config(cfg)))
    if latest.exists():
        payload = torch.load(latest,map_location='cpu',weights_only=True)
        if payload['signature'] != signature:
            previous=payload.get('config')
            if (previous is None or payload['signature']!=digest(dict(identity,config=previous))
                    or semantic_config(previous)!=semantic_config(cfg)):
                raise ValueError('training configuration/initialization changed; use a new run directory')
        scorer.load_state_dict(payload['scorer_state']); critic.load_state_dict(payload['critic_state'])
        actor_opt.load_state_dict(payload['actor_optimizer']); critic_opt.load_state_dict(payload['critic_optimizer'])
        counters = payload['counters']; torch.set_rng_state(payload['rng_state'])
        if learner_device.startswith('cuda') and payload.get('cuda_rng_states'):
            torch.cuda.set_rng_state_all(payload['cuda_rng_states'])
    elif initial_checkpoint:
        payload = torch.load(initial_checkpoint,map_location='cpu',weights_only=True)
        scorer.load_state_dict(payload['scorer_state'])
        if 'critic_state' in payload:
            critic.load_state_dict(payload['critic_state'])
    counters.setdefault('discarded_env_steps',0)
    families = _families(cfg)
    base_weights = cfg.get('train_family_weights',dict(planner=.4,scripted=.3,self_play=.2,direct=.1))
    weights = {f:w for f,w in base_weights.items() if f=='self_play' or families.get(f)}
    if not weights or any(float(w)<0 for w in weights.values()) or sum(weights.values())<=0:
        raise ValueError('training requires a nonnegative, nonempty opponent family mixture')
    weights={f:float(w) for f,w in weights.items() if w>0}
    total_weight=sum(weights.values()); weights={f:w/total_weight for f,w in weights.items()}
    split_file = ROOT/'code/v12/configs/splits.json'
    splits = cfg.get('splits') or (read_json(split_file) if split_file.exists() else {'train':[100000,999999]})
    bounds = splits[cfg.get('data_split','train')]
    if isinstance(bounds,dict): bounds=bounds['seeds']
    try:
        relative = output_dir.resolve().relative_to((ROOT/'logs/v12').resolve())
    except ValueError:
        relative = Path(output_dir.name+'-'+digest(str(output_dir.resolve()))[:12])
    data_root = ROOT/'artifacts/data/v12'/relative
    data_scheduling=dict(target_episode_batch_size=wave_size,seed_serial_source='committed episode count',
        batch_audit_filename='data_schedule.json',batch_root=str(data_root),
        dispatch_log=str(output_dir/'data_dispatches.jsonl'),
        allocation='opponent and side targets use actual steps after each completed batch; '
                   'changing data batch size can change future opponent and side assignments',
        ppo_updates=False) if data_collection else None
    while counters['env_steps'] < steps:
        if cfg.get('stop_file') and Path(cfg['stop_file']).exists():
            raise InterruptedError('training stopped before new wave')
        wave = counters['wave']
        directory = data_root/f'wave_{wave:05d}'
        directory.mkdir(parents=True,exist_ok=True)
        snapshot = directory/'behavior.pt'
        snapshot_payload = dict(scorer_state=cpu_state(scorer),config=cfg,arm=arm,seed=int(seed),wave=wave)
        if snapshot.exists():
            previous=torch.load(snapshot,map_location='cpu',weights_only=True)
            if (state_hash(previous['scorer_state']) != state_hash(snapshot_payload['scorer_state'])
                    or not compatible_manifest({k:previous.get(k) for k in ('config','arm','seed','wave')},
                                               {k:snapshot_payload[k] for k in ('config','arm','seed','wave')})):
                raise ValueError('immutable behavior snapshot differs from resumed learner')
        else:
            atomic_torch(snapshot,snapshot_payload)
        policy_hash=file_hash(snapshot)
        history=dict(kind='v12',id=f'v12-{arm}-wave{wave}',path=str(snapshot),sha256=policy_hash,family='self_play')
        plan_path=directory/'plan.json'
        previous_plan=read_json(plan_path) if plan_path.exists() else None
        # A stopped data batch keeps its original size and immutable tasks.
        # The larger runtime target applies only after that batch commits.
        batch_size=len(previous_plan['tasks']) if data_collection and previous_plan is not None else wave_size
        if batch_size<1:
            raise ValueError('immutable episode wave must contain tasks')
        tasks=[]; reserved={f:0 for f in weights}; side_reserved=[0,0]
        for index in range(batch_size):
            total=sum(counters['family_steps'].values())+sum(reserved.values())+10000
            family=max(weights,key=lambda f:weights[f]*total-counters['family_steps'][f]-reserved[f])
            pool=[history] if family=='self_play' else families[family]
            opponent=pool[counters['family_assigned'][family]%len(pool)]
            counters['family_assigned'][family]+=1; reserved[family]+=10000
            side=min((0,1),key=lambda s:counters['side_steps'][s]+side_reserved[s]); side_reserved[side]+=10000
            serial=counters['episodes']+index if data_collection else wave*wave_size+index
            h=int(digest(['v12',seed,serial])[:12],16)
            task=dict(id=f'w{wave:05d}_{index:02d}',directory=str(directory/f'episode_{index:02d}'),
                      config=cfg,side=side,opponent=opponent,family=family,arm=arm,
                      split=cfg.get('data_split','train'),
                      checkpoint=str(snapshot),policy_sha256=policy_hash,
                      requested_seed=bounds[0]+h%(bounds[1]-bounds[0]+1),action_seed=h%(2**31-1),
                      deterministic=arm.startswith('Q'))
            tasks.append(task)
        wave_plan=dict(tasks=tasks,policy_sha256=policy_hash)
        if previous_plan is not None:
            if not compatible_manifest(previous_plan,wave_plan):
                raise ValueError('immutable episode wave manifest changed')
            if data_collection:
                tasks=previous_plan['tasks']
        else:
            atomic_json(plan_path,wave_plan)
        if data_collection:
            audit=dict(schema='v12.data_schedule.v1',wave=wave,episode_count=len(tasks),
                       episode_serial_start=counters['episodes'],
                       episode_serial_end_exclusive=counters['episodes']+len(tasks),
                       plan_sha256=file_hash(plan_path),policy_sha256=policy_hash)
            audit_path=directory/'data_schedule.json'
            if audit_path.exists():
                if read_json(audit_path)!=audit:
                    raise ValueError('immutable data episode serial range changed')
            else:
                atomic_json(audit_path,audit)
            append(output_dir/'data_dispatches.jsonl',dict(audit,workers=int(workers),
                target_episode_batch_size=wave_size,reused_plan=previous_plan is not None,
                execution_revision=cfg.get('execution_revision')))
        summaries=run_wave(tasks,max(1,int(workers)))
        if cfg.get('stop_file') and Path(cfg['stop_file']).exists():
            raise InterruptedError('training stopped before optimizer commit')
        discarded_steps=0
        for task in tasks:
            audit=Path(task['directory'])/'discarded_attempts.jsonl'
            if audit.exists():
                discarded_steps+=sum(int(json.loads(line)['env_steps']) for line in audit.read_text(encoding='utf-8').splitlines() if line)
        episodes=[torch.load(row['trajectory'],map_location='cpu',weights_only=True) for row in summaries]
        if any(ep['policy_sha256']!=policy_hash for ep in episodes):
            raise ValueError('mixed behavior snapshots in wave')
        if len(episodes)!=len(tasks) or len({e['episode_id'] for e in episodes})!=len(tasks):
            raise ValueError('incomplete or duplicate episode wave')
        if not arm.startswith('Q'):
            with learner_lease(learner_device):
                batch=make_batch(episodes,critic,cfg.get('gamma',1.),cfg.get('gae_lambda',.9975))
                metrics=update(scorer,critic,actor_opt,critic_opt,batch,arm,cfg)
        else:
            metrics={'updates':0,'events':sum(len(e['events']) for e in episodes),'no_learning':True}
        for row in summaries:
            counters['env_steps']+=row['env_steps']; counters['events']+=row['event_count']; counters['episodes']+=1
            counters['family_steps'][row['family']]+=row['env_steps']; counters['side_steps'][row['side']]+=row['env_steps']
        counters['discarded_env_steps']+=discarded_steps
        counters['wave']+=1
        checkpoint=dict(schema='v12.training.v1',signature=signature,scorer_state=cpu_state(scorer),
                        critic_state=cpu_state(critic),actor_optimizer=actor_opt.state_dict(),critic_optimizer=critic_opt.state_dict(),
                        counters=counters,rng_state=torch.get_rng_state(),
                        cuda_rng_states=torch.cuda.get_rng_state_all() if learner_device.startswith('cuda') else [],
                        config=cfg,arm=arm,seed=int(seed))
        atomic_torch(latest,checkpoint)
        for row in summaries:
            append(output_dir/'episodes.jsonl',row)
        append(output_dir/'updates.jsonl',dict(wave=wave,metrics=metrics,counters=counters))
        atomic_json(output_dir/'progress.json',dict(counters,requested_steps=steps,
                    actual_exposure_steps=counters['env_steps']+counters['discarded_env_steps'],
                    overshoot=max(0,counters['env_steps']-steps),family_target=weights,
                    **({'data_scheduling':data_scheduling} if data_collection else {})))
    result=dict(checkpoint=str(latest),arm=arm,seed=int(seed),data_path=str(data_root),**counters)
    if data_collection:
        result['data_scheduling']=data_scheduling
    atomic_json(output_dir/'result.json',result)
    return result


run_training = train
