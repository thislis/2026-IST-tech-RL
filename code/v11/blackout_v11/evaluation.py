"""Canonical provided-runner evaluation, preserving raw reward and visible score."""
import hashlib
import time
from pathlib import Path
import numpy as np
import torch
from .io import atomic_json, digest, file_hash, read_json
from .opponents import make_opponent, frozen_catalog
from .behavior import SituationRecorder, decision_metadata


def initial_map_hash(obs):
    # Physical Team A view makes this invariant to candidate side assignment.
    g = np.asarray(obs['unit_0']['graphic'], dtype=np.float32)
    return hashlib.sha256(np.ascontiguousarray(g[..., [1,2,3,6,7,8,9,10]]).tobytes()).hexdigest()


def observation_hash(obs):
    h = hashlib.sha256()
    for n in sorted(obs):
        for k in ('vector','graphic'): h.update(np.asarray(obs[n][k]).tobytes())
    return h.hexdigest()


def evaluate(task, env_factory, heartbeat=lambda **_:None, stop=lambda:False):
    from .export import load_export
    from .windows import visible_windows
    if task.get('candidate_spec'):
        candidate = make_opponent(task['candidate_spec'],task['action_seed'])
    else:
        bundle = Path(task['bundle'])
        for name, expected in task.get('bundle_hashes',{}).items():
            if file_hash(bundle/name)!=expected: raise ValueError('evaluation bundle changed')
        candidate = load_export(bundle, device=task['config'].get('resolved_actor_device','cpu'),
                                planner_device=task['config'].get('resolved_planner_device','cpu'))
    opponent = make_opponent(task['opponent'],task['action_seed']+1)
    torch.manual_seed(task['action_seed'])
    side = task['side']; names = [f'unit_{i}' for i in range(10)]
    own = names[side*5:side*5+5]; other = [n for n in names if n not in own]
    totals = {n:0. for n in names}; start=time.monotonic(); env=None
    deliveries=0; moving=0; last_audit=0.; audits=[]; elapsed_inference=[]
    previous_time=None
    recorder=SituationRecorder.for_task(task)
    try:
        heartbeat(phase='environment_start',steps=0)
        env=env_factory(); obs,_=env.reset(seed=task['requested_seed'])
        if recorder is not None: recorder.start(obs)
        if set(obs)!=set(names): raise ValueError('incomplete initial team observations')
        map_hash=initial_map_hash(obs); obs_hash=observation_hash(obs)
        for step in range(task['config']['max_episode_steps']):
            if stop(): raise InterruptedError('evaluation interrupted')
            now=time.monotonic()
            if now-last_audit>=5:
                heartbeat(phase='evaluate',steps=step)
                if task.get('audit_windows',True):
                    audit=visible_windows(); audits.append(audit)
                    if not audit['verified'] or audit['onscreen_owned_windows']:
                        raise RuntimeError('hidden rendered window audit failed')
                last_audit=now
            v=np.asarray(obs[own[0]]['vector']); current=float(v[95])
            if previous_time is not None and current>previous_time+1e-3:
                raise ValueError('clock reset without terminal')
            previous_time=current
            if not np.isfinite(v).all(): raise ValueError('nonfinite observation')
            begin=time.perf_counter()
            actions=candidate.act({n:obs[n] for n in own})
            candidate_seconds=time.perf_counter()-begin
            elapsed_inference.append(candidate_seconds)
            decision=decision_metadata(candidate,side,np.stack([actions[n] for n in own])) if recorder is not None else None
            begin=time.perf_counter()
            actions.update(opponent.act({n:obs[n] for n in other}))
            opponent_seconds=time.perf_counter()-begin
            if set(actions)!=set(names): raise ValueError('incomplete actions')
            for a in actions.values():
                if a.shape!=(2,) or a.dtype!=np.float32 or not np.isfinite(a).all() or (np.abs(a)>1).any():
                    raise ValueError('invalid action')
            nxt,rewards,terms,truncs,infos=env.step(actions)
            if recorder is not None:
                recorder.observe(step,obs,nxt,actions,rewards,terms,truncs,infos,decision=decision,
                    inference_seconds={side:candidate_seconds,1-side:opponent_seconds})
            if any(set(x)!=set(names) for x in (rewards,terms,truncs)):
                raise ValueError('incomplete transition')
            if any(truncs.values()) or len(set(bool(x) for x in terms.values()))!=1:
                raise ValueError('truncated/partial terminal')
            if not all(np.isfinite(float(x)) for x in rewards.values()):
                raise ValueError('nonfinite raw reward')
            for n in names: totals[n]+=float(rewards[n])
            done=all(terms.values())
            if done != (not env.agents): raise ValueError('termination/active-agent mismatch')
            score_before=v[93:95].tolist(); score_after=None
            if set(nxt)==set(names):
                nv=np.asarray(nxt[own[0]]['vector']); score_after=nv[93:95].tolist()
                before=v[:90].reshape(10,9)[side*5:side*5+5]
                after=nv[:90].reshape(10,9)[side*5:side*5+5]
                moving+=int((np.linalg.norm(after[:,:2]-before[:,:2],axis=1)>1e-5).sum())
                for j,n in enumerate(own):
                    # Explicit proxy, class changes/teleports do not count as deposits.
                    lost=before[j,3:].argmax()>0 and after[j,3:].argmax()==0
                    unchanged=np.argmax(obs[n]['vector'][90:93])==np.argmax(nxt[n]['vector'][90:93])
                    near=np.argwhere(obs[n]['graphic'][...,2]>0)
                    pos=before[j,:2]
                    near_store=bool(len(near) and np.min(np.linalg.norm(np.stack((near[:,1]/96,1-near[:,0]/96),1)-pos,axis=1))<.10)
                    if lost and unchanged and near_store and np.linalg.norm(after[j,:2]-pos)<.2:
                        deliveries+=1
            elif not done: raise ValueError('missing nonterminal observations')
            if done:
                side_totals=[sum(totals[n] for n in set(names[s*5:s*5+5])) for s in (0,1)]
                own_total,opp_total=side_totals[side],side_totals[1-side]
                winner=0 if own_total>opp_total else 1 if own_total<opp_total else None
                score=score_after if score_after is not None else score_before
                score_winner=0 if score[0]>score[1] else 1 if score[0]<score[1] else None
                return dict(valid=True,natural_terminal=True,episode_steps=step+1,
                    winner=winner,runner_winner=winner,runner_reward_totals=[own_total,opp_total],
                    team_a_total_reward=own_total,team_b_total_reward=opp_total,
                    score_before_terminal=score_before,score_after_terminal=score_after,
                    visible_score_winner=score_winner,score_source='after' if score_after is not None else 'before',
                    official_winner=None,official_winner_available=False,
                    terminal_info_raw={n:{k:repr(x) for k,x in infos.get(n,{}).items()} for n in names},
                    side=side,replicate=task['replicate'],opponent=task['opponent'],
                    requested_seed=task['requested_seed'],action_seed=task['action_seed'],
                    initial_map_hash=map_hash,initial_observation_sha256=obs_hash,
                    applied_map_seed_verified=False,objective='provided_runner_v1',
                    delivery_proxy=deliveries,moving_unit_steps=moving,
                    inference_p95_ms=float(np.quantile(elapsed_inference,.95)*1000),
                    wall_seconds=time.monotonic()-start,window_audits=audits[-4:],
                    behavior_log=recorder.close(complete=True) if recorder is not None else {'enabled':False})
            obs=nxt
        raise RuntimeError('evaluation watchdog; incomplete match is not a draw')
    except BaseException as exc:
        if recorder is not None: recorder.close(error=exc)
        raise
    finally:
        if recorder is not None: recorder.close()
        if env is not None: env.close()


def evaluate_bundle(bundle,directory,cfg,replicates=None,split='dev',*,opponents=None,
                    candidate_spec=None,replicates_by_opponent=None,evaluation_scope=None,
                    stop=lambda:False,progress=lambda **_:None):
    from .runtime import run_tasks
    from .statistics import summarize
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    pool=frozen_catalog() if opponents is None else opponents
    pool_ids={s['id'] for s in pool}
    if len(pool_ids)!=len(pool): raise ValueError('duplicate evaluation opponents')
    if replicates_by_opponent is not None:
        if set(replicates_by_opponent)-pool_ids:
            raise ValueError('map counts name an opponent outside the evaluation pool')
        if any(type(value) is not int or value<1 for value in replicates_by_opponent.values()):
            raise ValueError('per-opponent map counts must be positive integers')
    if candidate_spec: pool=[s for s in pool if s['id']!=candidate_spec['id']]
    reps=cfg[split+'_replicates'] if replicates is None else replicates
    if type(reps) is not int or reps<1: raise ValueError('map count must be a positive integer')
    counts={s['id']:(replicates_by_opponent or {}).get(s['id'],reps) for s in pool}
    split_ranges=read_json(Path(__file__).parents[1]/'configs/splits.json')
    bounds=split_ranges[split]
    hashes={n:file_hash(Path(bundle)/n) for n in ('policy.py','checkpoint.pt')} if bundle else {}
    tasks=[]
    for spec in pool:
        for rep in range(counts[spec['id']]):
            h=int(digest([split_ranges['seed'],split,spec['id'],rep])[:12],16)
            for side in (0,1):
                task=dict(id=f"{spec['id']}__r{rep:03d}_s{side}",kind='evaluate',
                    config=cfg,bundle=str(bundle) if bundle else None,bundle_hashes=hashes,
                    candidate_spec=candidate_spec,opponent=spec,side=side,replicate=rep,
                    requested_seed=bounds[0]+h%(bounds[1]-bounds[0]+1),action_seed=h%(2**31-2),
                    split=split,audit_windows=True)
                tasks.append(task)
    plan=dict(tasks=tasks,split=split)
    # The original call path must keep the exact old plan shape so completed
    # baseline/pilot results remain reusable without changing their schedules.
    if replicates_by_opponent is not None: plan['replicates_by_opponent']=counts
    if evaluation_scope is not None: plan['evaluation_scope']=evaluation_scope
    plan_path=directory/'plan.json'
    if plan_path.exists() and read_json(plan_path)!=plan: raise ValueError('evaluation schedule/bundle changed')
    atomic_json(plan_path,plan)
    cells=directory/'cells'; cells.mkdir(exist_ok=True)
    results=[]; pending=[]
    invalid=sum(not read_json(p).get('valid',False) for p in (directory/'attempts').glob('**/attempt.json'))
    for t in tasks:
        p=cells/(t['id']+'.json')
        if p.exists(): results.append(read_json(p))
        else: pending.append(t)
    for retry in range(cfg.get('evaluation_retries',1)+1):
        if not pending: break
        if stop(): raise InterruptedError('evaluation stopped')
        attempt=directory/'attempts'/f'{time.time_ns()}-{retry}'
        rows=run_tasks(pending,attempt,cfg,stop=stop,progress=progress)
        failed=[]
        for row in rows:
            if row['valid'] and row['result'].get('natural_terminal'):
                result=row['result']; atomic_json(cells/(row['id']+'.json'),result); results.append(result)
            else:
                invalid+=1; failed.append(next(t for t in pending if t['id']==row['id']))
        returned={r['id'] for r in rows}
        failed.extend(t for t in pending if t['id'] not in returned)
        pending=failed
    if pending: raise RuntimeError(f'{len(pending)} evaluation cells incomplete; resume retains completed cells')
    report=summarize(results,invalid=invalid)
    report.update(split=split,rows_path=str(cells),bundle_hashes=hashes,
                  requested_map_pairs=next(iter(counts.values())) if len(set(counts.values()))==1 else None,complete=True)
    if replicates_by_opponent is not None or evaluation_scope is not None:
        report.update(requested_map_pairs_by_opponent=counts,requested_games=2*sum(counts.values()),
                      evaluation_scope=evaluation_scope or 'per_opponent_map_counts')
    atomic_json(directory/'summary.json',report)
    return report


def load_rows(report):
    return [read_json(p) for p in sorted(Path(report['rows_path']).glob('*.json'))]
