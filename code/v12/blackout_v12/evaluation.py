"""Paired evaluation with raw reward and separately valid terminal score."""
from concurrent.futures import CancelledError, FIRST_COMPLETED, ProcessPoolExecutor, wait
import multiprocessing
from pathlib import Path
import time
import numpy as np
import torch
from blackout_v11.trajectory import canonical_names, pack
from blackout_v11.evaluation import initial_map_hash, observation_hash
from .execution import compatible_manifest
from .io import ROOT, append, atomic_json, digest, file_hash, read_json
from .map_splits import MapSeedMismatch, MapSplitCollision, register_map
from .trips import TripLedger


def _evaluate_worker(task):
    if task['config'].get('stop_file') and Path(task['config']['stop_file']).exists():
        raise InterruptedError('evaluation stopped before environment start')
    from .env import make_env
    from .opponents import make_opponent
    from .policy import MyPolicy
    torch.set_num_threads(int(task['config'].get('actor_threads',1)))
    torch.manual_seed(task['action_seed'])
    frozen = task.get('candidate_spec')
    candidate = make_opponent(frozen,task['action_seed']) if frozen else MyPolicy(task['config'],arm=task['arm']).eval()
    if task.get('checkpoint'):
        if file_hash(task['checkpoint']) != task['policy_sha256']:
            raise ValueError('evaluation snapshot changed')
        candidate.scorer.load_state_dict(torch.load(task['checkpoint'],map_location='cpu',weights_only=True)['scorer_state'])
    opponent = make_opponent(task['opponent'],task['action_seed']+1)
    side=task['side']; own=canonical_names(side); other=canonical_names(1-side)
    expected=set(own+other); totals=[0.,0.]; env=None; start=time.monotonic(); timings=[]
    ledger=TripLedger(side)
    try:
        env=make_env(task); obs,_=env.reset(seed=task['requested_seed'])
        map_hash=initial_map_hash(obs); obs_hash=observation_hash(obs)
        split_audit=register_map(task['config'],task['split'],map_hash,obs_hash,task['requested_seed'],side)
        for step in range(task['config'].get('max_episode_steps',22000)):
            if time.monotonic()-start > float(task['config'].get('episode_wall_seconds',1800)):
                raise TimeoutError('evaluation wall watchdog; incomplete match is not a draw')
            if task['config'].get('stop_file') and Path(task['config']['stop_file']).exists():
                raise InterruptedError('evaluation interrupted')
            v,g=pack(obs,own)
            mark=time.perf_counter()
            if frozen:
                actions=candidate.act({n:obs[n] for n in own},own)
            else:
                with torch.inference_mode():
                    result=candidate.step(torch.from_numpy(v),torch.from_numpy(g)[None].expand(5,-1,-1,-1),
                                          deterministic=task.get('decoder','map')=='map')
                actions=dict(zip(own,result['action'].cpu().numpy().astype(np.float32)))
                if result['event']: ledger.decision(step,result['decision'])
            timings.append(time.perf_counter()-mark)
            actions.update(opponent.act({n:obs[n] for n in other},other))
            nxt,rewards,terms,truncs,_=env.step(actions)
            if any(set(x)!=expected for x in (rewards,terms,truncs)) or any(truncs.values()) or len(set(map(bool,terms.values())))!=1:
                raise ValueError('incomplete/truncated evaluation transition')
            for s in (0,1): totals[s]+=sum(float(rewards[n]) for n in canonical_names(s))
            if not np.isfinite(totals).all(): raise FloatingPointError('nonfinite raw evaluation reward')
            done=all(terms.values()); nv=None
            if set(nxt)==expected: nv,_=pack(nxt,own)
            elif not done: raise ValueError('nonterminal observations missing')
            reset=nv is not None and float(nv[0,95])>float(v[0,95])+1e-3
            if reset and not done: raise ValueError('unannounced episode reset')
            score_valid=nv is not None and not reset
            ledger.observe(v,nv,step+1,done,score_valid,g)
            if done:
                difference=totals[side]-totals[1-side]
                score=nv[0,93:95].tolist() if score_valid else None
                return dict(valid=True,natural_terminal=True,episode_id=task['id'],arm=task['arm'],
                    opponent=task['opponent'],side=side,replicate=task['replicate'],split=task['split'],
                    requested_seed=task['requested_seed'],initial_map_hash=map_hash,initial_observation_sha256=obs_hash,
                    split_audit=split_audit,action_seed=task['action_seed'],objective='provided_runner_v1',
                    env_steps=step+1,raw_points=(float(np.sign(difference))+1)/2,
                    runner_reward_totals=[totals[side],totals[1-side]],
                    runner_winner=0 if difference>0 else 1 if difference<0 else None,
                    score_points=(float(np.sign(score[0]-score[1]))+1)/2 if score_valid else None,
                    score_valid=score_valid,score_before_terminal=v[0,93:95].tolist(),score_after_terminal=score,
                    score_invalid_reason=None if score_valid else 'missing/reset terminal observation',
                    official_winner_available=False,trip_summary=ledger.report(),
                    inference_p50_ms=float(np.quantile(timings,.5)*1000),inference_p95_ms=float(np.quantile(timings,.95)*1000),
                    wall_seconds=time.monotonic()-start)
            obs=nxt
        raise RuntimeError('evaluation watchdog is not a draw')
    finally:
        if env is not None: env.close()


def summarize(rows):
    opponents=sorted({r['opponent']['id'] for r in rows})
    per={}
    for opponent in opponents:
        cells=[r for r in rows if r['opponent']['id']==opponent]
        valid=[r for r in cells if r.get('valid')]
        score=[r for r in valid if r.get('score_valid')]
        per[opponent]=dict(games=len(cells),failed_games=len(cells)-len(valid),
            raw=float(np.mean([r.get('raw_points',0.) for r in cells])),
            score_conservative=float(np.mean([r.get('score_points') if r.get('score_valid') else 0. for r in cells])),
            score_valid_only=float(np.mean([r['score_points'] for r in score])) if score else None,
            score_valid_games=len(score),sides={str(s):float(np.mean([r.get('raw_points',0.) for r in cells if r['side']==s]))
                for s in (0,1) if any(r['side']==s for r in cells)})
        per[opponent]['score']=per[opponent]['score_conservative']
    families={}
    for opponent,values in per.items():
        spec=next(r['opponent'] for r in rows if r['opponent']['id']==opponent)
        family=spec.get('family','scripted' if opponent.startswith('scripted-') else
                        'direct' if opponent.startswith(('v8-','v9-','v10-')) else 'planner')
        families.setdefault(family,[]).append(values)
    return dict(games=len(rows),failed_games=sum(not r.get('valid') for r in rows),opponents=per,
                opponent_macro_raw=float(np.mean([r['raw'] for r in per.values()])) if per else None,
                opponent_macro_score=float(np.mean([r['score_conservative'] for r in per.values()])) if per else None,
                score_valid_games=sum(r.get('score_valid',False) for r in rows),
                family_macro={f:dict(raw=float(np.mean([v['raw'] for v in values])),
                                    score=float(np.mean([v['score'] for v in values]))) for f,values in families.items()},
                completeness=all(r.get('valid',False) for r in rows),
                official_winner_available=False,judgement='unconfirmed; apply preregistered paired criteria')


def paired_macro_difference(candidate_rows, baseline_rows, metric='raw_points', samples=2000, seed=12):
    """Pair actual map hashes, jointly resampling maps reused by opponents."""
    def index(rows):
        return {(r['opponent']['id'],r['requested_seed'],r['side']):r for r in rows if r.get('valid')}
    candidate,baseline=index(candidate_rows),index(baseline_rows)
    pairs={}; unmatched=0
    for key,a in candidate.items():
        b=baseline.get(key)
        if b is None or a['initial_map_hash']!=b['initial_map_hash']:
            unmatched+=1; continue
        if metric=='score_points' and (not a.get('score_valid') or not b.get('score_valid')):
            unmatched+=1; continue
        pairs.setdefault((key[0],key[1]),[]).append((a['initial_map_hash'],float(a[metric]-b[metric]),key[2]))
    clusters={}
    for (opponent,_),values in pairs.items():
        if {x[2] for x in values}!={0,1} or len({x[0] for x in values})!=1:
            unmatched+=len(values);continue
        map_id=values[0][0]
        clusters.setdefault(map_id,{}).setdefault(opponent,[]).append(float(np.mean([x[1] for x in values])))
    if not clusters: return dict(estimate=None,confidence_interval=None,matched_map_clusters=0,unmatched=unmatched)
    keys=list(clusters); opponents=sorted({o for c in clusters.values() for o in c})
    def macro(chosen):
        values={o:[] for o in opponents}
        for key in chosen:
            for opponent,delta in clusters[key].items(): values[opponent].extend(delta)
        if any(not v for v in values.values()): return None
        return float(np.mean([np.mean(v) for v in values.values()]))
    rng=np.random.default_rng(seed); estimates=[]
    for _ in range(samples):
        value=macro(rng.choice(keys,size=len(keys),replace=True))
        if value is not None: estimates.append(value)
    return dict(estimate=macro(keys),confidence_interval=np.quantile(estimates,[.025,.975]).tolist() if estimates else None,
                matched_map_clusters=len(keys),unmatched=unmatched,opponents=opponents,bootstrap_samples=len(estimates),
                unit='map cluster including both sides and all reused opponents')


def interrupted_row(row):
    """Recognize deliberate stop records without hiding genuine cell failures."""
    if row.get('valid'): return False
    error_type=row.get('type',row.get('error_type'))
    if error_type in {'InterruptedError','CancelledError'}: return True
    return error_type is None and (row.get('interrupted',False) or row.get('error') in {
        'evaluation stopped before environment start','evaluation interrupted'})


def _worker_limit(cfg,workers):
    """The scheduler may lend idle execution slots without changing a cell."""
    capacity=max(int(workers),int(cfg.get('evaluation_worker_capacity',workers)))
    path=cfg.get('worker_allocation_file')
    allocation=read_json(path).get('workers',workers) if path and Path(path).exists() else workers
    return max(1,min(capacity,int(allocation)))


def evaluate(output_dir,cfg,checkpoint,arm,opponents,pairs,split,workers=1):
    if int(workers)<1: raise ValueError('positive evaluation worker count required')
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    cells=output_dir/'cells';cells.mkdir(exist_ok=True)
    splits=cfg.get('splits') or read_json(ROOT/'code/v12/configs/splits.json')
    bounds=splits[split]
    if isinstance(bounds,dict): bounds=bounds['seeds']
    candidate_spec=checkpoint if isinstance(checkpoint,dict) else None
    checkpoint=None if candidate_spec else checkpoint
    policy_hash=file_hash(checkpoint) if checkpoint else None
    tasks=[]
    for opponent in opponents:
        if candidate_spec and candidate_spec['id']==opponent['id']: continue
        count=int(pairs.get(opponent['id'],10) if isinstance(pairs,dict) else pairs)
        for rep in range(count):
            # Same map manifest across opponents and candidates, independent of side.
            h=int(digest(['v12-eval',split,rep])[:12],16)
            for side in (0,1):
                tasks.append(dict(id=f"{opponent['id']}_r{rep:03d}_s{side}",config=cfg,
                    checkpoint=None if checkpoint is None else str(checkpoint),candidate_spec=candidate_spec,
                    policy_sha256=policy_hash,arm=arm,opponent=opponent,side=side,replicate=rep,split=split,
                    requested_seed=bounds[0]+h%(bounds[1]-bounds[0]+1),action_seed=(h+side)%(2**31-1),
                    decoder=cfg.get('decoder','map')))
    plan=dict(tasks=tasks,policy_sha256=policy_hash)
    if (output_dir/'plan.json').exists():
        if not compatible_manifest(read_json(output_dir/'plan.json'),plan):
            raise ValueError('evaluation manifest changed')
    else:
        atomic_json(output_dir/'plan.json',plan)
    rows=[];pending=[]
    def archive(task,error=None,row=None):
        attempt=dict(episode_id=task['id'],interrupted=True,type=type(error).__name__ if error else None,
                     error=str(error) if error else row.get('error'),no_terminal_cell=True)
        if row is not None: attempt['previous_cell']=row
        append(output_dir/'interrupted_attempts.jsonl',attempt)
    def stopped():
        return bool(cfg.get('stop_file') and Path(cfg['stop_file']).exists())
    for task in tasks:
        path=cells/(task['id']+'.json')
        if path.exists():
            row=read_json(path)
            if interrupted_row(row):
                archive(task,row=row)
                path.unlink()
                pending.append(task)
                continue
            if row.get('valid'):
                try:
                    register_map(cfg,split,row['initial_map_hash'],row['initial_observation_sha256'],
                                 row['requested_seed'],row['side'])
                except MapSeedMismatch as exc:
                    row.update(valid=False,error=str(exc),raw_points=0.,score_valid=False,score_points=None,
                               actual_map_pairing_valid=False)
                    atomic_json(path,row)
            rows.append(row)
        else: pending.append(task)
    def accept(task,result=None,error=None):
        row=result if error is None else dict(valid=False,error=str(error),episode_id=task['id'],
            type=type(error).__name__,
            opponent=task['opponent'],side=task['side'],replicate=task['replicate'],split=split,
            requested_seed=task['requested_seed'],raw_points=0.,score_valid=False,score_points=None,
            actual_map_pairing_valid=False if isinstance(error,MapSeedMismatch) else None)
        atomic_json(cells/(task['id']+'.json'),row);rows.append(row)
    interrupted=None
    capacity=max(int(workers),int(cfg.get('evaluation_worker_capacity',workers)))
    if not pending:
        pass
    elif capacity==1:
        for task in pending:
            if stopped():
                interrupted=InterruptedError('evaluation stopped before new cell')
                break
            try: accept(task,_evaluate_worker(task))
            except MapSplitCollision: raise
            except (InterruptedError,CancelledError) as exc:
                archive(task,error=exc); interrupted=exc; break
            except Exception as exc: accept(task,error=exc)
    else:
        with ProcessPoolExecutor(max_workers=capacity,mp_context=multiprocessing.get_context('spawn')) as pool:
            remaining=iter(pending); futures={}; exhausted=False
            try:
                while futures or not exhausted:
                    if stopped() and interrupted is None:
                        interrupted=InterruptedError('evaluation stopped before new cell')
                    if interrupted is not None:
                        for future in futures: future.cancel()
                    while interrupted is None and not exhausted and len(futures)<_worker_limit(cfg,workers):
                        if stopped():
                            interrupted=InterruptedError('evaluation stopped before new cell')
                            break
                        task=next(remaining,None)
                        if task is None: exhausted=True; break
                        futures[pool.submit(_evaluate_worker,task)]=task
                    if not futures: break
                    done,_=wait(futures,timeout=.25,return_when=FIRST_COMPLETED)
                    for future in done:
                        task=futures.pop(future)
                        try: accept(task,future.result())
                        except MapSplitCollision: raise
                        except (InterruptedError,CancelledError) as exc:
                            archive(task,error=exc)
                            interrupted=exc
                        except Exception as exc: accept(task,error=exc)
            except BaseException:
                for future in futures: future.cancel()
                raise
    rows.sort(key=lambda row:row['episode_id'])
    rows_path=output_dir/'rows.json'
    atomic_json(rows_path,rows)
    report=summarize(rows)
    report.update(rows_path=str(rows_path),cells_path=str(cells),rows=rows,split=split,arm=arm,
                  planned_games=len(tasks),pending_games=len(tasks)-len(rows),
                  completeness=len(rows)==len(tasks) and report['completeness'],interrupted=interrupted is not None)
    atomic_json(output_dir/'summary.json',{k:v for k,v in report.items() if k!='rows'})
    if interrupted is not None:
        raise InterruptedError('evaluation interrupted; completed cells retained') from interrupted
    return report
