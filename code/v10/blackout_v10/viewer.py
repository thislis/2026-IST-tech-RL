"""One-line visible exhibition, separately supervised from training/evaluation."""
import argparse
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import signal
import time
import torch
from .io import ROOT, atomic_json, file_hash, read_json
from .contracts import config, verify_original
from .export import export_policy, verify_export, load_export
from .policy import MyPolicy
from .watch_adapters import available_models, load_opponent, verify_spec
from .runtime import run_tasks, check_legacy_processes
from .evaluation import ObservedPolicy


def candidate_bundle(study, checkpoint=None):
    if checkpoint:
        source=Path(checkpoint).expanduser().resolve()
        if source.name.startswith(('.tmp', 'pending')): raise ValueError('checkpoint is not an atomic completed snapshot')
        before_hash=file_hash(source)
        payload=torch.load(source,map_location='cpu',weights_only=True)
        if file_hash(source)!=before_hash: raise ValueError('checkpoint changed while loading')
        policy=MyPolicy().eval(); policy.load_state_dict(payload['policy_state'],strict=True)
        if payload.get('arm') not in ('C0','C1','C1-blind'): raise ValueError('v10 snapshot required')
        sha=file_hash(source); bundle=ROOT/'artifacts/submission/v10'/sha
        if not bundle.exists(): export_policy(policy,bundle)
        verify_export(bundle,policy)
        return bundle,dict(version='v10',run=f"{payload['arm']}_s{payload['seed']}",step=payload['step'],
                           checkpoint_sha256=sha,snapshot_explicit=True)
    from .runner import directory_for
    path=directory_for(config(),study)/'selection_lock.json'
    if not path.exists(): raise FileNotFoundError('No selected v10 yet; finish the study or use --checkpoint PATH to an atomic saved snapshot')
    lock=read_json(path); bundle=Path(lock['bundle'])
    if file_hash(bundle/'checkpoint.pt')!=lock['checkpoint_sha256']: raise ValueError('selection snapshot changed')
    verify_export(bundle)
    return bundle,dict(version='v10',run=lock['run'],step=lock.get('step'),checkpoint_sha256=lock['checkpoint_sha256'],snapshot_explicit=False)


def visible_env(speed):
    from blackout_env import BlackOutEnv
    return BlackOutEnv(env_path=str(ROOT/'artifacts/builds/BlackOut.app'),no_graphics=False,time_scale=speed)


def play_task(task, heartbeat=lambda **_:None, stop=lambda:False, env_factory=visible_env):
    from .match import validated_match as run_match
    for name, checksum in task['bundle_hashes'].items():
        if file_hash(Path(task['bundle'])/name)!=checksum: raise ValueError('viewer actor changed')
    actor=ObservedPolicy(load_export(task['bundle']),task['config']['max_episode_steps'],heartbeat,stop)
    other=load_opponent(task['opponent_spec'])
    torch.manual_seed(task['action_seed']); env=None; start=time.monotonic()
    try:
        heartbeat(phase='visible_environment_start',steps=0)
        env=env_factory(task['speed'])
        with torch.inference_mode(): result=run_match(env,actor,other,swap_teams=bool(task['side']))
        return dict(asdict(result),valid=True,side=task['side'],models=task['model_labels'],
                    wall_seconds=time.monotonic()-start,exhibition_only=True,training_data=False,
                    game_outcome_verified=False)
    finally:
        if env is not None: env.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',default='mapcritic_v1')
    parser.add_argument('--opponent',choices=('v9','v8','v6','v7-1','all'),default='v9')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--games',type=int,default=2)
    parser.add_argument('--speed',type=float,default=1.)
    parser.add_argument('--list',action='store_true'); parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    if args.games<1 or not 0<args.speed<=50: parser.error('games >= 1 and 0 < speed <= 50 required')
    torch.set_num_threads(1)
    models=available_models()
    if args.list:
        print(json.dumps(models,ensure_ascii=False,indent=2)); return
    verify_original()
    requested=list(models) if args.opponent=='all' else [args.opponent]
    selected=[]
    for key in requested:
        spec=models[key]
        if not spec['available']:
            if args.opponent!='all': raise FileNotFoundError(spec['error'])
            print(f"Skipping unavailable {key}: {spec['error']}",flush=True); continue
        # Validate all sources before launching a visible game.
        load_opponent(spec); selected.append(spec)
    if not selected: raise FileNotFoundError('no available historical opponent')
    bundle, candidate=candidate_bundle(args.study,args.checkpoint)
    if args.check:
        print(json.dumps(dict(passed=True,candidate=candidate,opponents=selected,unity_started=False,
                              live_watch_verified=False),ensure_ascii=False,indent=2)); return
    lock_path=ROOT/'logs/runtime/compute.lock'; lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('experiment/viewer owns the worker budget; retry after it stops')
        check_legacy_processes()
        session=ROOT/'logs/v10/watch'/str(time.time_ns()); session.mkdir(parents=True)
        atomic_json(session/'manifest.json',dict(candidate=candidate,opponents=selected,availability=models))
        stop_requested=False
        def request_stop(*_):
            nonlocal stop_requested
            stop_requested=True
        signal.signal(signal.SIGINT,request_stop); signal.signal(signal.SIGTERM,request_stop)
        cfg=dict(config(),workers=1,episode_wall_seconds=max(1800,1800/args.speed))
        results=[]
        for spec in selected:
            for game in range(args.games):
                if stop_requested: break
                side=game%2; labels={'Blue':candidate if side==0 else spec,'Red':spec if side==0 else candidate}
                print(json.dumps(dict(game=game+1,**labels),ensure_ascii=False),flush=True)
                task=dict(id=f"{spec['version']}-{game:03d}",kind='watch',config=cfg,speed=args.speed,
                    side=side,action_seed=151000+game,bundle=str(bundle),
                    bundle_hashes={n:file_hash(bundle/n) for n in ('policy.py','checkpoint.pt')},
                    opponent_spec=spec,model_labels=labels)
                rows=run_tasks([task],session,cfg,lambda:stop_requested)
                results.extend(rows)
                if stop_requested: break
                if not rows or not rows[0]['valid']:
                    raise RuntimeError('exhibition failed; inspect '+str(session))
                print(json.dumps(rows[0]['result'],ensure_ascii=False),flush=True)
        atomic_json(session/'summary.json',dict(watch_ready=bool(results) and all(r['valid'] for r in results),
                    interrupted=stop_requested,exhibition_only=True,results=results))
        print('관전 종료. 학습·공식 평가 자료에는 포함하지 않았습니다. '+str(session),flush=True)


if __name__=='__main__': main()
