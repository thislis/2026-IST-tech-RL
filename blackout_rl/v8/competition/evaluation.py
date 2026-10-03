"""Use the provided run_match verbatim; watchdogs invalidate, never award games."""
from dataclasses import asdict
import hashlib
import importlib.util
from pathlib import Path
import time
import sys
from blackout_env import load_checkpoint,run_match
from blackout_rl.v8.opponents import make_opponent
from blackout_rl.v8.checkpoints import atomic_json
from blackout_rl.v8.telemetry import append
from blackout_rl.v8.contracts import digest


class HistoricalOpponent:
    def __init__(self,kind,config):self.kind,self.config=kind,config;self.policy=None
    def act(self,obs):
        names=list(obs)
        if self.policy is None:
            team=int(names[0].split('_')[1])//5
            self.policy,self.hash=make_opponent(self.kind,team,self.config)
        return self.policy.act(obs,names)


class ObservedPolicy:
    def __init__(self,model,max_steps,progress=None,stop=lambda:False):
        self.model=model;self.limit=max_steps;self.calls=0;self.first_hash=None;self.progress=progress;self.stop=stop
    def act(self,obs):
        if self.stop():raise InterruptedError('evaluation stopped; whole shard remains incomplete')
        if self.calls>=self.limit:raise RuntimeError('match watchdog; invalid attempt, not a draw')
        if self.first_hash is None:
            h=hashlib.sha256()
            for name in sorted(obs):
                h.update(name.encode());h.update(obs[name]['vector'].tobytes());h.update(obs[name]['graphic'].tobytes())
            self.first_hash=h.hexdigest()
        self.calls+=1
        if self.progress is not None:self.progress(self.calls)
        return self.model.act(obs)


def load_submitted(bundle):
    bundle=Path(bundle)
    spec=importlib.util.spec_from_file_location('v8_submitted',bundle/'policy.py')
    module=importlib.util.module_from_spec(spec)
    previous=sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode=True;spec.loader.exec_module(module)
    finally:sys.dont_write_bytecode=previous
    return load_checkpoint(module.MyPolicy,str(bundle/'checkpoint.pt'),state_dict_key='policy_state',device='cpu',vector_size=96,n_channels=11)


def evaluate(bundle,cfg,replicates,env_factory,directory,*,planner=False,progress=None,stop=lambda:False,lock=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);episodes=[]
    lock=lock or {'scope':'synthetic_fixture'};lock_sha=digest(lock)
    if (directory/'model_lock.json').exists():raise FileExistsError('evaluation attempt already exists')
    atomic_json(directory/'model_lock.json',dict(lock=lock,sha256=lock_sha))
    for replicate in replicates:
        for side in (0,1):
            cell=dict(replicate=replicate,side=side,lock_sha256=lock_sha)
            append(directory/'attempts.jsonl',dict(event='start',**cell))
            # No seed injection, hidden reset, private cache access or env wrappers.
            env=env_factory()
            try:
                candidate=HistoricalOpponent('scripted',cfg['opponent']) if planner else load_submitted(bundle)
                candidate=ObservedPolicy(candidate,cfg['max_episode_steps'],progress,stop)
                target=HistoricalOpponent('target',cfg['opponent'])
                started=time.monotonic();result=run_match(env,candidate,target,swap_teams=bool(side))
                row=dict(**cell,status='valid',**asdict(result),wall_seconds=time.monotonic()-started,initial_observation_sha256=candidate.first_hash)
                episodes.append(row);append(directory/'attempts.jsonl',dict(event='finish',**row))
            except BaseException as exc:
                append(directory/'attempts.jsonl',dict(event='invalid',**cell,error=str(exc)));raise
            finally:env.close()
    result=dict(metric='provided_run_match_reward_total_W_over_N',episodes=episodes,n=len(episodes),
                lock_sha256=lock_sha,
                wins=sum(x['winner']==0 for x in episodes),draws=sum(x['winner'] is None for x in episodes),
                actual_map_seeds_verified=False,replicates_are_not_paired_maps=True)
    atomic_json(directory/'result.json',result);return result
