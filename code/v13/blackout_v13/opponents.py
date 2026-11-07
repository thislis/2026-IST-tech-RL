"""Pinned teacher and league adapters; no network or game process on import."""
import importlib.util
import sys
from functools import lru_cache
import numpy as np
import torch
from .io import ROOT,read_json,file_hash
from .contracts import KINDS


@lru_cache(maxsize=1)
def catalog():
    from agent_league.new_game import selected_roster
    # Reuse registered portable dependencies, not the historical pre-relocation paths.
    return selected_roster()['models']


class Teacher:
    def __init__(self):
        bundle=ROOT/'artifacts/submission/v12/scripted9_before_learned_latest'
        # The same frozen two-file package used by the league; avoid importing live v12 policy.
        spec=importlib.util.spec_from_file_location('_v13_frozen_teacher',bundle/'policy.py')
        module=importlib.util.module_from_spec(spec)
        previous=sys.dont_write_bytecode
        try:
            sys.dont_write_bytecode=True;spec.loader.exec_module(module)
        finally:sys.dont_write_bytecode=previous
        self.model=module.MyPolicy()
        payload=torch.load(bundle/'checkpoint.pt',map_location='cpu',weights_only=True)
        self.model.load_state_dict(payload['policy_state'])
        self.model.eval()

    def reset(self):
        self.model.reset()

    @torch.no_grad()
    def act(self,vector,graphic):
        action=self.model(torch.from_numpy(vector),torch.from_numpy(graphic)).numpy()
        units=self.model.runtime.units
        labels=np.array([[u.destination if u.kind=='deliver' else u.source,
                           KINDS.index(u.kind) if u.kind in KINDS else 8] for u in units],np.int64)
        return action,labels


def make_opponent(spec,side,seed,episode):
    if spec.get('kind')=='v13':return SnapshotOpponent(spec,side)
    from agent_league.adapters import Adapter
    return Adapter(spec,side,seed,episode,device='cpu')


@lru_cache(maxsize=4)
def snapshot_model(path,checksum):
    from .policy import Policy
    if file_hash(path)!=checksum:raise ValueError('self-play snapshot changed')
    payload=torch.load(path,map_location='cpu',weights_only=True)
    with torch.random.fork_rng(devices=[]):policy=Policy(payload['config']).eval()
    policy.load_state_dict(payload['policy'])
    return torch.jit.script(policy)


class SnapshotOpponent:
    def __init__(self,spec,side):
        from .contracts import initial_state
        self.model=snapshot_model(spec['path'],spec['sha256'])
        self.state=initial_state();self.side=side

    @torch.no_grad()
    def act(self,obs,step):
        from .env import pack
        v,g,names=pack(obs,self.side)
        result=self.model(torch.from_numpy(v),torch.from_numpy(g),torch.arange(5),*self.state)
        self.state=result[1:]
        return dict(zip(names,result[0].numpy()))
