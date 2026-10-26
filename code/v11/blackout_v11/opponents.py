"""Pinned league policies, each owning one team's state and independent RNG."""
from functools import lru_cache
import numpy as np
import torch
from .io import ROOT, file_hash, read_json


def family(name):
    if name.startswith('scripted-'): return 'scripted'
    if name.startswith(('v8-', 'v9-', 'v10-')): return 'direct'
    return 'planner'


@lru_cache(maxsize=1)
def _catalog():
    rows = read_json(ROOT/'code/v11/configs/opponents.json')
    if len(rows) != 21 or len({r['id'] for r in rows}) != 21:
        raise ValueError('fixed league requires 21 unique opponents')
    return tuple(dict(kind='league', id=r['id'], family=family(r['id']), model=r) for r in rows)


def frozen_catalog(): return list(_catalog())


def dev_catalog():
    return [s for s in _catalog() if s['id'].startswith(('scripted-', 'v4-', 'v6-', 'v7-', 'v9-', 'v10-'))]


def training_catalog():
    rows = _catalog()
    return dict(planner=[s for s in rows if s['id'].startswith(('v4-', 'v6-', 'v7-'))],
                scripted=[s for s in rows if s['family']=='scripted'],
                direct=[s for s in rows if s['family']=='direct'], self_play=[])


def reference_spec():
    return next(s for s in _catalog() if s['id']=='v4-29794421')


def benchmark_opponent():
    return next(s for s in _catalog() if s['id'].startswith('v6-'))


class LeagueOpponent:
    def __init__(self, spec, seed):
        self.spec, self.seed, self.adapter, self.step = spec, seed, None, 0

    def act(self, obs, names=None):
        from agent_league.adapters import Adapter
        names = sorted(obs if names is None else names, key=lambda n: int(n.split('_')[1]))
        team = int(names[0].split('_')[1])//5
        if self.adapter is None or self.adapter.team != team:
            self.adapter = Adapter(self.spec['model'], team, self.seed, episode='v11')
            self.step = 0
        out = self.adapter.act({n: obs[n] for n in names}, self.step)
        self.step += 1
        return out


class NeuralOpponent:
    def __init__(self, spec, seed):
        from .policy import MyPolicy
        if file_hash(spec['path']) != spec['sha256']:
            raise ValueError('frozen self-play snapshot changed')
        self.policy = MyPolicy().eval()
        payload = torch.load(spec['path'], map_location='cpu', weights_only=True)
        self.policy.load_state_dict(payload['policy_state'])
        self.rng = torch.Generator().manual_seed(seed).get_state()

    @torch.inference_mode()
    def act(self, obs, names=None):
        names = sorted(obs if names is None else names, key=lambda n:int(n.split('_')[1]))
        v = torch.from_numpy(np.stack([obs[n]['vector'] for n in names]))
        g = torch.from_numpy(np.stack([obs[n]['graphic'] for n in names])).permute(0,3,1,2)
        outer = torch.get_rng_state()
        try:
            torch.set_rng_state(self.rng)
            actions = self.policy(v,g).cpu().numpy()
            self.rng = torch.get_rng_state()
        finally: torch.set_rng_state(outer)
        return dict(zip(names, actions))


def make_opponent(spec, seed=0):
    if isinstance(spec, str):
        spec = next(s for s in _catalog() if s['id']==spec)
    if spec['kind']=='league': return LeagueOpponent(spec,seed)
    if spec['kind']=='history': return NeuralOpponent(spec,seed)
    # Small runtime probes use these same public-observation baseline policies.
    from blackout_v10.opponents import StrategyOpponent, HistoricalOpponent
    if spec['kind'] in ('target','scripted'): return HistoricalOpponent(spec['kind'])
    if spec['kind'] not in ('noop','random','collector','rush','raider','defender'):
        raise ValueError('unsupported opponent '+str(spec))
    return StrategyOpponent(spec['kind'], seed)
