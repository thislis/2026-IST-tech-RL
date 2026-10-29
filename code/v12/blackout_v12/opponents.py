"""Frozen historical execution policies plus complete v12 team streams."""
import numpy as np
import torch
from functools import lru_cache
from blackout_v11.opponents import make_opponent as historical_opponent, family
from project_paths import project_path
from .io import ROOT, read_json, file_hash


@lru_cache(maxsize=1)
def frozen_catalog():
    return [dict(kind='league', id=row['id'], family=family(row['id']), model=row)
            for row in read_json(ROOT / 'code/v12/configs/opponents.json')]


def training_catalog():
    rows = frozen_catalog()
    return dict(planner=[s for s in rows if s['id'].startswith(('v4-', 'v6-', 'v7-'))],
                scripted=[s for s in rows if s['family']=='scripted'],
                direct=[s for s in rows if s['family']=='direct'], self_play=[])


def confirmation_catalog():
    names = {'scripted-noop', 'scripted-random', 'scripted-collector'}
    return [row for row in frozen_catalog() if row['id'] in names or row['id'].startswith(('v4-', 'v6-'))]


def missing_assets(catalog=None):
    missing = []
    for spec in frozen_catalog() if catalog is None else catalog:
        model = spec.get('model', spec)
        paths = list(model.get('assets', {}))
        if model.get('path'):
            paths.append(model['path'])
        for value in dict.fromkeys(paths):
            path = project_path(value, root=ROOT)
            if not path.is_file():
                missing.append(dict(opponent=spec['id'], path=str(path)))
    return missing


def dev_catalog():
    catalog = frozen_catalog()
    ids = ['scripted-noop', 'scripted-random', 'scripted-collector', 'scripted-rush', 'scripted-weak']
    rows = [next(s for s in catalog if s['id'] == name) for name in ids]
    rows += [next(s for s in catalog if s['id'].startswith(prefix)) for prefix in ('v4-', 'v6-', 'v10-')]
    return rows


class SelfOpponent:
    def __init__(self, spec, seed):
        from .policy import MyPolicy
        if spec.get('sha256') and file_hash(spec['path']) != spec['sha256']:
            raise ValueError('self-play snapshot changed')
        payload = torch.load(spec['path'], map_location='cpu', weights_only=True)
        self.policy = MyPolicy(payload.get('config'), arm=payload.get('arm', 'F')).eval()
        self.policy.scorer.load_state_dict(payload['scorer_state'])
        self.rng = torch.Generator().manual_seed(seed).get_state()

    @torch.inference_mode()
    def act(self, obs, names=None):
        names = sorted(obs if names is None else names, key=lambda n:int(n.split('_')[-1]))
        vectors = torch.from_numpy(np.stack([obs[n]['vector'] for n in names]))
        graphics = torch.from_numpy(np.stack([obs[n]['graphic'] for n in names])).permute(0,3,1,2)
        outer = torch.get_rng_state()
        try:
            torch.set_rng_state(self.rng)
            action = self.policy.step(vectors, graphics, deterministic=False)['action'].cpu().numpy()
            self.rng = torch.get_rng_state()
        finally:
            torch.set_rng_state(outer)
        return dict(zip(names, action))


def make_opponent(spec, seed=0):
    return SelfOpponent(spec, seed) if isinstance(spec, dict) and spec.get('kind') == 'v12' else historical_opponent(spec, seed)
