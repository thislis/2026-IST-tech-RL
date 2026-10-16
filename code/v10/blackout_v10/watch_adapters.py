"""Historical execution contracts are confined to exhibition opponents."""
from pathlib import Path
from project_paths import project_path
from .io import ROOT, file_hash, read_json
from .export import load_export

MANIFEST = ROOT/'code/v10/configs/watch_models.json'


def available_models():
    result = {}
    for key, spec in read_json(MANIFEST)['models'].items():
        try:
            verify_spec(spec)
            row = dict(spec, available=True, error=None)
        except (FileNotFoundError, ValueError) as exc:
            row = dict(spec, available=False, error=str(exc))
        result[key] = row
    return result


def verify_spec(spec):
    if spec['kind']=='submission':
        base=project_path(spec['bundle'],root=ROOT)
        for name, checksum in spec['files'].items():
            if file_hash(base/name)!=checksum: raise ValueError('historical bundle changed: '+str(base/name))
    else:
        path=project_path(spec['checkpoint'],root=ROOT)
        if file_hash(path)!=spec['sha256']: raise ValueError('legacy checkpoint changed: '+str(path))
    return spec


class LegacyAdapter:
    def __init__(self, version):
        from tools import watch_best_models as old
        self.models, self.planner = old.load_models()  # strict checkpoint and historical source checks
        self.version=version; self.policy=None

    def act(self, obs):
        from blackout_rl.mappo_v6 import V6Policy, ROLES
        names=sorted(obs,key=lambda n:int(n.split('_')[1]))
        team=int(names[0].split('_')[1])//5
        if self.policy is None:
            self.policy=V6Policy(self.models[0 if self.version=='v6' else 1],team=team)
            if self.version=='v6':
                self.policy.context.planner=self.planner(team,roles=ROLES,chase_radius_cells=48)
            self.policy.reset()
        return self.policy.act(obs,names)


def load_opponent(spec):
    verify_spec(spec)
    if spec['kind']=='submission': return load_export(project_path(spec['bundle'],root=ROOT))
    return LegacyAdapter(spec['version'])
