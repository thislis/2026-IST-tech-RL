import hashlib
import json
import os
from pathlib import Path
from project_paths import project_root, project_path

ROOT = project_root()
BASE = ROOT / 'logs/shared/agent_league'
DEFAULTS = dict(rounds=5, seed=20261016, workers=4, threads=1, device='cpu',
                max_steps=22000, episode_seconds=1800, stall_seconds=120,
                memory_gib=36, reserve_gib=15, chunk_steps=512, snapshot_every=1000,
                time_scale=50., movement_epsilon=1e-5, teleport_distance=.2,
                stop_epsilon=1e-6, near_base_distance=.12, direction_change_cos=.70710678,
                probe_seconds=30, retries=1)

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def resolve(path):
    return project_path(path, root=ROOT)

def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))

def sources():
    # Pin the actual inference dependencies, including historical resolution tables.
    suffixes = {'.py', '.sh', '.json', '.yaml', '.lock'}
    return {relative(p): sha(p) for p in sorted((ROOT/'code').rglob('*'))
            if p.is_file() and p.suffix in suffixes and '__pycache__' not in p.parts}

def task_id(a, b, rep, side):
    return f'{a}__{b}__r{rep:02d}_s{side}'

def schedule(models, cfg):
    ids = sorted(m['id'] for m in models)
    tasks = []
    for rep in range(cfg['rounds']):
        for i, a in enumerate(ids):
            for b in ids[i+1:]:
                base = int(digest([cfg['seed'], a, b, rep])[:8], 16) % (2**31-3)
                for side in (0, 1):
                    tasks.append(dict(id=task_id(a,b,rep,side), a=a, b=b, repeat=rep,
                                      side=side, action_seeds=[base,base+1], environment_seed=base+2))
    return tasks

def verify_assets(spec):
    for path, expected in spec.get('assets', {}).items():
        if sha(resolve(path)) != expected: raise ValueError('Model asset changed: '+path)
