"""Study description and integrity checks. Standard library only: safe before native import."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT / 'reports/v8/acceleration/registration.json'
DIRECTORY = ROOT / 'logs/v8/accelerated_pilot_v1'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def sources():
    paths = list((ROOT / 'blackout_rl').rglob('*.py')) + list((ROOT / 'scripts').glob('*v8*.py'))
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def validate():
    r = read(REGISTRATION)
    if r['schema'] != 'blackout.v8.accelerated_study.v1' or r['sources'] != sources():
        raise ValueError('v8 accelerated source registration changed; do not overwrite a running study lock')
    for p, h in r['files'].items():
        if sha(ROOT / p) != h:
            raise ValueError('registered study input changed: ' + p)
    return r


def tasks(r):
    result = [dict(id=j['id'] + '-train', kind='train', run=j['id'], config=j['config'], depends=[]) for j in r['runs']]
    references = r['runs'] + [r['planner']]
    for run in references:
        endpoints = r['endpoints'] if run['id'] != 'planner' else [0]
        for step in endpoints:
            for shard, offset in enumerate(range(0, len(r['dev_maps']), r['maps_per_shard'])):
                result.append(dict(id=f"{run['id']}-eval-{step}-{shard}", kind='eval', run=run['id'], config=run['config'],
                    step=step, maps=r['dev_maps'][offset:offset + r['maps_per_shard']],
                    depends=[] if run['id'] == 'planner' else [run['id'] + '-train']))
    return result


def completed(task):
    path = DIRECTORY / 'jobs' / task['id'] / 'done.json'
    if not path.exists():
        return False
    record = read(path)
    if record['registration_sha256'] != sha(REGISTRATION):
        raise ValueError('job belongs to another registration')
    if task['kind'] == 'train':
        run = DIRECTORY / 'runs' / task['run']
        latest = read(run / 'checkpoints/latest.json')
        if read(run / 'status.json')['state'] != 'complete' or latest['sha256'] != record['checkpoint_sha256']:
            raise ValueError('completed training pointer changed')
        if sha(run / 'checkpoints' / latest['path']) != latest['sha256']:
            raise ValueError('completed checkpoint corrupted')
    else:
        if sha(ROOT / record['result']) != record['result_sha256']:
            raise ValueError('completed evaluation result changed')
    return True


def attempt_directory(task_id):
    parent = DIRECTORY / 'jobs' / task_id
    parent.mkdir(parents=True, exist_ok=True)
    for i in range(100000):
        target = parent / f'attempt-{i:04d}'
        try:
            target.mkdir()
            return target
        except FileExistsError:
            continue
    raise RuntimeError('attempt index exhausted')
