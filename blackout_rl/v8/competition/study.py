"""Separate immutable registration for the unchanged-game competition branch."""
import json
import importlib.metadata
from pathlib import Path
from blackout_rl.v8.checkpoints import atomic_json,fingerprints
from blackout_rl.v8.contracts import file_hash
from blackout_rl.v8.provided_environment import ROOT,verify_original,LAUNCHER_EXECUTABLE

DIRECTORY=ROOT/'logs/v8/provided_competition_v1'
REGISTRATION=ROOT/'reports/v8/competition/registration.json'
CONFIG=ROOT/'configs/v8/competition/study.json'
PACKAGES=('torch','numpy','protobuf','mlagents-envs','grpcio','blackout-env')


def read(path):return json.loads(Path(path).read_text())


def prepare():
    if (DIRECTORY/'jobs').exists() or (DIRECTORY/'runs').exists():raise RuntimeError('attempted study cannot be re-registered')
    verify_original();cfg=read(CONFIG)
    training=cfg['training']
    if (cfg['seeds']!=[11,22,33] or not 1<=cfg['workers']<=6 or
        not cfg['endpoints'] or cfg['endpoints']!=sorted(set(cfg['endpoints'])) or cfg['endpoints'][-1]!=training['steps'] or
        training['rollout']<1 or training['save_every']<1 or training['save_every']%training['rollout'] or
        any(step%training['save_every'] for step in cfg['endpoints']) or
        cfg['evaluation_replicates']<1 or cfg['replicates_per_shard']<1 or
        training['schedule'][-1]['until']<=training['steps'] or
        not training['side_cycle'] or not set(training['side_cycle'])<={0,1}):
        raise ValueError('invalid registered budget/checkpoint/worker/evaluation settings')
    protected=[CONFIG,LAUNCHER_EXECUTABLE,ROOT/'blackout_last_4_pages.md',ROOT/'contracts/v8/submission_contract.json',
               ROOT/'scripts/run_v8_fast.sh',ROOT/'checkpoints/win_70_vs_scripted.pt',
               ROOT/'reports/v8/environment_restoration/restoration.json']
    registration=dict(schema='blackout.v8.provided_competition.v1',sources=fingerprints(),
                      files={str(p.relative_to(ROOT)):file_hash(p) for p in protected},config=cfg,
                      packages={name:importlib.metadata.version(name) for name in PACKAGES},
                      environment='unchanged provided BlackOutEnv and original app',
                      trained=False,live_validation_performed=False)
    atomic_json(REGISTRATION,registration);return registration


def validate():
    r=read(REGISTRATION)
    if r['schema']!='blackout.v8.provided_competition.v1' or r['sources']!=fingerprints():raise ValueError('competition source registration changed')
    for name,h in r['files'].items():
        if file_hash(ROOT/name)!=h:raise ValueError('registered input changed: '+name)
    if r['packages']!={name:importlib.metadata.version(name) for name in PACKAGES}:raise ValueError('registered package version changed')
    verify_original();return r


def run_config(r,run):
    cfg=dict(r['config']['training']);cfg.update(seed=int(run.split('_s')[1]),arm=run.split('_s')[0])
    cfg['opponent']=dict(path=str(ROOT/'checkpoints/win_70_vs_scripted.pt'),sha256=r['files']['checkpoints/win_70_vs_scripted.pt'],
        source_sha256={p:h for p,h in r['sources'].items() if p.startswith('blackout_rl/') and '/v8/' not in p})
    return cfg


def tasks(r):
    cfg=r['config'];runs=[f'{arm}_s{seed}' for arm in ('c1','flat') for seed in cfg['seeds']]
    jobs=[dict(id=run+'-train',kind='train',run=run,depends=[]) for run in runs]
    for run in runs+['planner']:
        for step in cfg['endpoints'] if run!='planner' else [0]:
            for start in range(0,cfg['evaluation_replicates'],cfg['replicates_per_shard']):
                jobs.append(dict(id=f'{run}-eval-{step}-{start}',kind='eval',run=run,step=step,
                    replicates=list(range(start,min(start+cfg['replicates_per_shard'],cfg['evaluation_replicates']))),
                    depends=[] if run=='planner' else [run+'-train']))
    return jobs


def completed(job):
    p=DIRECTORY/'jobs'/job['id']/'done.json'
    if not p.exists():return False
    done=read(p)
    if done['registration_sha256']!=file_hash(REGISTRATION):raise ValueError('foreign completion record')
    if file_hash(ROOT/done['artifact'])!=done['artifact_sha256']:raise ValueError('completion artifact changed')
    if job['kind']=='train':
        run=DIRECTORY/'runs'/job['run']
        if read(run/'status.json')['state']!='complete' or read(run/'checkpoints/latest.json')['sha256']!=done['artifact_sha256']:
            raise ValueError('completed training pointer changed')
    return True


def attempt(job):
    parent=DIRECTORY/'jobs'/job['id'];parent.mkdir(parents=True,exist_ok=True)
    for i in range(100000):
        path=parent/f'attempt-{i:04d}'
        try:path.mkdir();return path
        except FileExistsError:pass
    raise RuntimeError('attempt limit')
