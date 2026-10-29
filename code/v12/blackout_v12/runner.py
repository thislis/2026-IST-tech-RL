"""Portable detached experiment CLI with immutable registration and safe probes."""
import argparse
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
import uuid
from .io import ROOT, atomic_json, digest, file_hash, read_json
from .runtime import FileLease, child_environment, process_options

CONFIG = ROOT / 'code/v12/configs/default.json'


def load_config(path=CONFIG, scope=None, device=None):
    from .opponents import frozen_catalog, missing_assets
    from project_paths import project_path
    cfg = read_json(path)
    cfg['scope'] = scope or cfg.get('scope', 'available')
    if cfg['scope'] not in ('available', 'full'):
        raise ValueError('scope must be available or full')
    if device:
        cfg['device'] = device
    if int(cfg.get('max_tasks', 24)) < 10 or int(cfg.get('event_period', 25)) < 1:
        raise ValueError('invalid max_tasks/event_period')
    catalog = frozen_catalog()
    missing = missing_assets(catalog)
    missing_ids = sorted({row['opponent'] for row in missing})
    if cfg['scope'] == 'full' and missing:
        raise FileNotFoundError('Full frozen league assets are missing: ' + ', '.join(missing_ids) +
                                '; use --scope available for the installed opponents')
    available = [row for row in catalog if row['id'] not in missing_ids]
    for spec in available:
        for value, expected in spec['model'].get('assets', {}).items():
            if file_hash(project_path(value, root=ROOT)) != expected:
                raise ValueError('Frozen opponent asset hash changed: ' + value)
    if not available:
        raise RuntimeError('No runnable opponent is available')
    cfg.update(opponents=available, missing_opponent_ids=missing_ids,
               missing_opponent_assets=missing,
               splits=read_json(ROOT / 'code/v12/configs/splits.json'))
    return cfg


def directory_for(cfg, name=None):
    name = name or cfg['experiment']
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', name) or name in ('.', '..'):
        raise ValueError('Use a simple experiment name (letters, numbers, _, -, .)')
    return ROOT / 'logs/v12' / name


def registration(cfg):
    suffixes = {'.py', '.json', '.sh', '.ps1', '.yaml', '.yml', '.lock'}
    sources = {p.relative_to(ROOT).as_posix(): file_hash(p)
               for p in sorted((ROOT / 'code').rglob('*'))
               if p.is_file() and p.suffix in suffixes and '__pycache__' not in p.parts}
    packages = {}
    for name in ('torch', 'numpy', 'onnx', 'onnxruntime', 'mlagents-envs', 'blackout-env'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return dict(schema='v12.registration.1', config=cfg, source_sha256=sources,
                packages=packages, python=sys.version, executable=sys.executable,
                feature_schema='blackout.v12.features.1', state_schema='blackout.v12.state.1',
                objective=cfg['objective'], source_policy='resume refuses any registered change')


def prepare_registration(directory, cfg):
    actual = registration(cfg)
    path = Path(directory) / 'registration.json'
    if path.exists():
        from .resume import effective_registration
        if effective_registration(directory) != actual:
            raise ValueError('Registered source/config/packages changed; choose a new --name')
    else:
        atomic_json(path, actual)
    return digest(actual)


def is_active(directory):
    path = Path(directory) / 'experiment.lock'
    if not path.exists():
        return False
    try:
        with FileLease(path, blocking=False):
            return False
    except RuntimeError:
        return True


def status(directory):
    directory = Path(directory)
    path = directory / 'status.json'
    state = read_json(path) if path.exists() else dict(state='not_started')
    active = is_active(directory)
    if not active and state.get('state') in ('running', 'starting', 'stopping'):
        state = dict(state, state='interrupted', reason='supervisor is no longer holding its lease')
    return dict(directory=str(directory), active=active, status=state,
                stop_requested=(directory / 'stop.request').exists(),
                console_log=str(directory / 'console.log'),
                progress={str(p.parent.relative_to(directory)): read_json(p)
                          for p in directory.glob('**/progress.json')})


def dry_run(cfg, directory):
    from .study import plan
    from .env import executable
    from .runtime import hardware_identity, budget
    hardware = hardware_identity()
    return dict(schema='v12.dry_run.1', directory=str(directory), scope=cfg['scope'],
                opponent_ids=[row['id'] for row in cfg['opponents']],
                missing_opponent_ids=cfg['missing_opponent_ids'],
                plan=plan(cfg), hardware=hardware, resource_bounds=budget(cfg, hardware),
                game_build=str(executable(cfg)), game_build_present=executable(cfg).is_file(),
                unity_started=False, training_started=False,
                limitation='available scope cannot certify frozen-v4/v6 noninferiority or the full 21-opponent league')


def check(cfg):
    """Small CPU contract/probability checks; no export, Unity, or PPO update."""
    import torch
    from .assignment import log_partition, map_assignment, joint_log_prob
    from .benchmark import fixture
    from .policy import MyPolicy
    torch.set_num_threads(1)
    scores = torch.tensor([[1.2,.7,-.4,0.],[.9,1.1,.2,0.],[-.3,.6,1.4,0.]], dtype=torch.double)
    caps = torch.tensor([1,2,1,3])
    legal = torch.ones(3,4,dtype=torch.bool)
    legal[0,2] = legal[2,0] = False
    partition = log_partition(scores, caps, legal, .7)
    if abs(partition.item()-6.5988516569678035) > 1e-10:
        raise AssertionError('29-assignment fixture logZ mismatch')
    choice = map_assignment(scores, caps, legal, .7)
    logp = joint_log_prob(scores, choice, caps, legal, .7)
    policy = MyPolicy(cfg, arm='Q2').eval()
    vector, graphic = fixture()
    with torch.inference_mode():
        first = policy.step(vector, graphic, deterministic=True)
        solves = policy.navigation.solves
        vector[:,95] -= .02/420
        second = policy.step(vector, graphic, deterministic=True)
    parameters = sum(p.numel() for p in policy.scorer.parameters())
    if parameters != 6273 or choice.tolist() != [0,1,2] or not torch.isfinite(logp):
        raise AssertionError('scorer/assignment contract mismatch')
    if not first['event'] or second['event'] or policy.navigation.solves != solves:
        raise AssertionError('maintenance call recomputed a decision/path')
    if first['action'].shape != (5,2) or not torch.isfinite(first['action']).all():
        raise AssertionError('public action contract mismatch')
    return dict(passed=True, actor_parameters=parameters, log_partition=partition.item(),
                maintenance_skip=True, scope=cfg['scope'],
                opponent_ids=[s['id'] for s in cfg['opponents']],
                missing_opponent_ids=cfg['missing_opponent_ids'],
                unity_started=False, training_started=False, onnx_sequence_certified=False)


def supervise(args, cfg, directory):
    from .benchmark import apply_profile, calibrate
    from .runtime import process_tree_guard
    directory = Path(directory)
    stop_file = directory / 'stop.request'
    operation = 'benchmark' if args.benchmark else 'study'
    started = time.monotonic()
    state = {}
    last_write = 0.
    def progress(**data):
        nonlocal last_write
        state.update(data)
        now = time.monotonic()
        if now-last_write >= 1:
            atomic_json(directory / 'status.json', dict(state,
                state='stopping' if stop_file.exists() else 'running', operation=operation,
                pid=os.getpid(), elapsed_seconds=now-started, updated_at=time.time()))
            print(json.dumps(dict(state, elapsed_seconds=round(now-started,1)), ensure_ascii=False), flush=True)
            last_write = now
        if stop_file.exists():
            raise InterruptedError('stop requested')
    signal.signal(signal.SIGTERM, lambda *_: stop_file.touch())
    signal.signal(signal.SIGINT, lambda *_: stop_file.touch())
    with FileLease(directory / 'experiment.lock', blocking=False), \
         FileLease(ROOT / 'logs/runtime/compute.lock', blocking=False), process_tree_guard():
        try:
            signature = prepare_registration(directory, cfg)
            atomic_json(directory / 'supervisor.json', dict(pid=os.getpid(), token=args.start_token,
                        operation=operation, registration_sha256=signature))
            progress(phase='preflight')
            atomic_json(directory / 'preflight.json', check(cfg))
            runtime_path = directory / 'runtime_profile.json'
            prepared_path=directory/'resume_preparation.json'
            if prepared_path.exists():
                runtime_path=directory/read_json(prepared_path)['revision']/'runtime_profile.json'
            profile = read_json(runtime_path) if runtime_path.exists() else calibrate(directory, cfg, progress)
            if operation == 'benchmark':
                result = dict(runtime_profile=str(runtime_path), synthetic_only=True,
                              unity_started=False, training_started=False)
            else:
                from .study import run_study
                resolved = apply_profile(cfg, profile)
                resolved.update(stop_file=str(stop_file), map_split_registry=str(directory / 'map_splits.json'))
                atomic_json(directory / 'resolved_config.json', resolved)
                result = run_study(directory, resolved, stop_file.exists, progress)
            atomic_json(directory / 'status.json', dict(state='complete', operation=operation,
                        result=result, elapsed_seconds=time.monotonic()-started, updated_at=time.time()))
        except InterruptedError as exc:
            stop_file.touch()
            atomic_json(directory / 'status.json', dict(state='stopped', operation=operation,
                        reason=str(exc), updated_at=time.time()))
        except BaseException as exc:
            stop_file.touch()
            atomic_json(directory / 'status.json', dict(state='failed', operation=operation,
                        error=str(exc), traceback=traceback.format_exc(), updated_at=time.time()))
            raise
        finally:
            (directory / 'supervisor.json').unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description='v12 experiments; detached rendered Unity study by default')
    group = parser.add_mutually_exclusive_group()
    for flag in ('check', 'dry-run', 'benchmark', 'status', 'stop', 'resume', 'resume-check', 'prepare-resume', 'prepare-scale'):
        group.add_argument('--'+flag, action='store_true')
    parser.add_argument('--name')
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--scope', choices=('available','full'))
    parser.add_argument('--device', choices=('auto','cpu','cuda','mps'))
    parser.add_argument('--env-workers',type=int,help='global worker budget for an explicit prepared runtime revision')
    parser.add_argument('--tuning-evidence',type=Path,help='isolated resource plan/evidence for --prepare-resume')
    parser.add_argument('--foreground', action='store_true', help='keep benchmark/study attached to this terminal')
    parser.add_argument('--supervisor', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--start-token', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    # Status and cooperative stop work even when an optional asset disappeared.
    if args.status or args.stop:
        base = read_json(args.config)
        directory = directory_for(base, args.name)
        if args.stop:
            if is_active(directory):
                (directory/'stop.request').touch()
                print('Stop requested; complete checkpoints are retained. Use --status to confirm shutdown.')
            else:
                print('v12 is not running.')
        else:
            print(json.dumps(status(directory), ensure_ascii=False, indent=2))
        return
    cfg = load_config(args.config.resolve(), args.scope, args.device)
    directory = directory_for(cfg, args.name)
    if args.prepare_scale:
        if args.tuning_evidence is None:
            parser.error('--prepare-scale requires --tuning-evidence')
        from .resume import prepare_scale
        print(json.dumps(prepare_scale(directory,args.tuning_evidence),ensure_ascii=False,indent=2))
        return
    if args.prepare_resume:
        if args.env_workers is None or args.tuning_evidence is None:
            parser.error('--prepare-resume requires --env-workers and --tuning-evidence')
        from .resume import prepare
        print(json.dumps(prepare(directory,args.env_workers,args.tuning_evidence),ensure_ascii=False,indent=2))
        return
    prepared=(directory/'resume_preparation.json').exists()
    if prepared and (args.resume or args.resume_check or args.supervisor):
        from .resume import effective_registration
        cfg=copy.deepcopy(effective_registration(directory)['config'])
        if ((args.scope and args.scope!=cfg['scope']) or
                (args.device and args.device!=cfg.get('device','auto'))):
            raise ValueError('prepared resume does not accept unregistered scope/device changes')
    if args.resume_check:
        from .resume import check as check_resume
        print(json.dumps(check_resume(directory),ensure_ascii=False,indent=2))
        return
    if args.check or args.dry_run:
        print(json.dumps(check(cfg) if args.check else dry_run(cfg,directory), ensure_ascii=False, indent=2))
        return
    directory.mkdir(parents=True, exist_ok=True)
    if args.supervisor:
        supervise(args, cfg, directory)
        return
    if shutil.disk_usage(ROOT).free < cfg.get('disk_reserve_gib',8)*1024**3:
        raise RuntimeError('Insufficient disk reserve to start')
    with FileLease(directory/'launch.lock', blocking=False):
        if is_active(directory):
            raise RuntimeError('Experiment is already active; inspect --status')
        previous = read_json(directory/'status.json') if (directory/'status.json').exists() else {}
        if previous and not args.resume:
            raise ValueError('Existing experiment found; use --resume or a new --name')
        if args.resume and previous.get('state') == 'complete':
            print('Experiment already completed: '+str(directory))
            return
        if args.resume and previous.get('operation') == 'benchmark':
            args.benchmark = True
        # Check the global budget before acknowledging a detached startup.
        with FileLease(ROOT/'logs/runtime/compute.lock', blocking=False):
            prepare_registration(directory, cfg)
        if args.resume and cfg.get('tune_on_resume'):
            # This path runs ONLY when the user explicitly resumes. Preparing
            # the revision or checking readiness never launches Unity.
            from .throughput_probe import main as probe
            from .resume import activate_tuning
            print('Measuring 4/6/8 Unity workers before resume; no training during this short probe.',flush=True)
            measured=probe(['--study',directory.name,'--workers',
                *map(str,cfg['resume_tuning_candidates']),'--seconds',str(cfg['resume_tuning_seconds'])])
            cfg=activate_tuning(directory,measured['summary_path'])
            prepare_registration(directory,cfg)
        (directory/'stop.request').unlink(missing_ok=True)
        token = uuid.uuid4().hex
        args.start_token = token
        if args.foreground:
            supervise(args,cfg,directory)
            return
        operation = 'benchmark' if args.benchmark else 'study'
        command = [sys.executable,'-u',str(ROOT/'code/v12/scripts/v12_experiments.py'),
                   '--config',str(args.config.resolve()),'--name',directory.name,
                   '--scope',cfg['scope'],'--device',cfg.get('device','auto'),
                   '--supervisor','--start-token',token]
        if args.benchmark:
            command.append('--benchmark')
        atomic_json(directory/'status.json',dict(state='starting',operation=operation,updated_at=time.time()))
        with (directory/'console.log').open('ab',buffering=0) as log:
            child = subprocess.Popen(command,cwd=ROOT,env=child_environment(),stdin=subprocess.DEVNULL,
                                     stdout=log,stderr=log,**process_options(detached=True))
        for _ in range(200):
            ready = directory/'supervisor.json'
            supervisor = read_json(ready) if ready.exists() else {}
            if supervisor.get('token') == token:
                print(f'v12 {operation} started in background (PID {supervisor["pid"]}).\nLogs: {directory}\n'
                      f'Status: --name {directory.name} --status\nStop: --name {directory.name} --stop')
                return
            if child.poll() is not None:
                if child.returncode == 0:
                    print(json.dumps(status(directory),ensure_ascii=False,indent=2))
                    return
                raise RuntimeError('Background startup failed; inspect '+str(directory/'console.log'))
            time.sleep(.1)
        raise RuntimeError('Background startup pending; inspect --status before retrying')


if __name__ == '__main__':
    main()
