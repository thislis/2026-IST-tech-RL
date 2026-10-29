"""Explicit, detached scripted-nine fine tuning; checks never launch Unity."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import uuid

from .io import ROOT, atomic_json, file_hash, read_json
from .runtime import FileLease, child_environment, process_options

CONFIG = ROOT / 'code/v12/configs/finetune_scripted.json'
ENTRY = ROOT / 'code/v12/scripts/finetune_scripted.py'


def configuration(path):
    from .opponents import frozen_catalog
    base = read_json(ROOT / 'code/v12/configs/default.json')
    overrides = read_json(path)
    ppo = dict(base['ppo'], **overrides.get('ppo', {}))
    base.update(overrides)
    base['ppo'] = ppo
    base['opponents'] = [row for row in frozen_catalog() if row['id'].startswith('scripted-')]
    base['splits'] = overrides.get('splits') or read_json(ROOT / 'code/v12/configs/splits.json')
    return base


def available_memory_gib():
    if os.name == 'nt':
        class MemoryStatus(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong),
                        *[(key, ctypes.c_ulonglong) for key in
                          ('total', 'available', 'page', 'page_available',
                           'virtual', 'virtual_available', 'extended')]]
        value = MemoryStatus()
        value.length = ctypes.sizeof(value)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
            return value.available / 1024**3
    elif Path('/proc/meminfo').exists():
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / 1024**2
    return None


def preflight(cfg, checkpoint):
    """Readiness only: no policy forward, optimizer, simulator, or probe."""
    import torch
    from .env import executable
    if not checkpoint.is_file():
        raise FileNotFoundError('Warm-start checkpoint missing: ' + str(checkpoint))
    if cfg.get('initial_checkpoint_sha256') and file_hash(checkpoint) != cfg['initial_checkpoint_sha256']:
        raise ValueError('Frozen warm-start checkpoint changed')
    if not executable(cfg).is_file():
        raise FileNotFoundError('Game build missing: ' + str(executable(cfg)))
    if cfg['device'] == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; fix the CUDA environment or explicitly select --device cpu')
    for key in ('workers', 'max_steps', 'eval_interval', 'dev_pairs', 'confirmation_pairs'):
        if isinstance(cfg[key], bool) or int(cfg[key]) != cfg[key] or int(cfg[key]) < 1:
            raise ValueError(key + ' must be a positive integer')
    if cfg['workers'] > 16:
        raise ValueError('This measured configuration supports at most 16 parallel workers')
    if len(cfg['opponents']) != 9 or len({r['id'] for r in cfg['opponents']}) != 9:
        raise ValueError('Exactly nine distinct scripted opponents are required')
    if not 0 < float(cfg['time_scale']) <= 50:
        raise ValueError('time_scale must be in (0, 50]')
    memory = available_memory_gib()
    required = float(cfg['memory_reserve_gib']) + cfg['workers'] * float(cfg['worker_memory_gib'])
    if memory is not None and memory < required:
        raise RuntimeError(f'Insufficient available RAM: {memory:.1f} GiB; need approximately {required:.1f} GiB including reserve')
    if shutil.disk_usage(ROOT).free < float(cfg['disk_reserve_gib']) * 1024**3:
        raise RuntimeError('Insufficient free disk space for the configured reserve')
    # Reject a wrong or corrupted warm start before acknowledging detached launch.
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if payload.get('arm') != 'F' or not {'scorer_state', 'critic_state'} <= payload.keys():
        raise ValueError('Expected an F-arm training checkpoint with scorer and critic')
    from .policy import MyPolicy
    from .critic import Critic
    MyPolicy(cfg, arm='F').scorer.load_state_dict(payload['scorer_state'])
    Critic().load_state_dict(payload['critic_state'])
    return dict(ready=True, unity_started=False, training_started=False,
                initial_checkpoint=str(checkpoint), device=cfg['device'], workers=cfg['workers'],
                policy_mode=cfg.get('policy_mode', 'residual'),
                parent_env_steps=cfg.get('parent_env_steps', 0),
                visible_training_windows=1 if cfg.get('visible_training', True) else 0,
                background_training_workers=cfg['workers'] - int(cfg.get('visible_training', True)),
                time_scale=cfg['time_scale'], available_memory_gib=memory,
                max_steps=cfg['max_steps'], dev_games_per_round=18 * cfg['dev_pairs'],
                confirmation_games=18 * cfg['confirmation_pairs'],
                opponent_ids=[row['id'] for row in cfg['opponents']],
                tuning_evidence='logs/v12/resource_tuning_20261026_162623/scale_summary.json',
                limitation='16 workers was the best prior Q2 collection probe; the new training workload has not been benchmarked')


def registration(cfg, checkpoint):
    from .runner import registration as source_registration
    sources = source_registration(cfg)
    if cfg.get('policy_mode') == 'learned':
        from .features import LEARNED_FEATURE_SCHEMA
        sources['feature_schema'] = LEARNED_FEATURE_SCHEMA['id']
    prior = Path(cfg['prior_map_registry']) if cfg.get('prior_map_registry') else None
    if prior is not None and not prior.is_absolute():
        prior = ROOT / prior
    return dict(schema='v12.scripted_finetune.launch.1',
                initial_checkpoint=str(checkpoint), initial_sha256=file_hash(checkpoint),
                prior_map_registry_sha256=file_hash(prior) if prior else None,
                registration=sources)


def prepare_map_registry(directory, cfg, manifest):
    """Reject physical maps already observed anywhere in the source study."""
    target = directory / 'map_splits.json'
    if target.exists() or not cfg.get('prior_map_registry'):
        return
    prior = Path(cfg['prior_map_registry'])
    if not prior.is_absolute():
        prior = ROOT / prior
    if file_hash(prior) != manifest['prior_map_registry_sha256']:
        raise ValueError('Source study map registry changed')
    previous = read_json(prior)
    atomic_json(target, dict(schema='v12.actual_maps.1',
        maps={key:dict(split='prior_study', observations=[], seeds=[])
              for key in previous['maps']}, collisions=[], seed_maps={},
        excluded_prior_registry=str(prior), excluded_prior_sha256=file_hash(prior)))


def state(directory):
    from .runner import is_active
    result = read_json(directory / 'status.json') if (directory / 'status.json').exists() else {'state': 'not_started'}
    active = is_active(directory)
    if not active and result.get('state') in ('starting', 'running', 'stopping'):
        result = dict(result, state='interrupted')
    progress = read_json(directory / 'progress.json') if (directory / 'progress.json').exists() else None
    return dict(directory=str(directory), active=active, status=result, progress=progress,
                stop_requested=(directory / 'stop.request').exists(), console_log=str(directory / 'console.log'))


def worker(directory):
    from .finetune import run
    manifest = read_json(directory / 'finetune_registration.json')
    cfg = manifest['registration']['config']
    stop_file = directory / 'stop.request'
    signal.signal(signal.SIGINT, lambda *_: stop_file.touch())
    signal.signal(signal.SIGTERM, lambda *_: stop_file.touch())
    try:
        prepare_map_registry(directory, cfg, manifest)
        result = run(directory, cfg, Path(manifest['initial_checkpoint']),
                     workers=cfg['workers'], learner_device=cfg['device'], seed=cfg['seed'])
        from .export import export_checkpoint
        bundle = ROOT / 'artifacts/submission/v12' / directory.name
        result['submission'] = export_checkpoint(result['checkpoint'], bundle, arm='F', decoder='map')
        atomic_json(directory / 'result.json', result)
        atomic_json(directory / 'status.json', dict(state='complete', result=result, updated_at=time.time()))
    except InterruptedError as exc:
        stop_file.touch()
        atomic_json(directory / 'status.json', dict(state='stopped', reason=str(exc), updated_at=time.time()))
    except BaseException as exc:
        stop_file.touch()
        atomic_json(directory / 'status.json', dict(state='failed', error=str(exc),
                    traceback=traceback.format_exc(), updated_at=time.time()))
        raise


def supervise(directory, token):
    from .runtime import managed_popen, process_tree_guard
    stop_file = directory / 'stop.request'
    signal.signal(signal.SIGINT, lambda *_: stop_file.touch())
    signal.signal(signal.SIGTERM, lambda *_: stop_file.touch())
    with FileLease(directory / 'experiment.lock', blocking=False), \
         FileLease(ROOT / 'logs/runtime/compute.lock', blocking=False), process_tree_guard():
        # The guarded child and its spawn/Unity descendants inherit one Windows
        # Job. A killed supervisor therefore cannot leave sixteen orphan games.
        try:
            atomic_json(directory / 'status.json', dict(state='running', pid=os.getpid(), updated_at=time.time()))
            child = managed_popen([sys.executable, '-u', str(ENTRY), '--name', directory.name, '--worker'],
                cwd=ROOT, env=child_environment(), stdin=subprocess.DEVNULL, **process_options())
            atomic_json(directory / 'supervisor.json', dict(pid=os.getpid(), worker_pid=child.pid, token=token))
            stopped_at = None
            cfg = read_json(directory / 'finetune_registration.json')['registration']['config']
            while child.poll() is None:
                if stop_file.exists():
                    stopped_at = stopped_at or time.monotonic()
                    if time.monotonic() - stopped_at >= float(cfg.get('stop_grace_seconds', 20)):
                        child.terminate()
                        try:
                            child.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            child.kill()
                        atomic_json(directory / 'status.json', dict(state='stopped',
                            reason='Stop grace elapsed; incomplete wave discarded; last committed checkpoint retained',
                            updated_at=time.time()))
                        return
                time.sleep(.25)
            code = child.returncode
            if code:
                current = read_json(directory / 'status.json')
                if current.get('state') != 'failed':
                    atomic_json(directory / 'status.json', dict(state='failed', error=f'Worker exited {code}', updated_at=time.time()))
                raise RuntimeError(f'Fine-tuning worker exited {code}; inspect console.log')
        finally:
            (directory / 'supervisor.json').unlink(missing_ok=True)


def main(argv=None):
    from .runner import directory_for, is_active
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    for flag in ('dry-run', 'check', 'status', 'stop'):
        actions.add_argument('--' + flag, action='store_true')
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--name')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--device', choices=('cuda', 'cpu'))
    parser.add_argument('--resume', action='store_true', help='resume the saved run (also automatic with the same command)')
    parser.add_argument('--foreground', action='store_true')
    parser.add_argument('--supervisor', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--start-token', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    base = read_json(args.config)
    directory = directory_for(base, args.name)
    if args.status or args.stop:
        if args.stop and is_active(directory):
            (directory / 'stop.request').touch()
            print('Stop requested. The last committed update is retained; use --status to confirm shutdown.')
        else:
            print(json.dumps(state(directory), ensure_ascii=False, indent=2))
        return
    if args.worker:
        worker(directory)
        return
    if args.supervisor:
        supervise(directory, args.start_token)
        return
    cfg = configuration(args.config)
    cfg['workers'] = args.workers if args.workers is not None else cfg['workers']
    cfg['rollout_episodes_per_wave'] = 16
    cfg['device'] = args.device or cfg['device']
    cfg.update(stop_file=str(directory / 'stop.request'), map_split_registry=str(directory / 'map_splits.json'),
               data_directory=str(ROOT / 'artifacts/data/v12' / directory.name))
    checkpoint = (args.checkpoint or Path(cfg['initial_checkpoint'])).expanduser()
    checkpoint = (checkpoint if checkpoint.is_absolute() else ROOT / checkpoint).resolve()
    readiness = preflight(cfg, checkpoint)
    if args.dry_run or args.check:
        print(json.dumps(dict(readiness, directory=str(directory)), ensure_ascii=False, indent=2))
        return
    directory.mkdir(parents=True, exist_ok=True)
    with FileLease(directory / 'launch.lock', blocking=False):
        if is_active(directory):
            raise RuntimeError('Fine tuning is already active; use --status')
        manifest = registration(cfg, checkpoint)
        path = directory / 'finetune_registration.json'
        if path.exists() and read_json(path) != manifest:
            raise ValueError('Registered source/config/checkpoint changed; use a new --name')
        if ((directory / 'result.json').exists() and (directory / 'status.json').exists()
                and read_json(directory / 'status.json').get('state') == 'complete'):
            print(json.dumps(state(directory), ensure_ascii=False, indent=2))
            return
        with FileLease(ROOT / 'logs/runtime/compute.lock', blocking=False):
            if not path.exists():
                atomic_json(path, manifest)
        (directory / 'stop.request').unlink(missing_ok=True)
        token = uuid.uuid4().hex
        if args.foreground:
            supervise(directory, token)
            return
        atomic_json(directory / 'status.json', dict(state='starting', updated_at=time.time()))
        with (directory / 'console.log').open('ab', buffering=0) as log:
            child = subprocess.Popen([sys.executable, '-u', str(ENTRY), '--name', directory.name,
                                      '--supervisor', '--start-token', token], cwd=ROOT,
                env=child_environment(), stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                **process_options(detached=True))
        for _ in range(200):
            ready = read_json(directory / 'supervisor.json') if (directory / 'supervisor.json').exists() else {}
            if ready.get('token') == token:
                print(f'Fine tuning started in background (PID {ready["pid"]}).\n'
                      f'Logs: {directory}\nOne game window opens during collection, after the baseline evaluation.\n'
                      'Use the same command with --status or --stop; run the command again to resume.')
                return
            if child.poll() is not None:
                if child.returncode:
                    raise RuntimeError('Background startup failed; inspect ' + str(directory / 'console.log'))
                print(json.dumps(state(directory), ensure_ascii=False, indent=2))
                return
            time.sleep(.1)
        raise RuntimeError('Startup still pending; inspect --status before retrying')


if __name__ == '__main__':
    main()
