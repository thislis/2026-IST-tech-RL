"""Frozen full league on Windows; preparation and checks never launch Unity.

The portable package's macOS supervisor is intentionally not imported. Match
semantics and trace/report formats come from the shared league implementation;
native process ownership and the environment come from blackout_v12.
"""
import argparse
import copy
from collections import deque
from contextlib import ExitStack
import ctypes
import importlib.metadata
import importlib.util
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

from .common import BASE, DEFAULTS, ROOT, digest, schedule, sha, verify_assets
from blackout_v12.io import atomic_json as atomic, read_json as read, transient_file_access
from blackout_v12.runtime import (FileLease, child_environment, managed_popen,
                                 process_options, process_tree_guard)


DEFAULT_NAME = 'v12_full_league_20261028'
ROSTER = ROOT / 'artifacts/league/portable_20261028/roster.json'
BUNDLE = ROOT / 'artifacts/submission/v12/scripted9_before_learned_latest'
EXTRA_ID = 'v12-before-learned'
COMPUTE_LOCK = ROOT / 'logs/runtime/compute.lock'
GIB = 1024 ** 3
BUILD_RUNTIME_OUTPUTS = frozenset({'Blackout_Data/ML-Agents/Timers/Prototype_timers.json'})
HEARTBEAT_FIX_SOURCES = frozenset({'code/shared/agent_league/windows.py',
    'code/shared/agent_league/windows_worker.py', 'code/v12/blackout_v12/io.py'})


def parser():
    p = argparse.ArgumentParser(description='Frozen 21 opponents + v12-before-learned, full paired-side league on Windows')
    p.add_argument('--name', default=DEFAULT_NAME)
    p.add_argument('--workers', type=int, help='CPU match workers, 1..16 (default 16; one thread each)')
    group = p.add_mutually_exclusive_group()
    for flag in ('list', 'prepare', 'check', 'status', 'stop', 'resume', 'resume-check', 'report'):
        group.add_argument('--' + flag, action='store_true')
    group.add_argument('--trace', metavar='MATCH_ID', help='Inspect one stored transition; no game is launched')
    p.add_argument('--step', type=int, default=0)
    p.add_argument('--supervisor', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--launch-token', help=argparse.SUPPRESS)
    return p


def configuration(workers=16):
    cfg = dict(DEFAULTS, workers=workers, threads=1, device='cpu', rounds=5,
                reserve_gib=15, ram_reserve_gib=4,
                startup_reserve_gib=1,
                trace_reserve_per_active_gib=.25,
                execution_adapter='blackout_v12.env.make_env(background=True)',
                window_audit='Windows CREATE_NO_WINDOW + Unity background; macOS audit disabled')
    cfg.pop('memory_gib', None)  # Admission uses system available RAM, not the macOS RSS budget.
    return cfg


def selected_roster():
    roster = read(ROSTER)
    if len(roster['models']) != 21 or len({m['id'] for m in roster['models']}) != 21:
        raise ValueError('Expected the unchanged frozen 21-model roster: ' + str(ROSTER))
    if EXTRA_ID in {m['id'] for m in roster['models']}:
        raise ValueError('The frozen roster already contains the added v12 ID')
    # The original v7 config digest remains unchanged. Only its historical
    # validation paths use native, separately pinned Windows dependencies.
    from blackout_v12.env import executable
    game = executable(configuration()).resolve()
    for model in roster['models']:
        if model['kind'] == 'v7':
            model['validation_overrides'] = dict(environment_executable=str(game),
                environment_sha256=sha(game),
                split_manifest='artifacts/league/portable_20261028/dependencies/code/v7/configs/splits.json')
    assets = {}
    for name in ('policy.py', 'checkpoint.pt'):
        path = BUNDLE / name
        if not path.is_file():
            raise FileNotFoundError(path)
        assets[path.relative_to(ROOT).as_posix()] = sha(path)
    spec = dict(id=EXTRA_ID, kind='bundle',
                path=(BUNDLE / 'checkpoint.pt').relative_to(ROOT).as_posix(),
                config={}, seed=0, step=0, selected=True, complete=True, main=True,
                contract='submission_tensor', assets=assets,
                label='v12 stronger scripted9 before learned fine-tuning',
                selection_reason='Explicit frozen before-learned v12 submission bundle')
    return dict(roster, models=[*roster['models'], spec],
                excluded=roster.get('excluded', []),
                extra_model=EXTRA_ID,
                frozen_roster_path=ROSTER.relative_to(ROOT).as_posix(),
                frozen_roster_sha256=sha(ROSTER))


def source_hashes():
    suffixes = {'.py', '.json', '.yaml', '.yml', '.lock', '.sh', '.ps1'}
    return {p.relative_to(ROOT).as_posix(): sha(p)
            for p in sorted((ROOT / 'code').rglob('*'))
            if p.is_file() and p.suffix in suffixes and '__pycache__' not in p.parts}


def build_identity(cfg):
    from blackout_v12.env import executable
    path = executable(cfg).resolve()
    if not path.is_file() or path.suffix.lower() != '.exe':
        raise FileNotFoundError('Native Windows game build required: ' + str(path))
    files = {p.relative_to(path.parent).as_posix(): sha(p)
             for p in sorted(path.parent.rglob('*')) if p.is_file()
             and p.relative_to(path.parent).as_posix() not in BUILD_RUNTIME_OUTPUTS}
    return dict(executable=str(path), files=files, sha256=digest(files))


def runtime_identity():
    packages = {}
    for name in ('torch', 'numpy', 'onnx', 'onnxruntime', 'mlagents-envs', 'blackout-env', 'protobuf'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    # The editable environment package is outside code/ and is part of the API.
    api = {}
    spec = importlib.util.find_spec('blackout_env')
    if spec and spec.origin:
        folder = Path(spec.origin).parent
        api = {str(p.resolve()): sha(p) for p in sorted(folder.rglob('*.py'))}
    return dict(python=sys.version, executable=str(Path(sys.executable).resolve()),
                packages=packages, blackout_env_sources=api)


def registration(cfg, roster):
    for spec in roster['models']:
        verify_assets(spec)
    return dict(schema='blackout.league.windows.inputs.v1', config=cfg, roster=roster,
                sources=source_hashes(), runtime=runtime_identity(), build=build_identity(cfg))


def validate_inputs(manifest, directory=None):
    """Hash once before supervision, never every match or every game step."""
    if manifest['inputs_sha256'] != digest(manifest['inputs']):
        raise ValueError('Original input manifest digest mismatch')
    expected = copy.deepcopy(manifest['inputs'])
    revision_path = Path(directory) / 'runtime_revision.json' if directory is not None else None
    if revision_path is not None and revision_path.exists():
        revision = read(revision_path)
        if (revision.get('schema') != 'blackout.league.heartbeat_fix.v1'
                or revision.get('manifest_digest') != digest(manifest)
                or revision.get('original_inputs_sha256') != manifest['inputs_sha256']):
            raise ValueError('Runtime revision does not belong to this original manifest')
        changes = revision.get('source_changes', {})
        if not changes or set(changes) - HEARTBEAT_FIX_SOURCES:
            raise ValueError('Runtime revision may change only the heartbeat I/O fix files')
        for name, change in changes.items():
            if expected['sources'].get(name) != change['before']:
                raise ValueError('Runtime revision source baseline mismatch: ' + name)
            expected['sources'][name] = change['after']
        ignored = set(revision.get('ignored_build_runtime_outputs', []))
        if ignored != BUILD_RUNTIME_OUTPUTS:
            raise ValueError('Unexpected build-file exclusion in runtime revision')
        expected['build']['files'] = {k:v for k,v in expected['build']['files'].items() if k not in ignored}
        expected['build']['sha256'] = digest(expected['build']['files'])
        if digest(expected) != revision.get('revised_inputs_sha256'):
            raise ValueError('Runtime revision input digest mismatch')
    current = registration(manifest['config'], selected_roster())
    if current != expected:
        changed = [k for k in current if current[k] != expected.get(k)]
        raise ValueError('Registered inputs changed (' + ', '.join(changed) + '); choose a new --name')
    if manifest['tasks'] != schedule(manifest['roster']['models'], manifest['config']):
        raise ValueError('Registered paired-side schedule changed')


def prepared(directory, workers=None):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'manifest.json'
    if path.exists():
        manifest = read(path)
        if workers is not None and workers != manifest['config']['workers']:
            raise ValueError('Worker count is registered; use a new --name to change it')
        validate_inputs(manifest, directory)
        return manifest
    if (directory / 'request.json').exists():
        raise ValueError('Incomplete or incompatible registration; choose a new --name')
    roster = selected_roster()
    cfg = configuration(workers or 16)
    inputs = registration(cfg, roster)
    tasks = schedule(roster['models'], cfg)
    if len(tasks) != 2310:
        raise ValueError('Full league must contain 2,310 games')
    manifest = dict(schema='blackout.league.windows.v1', experiment_kind='league',
                    roster=roster, config=cfg, tasks=tasks, sources=inputs['sources'],
                    inputs=inputs, inputs_sha256=digest(inputs), created_at=time.time(),
                    concurrency=dict(workers=cfg['workers'], threads=1, device='cpu',
                                     selection='fixed prior fastest configuration; no live benchmark'))
    atomic(directory / 'request.json', dict(roster=roster, config=cfg, sources=inputs['sources'],
                                           inputs_sha256=digest(inputs)))
    atomic(path, manifest)
    atomic(directory / 'status.json', dict(state='prepared', updated_at=time.time(),
                                          completed=0, expected=len(tasks), full_league_started=False))
    return manifest


def active(directory):
    if not (directory / 'run.lock').exists():
        return False
    try:
        with FileLease(directory / 'run.lock', blocking=False):
            return False
    except RuntimeError:
        return True


def status(directory):
    state = read(directory / 'status.json') if (directory / 'status.json').exists() else dict(state='not_prepared')
    running = active(directory)
    if not running and state.get('state') in ('starting', 'running', 'stopping', 'draining_resources', 'checking'):
        state = dict(state, state='interrupted', reason='supervisor no longer holds the run lease')
    return dict(directory=str(directory), active=running, stop_requested=(directory / 'STOP').exists(),
                status=state, console_log=str(directory / 'console.log'))


def resources():
    if os.name == 'nt':
        class MemoryStatus(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong),
                        *[(n, ctypes.c_ulonglong) for n in
                          ('total', 'available', 'page', 'page_available', 'virtual', 'virtual_available', 'extended')]]
        value = MemoryStatus()
        value.length = ctypes.sizeof(value)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
            raise ctypes.WinError()
        available = value.available
    else:
        try:
            available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                                 if line.startswith('MemAvailable:'))) * 1024
        except (OSError, StopIteration):
            raise RuntimeError('System available RAM cannot be measured')
    return dict(available_ram_gib=available / GIB, free_disk_gib=shutil.disk_usage(ROOT).free / GIB)


def can_launch(sample, cfg, count=0):
    # Keep room for the bounded traces of matches already admitted to this wave.
    disk_floor = cfg['reserve_gib'] + (count + 1) * cfg['trace_reserve_per_active_gib']
    # Reserve startup memory before torch and Unity have allocated their working
    # sets. Counting every admitted match is intentionally conservative.
    ram_floor = cfg['ram_reserve_gib'] + (count + 1) * cfg['startup_reserve_gib']
    return sample['available_ram_gib'] > ram_floor and sample['free_disk_gib'] > disk_floor


def terminate(process):
    """Kill only this owned child tree, including the Windows venv redirector."""
    if process.poll() is None:
        if os.name == 'nt':
            # python.exe in a venv can be a redirector with a real Python child.
            # Killing only the redirector leaves that worker and its Job alive.
            result = subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                                    stdin=subprocess.DEVNULL, capture_output=True,
                                    env=child_environment(), timeout=15, **process_options())
            if result.returncode and process.poll() is None:
                raise RuntimeError(f'Cannot stop owned child process tree {process.pid}; outer Job guard will reap it')
        else:
            process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def trace_query(directory, match_id, step):
    from .trace import load_chunk
    if step < 0 or not re.fullmatch(r'[A-Za-z0-9_.-]+', match_id):
        raise ValueError('Invalid match ID or negative step')
    record = read(directory / 'cells' / (match_id + '.json'))
    folder = Path(record['directory']) / 'trace'
    index = read(folder / 'index.json')
    chunk = next((c for c in index['chunks'] if c['first'] <= step < c['first'] + c['count']), None)
    if chunk is None:
        raise ValueError('Step outside saved trace')
    data = load_chunk(folder / chunk['path'])
    offset = step - chunk['first']
    print(json.dumps({key: value[offset].tolist() for key, value in data.items()},
                     ensure_ascii=False, indent=2, allow_nan=True))


def offline_check(directory, manifest, stop=lambda: False):
    rows = []
    folder = directory / 'checks'
    folder.mkdir(parents=True, exist_ok=True)
    for spec in manifest['roster']['models']:
        if stop():
            raise InterruptedError('Offline check stopped')
        cell = folder / spec['id']
        cell.mkdir(exist_ok=True)
        atomic(cell / 'spec.json', spec)
        (cell / 'result.json').unlink(missing_ok=True)
        with (cell / 'console.log').open('wb') as log:
            command = [sys.executable, str(ROOT / 'code/run.py'), '-m', 'agent_league.check',
                       str(cell / 'spec.json'), str(cell / 'result.json')]
            process = managed_popen(command, cwd=ROOT, env=child_environment(), stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=log, **process_options())
            started = time.monotonic()
            try:
                while process.poll() is None:
                    if stop() or time.monotonic() - started > 240:
                        raise RuntimeError('Offline check stopped/timed out: ' + spec['id'])
                    time.sleep(.1)
            finally:
                terminate(process)
        row = read(cell / 'result.json') if (cell / 'result.json').exists() else dict(passed=False, error='checker exited without result')
        row.update(id=spec['id'], live_game=False)
        rows.append(row)
        atomic(folder / 'result.json', dict(passed=all(r['passed'] for r in rows), checked=len(rows),
                                           expected=len(manifest['roster']['models']), models=rows, live_game=False))
        print(f"CHECK {len(rows)}/{len(manifest['roster']['models'])} {spec['id']}: {'PASS' if row['passed'] else 'FAIL'}", flush=True)
    if not all(r['passed'] for r in rows):
        raise RuntimeError('Offline model check failed; see ' + str(folder / 'result.json'))
    return rows


def task_for(base, manifest):
    models = {m['id']: m for m in manifest['roster']['models']}
    return dict(base, config=manifest['config'], models=[models[base['a']], models[base['b']]],
                inputs_sha256=manifest['inputs_sha256'], audit_windows=False)


def result_record(folder, task, manifest, error=None):
    result = {}
    envelope_path = folder / 'result.json'
    if envelope_path.exists() and not (folder / 'error.json').exists() and error is None:
        envelope = read(envelope_path)
        if envelope.get('task_sha256') == digest(task):
            result = envelope.get('result', {})
    index_path = folder / 'trace/index.json'
    valid = bool(result.get('valid') and result.get('natural_terminal') and not result.get('probe')
                 and index_path.exists() and read(index_path).get('complete'))
    return dict(id=task['id'], task={k: v for k, v in task.items() if k != 'models'}, result=result,
                directory=str(folder), valid=valid,
                error=error or (read(folder / 'error.json') if (folder / 'error.json').exists() else None),
                manifest_sha256=digest(manifest))


def recover_completed(directory, manifest):
    expected = {t['id']: task_for(t, manifest) for t in manifest['tasks']}
    for path in sorted((directory / 'attempts').glob('*/*/result.json')):
        folder = path.parent
        if not (folder / 'task.json').exists():
            continue
        task = read(folder / 'task.json')
        if task != expected.get(task.get('id')) or (directory / 'cells' / (task['id'] + '.json')).exists():
            continue
        if (folder / 'attempt.json').exists() and not read(folder / 'attempt.json').get('valid'):
            continue
        row = result_record(folder, task, manifest)
        if row['valid']:
            row['recovered_after_supervisor_exit'] = True
            atomic(directory / 'cells' / (task['id'] + '.json'), row)


def completed_rows(directory, manifest):
    expected = {t['id']: t for t in manifest['tasks']}
    rows = []
    for path in sorted((directory / 'cells').glob('*.json')):
        row = read(path)
        base = expected.get(row.get('id'))
        if (base is None or row.get('manifest_sha256') != digest(manifest)
                or row.get('task') != {k: v for k, v in task_for(base, manifest).items() if k != 'models'}
                or not row.get('valid') or not row.get('result', {}).get('natural_terminal')):
            raise ValueError('Invalid committed match: ' + str(path))
        rows.append(row)
    return rows


def heartbeat_age(job, elapsed):
    """A missed telemetry read must not kill all other matches or refresh liveness."""
    try:
        value = read(job['folder'] / 'heartbeat.json')
        job['last_heartbeat_time'] = float(value['time'])
    except FileNotFoundError:
        pass  # Worker has not published its first heartbeat yet.
    except PermissionError as exc:
        if not transient_file_access(exc):
            raise
    last = job.get('last_heartbeat_time')
    return elapsed if last is None else max(0., time.time() - last)


def run_league(directory, manifest, stop):
    from .report import generate
    cfg = manifest['config']
    recover_completed(directory, manifest)
    completed = completed_rows(directory, manifest)
    complete_ids = {r['id'] for r in completed}
    pending = deque((t, 1) for t in manifest['tasks'] if t['id'] not in complete_ids)
    live = {}
    slots = set(range(cfg['workers']))
    session = directory / 'attempts' / str(time.time_ns())
    revision = directory / 'runtime_revision.json'
    if revision.exists():
        atomic(session / 'runtime_revision.json', read(revision))
    started = time.monotonic()
    last_report, last_status = started, 0
    paused = False
    try:
        while pending or live:
            sample = resources()
            stopping = stop()
            admission = can_launch(sample, cfg, len(live))
            while pending and slots and not stopping and admission:
                base, number = pending.popleft()
                slot = min(slots)
                slots.remove(slot)
                folder = session / (base['id'] + f'__attempt{number}')
                folder.mkdir(parents=True)
                task = task_for(base, manifest)
                atomic(folder / 'task.json', task)
                stream = (folder / 'console.log').open('wb')
                try:
                    command = [sys.executable, str(ROOT / 'code/run.py'), '-m', 'agent_league.windows_worker',
                               '--task', str(folder / 'task.json'), '--slot', str(slot)]
                    process = managed_popen(command, cwd=ROOT, env=child_environment(), stdin=subprocess.DEVNULL,
                                            stdout=stream, stderr=stream, **process_options())
                except BaseException:
                    stream.close()
                    raise
                live[process.pid] = dict(process=process, stream=stream, folder=folder, task=task,
                                         base=base, number=number, slot=slot, started=time.monotonic())
                # Stagger model/Unity construction and remeasure actual system RAM.
                time.sleep(.25)
                sample = resources()
                stopping = stop()
                admission = can_launch(sample, cfg, len(live))
            for pid, job in list(live.items()):
                process = job['process']
                elapsed = time.monotonic() - job['started']
                error = None
                age = heartbeat_age(job, elapsed) if process.poll() is None else 0.
                if process.poll() is None and elapsed > cfg['episode_seconds']:
                    error = 'episode wall timeout; partial match excluded'
                elif process.poll() is None and age > cfg['stall_seconds']:
                    error = 'heartbeat timeout; partial match excluded'
                if error:
                    terminate(process)
                if process.poll() is None:
                    continue
                job['stream'].close()
                if process.returncode and error is None:
                    error = f'worker exit {process.returncode}'
                row = result_record(job['folder'], job['task'], manifest, error)
                atomic(job['folder'] / 'attempt.json', row)
                if row['valid']:
                    atomic(directory / 'cells' / (row['id'] + '.json'), row)
                    (directory / 'failures' / (row['id'] + '.json')).unlink(missing_ok=True)
                    completed.append(row)
                else:
                    atomic(directory / 'failures' / (row['id'] + '.json'),
                           dict(task=job['base'], attempt=row, attempt_number=job['number']))
                    if job['number'] <= cfg['retries'] and not stop():
                        pending.appendleft((job['base'], job['number'] + 1))
                slots.add(job['slot'])
                del live[pid]
            if time.monotonic() - last_status >= 2:
                last_status = time.monotonic()
                mean = sum(r['result']['wall_seconds'] for r in completed) / len(completed) if completed else None
                atomic(directory / 'status.json', dict(state='stopping' if stop() else 'running' if admission else 'draining_resources',
                       updated_at=time.time(), pid=os.getpid(), completed=len(completed), expected=len(manifest['tasks']),
                       active=len(live), queued=len(pending), resources=sample,
                       elapsed_this_session=time.monotonic() - started,
                       eta_seconds=(len(manifest['tasks']) - len(completed)) * mean / cfg['workers'] if mean else None,
                       matches=[dict(id=j['task']['id'], pid=j['process'].pid, attempt=j['number'],
                                     elapsed=time.monotonic() - j['started']) for j in live.values()]))
            if not live and (stop() or not admission):
                paused = not stop() and bool(pending)
                break
            if time.monotonic() - last_report >= 300:
                generate(directory, detailed=False)
                last_report = time.monotonic()
            if live:
                time.sleep(.5)
    finally:
        # Only our own children are touched; the outer Job guard also reaps their descendants.
        for job in live.values():
            terminate(job['process'])
            job['stream'].close()
    state = ('complete' if len(completed) == len(manifest['tasks']) else 'stopped' if stop()
             else 'paused_resources' if paused else 'incomplete')
    atomic(directory / 'status.json', dict(state=state, updated_at=time.time(), pid=os.getpid(),
           completed=len(completed), expected=len(manifest['tasks']), resources=resources(),
           elapsed_this_session=time.monotonic() - started, active=0))
    generate(directory)


def supervise(directory, token):
    if os.name != 'nt':
        raise RuntimeError('The native Windows league runner requires Windows')
    stopped = False
    def request_stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    stop = lambda: stopped or (directory / 'STOP').exists()
    with FileLease(directory / 'run.lock', blocking=False), FileLease(COMPUTE_LOCK, blocking=False), process_tree_guard():
        try:
            manifest = read(directory / 'manifest.json')
            validate_inputs(manifest, directory)
            atomic(directory / 'supervisor.json', dict(pid=os.getpid(), started_at=time.time(), launch_token=token))
            atomic(directory / 'status.json', dict(state='checking', pid=os.getpid(), updated_at=time.time(),
                   completed=len(completed_rows(directory, manifest)), expected=len(manifest['tasks'])))
            checked = directory / 'checks/input_digest.json'
            if not checked.exists() or read(checked).get('inputs_sha256') != manifest['inputs_sha256']:
                offline_check(directory, manifest, stop)
                atomic(checked, dict(inputs_sha256=manifest['inputs_sha256'], checked_at=time.time(), live_game=False))
            if stop():
                atomic(directory / 'status.json', dict(state='stopped', updated_at=time.time(), full_league_started=False))
                return
            run_league(directory, manifest, stop)
        except BaseException as exc:
            atomic(directory / 'status.json', dict(state='stopped' if stop() else 'failed', updated_at=time.time(),
                   error=str(exc), traceback=traceback.format_exc()))
            if (directory / 'manifest.json').exists():
                from .report import generate
                generate(directory)
            raise


def launch(directory, manifest):
    if os.name != 'nt':
        raise RuntimeError('Execution requires native Windows; --prepare/--check/--list are offline')
    with FileLease(directory / 'run.lock', blocking=False), FileLease(COMPUTE_LOCK, blocking=False):
        pass
    (directory / 'STOP').unlink(missing_ok=True)
    token = uuid.uuid4().hex
    atomic(directory / 'status.json', dict(state='starting', updated_at=time.time(), expected=len(manifest['tasks'])))
    command = [sys.executable, str(ROOT / 'code/run.py'), '-m', 'agent_league.windows',
               '--name', directory.name, '--supervisor', '--launch-token', token]
    with (directory / 'console.log').open('ab', buffering=0) as log:
        process = subprocess.Popen(command, cwd=ROOT, env=child_environment(), stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=log, **process_options(detached=True))
    # A second command cannot race launch: the caller holds launch.lock through acknowledgement.
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        path = directory / 'supervisor.json'
        if path.exists() and read(path).get('launch_token') == token:
            print(f'PID {process.pid}\nLog: {directory / "console.log"}\nReport: {directory / "report.md"}', flush=True)
            return
        if process.poll() is not None:
            raise RuntimeError('Supervisor did not start; inspect ' + str(directory / 'console.log'))
        time.sleep(.1)
    # Hashing a large build may outlast acknowledgement; ownership remains unambiguous.
    print(f'PID {process.pid} (validating registered inputs)\nLog: {directory / "console.log"}', flush=True)


def main():
    args = parser().parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.name):
        raise ValueError('name: letters/digits/_/- only')
    if args.workers is not None and not 1 <= args.workers <= 16:
        raise ValueError('workers must be 1..16')
    directory = BASE / args.name
    if args.trace:
        trace_query(directory, args.trace, args.step)
        return
    if args.status:
        print(json.dumps(status(directory), ensure_ascii=False, indent=2))
        return
    if args.stop:
        if not directory.exists():
            raise FileNotFoundError(directory)
        atomic(directory / 'STOP', dict(requested_at=time.time()))
        print('Stop requested. Active matches finish within their existing watchdog limits; no new matches start.')
        return
    if args.report:
        # Report generation is read-only with respect to matches and never launches a game.
        with FileLease(directory / 'launch.lock', blocking=False), FileLease(directory / 'run.lock', blocking=False):
            from .report import generate
            generate(directory)
        print(directory / 'report.md')
        return
    if args.list:
        roster = selected_roster()
        for spec in roster['models']:
            print(f"{spec['id']}\t{spec['kind']}\t{spec['label']}")
        print('22 models; 5 repeats, both sides; 2,310 games; no live game launched')
        return
    if args.supervisor:
        supervise(directory, args.launch_token)
        return
    if (args.resume or args.resume_check) and not (directory / 'manifest.json').exists():
        raise FileNotFoundError('No prepared league: ' + str(directory))
    directory.mkdir(parents=True, exist_ok=True)
    with FileLease(directory / 'launch.lock', blocking=False):
        if active(directory):
            print(json.dumps(status(directory), ensure_ascii=False, indent=2))
            return
        with FileLease(directory / 'run.lock', blocking=False):
            manifest = prepared(directory, args.workers)
            if args.resume_check:
                count = len(completed_rows(directory, manifest))
                print(f'Resume ready: {count}/{len(manifest["tasks"])} completed, '
                      f'{len(manifest["tasks"])-count} pending. No game launched.')
                return
        if args.prepare:
            print(f'Prepared {len(manifest["tasks"]):,} games. No game launched.\n{directory / "manifest.json"}')
            return
        if args.check:
            with FileLease(directory / 'run.lock', blocking=False), FileLease(COMPUTE_LOCK, blocking=False), process_tree_guard():
                offline_check(directory, manifest)
                atomic(directory / 'checks/input_digest.json', dict(inputs_sha256=manifest['inputs_sha256'], checked_at=time.time(), live_game=False))
                atomic(directory / 'status.json', dict(state='checked', updated_at=time.time(), full_league_started=False,
                       completed=len(completed_rows(directory, manifest)), expected=len(manifest['tasks'])))
            print('All 22 models passed offline checks; no live game launched.')
            return
        launch(directory, manifest)


if __name__ == '__main__':
    main()
