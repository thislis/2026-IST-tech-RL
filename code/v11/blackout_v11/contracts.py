"""Pinned supplied runtime and public two-tensor team policy contract."""
import importlib.metadata
import importlib.util
import os
from pathlib import Path
from project_paths import project_root, project_path, source_files
from .io import digest, file_hash, read_json

ROOT = project_root()
CONFIG = ROOT / 'code/v11/configs/default.json'
ORIGINAL = ROOT / 'code/v11/contracts/original.json'
LAUNCHER = ROOT / 'code/v11/launchers/BlackOutRendered.app'


def config(path=CONFIG):
    c = read_json(path)
    if c['contract'] != 'v11-canonical-team-v1' or c['objective'] != 'provided_runner_v1':
        raise ValueError('unsupported input/outcome contract')
    if not 1 <= c['workers'] <= c['worker_cap'] <= 8:
        raise ValueError('invalid worker count')
    if c['arms'] != ['R', 'G', 'F', 'D']:
        raise ValueError('primary comparison requires R/G/F/D')
    if c['gamma'] != 1 or not 0 < c['gae_lambda'] <= 1:
        raise ValueError('invalid duration discount')
    if c['max_episode_steps'] != 22000:
        raise ValueError('natural episode watchdog changed')
    for key in ('device', 'actor_device', 'planner_device'):
        if c[key] not in ('auto', 'cpu', 'mps'):
            raise ValueError('invalid device: '+key)
    for key in ('teacher_steps', 'pilot_steps', 'training_steps', 'lifecycle_games', 'dev_replicates', 'test_replicates'):
        if c[key] <= 0:
            raise ValueError('nonpositive budget: '+key)
    return c


def verify_original():
    """Verify the runtime closure, without reauditing all Unity source/assets."""
    record = read_json(ORIGINAL)
    checked = 0
    for group in ('api_files', 'bundle_files'):
        for rel, checksum in record[group].items():
            path = project_path(rel, root=ROOT)
            if file_hash(path) != checksum:
                raise ValueError('provided runtime changed: '+rel)
            checked += 1
    for package, relative in (('blackout_env', 'blackout_env'), ('mlagents_envs', 'mlagents_envs')):
        origin = Path(importlib.util.find_spec(package).origin).resolve()
        expected = ROOT / '.venv/lib/python3.10/site-packages' / relative
        if not origin.is_relative_to(expected):
            raise ValueError('unexpected runtime import: '+str(origin))
    launcher = LAUNCHER / 'Contents/MacOS/BlackOutRendered'
    if not launcher.is_file() or not os.access(launcher, os.X_OK):
        raise ValueError('hidden rendered launcher missing/not executable')
    return dict(original_verified=True, files=checked,
                executable_sha256=record['executable_sha256'],
                routing='provided run_match canonical env.agents order; five rows per team')


def source_fingerprint():
    files = [p for p in (ROOT/'code/v11').rglob('*') if p.is_file()
             and '__pycache__' not in p.parts and 'tests' not in p.parts
             and (p.suffix in {'.py', '.sh', '.json', '.lock'} or p.name == 'BlackOutRendered')]
    files += source_files(ROOT, 'blackout_rl')
    files += list((ROOT/'code/shared/agent_league').glob('*.py'))
    files += [ROOT/'code/shared/project_paths.py', ROOT/'code/shared/legacy_paths.json',
              ROOT/'code/shared/relocation_sources.json', ROOT/'code/v10/blackout_v10/opponents.py']
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(set(files))}


def registration(c):
    packages = {}
    for name in ('torch', 'numpy', 'blackout-env', 'mlagents-envs', 'protobuf', 'grpcio', 'onnx', 'onnxruntime'):
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = None
    return dict(schema='blackout.v11.1', config=c, sources=source_fingerprint(), packages=packages,
                public_inputs=['vector[B,96]', 'graphic[B,11,96,96]'], public_output='action[B,2]',
                canonical_team_batch=5, state='instance-owned; observed clock/side reset',
                objective='provided_runner_v1', official_server_certified=False,
                seed_application_verified=False)


def make_env(time_scale=50.):
    from blackout_env import BlackOutEnv
    return BlackOutEnv(env_path=str(LAUNCHER), no_graphics=False,
                       time_scale=time_scale, worker_id=0, base_port=None)
