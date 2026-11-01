"""Select the separately staged November game/API without changing the venv.

Importing this module never imports blackout_env or launches a game. Every
worker/checker receives the selected API on PYTHONPATH before its first import.
"""
import importlib
import importlib.util
import os
from pathlib import Path
import re
import sys
import time

from .common import ROOT, digest, read, sha


RUNTIME = ROOT / 'artifacts/new_game/20261101/runtime.json'
UNITY_COMMIT = 'af4c5faaf2ef9eae24f536860737099ad56acba3'
API_COMMIT = 'bcd8aadad2f293bf45f3999286b1f174bae5586b'


def _inside_project(value, label):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError(label + ' must be inside the project')
    return path


def runtime_spec():
    if not RUNTIME.is_file():
        raise FileNotFoundError('Separately staged new-game runtime required: ' + str(RUNTIME))
    metadata = read(RUNTIME)
    if metadata.get('unity_commit') != UNITY_COMMIT:
        raise ValueError('New league requires Unity source ' + UNITY_COMMIT)
    if metadata.get('api_commit') != API_COMMIT:
        raise ValueError('New league requires API source ' + API_COMMIT)
    api = _inside_project(metadata['api_package_parent'], 'API snapshot')
    if not (api / 'blackout_env/__init__.py').is_file():
        raise FileNotFoundError('Isolated API snapshot missing: ' + str(api))
    return metadata


def executable():
    metadata = runtime_spec()
    if metadata.get('ready') is not True or not metadata.get('executable'):
        raise FileNotFoundError('Matching new-game Windows build is not staged (Unity source '
                                + UNITY_COMMIT + '); see ' + str(RUNTIME))
    game = _inside_project(metadata['executable'], 'executable')
    if not game.is_file() or game.suffix.lower() != '.exe':
        raise FileNotFoundError('New Windows executable required: ' + str(game))
    return game


def configure_runtime():
    """Select the snapshot before any blackout_env-dependent project imports."""
    metadata = runtime_spec()
    api = _inside_project(metadata['api_package_parent'], 'API snapshot')
    sys.dont_write_bytecode = True
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    imported = sys.modules.get('blackout_env')
    if imported is not None:
        source = Path(imported.__file__).resolve()
        if not source.is_relative_to(api):
            raise RuntimeError('blackout_env already imported from another runtime: ' + str(source))
    value = str(api)
    sys.path[:] = [value, *[p for p in sys.path if p != value]]
    inherited = [p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) if p]
    os.environ['PYTHONPATH'] = os.pathsep.join(dict.fromkeys([value, *inherited]))
    importlib.invalidate_caches()
    selected = importlib.util.find_spec('blackout_env')
    if selected is None or not Path(selected.origin).resolve().is_relative_to(api):
        raise RuntimeError('New-game API import does not resolve to its isolated snapshot')
    return metadata


def verify_api_snapshot():
    """Verify archive provenance and every staged source before registration."""
    metadata = runtime_spec()
    api = _inside_project(metadata['api_package_parent'], 'API snapshot')
    manifest_path = _inside_project(metadata['api_manifest'], 'API manifest')
    if sha(manifest_path) != metadata['api_manifest_sha256']:
        raise ValueError('New-game API manifest checksum mismatch')
    manifest = read(manifest_path)
    files = manifest['files']
    if (manifest.get('commit') != metadata['api_commit']
            or digest(files) != metadata['api_sha256']):
        raise ValueError('New-game API content identity mismatch')
    actual_names = {path.relative_to(api).as_posix() for path in api.rglob('*')
                    if path.is_file() and '__pycache__' not in path.parts}
    if actual_names != set(files):
        raise ValueError('New-game API snapshot file set changed')
    for name, expected in files.items():
        path = (api / name).resolve()
        if not path.is_relative_to(api) or sha(path) != expected:
            raise ValueError('New-game API source checksum mismatch: ' + name)
    archive = _inside_project(metadata['api_archive'], 'API archive')
    if sha(archive) != metadata['api_archive_sha256']:
        raise ValueError('New-game API archive checksum mismatch')
    return dict(manifest=metadata['api_manifest'], manifest_sha256=sha(manifest_path),
                source_sha256=digest(files), files=len(files))


def game_content_files(root):
    """Hash immutable game files while excluding Unity's runtime timer output."""
    root = Path(root)
    return {path.relative_to(root).as_posix(): sha(path) for path in sorted(root.rglob('*'))
            if path.is_file() and not re.fullmatch(
                r'[^/]+_Data/ML-Agents/Timers/.*', path.relative_to(root).as_posix(), re.IGNORECASE)}


def verify_game_snapshot():
    metadata = runtime_spec()
    game = executable()
    root = _inside_project(metadata['game_root'], 'game root')
    manifest_path = _inside_project(metadata['game_manifest'], 'game manifest')
    if manifest_path.is_relative_to(root):
        raise ValueError('Game content manifest must be outside the immutable game root')
    if sha(manifest_path) != metadata['game_manifest_sha256']:
        raise ValueError('New-game build manifest checksum mismatch')
    manifest = read(manifest_path)
    if (manifest.get('schema') != 'blackout.game-snapshot.v1'
            or manifest.get('source_commit') != metadata['unity_commit']
            or manifest.get('source_proof') != 'user_attestation'
            or manifest.get('excludes') != ['*_Data/ML-Agents/Timers/*']):
        raise ValueError('New-game build registration provenance mismatch')
    if (root / manifest['executable']).resolve() != game:
        raise ValueError('New-game executable does not match the registered build')
    if sha(game) != metadata['executable_sha256']:
        raise ValueError('New-game executable checksum mismatch')
    files = game_content_files(root)
    if files != manifest['files'] or digest(files) != metadata['game_sha256']:
        raise ValueError('New-game build content checksum mismatch')
    return dict(executable=str(game), files=files, sha256=digest(files),
                manifest=metadata['game_manifest'], manifest_sha256=sha(manifest_path),
                source_commit=metadata['unity_commit'], source_proof=manifest['source_proof'])


def snapshot_environment():
    configure_runtime()
    from blackout_v12.runtime import child_environment
    environment = child_environment()
    api = str(_inside_project(runtime_spec()['api_package_parent'], 'API snapshot'))
    inherited = [p for p in environment.get('PYTHONPATH', '').split(os.pathsep) if p]
    environment['PYTHONPATH'] = os.pathsep.join(dict.fromkeys([api, *inherited]))
    return environment


def make_env(task, *, background, worker_id):
    """Create one rendered game only when explicitly called by a match worker."""
    configure_runtime()
    from blackout_rl.env import ContractBlackOutEnv
    config = task['config']
    if config.get('game_runtime') != RUNTIME.relative_to(ROOT).as_posix():
        raise ValueError('Match does not select the isolated new-game runtime')
    scale = config['time_scale'] if background else config['visible_time_scale']
    class AuditedEnvironment(ContractBlackOutEnv):
        def reset(self, *args, **kwargs):
            observation = super().reset(*args, **kwargs)
            deadline = time.monotonic() + 5
            while True:
                audit = window_audit(self, background=background)
                if audit['verified'] or background or time.monotonic() >= deadline:
                    break
                time.sleep(.1)
            self.native_window_audit = audit
            if not audit['verified']:
                raise RuntimeError('New-game visible/background window audit failed: ' + str(audit))
            return observation

    return AuditedEnvironment(env_path=str(executable()), no_graphics=False,
                             background=background, time_scale=float(scale),
                             worker_id=int(worker_id), base_port=None)


def window_audit(environment, *, background):
    """Check only the Unity PID owned by this match worker."""
    if os.name != 'nt':
        raise RuntimeError('The new-game window audit requires Windows')
    import ctypes
    from ctypes import wintypes
    process = environment._unity_env._process
    if process is None or process.poll() is not None:
        raise RuntimeError('Owned Unity process is unavailable for window audit')
    user = ctypes.WinDLL('user32', use_last_error=True)
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user.EnumWindows.argtypes = [callback_type, wintypes.LPARAM]
    user.EnumWindows.restype = wintypes.BOOL
    user.IsWindowVisible.argtypes = [wintypes.HWND]
    user.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    windows = []

    @callback_type
    def inspect(handle, _):
        pid = wintypes.DWORD()
        user.GetWindowThreadProcessId(handle, ctypes.byref(pid))
        if pid.value == process.pid and user.IsWindowVisible(handle):
            title = ctypes.create_unicode_buffer(user.GetWindowTextLengthW(handle) + 1)
            user.GetWindowTextW(handle, title, len(title))
            windows.append(dict(handle=int(handle), title=title.value))
        return True

    if not user.EnumWindows(inspect, 0):
        raise ctypes.WinError(ctypes.get_last_error())
    return dict(pid=process.pid, background=background, visible_windows=windows,
                verified=not windows if background else len(windows) >= 1,
                checked_at=time.time())


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Inspect the isolated API without starting Unity')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--pythonpath', action='store_true')
    group.add_argument('--verify-api', action='store_true')
    arguments = parser.parse_args()
    configure_runtime()
    if arguments.pythonpath:
        print(os.environ['PYTHONPATH'])
    else:
        import json
        print(json.dumps(verify_api_snapshot(), indent=2))


if __name__ == '__main__':
    main()
