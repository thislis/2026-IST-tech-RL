"""Register a separately supplied Windows build without executing it.

Run through code/run.py with --build DIRECTORY and the full --source-commit.
The commit is an explicit user attestation; file hashes identify the supplied
binary payload, but do not prove which source revision produced those binaries.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile

from .common import ROOT, atomic, digest, read, sha


UNITY_COMMIT = 'af4c5faaf2ef9eae24f536860737099ad56acba3'
RUNTIME_REL = 'artifacts/new_game/20261101/runtime.json'
BUILD_PARENT_REL = 'artifacts/builds/windows-af4c5faaf2ef'
MANIFEST_REL = 'artifacts/new_game/20261101/game_manifest.json'
EXCLUDES = ['*_Data/ML-Agents/Timers/*']


def _excluded(name):
    return bool(re.fullmatch(r'[^/]+_Data/ML-Agents/Timers/.*', name, re.IGNORECASE))


def _reparse(path):
    info = path.lstat()
    return path.is_symlink() or bool(getattr(info, 'st_file_attributes', 0)
                                    & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400))


def _project_path(root, value):
    path = root / value
    # Refuse directory junctions as well as symlinks before resolving them.
    for candidate in [path, *path.parents]:
        if candidate.exists() and _reparse(candidate):
            raise ValueError('Registration destination contains a link: ' + str(candidate))
        if candidate == root:
            break
    resolved = path.resolve()
    if not resolved.is_relative_to(root) or resolved == root:
        raise ValueError('Registration destination must stay inside the project')
    return resolved


def content_files(directory):
    """Hash every regular build file except the documented runtime timers."""
    files = {}
    # Check directory entries before descending: Path.rglob can follow Windows
    # junctions before its caller gets a chance to inspect the junction itself.
    for current, directories, names in os.walk(directory, topdown=True, followlinks=False):
        directories.sort()
        for name in [*directories, *sorted(names)]:
            path = Path(current) / name
            if _reparse(path):
                raise ValueError('Build links and directory junctions are unsupported: ' + str(path))
            if path.is_file():
                relative = path.relative_to(directory).as_posix()
                if not _excluded(relative):
                    files[relative] = sha(path)
            elif not path.is_dir():
                raise ValueError('Build contains a non-regular entry: ' + str(path))
    return dict(sorted(files.items()))


def _select_executable(directory, selected=None):
    candidates = [path for path in directory.iterdir()
                  if path.is_file() and path.suffix.lower() == '.exe'
                  and (directory / (path.stem + '_Data')).is_dir()]
    if selected is not None:
        if Path(selected).name != selected:
            raise ValueError('--executable must be a filename directly inside --build')
        candidates = [path for path in candidates if path.name == selected]
    if len(candidates) != 1:
        raise ValueError('Expected one game .exe with a matching <name>_Data directory; '
                         'use --executable FILENAME if necessary')
    executable = candidates[0]
    if not (directory / 'UnityPlayer.dll').is_file():
        raise ValueError('Complete Windows build required: UnityPlayer.dll is missing')
    return executable


def _payload(files, data_name):
    prefix = data_name + '/'
    payload = {'_Data/' + name[len(prefix):]: value
               for name, value in files.items() if name.startswith(prefix)}
    if 'GameAssembly.dll' in files:
        payload['GameAssembly.dll'] = files['GameAssembly.dll']
    return payload


def _reject_old_payload(root, files, data_name):
    old = root / 'artifacts/builds/windows-v1.1'
    if not old.is_dir():
        return
    old_executable = _select_executable(old)
    old_files = content_files(old)
    old_data = old_executable.stem + '_Data'
    if _payload(files, data_name) == _payload(old_files, old_data):
        raise ValueError('Rejected the known v1.1 game payload; a new source build is required')
    # Unity executable stubs may be identical across builds. Compare the actual
    # gameplay assembly, never the launcher stub, to catch repackaged old games.
    for relative in ('Managed/Assembly-CSharp.dll',):
        candidate = files.get(data_name + '/' + relative)
        previous = old_files.get(old_data + '/' + relative)
        if candidate is not None and candidate == previous:
            raise ValueError('Rejected the known v1.1 gameplay assembly: ' + relative)
    if files.get('GameAssembly.dll') is not None and files['GameAssembly.dll'] == old_files.get('GameAssembly.dll'):
        raise ValueError('Rejected the known v1.1 GameAssembly.dll')


def _remove_owned_directory(path, parent):
    """Remove only a fresh directory created by this registration attempt."""
    resolved = path.resolve()
    if (resolved == parent or not resolved.is_relative_to(parent.resolve())
            or not (path.name.startswith('.stage-') or path.name.startswith('build-'))):
        raise ValueError('Refusing cleanup outside the new build directory')
    for child in path.rglob('*'):
        if child.is_file():
            child.chmod(stat.S_IREAD | stat.S_IWRITE)
    shutil.rmtree(path)


def register_build(build, source_commit, *, selected_executable=None, root=None):
    """Copy to a fresh snapshot and atomically select it after all checks pass."""
    if source_commit != UNITY_COMMIT:
        raise ValueError('--source-commit must be the exact full commit ' + UNITY_COMMIT)
    root = Path(root or ROOT).resolve()
    raw_source = Path(build).expanduser().absolute()
    if not raw_source.is_dir() or _reparse(raw_source):
        raise ValueError('--build must be a complete ordinary Windows build directory')
    source = raw_source.resolve()
    runtime_path = _project_path(root, RUNTIME_REL)
    metadata = read(runtime_path)
    if metadata.get('unity_commit') != UNITY_COMMIT or metadata.get('api_ready') is not True:
        raise ValueError('The exact new-game source and isolated API metadata must be staged first')
    if metadata.get('ready') is True:
        raise FileExistsError('A build is already registered; refusing to replace its provenance')
    parent = _project_path(root, BUILD_PARENT_REL)
    if source.is_relative_to(parent) or parent.is_relative_to(source):
        raise ValueError('--build must be separate from the new build destination')
    executable = _select_executable(source, selected_executable)
    files = content_files(source)
    data_name = executable.stem + '_Data'
    if not _payload(files, data_name):
        raise ValueError('The matching game _Data directory contains no build payload')
    _reject_old_payload(root, files, data_name)
    content_sha = digest(files)
    destination = _project_path(root, BUILD_PARENT_REL + '/build-' + content_sha[:16])
    manifest_path = _project_path(root, MANIFEST_REL)
    if destination.exists() or manifest_path.exists():
        raise FileExistsError('Build snapshot or registration manifest already exists; refusing to overwrite it')
    parent.mkdir(parents=True, exist_ok=True)
    lock_path = runtime_path.with_name('build_registration.lock')
    descriptor_before = sha(runtime_path)
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(lock_fd)
    temporary = None
    created_manifest = False
    moved = False
    registered = False
    try:
        temporary = Path(tempfile.mkdtemp(prefix='.stage-', dir=parent))
        for name in files:
            target = temporary / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, target)
        # A running or changing source directory must not yield a mixed snapshot.
        if content_files(temporary) != files or content_files(source) != files:
            raise ValueError('Build changed while being copied; no runtime was registered')
        if sha(runtime_path) != descriptor_before:
            raise ValueError('Runtime metadata changed during registration; refusing to replace it')
        if destination.exists():
            raise FileExistsError('Build destination appeared during registration')
        stamp = datetime.now(timezone.utc).isoformat()
        manifest = dict(schema='blackout.game-snapshot.v1',
                        root=destination.relative_to(root).as_posix(),
                        source_commit=UNITY_COMMIT, source_proof='user_attestation',
                        executable=executable.name, excludes=EXCLUDES,
                        files=files, content_sha256=content_sha,
                        provenance=dict(registered_at_utc=stamp,
                            supplied_build=str(source), source_commit_attestation=source_commit,
                            cryptographic_source_proof=False,
                            statement='The user supplied this build and attested its source commit; '
                                      'binary SHA256 hashes do not prove the compiled source revision.'))
        # Exclusive creation protects any prior registration, including one left
        # by an interrupted process. The descriptor is committed last.
        with manifest_path.open('x', encoding='utf-8') as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        created_manifest = True
        temporary.rename(destination)
        moved = True
        for name in files:
            (destination / name).chmod(stat.S_IREAD)
        manifest_path.chmod(stat.S_IREAD)
        updated = dict(metadata, ready=True, game_ready=True,
                       status='user_attested_build_registered',
                       reason='User-supplied build registered with source commit attestation and SHA256 hashes; '
                              'compiled source revision is not cryptographically proven.',
                       executable=(destination / executable.name).relative_to(root).as_posix(),
                       executable_sha256=files[executable.name],
                       game_root=destination.relative_to(root).as_posix(),
                       game_manifest=MANIFEST_REL, game_manifest_sha256=sha(manifest_path),
                       game_sha256=content_sha, game_source_commit=UNITY_COMMIT,
                       game_source_proof='user_attestation', game_registered_at_utc=stamp)
        atomic(runtime_path, updated)
        registered = True
        return dict(runtime=RUNTIME_REL, game_root=updated['game_root'],
                    executable=updated['executable'], files=len(files),
                    game_sha256=content_sha, source_commit=UNITY_COMMIT,
                    source_proof='user_attestation', unity_started=False)
    finally:
        if not registered:
            if moved:
                _remove_owned_directory(destination, parent)
            elif temporary is not None and temporary.exists():
                _remove_owned_directory(temporary, parent)
            if created_manifest:
                manifest_path.chmod(stat.S_IREAD | stat.S_IWRITE)
                manifest_path.unlink()
        lock_path.unlink()


def main():
    parser = argparse.ArgumentParser(description='Copy and register a user-supplied new Windows build; never launches Unity')
    parser.add_argument('--build', required=True, help='Full Windows build directory containing .exe, matching _Data, and UnityPlayer.dll')
    parser.add_argument('--source-commit', required=True, help='Explicit user attestation; must equal ' + UNITY_COMMIT)
    parser.add_argument('--executable', help='Game .exe filename if the build contains multiple games')
    args = parser.parse_args()
    result = register_build(args.build, args.source_commit, selected_executable=args.executable)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
