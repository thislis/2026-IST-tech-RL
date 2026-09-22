"""Restore the exact registered pilot source for evaluating pre-fix checkpoints."""
import fcntl
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_files(root):
    files = sorted(list((root/'blackout_rl').rglob('*.py')) + list((root/'scripts').glob('*v7*')))
    return {str(p.relative_to(root)): sha256(p) for p in files if p.is_file()}


def resolve():
    registration = json.loads((ROOT/'reports/v7/pilot_registration.json').read_text())
    expected = registration['source_files']
    if source_files(ROOT) == expected:
        return ROOT
    archive = ROOT/registration['source_snapshot']
    if sha256(archive) != registration['source_snapshot_sha256']:
        raise ValueError('registered legacy source archive checksum mismatch')
    parent = ROOT/'logs/v7/frozen_runtimes'
    parent.mkdir(parents=True, exist_ok=True)
    target = parent/registration['source_snapshot_sha256']
    with (parent/'restore.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not target.exists():
            temporary = parent/(target.name+'.tmp')
            temporary.mkdir(exist_ok=True)
            with zipfile.ZipFile(archive) as bundle:
                for member in bundle.infolist():
                    relative = Path(member.filename)
                    if relative.is_absolute() or '..' in relative.parts or not relative.parts or relative.parts[0] not in ('blackout_rl','scripts','configs','requirements.lock','requirements-v7.lock'):
                        raise ValueError('unexpected path in registered source archive')
                bundle.extractall(temporary)
            if source_files(temporary) != expected:
                raise ValueError('archived pilot sources do not match registration')
            for name in ('builds','data','checkpoints','reports','.venv'):
                (temporary/name).symlink_to(ROOT/name, target_is_directory=True)
            temporary.rename(target)
        if source_files(target) != expected:
            raise ValueError('restored frozen runtime has been modified')
    return target
