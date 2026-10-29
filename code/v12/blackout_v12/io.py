"""Small UTF-8, atomic artifact helpers for Windows and POSIX."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

ROOT = Path(__file__).resolve().parents[3]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def transient_file_access(exc):
    """Windows sharing/delete-pending failures, not arbitrary I/O errors."""
    return (os.name == 'nt' and isinstance(exc, PermissionError)
            and getattr(exc, 'winerror', None) in (None, 5, 32, 33))


def retry_file_access(operation):
    # Readers can briefly hold a target while another process atomically replaces
    # it. Retry only access/sharing failures, for at most 0.71 seconds of sleep.
    delays = (.01, .02, .04, .08, .16, .2, .2)
    for attempt in range(len(delays) + 1):
        try:
            return operation()
        except PermissionError as exc:
            if not transient_file_access(exc) or attempt == len(delays):
                raise
            time.sleep(delays[attempt])


def read_json(path):
    return json.loads(retry_file_access(lambda: Path(path).read_text(encoding="utf-8-sig")))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        retry_file_access(lambda: os.replace(temporary, path))
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_torch(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def cpu_state(module):
    return {k:v.detach().cpu().clone() for k,v in module.state_dict().items()}


def state_hash(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str((value.dtype,tuple(value.shape))).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()
