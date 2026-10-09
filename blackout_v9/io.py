"""Atomic, checksummed local artifacts and bounded telemetry."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import torch

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".tmp-", delete=False) as f:
        tmp = Path(f.name)
        try:
            json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            f.write("\n"); f.flush(); os.fsync(f.fileno())
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)


def append(path, value, max_bytes=2*1024**3):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > max_bytes:
        raise RuntimeError("telemetry budget exceeded: " + str(path))
    with path.open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def atomic_torch(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    os.close(fd)
    tmp = Path(name)
    try:
        with tmp.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def cpu_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def save_checkpoint(directory, payload):
    directory = Path(directory)
    tmp = directory / "pending.pt"
    atomic_torch(tmp, payload)
    checksum = file_hash(tmp)
    final = directory / (checksum + ".pt")
    if final.exists():
        tmp.unlink()
    else:
        os.replace(tmp, final)
    atomic_json(directory / "latest.json", {"sha256": checksum, "step": payload["step"]})
    return final


def load_checkpoint(directory):
    directory = Path(directory)
    pointer = read_json(directory / "latest.json")
    path = directory / (pointer["sha256"] + ".pt")
    if file_hash(path) != pointer["sha256"]:
        raise ValueError("checkpoint checksum mismatch")
    return torch.load(path, map_location="cpu", weights_only=True), path
