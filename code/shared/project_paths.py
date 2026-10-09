"""Physical project layout and read-only resolution of historical artifact paths.

No filesystem aliases, environment patches, or changes to recorded experiments.
New source registration always fingerprints the relocated source files.
"""
from functools import lru_cache
import json
import os
from pathlib import Path
import sys


def project_root():
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def legacy_paths():
    return json.loads(Path(__file__).with_name("legacy_paths.json").read_text())


def project_path(value, root=None):
    """Resolve old project-relative/absolute metadata without rewriting it.

    Unknown relative paths keep normal Path semantics unless root is supplied.
    Absolute paths outside the project are never redirected.
    """
    path = Path(value).expanduser()
    base = Path(root) if root is not None else project_root()
    if path.is_absolute():
        try:
            relative = path.relative_to(base).as_posix()
        except ValueError:
            return path
    else:
        relative = path.as_posix()
    # An actual new-layout path must not be remapped by a legacy alias key.
    if relative.startswith(("code/", "docs/", "artifacts/")) and (base / relative).exists():
        return base / relative
    table = legacy_paths()
    original = relative
    seen = set()
    while relative not in seen:
        seen.add(relative)
        if relative in table:
            candidate = table[relative]
        else:
            candidate = relative
            for prefix in sorted(table, key=len, reverse=True):
                if relative.startswith(prefix + "/"):
                    candidate = table[prefix] + "/" + relative[len(prefix) + 1:]
                    break
        if candidate == relative:
            break
        relative = candidate
        if relative.startswith(("code/", "docs/", "artifacts/", "logs/")) and (base / relative).exists():
            break
    if relative != original:
        return base / relative
    return base / path if root is not None or path.is_absolute() else path


def package_paths(package):
    base = project_root() / "code"
    return [str(p) for p in [base / "shared" / package,
            *[base / f"v{i}" / package for i in range(1, 10)]] if p.is_dir()]


def activate():
    """Expose versioned Python packages and script modules to child processes."""
    base = project_root() / "code"
    folders = [base / "shared", *[base / f"v{i}" for i in range(1, 10)], base / "pre_v1"]
    paths = [str(p) for p in folders + [p / "scripts" for p in folders] if p.is_dir()]
    for path in reversed(paths):
        if path not in sys.path:
            sys.path.insert(0, path)
    inherited = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    os.environ["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(paths + inherited))


def source_files(root=None, package=None):
    root = project_root() if root is None else Path(root)
    return sorted(p for p in (root / "code").rglob("*")
                  if p.is_file() and "__pycache__" not in p.parts
                  and p.suffix in {".py", ".sh", ".json", ".yaml", ".lock"}
                  and (package is None or package in p.parts))


def historical_source_path(name, expected_sha256):
    """Use an exact archived source for a pinned historical checkpoint check."""
    import hashlib
    for candidate in (project_path(name, root=project_root()),
                      project_root() / "artifacts/source_snapshots/before_direct_layout" / name):
        if candidate.is_file() and hashlib.sha256(candidate.read_bytes()).hexdigest() == expected_sha256:
            return candidate
    raise ValueError("Historical source checksum unavailable: " + name)


def verify_relocated_source(name, expected_sha256):
    """Verify a pinned source against the reviewed before/after relocation map."""
    import hashlib
    current = project_path(name, root=project_root())
    actual = hashlib.sha256(current.read_bytes()).hexdigest()
    if actual == expected_sha256:
        return
    records = json.loads(Path(__file__).with_name("relocation_sources.json").read_text())
    record = records.get(name, {})
    if record.get("before") != expected_sha256 or record.get("after") != actual:
        raise ValueError("Historical inference source changed: " + name)
