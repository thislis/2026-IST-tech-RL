"""Read-only verification of supplied artifacts. No runtime/env monkey patches."""
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import subprocess
from .io import ROOT, digest, file_hash, read_json

CONFIG = ROOT / "configs/v9/default.json"
ORIGINAL = ROOT / "contracts/v9/original.json"
LAUNCHER = ROOT / "launchers/v9/BlackOutRendered.app"


def config(path=CONFIG):
    c = read_json(path)
    if c["contract"] != "v9-s-provided-runner-v1" or c["objective"] != "provided_runner_v1":
        raise ValueError("unsupported outcome/input contract")
    if not 1 <= c["workers"] <= c["worker_cap"] <= 8:
        raise ValueError("invalid global worker cap")
    if not c["arms"] or set(c["arms"]) - {"A", "B", "C"}:
        raise ValueError("invalid arms")
    if c["decoder"] != "sample" or c["gamma"] not in (1., .99995):
        raise ValueError("on-policy experiment requires registered sampling decoder/gamma")
    if c["max_episode_steps"] != 22000 or not 0 < c["gae_lambda"] <= 1:
        raise ValueError("invalid termination/GAE contract")
    for key in ("pilot_steps", "training_steps", "lifecycle_games", "dev_replicates", "test_replicates"):
        if c[key] <= 0:
            raise ValueError("nonpositive budget: " + key)
    if c["lifecycle_games"] < 20 or c["dev_replicates"] < 30:
        raise ValueError("plan lifecycle/dev minimum not met")
    return c


def verify_original():
    record = read_json(ORIGINAL)
    for name, repo in record["upstream"].items():
        path = ROOT.parent / name
        commit = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True).strip()
        if commit != repo["commit"] or dirty:
            raise ValueError("provided source repository changed: " + str(path))
        for rel, checksum in repo["files"].items():
            if file_hash(path / rel) != checksum:
                raise ValueError("provided source changed: " + rel)
    for group in ("api_files", "bundle_files"):
        for rel, checksum in record[group].items():
            if file_hash(ROOT / rel) != checksum:
                raise ValueError("provided runtime changed: " + rel)
    for package, directory in (("blackout_env", "blackout_env"), ("mlagents_envs", "mlagents_envs"),
                               ("google.protobuf", "google/protobuf")):
        origin = Path(importlib.util.find_spec(package).origin).resolve()
        expected = ROOT / ".venv/lib/python3.10/site-packages" / directory
        if not origin.is_relative_to(expected):
            raise ValueError("non-original import: " + str(origin))
    executable = LAUNCHER / "Contents/MacOS/BlackOutRendered"
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError("rendered launcher not executable")
    return {"original_verified": True, "executable_sha256": record["executable_sha256"]}


def source_fingerprint():
    files = list((ROOT / "blackout_v9").glob("*.py"))
    # Historical opponents are used as-is and must remain pinned too.
    files += list((ROOT / "blackout_rl").rglob("*.py"))
    files += [ORIGINAL, CONFIG, ROOT / "scripts/run_v9_fast.sh", ROOT / "scripts/v9_experiments.py",
              ROOT / "run_v9_fast.sh", LAUNCHER / "Contents/MacOS/BlackOutRendered",
              ROOT / "blackout_last_4_pages.md", ROOT / "checkpoints/win_70_vs_scripted.pt"]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(files)}


def registration(c):
    return {"schema": "blackout.v9.1", "config": c, "sources": source_fingerprint(),
            "packages": {name: importlib.metadata.version(name) for name in
                         ("torch", "numpy", "blackout-env", "mlagents-envs", "protobuf", "grpcio")},
            "game_outcome_verified": False, "self_id_available": False,
            "seed_application_verified": False, "official_server_certified": False}


def make_env(time_scale=50.):
    # Called only in isolated episode workers after supervisor verification.
    # The supplied class itself owns reset, step, seed channel and observations.
    from blackout_env import BlackOutEnv
    return BlackOutEnv(env_path=str(LAUNCHER), no_graphics=False,
                       time_scale=time_scale, worker_id=0, base_port=None)
