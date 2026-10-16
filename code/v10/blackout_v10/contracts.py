"""Read-only verification of supplied artifacts. No runtime/env monkey patches."""

from project_paths import project_root, project_path, source_files

import importlib.metadata
import importlib.util
import os
from pathlib import Path
import subprocess
from .io import ROOT, digest, file_hash, read_json

CONFIG = project_path('code/v10/configs/default.json', root=ROOT)
ORIGINAL = project_path('code/v10/contracts/original.json', root=ROOT)
LAUNCHER = project_path('code/v10/launchers/BlackOutRendered.app', root=ROOT)


def config(path=CONFIG):
    c = read_json(path)
    if c["contract"] != "v10-s-provided-runner-v1" or c["objective"] != "provided_runner_v1":
        raise ValueError("unsupported outcome/input contract")
    if not 1 <= c["workers"] <= c["worker_cap"] <= 8:
        raise ValueError("invalid global worker cap")
    if not c["arms"] or set(c["arms"]) - {"C0", "C1"}:
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
    if c["arms"] != ["C0", "C1"]: raise ValueError("primary study requires both C0 and C1")
    if c["device"] not in ("auto", "cpu", "mps") or c["actor_device"] not in ("auto", "cpu", "mps"):
        raise ValueError("invalid device")
    if c["ppo"]["aux_coef"] != 0: raise ValueError("auxiliary learning is not part of primary study")
    if c["diagnostic_games"] != 16 or c["diagnostic_fit_games"] != 12: raise ValueError("P2 split changed")
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
            if file_hash(project_path(rel, root=ROOT)) != checksum:
                raise ValueError("provided runtime changed: " + rel)
    for package, directory in (("blackout_env", "blackout_env"), ("mlagents_envs", "mlagents_envs"),
                               ("google.protobuf", "google/protobuf")):
        origin = Path(importlib.util.find_spec(package).origin).resolve()
        expected = project_path(Path(".venv/lib/python3.10/site-packages") / directory, root=ROOT)
        if not origin.is_relative_to(expected):
            raise ValueError("non-original import: " + str(origin))
    executable = LAUNCHER / "Contents/MacOS/BlackOutRendered"
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError("rendered launcher not executable")
    return {"original_verified": True, "executable_sha256": record["executable_sha256"]}


def source_fingerprint():
    files = list((project_path('code/v10/blackout_v10', root=ROOT)).glob("*.py"))
    # Historical opponents are used as-is and must remain pinned too.
    files += source_files(ROOT, "blackout_rl")
    files += [ROOT / "code/shared/project_paths.py", ROOT / "code/shared/legacy_paths.json", ROOT / "code/shared/relocation_sources.json"]
    files += list((ROOT / "code/v10/configs").glob("*.json"))
    files += list((ROOT / "code/v10/contracts").glob("*.json"))
    files += list((ROOT / "code/v10/scripts").glob("*.py"))
    files += list((ROOT / "code/v10").glob("*.sh"))
    files += [ROOT / "code/shared/tools/watch_best_models.py"]
    files += [ORIGINAL, CONFIG, project_path('code/v10/scripts/run_v10_fast.sh', root=ROOT), project_path('code/v10/scripts/v10_experiments.py', root=ROOT),
              project_path('code/v10/run_v10_fast.sh', root=ROOT), LAUNCHER / "Contents/MacOS/BlackOutRendered",
              project_path('docs/common/blackout_last_4_pages.md', root=ROOT), project_path('artifacts/checkpoints/win_70_vs_scripted.pt', root=ROOT)]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(files)}


def registration(c):
    return {"schema": "blackout.v10.1", "config": c, "sources": source_fingerprint(),
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
