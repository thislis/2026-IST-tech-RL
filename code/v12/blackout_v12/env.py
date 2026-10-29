"""Launch the supplied game with the semantic-map renderer enabled."""
import os
from pathlib import Path
from .io import ROOT


def executable(config):
    configured = config.get("env_path") or os.environ.get("BLACKOUT_ENV_PATH")
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_absolute() else ROOT / path
    if os.name == "nt":
        return ROOT / "artifacts/builds/windows-v1.1/Blackout.exe"
    candidates = list((ROOT / "artifacts/builds").glob("**/BlackOut.x86_64"))
    candidates += list((ROOT / "artifacts/builds").glob("**/BlackOut.app"))
    return candidates[0] if candidates else ROOT / "artifacts/builds/BlackOut.x86_64"


def make_env(task, *, background=True, worker_id=0):
    """Only called by explicit experiment workers, never by check/benchmark."""
    from blackout_rl.env import ContractBlackOutEnv
    config = task.get("config", task)
    path = executable(config)
    if not path.exists():
        raise FileNotFoundError(f"Game build missing: {path}; set env_path or BLACKOUT_ENV_PATH")
    # no_graphics disables the semantic-map camera; keep the GPU renderer alive.
    return ContractBlackOutEnv(env_path=str(path), no_graphics=False, background=background,
                               time_scale=float(config.get("time_scale", 50)),
                               worker_id=int(worker_id), base_port=None)
