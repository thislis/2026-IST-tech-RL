"""Verbatim provided run_match; visible diagnostics are separately named."""
from dataclasses import asdict
import time
import torch
from .collector import observation_hash
from .export import load_export
from .opponents import make_opponent


class ObservedPolicy:
    def __init__(self, model, limit, heartbeat, stop, audit_windows=False):
        self.model, self.limit, self.heartbeat, self.stop = model, limit, heartbeat, stop
        self.calls = 0
        self.initial_hash = None
        self.last = 0.
        self.audit_windows = audit_windows
        self.windows = []

    def act(self, obs):
        if self.stop(): raise InterruptedError("evaluation interrupted; invalid attempt")
        if self.calls >= self.limit: raise RuntimeError("evaluation watchdog; not a draw")
        if self.initial_hash is None: self.initial_hash = observation_hash(obs)
        self.calls += 1
        now = time.monotonic()
        if now-self.last >= 5:
            self.heartbeat(phase="evaluation", steps=self.calls)
            self.last = now
            if self.audit_windows:
                from .windows import visible_windows
                report = visible_windows()
                if len(self.windows) < 32: self.windows.append(report)
                if not report["verified"] or report["onscreen_owned_windows"]:
                    raise RuntimeError("visible Unity window or unavailable window audit")
        return self.model.act(obs)


def evaluate(task, env_factory, heartbeat=lambda **_: None, stop=lambda: False):
    from blackout_env import run_match
    if "bundle_hashes" in task:
        from pathlib import Path
        from .io import file_hash
        for name, checksum in task["bundle_hashes"].items():
            if file_hash(Path(task["bundle"]) / name) != checksum:
                raise ValueError("evaluation export changed")
    candidate = load_export(task["bundle"])
    candidate = ObservedPolicy(candidate, task["config"]["max_episode_steps"], heartbeat, stop, task.get("audit_windows", False))
    opponent = make_opponent(task["opponent"], task["action_seed"]+1)
    torch.manual_seed(task["action_seed"])
    env = None
    start = time.monotonic()
    try:
        heartbeat(phase="environment_start", steps=0)
        env = env_factory()
        result = run_match(env, candidate, opponent, swap_teams=bool(task["side"]))
        return dict(asdict(result), valid=True, replicate=task["replicate"], side=task["side"],
                    opponent=task["opponent"], objective="provided_runner_v1", natural_terminal=True,
                    initial_observation_sha256=candidate.initial_hash, action_seed=task["action_seed"],
                    applied_map_seed_verified=False, requested_seed=None, wall_seconds=time.monotonic()-start,
                    window_audits=candidate.windows, game_outcome_verified=False)
    finally:
        if env is not None: env.close()
