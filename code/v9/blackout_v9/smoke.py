"""Explicit bounded original-Unity connection diagnostic, never a training run."""
import time
import numpy as np
import torch
from .collector import observation_hash
from .policy import MyPolicy
from .trajectory import pack
from .windows import visible_windows


def smoke(task, env_factory, heartbeat=lambda **_: None, stop=lambda: False):
    env = None
    policy = MyPolicy().eval()
    audits = []
    first = None
    steps = 0
    started = time.monotonic()
    try:
        heartbeat(phase="smoke_start", steps=0)
        env = env_factory()
        obs, _ = env.reset(seed=90909)
        with torch.inference_mode():
            for step in range(task.get("diagnostic_steps", 32)):
                if stop(): raise InterruptedError("smoke stopped")
                if step % 8 == 0:
                    audit = visible_windows(); audits.append(audit)
                    if not audit["verified"] or audit["onscreen_owned_windows"]:
                        raise RuntimeError("window audit failed: " + str(audit))
                before = observation_hash(obs)
                if first is None: first = before
                own = [n for n in obs if int(n.split("_")[1]) < 5]
                vectors, graphic = pack(obs, own)
                if graphic.shape != (11, 96, 96) or not np.any(graphic[1:] > 0):
                    raise ValueError("missing rendered graphic content")
                vt, gt = torch.from_numpy(vectors), torch.from_numpy(np.repeat(graphic[None], 5, axis=0))
                action = policy(vt, gt).numpy()
                actions = {n: np.zeros(2, np.float32) for n in obs}
                actions.update(dict(zip(own, action)))
                if observation_hash(obs) != before: raise ValueError("model mutated supplied observations")
                obs, raw, terms, truncs, infos = env.step(actions)
                steps += 1
                heartbeat(phase="smoke", steps=steps)
                if all(terms.values()): break
        changed = observation_hash(obs) != first
        if not changed: raise ValueError("observations did not progress")
        return {"valid": True, "diagnostic_only": True, "steps": steps,
                "graphic_shape": [5, 11, 96, 96], "vector_shape": [5, 96],
                "observation_changed": changed, "window_audits": audits,
                "training_updates": 0, "full_match_evaluation": False,
                "wall_seconds": time.monotonic()-started}
    finally:
        if env is not None: env.close()


def main():
    from .contracts import config, verify_original
    from .io import ROOT, atomic_json
    from .runtime import run_tasks
    cfg = config(); cfg = dict(cfg, workers=1, episode_wall_seconds=90, stall_seconds=45)
    verify_original()
    directory = ROOT / "reports/v9/live_smoke" / str(time.time_ns())
    task = {"id": "original-rendered-32-steps", "kind": "smoke", "config": cfg,
            "action_seed": 9009, "diagnostic_steps": 32}
    results = run_tasks([task], directory, cfg)
    report = {"results": results, "original_after": verify_original(),
              "passed": all(r["valid"] for r in results), "training_started": False}
    atomic_json(ROOT / "reports/v9/live_smoke.json", report)
    print(__import__("json").dumps(report, indent=2))
    if not report["passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
