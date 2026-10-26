"""Isolated episode worker and its owned, rendered-but-hidden Unity child."""
import argparse
import os
from pathlib import Path
import signal
import time
import traceback

STOP = False


def probe_episode(policy, opponent, task, env_factory, heartbeat, stop):
    """Bounded real-environment throughput probe; never a training episode."""
    import numpy as np
    import torch
    from .trajectory import pack, canonical_names
    from .windows import visible_windows
    from .telemetry import mps_memory
    from .behavior import SituationRecorder, decision_metadata
    env = None
    start = time.perf_counter()
    side = task["side"]
    own = canonical_names(side)
    other = canonical_names(1-side)
    names = set(own + other)
    steps = 0
    audits = []
    natural = False
    actor_seconds = opponent_seconds = env_seconds = 0.
    recorder = SituationRecorder.for_task(task)
    try:
        heartbeat(phase="environment_start", steps=0)
        env = env_factory()
        obs, _ = env.reset(seed=task["requested_seed"])
        if recorder is not None:
            recorder.start(obs)
        if set(obs) != names:
            raise ValueError("incomplete probe observations")
        reset_seconds = time.perf_counter() - start
        begun = time.perf_counter()
        last_audit = 0.
        while steps < task["probe_steps"] and time.perf_counter()-begun < task["probe_seconds"]:
            if stop():
                raise InterruptedError("probe stopped")
            now = time.perf_counter()
            if now-last_audit >= 5:
                audit = visible_windows()
                audits.append(audit)
                if not audit["verified"] or audit["onscreen_owned_windows"]:
                    raise RuntimeError("rendered probe window audit failed: " + str(audit))
                heartbeat(phase="probe", steps=steps)
                last_audit = now
            vector, graphic = pack(obs, own)
            if not (graphic[1:] > 0).any():
                raise ValueError("graphic has no visible game content")
            mark = time.perf_counter()
            device = next(policy.parameters()).device
            with torch.inference_mode():
                action = policy(torch.from_numpy(vector).to(device),
                                torch.from_numpy(graphic[None]).expand(5, -1, -1, -1).to(device)).cpu().numpy()
            if action.shape != (5, 2) or not np.isfinite(action).all() or (np.abs(action) > 1).any():
                raise ValueError("invalid probe action")
            current_actor_seconds = time.perf_counter()-mark
            actor_seconds += current_actor_seconds
            actions = dict(zip(own, action))
            mark = time.perf_counter()
            actions.update(opponent.act({n: obs[n] for n in other}, other))
            current_opponent_seconds = time.perf_counter()-mark
            opponent_seconds += current_opponent_seconds
            mark = time.perf_counter()
            next_obs, raw, terms, truncs, infos = env.step(actions)
            env_seconds += time.perf_counter()-mark
            if recorder is not None:
                recorder.observe(steps, obs, next_obs, actions, raw, terms, truncs, infos,
                    decision=decision_metadata(policy, side, action),
                    inference_seconds={side:current_actor_seconds, 1-side:current_opponent_seconds})
            obs = next_obs
            steps += 1
            if any(set(row) != names for row in (raw, terms, truncs)):
                raise ValueError("incomplete probe transition")
            if any(truncs.values()) or len(set(map(bool, terms.values()))) != 1:
                raise ValueError("unexpected truncation or partial termination")
            if all(terms.values()):
                natural = True
                break
        seconds = time.perf_counter()-begun
        heartbeat(phase="probe_complete", steps=steps)
        if not steps:
            raise RuntimeError("probe made no environment progress")
        return dict(valid=True, env_steps=steps, episode_steps=steps, training_steps=0,
                    natural_terminal=natural, truncated_for_benchmark=not natural,
                    wall_seconds=time.perf_counter()-start, simulation_seconds=seconds,
                    env_steps_per_second=steps/max(seconds, 1e-9), reset_seconds=reset_seconds,
                    inference_seconds=actor_seconds, opponent_seconds=opponent_seconds,
                    environment_seconds=env_seconds, window_audits=audits,
                    mps_memory=mps_memory(), graphic_verified=True,
                    behavior_log=recorder.close(complete=natural) if recorder is not None else {"enabled":False})
    except BaseException as exc:
        if recorder is not None:
            recorder.close(error=exc)
        raise
    finally:
        if recorder is not None:
            recorder.close()
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    import torch
    from .contracts import make_env
    from .io import atomic_json, digest, file_hash, read_json
    from .opponents import make_opponent
    from .policy import MyPolicy
    from .telemetry import mps_memory
    task_path = Path(args.task).resolve()
    task = read_json(task_path)
    cfg = task["config"]
    torch.set_num_threads(cfg.get("actor_threads", 1))
    torch.set_num_interop_threads(1)
    torch.manual_seed(task["action_seed"] + 2)
    def request_stop(*_):
        global STOP
        STOP = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    def heartbeat(**data):
        metal = mps_memory()
        atomic_json(task_path.parent / "heartbeat.json", dict(time=time.time(), pid=os.getpid(),
                    mps_driver_bytes=metal["driver_bytes"], **data))
    heartbeat(phase="starting", steps=0)
    try:
        if task["kind"] in ("collect", "probe"):
            policy = MyPolicy().eval()
            if task["kind"] == "collect":
                if file_hash(task["policy_path"]) != task["policy_sha256"]:
                    raise ValueError("behavior snapshot changed")
                payload = torch.load(task["policy_path"], map_location="cpu", weights_only=True)
                policy.configure(payload.get("arm", task.get("arm", "F")), cfg.get("decoder", "sample"))
                policy.load_state_dict(payload["policy_state"])
            else:
                policy.configure("F", "greedy")
            policy.to(cfg.get("resolved_actor_device", "cpu"))
            policy.configure_devices(planner_device=cfg.get("resolved_planner_device", "cpu"))
            opponent = make_opponent(task["opponent"], task["action_seed"] + 1)
            env_factory = lambda: make_env(cfg["time_scale"])
            if task["kind"] == "collect":
                from .collector import collect_episode
                result = collect_episode(policy, None, opponent, task, env_factory, heartbeat, lambda: STOP)
            else:
                result = probe_episode(policy, opponent, task, env_factory, heartbeat, lambda: STOP)
        elif task["kind"] == "evaluate":
            from .evaluation import evaluate
            result = evaluate(task, lambda: make_env(cfg["time_scale"]), heartbeat, lambda: STOP)
        else:
            raise ValueError("unknown worker task: " + str(task["kind"]))
        atomic_json(task_path.parent / "result.json", dict(task_sha256=digest(task), result=result))
    except BaseException as exc:
        atomic_json(task_path.parent / "error.json", dict(task_sha256=digest(task), error=str(exc),
                    type=type(exc).__name__, traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
