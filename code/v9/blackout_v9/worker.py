"""Isolated CPU episode worker; original Unity is its owned process-group child."""
import argparse
import os
from pathlib import Path
import signal
import time
import traceback

STOP = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    from .io import atomic_json, digest, file_hash, read_json
    import torch
    from .contracts import make_env
    from .policy import MyPolicy
    from .value import Critic
    from .opponents import make_opponent
    from .collector import collect_episode
    from .evaluation import evaluate
    task_path = Path(args.task).resolve()
    task = read_json(task_path)
    torch.set_num_threads(task["config"]["actor_threads"])
    torch.set_num_interop_threads(1)
    torch.manual_seed(task["action_seed"]+2)
    def request_stop(*_):
        global STOP
        STOP = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    def heartbeat(**data):
        atomic_json(task_path.parent / "heartbeat.json", dict(time=time.time(), pid=os.getpid(), **data))
    heartbeat(phase="starting", steps=0)
    try:
        if task["kind"] == "collect":
            if file_hash(task["policy_path"]) != task["policy_sha256"]:
                raise ValueError("behavior snapshot changed")
            payload = torch.load(task["policy_path"], map_location="cpu", weights_only=True)
            policy, critic = MyPolicy().eval(), Critic().eval()
            policy.load_state_dict(payload["policy_state"])
            critic.load_state_dict(payload["critic_state"])
            opponent = make_opponent(task["opponent"], task["action_seed"]+1)
            result = collect_episode(policy, critic, opponent, task,
                        lambda: make_env(task["config"]["time_scale"]), heartbeat, lambda: STOP)
        elif task["kind"] == "evaluate":
            result = evaluate(task, lambda: make_env(task["config"]["time_scale"]), heartbeat, lambda: STOP)
        elif task["kind"] == "smoke":
            from .smoke import smoke
            result = smoke(task, lambda: make_env(task["config"]["time_scale"]), heartbeat, lambda: STOP)
        else:
            raise ValueError("unknown worker task")
        atomic_json(task_path.parent / "result.json", {"task_sha256": digest(task), "result": result})
    except BaseException as exc:
        atomic_json(task_path.parent / "error.json", {"task_sha256": digest(task), "error": str(exc),
                    "type": type(exc).__name__, "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
