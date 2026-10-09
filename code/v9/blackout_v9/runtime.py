"""Bounded spawn scheduler with external watchdogs and owned-process cleanup."""

from project_paths import project_root, project_path

import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .io import ROOT, append, atomic_json, digest, read_json


def process_birth(pid):
    return subprocess.run(["ps", "-p", str(pid), "-o", "lstart="], capture_output=True, text=True).stdout.strip()


def owned_members(record):
    result = subprocess.run(["ps", "-axo", "pid=,pgid=,command="], capture_output=True, text=True)
    members = []
    for line in result.stdout.splitlines():
        parts = line.split(None, 2)
        if len(parts) != 3 or int(parts[1]) != record["pid"]:
            continue
        if ("blackout_v9.worker" in parts[2] or str(project_path('artifacts/builds/BlackOut.app', root=ROOT)) in parts[2]
                or "tests.v9.runtime_fixture" in parts[2]):
            members.append({"pid": int(parts[0]), "birth": process_birth(int(parts[0]))})
    return members


def terminate_owned(record, sig=signal.SIGTERM):
    if record.get("birth") and process_birth(record["pid"]) == record["birth"]:
        try: os.killpg(record["pid"], sig)
        except ProcessLookupError: pass
    # Also handle a dead worker leader with recorded surviving Unity children.
    for child in record.get("children", []):
        if child.get("birth") and process_birth(child["pid"]) == child["birth"]:
            try: os.kill(child["pid"], sig)
            except ProcessLookupError: pass


def child_environment():
    env = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[name] = "1"
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def clean_orphans(directory):
    for path in Path(directory).glob("**/ownership.json"):
        record = read_json(path)
        if not record.get("closed"):
            terminate_owned(record)
            time.sleep(.1)
            terminate_owned(record, signal.SIGKILL)
            atomic_json(path, dict(record, closed=True, recovered=True))


def run_tasks(tasks, directory, cfg, stop=lambda: False, progress=lambda **_: None,
              worker_module="blackout_v9.worker"):
    """Each task is one complete episode; no results cross a wave/version."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Fail before spawning if the current sandbox cannot audit process ownership.
    if not process_birth(os.getpid()):
        raise RuntimeError("process ownership audit unavailable")
    pending, active, completed = list(tasks), {}, []
    stopping_at = None
    try:
        while pending or active:
            if stop() and stopping_at is None:
                stopping_at = time.monotonic()
                pending.clear()
                for item in active.values():
                    terminate_owned(item["owner"])
            while pending and len(active) < cfg["workers"] and stopping_at is None:
                task = pending.pop(0)
                path = directory / task["id"]
                path.mkdir(parents=True, exist_ok=False)
                task = dict(task, trajectory=str(path / "trajectory"), invalid_path=str(path / "invalid.json"))
                atomic_json(path / "task.json", task)
                log = (path / "console.log").open("ab", buffering=0)
                process = subprocess.Popen([sys.executable, "-u", "-m", worker_module, "--task", str(path / "task.json")],
                    cwd=ROOT, env=child_environment(), stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
                log.close()
                owner = {"pid": process.pid, "birth": process_birth(process.pid), "children": [], "closed": False}
                atomic_json(path / "ownership.json", owner)
                active[task["id"]] = dict(process=process, path=path, task=task, owner=owner,
                                           started=time.monotonic(), last_scan=0., watchdog=None)
            for name, item in list(active.items()):
                process, path = item["process"], item["path"]
                now = time.monotonic()
                if now-item["last_scan"] > 5:
                    item["owner"]["children"] = owned_members(item["owner"])
                    atomic_json(path / "ownership.json", item["owner"])
                    item["last_scan"] = now
                heartbeat_path = path / "heartbeat.json"
                heartbeat = read_json(heartbeat_path) if heartbeat_path.exists() else {}
                age = time.time()-heartbeat.get("time", time.time()-(now-item["started"]))
                reason = None
                if now-item["started"] > cfg["episode_wall_seconds"]:
                    reason = "external episode wall watchdog"
                if age > cfg["stall_seconds"]:
                    reason = "external heartbeat watchdog"
                if (path / "console.log").stat().st_size > 64*1024**2:
                    reason = "worker console budget exceeded"
                if stopping_at is not None and now-stopping_at > cfg["stop_grace_seconds"]:
                    reason = "stop grace expired"
                if reason and process.poll() is None:
                    item["watchdog"] = reason
                    terminate_owned(item["owner"], signal.SIGKILL)
                code = process.poll()
                if code is None: continue
                current_birth = process_birth(item["owner"]["pid"])
                if not current_birth or current_birth == item["owner"]["birth"]:
                    item["owner"]["children"] = owned_members(item["owner"])
                terminate_owned(item["owner"])
                atomic_json(path / "ownership.json", dict(item["owner"], closed=True))
                result = None
                if code == 0 and not item["watchdog"] and (path / "result.json").exists():
                    result = read_json(path / "result.json")
                    if result["task_sha256"] != digest(item["task"]):
                        raise ValueError("worker result/task mismatch")
                error_path = path / "error.json"
                error = read_json(error_path).get("error") if error_path.exists() else None
                invalid_path = path / "invalid.json"
                invalid = read_json(invalid_path) if invalid_path.exists() else {}
                actual = result["result"] if result is not None else {}
                measured_steps = actual.get("length", actual.get("episode_steps", actual.get("steps", invalid.get("steps"))))
                row = {"id": name, "task": item["task"], "directory": str(path),
                       "valid": result is not None and result["result"].get("valid", False),
                       "result": None if result is None else result["result"],
                       "error": item["watchdog"] or error or (None if result else f"worker exit {code}"),
                       "attempt_steps": heartbeat.get("steps", 0) if measured_steps is None else measured_steps,
                       "step_count_complete": measured_steps is not None, "interrupted": stopping_at is not None}
                atomic_json(path / "attempt.json", row)
                completed.append(row)
                del active[name]
            progress(active=[dict(id=name, **(read_json(item["path"] / "heartbeat.json")
                               if (item["path"] / "heartbeat.json").exists() else {})) for name, item in active.items()],
                     pending=len(pending), completed=len(completed))
            if active: time.sleep(.25)
        return completed
    finally:
        for item in active.values():
            terminate_owned(item["owner"])
        deadline = time.monotonic()+min(cfg["stop_grace_seconds"], 5)
        while active and time.monotonic() < deadline:
            active = {k: v for k, v in active.items() if v["process"].poll() is None}
            if active: time.sleep(.1)
        for item in active.values():
            terminate_owned(item["owner"], signal.SIGKILL)
            item["process"].wait(timeout=5)
