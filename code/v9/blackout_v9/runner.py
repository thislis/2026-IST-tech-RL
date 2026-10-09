"""Detached v9 supervisor. Starting this CLI does not require shell '&' or nohup."""

from project_paths import project_root, project_path

import argparse
import fcntl
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from .contracts import CONFIG, config, registration, verify_original
from .io import ROOT, atomic_json, digest, read_json
from .runtime import child_environment, clean_orphans, process_birth


def directory_for(cfg, name=None):
    name = cfg["experiment"] if name is None else name
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name): raise ValueError("invalid experiment name")
    return project_path(Path("logs/v9") / name, root=ROOT)


def is_active(directory):
    path = Path(directory) / "experiment.lock"
    if not path.exists(): return False
    with path.open("a") as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: return True
    return False


def status(directory):
    directory = Path(directory)
    path = directory / "status.json"
    record = read_json(path) if path.exists() else {"state": "not_started"}
    if record.get("submission"):
        record["submission"] = str(project_path(record["submission"], root=ROOT))
    return {"active": is_active(directory), "directory": str(directory),
            "status": record,
            "runs": {str(p.parent.relative_to(directory)): read_json(p)
                     for p in directory.glob("**/progress.json")}}


def check(cfg):
    import torch
    from .benchmark import fixture
    from .export import export_policy, verify_export
    from .policy import MyPolicy
    import tempfile
    torch.set_num_threads(1)
    original = verify_original()
    if shutil.disk_usage(ROOT).free < cfg["disk_start_gib"]*1024**3:
        raise RuntimeError("insufficient disk reserve to start")
    with tempfile.TemporaryDirectory(prefix="v9-preflight-") as temp:
        torch.manual_seed(909)
        policy = MyPolicy().eval()
        bundle = export_policy(policy, Path(temp) / "submission")
        exported = verify_export(bundle, policy, fixture())
    report = {"passed": True, "original": original, "submission_fixture": exported,
              "unity_started": False, "training_started": False,
              "main_target_steps": len(cfg["arms"])*len(cfg["seeds"])*cfg["training_steps"],
              "pilot_budget": cfg["pilot_steps"], "confirmation_target_steps": len(cfg["confirmation_seeds"])*cfg["training_steps"],
              "workers_initial": cfg["workers"], "workers_cap": cfg["worker_cap"],
              "mps_available_in_this_process": torch.backends.mps.is_available(),
              "contract_limitations": ["self ID absent", "official scoring unverified", "seed application unverified"]}
    atomic_json(project_path('logs/v9/validation/current_layout/preflight.json', root=ROOT), report)
    return report


def prepare_registration(directory, cfg):
    actual = registration(cfg)
    path = directory / "registration.json"
    if path.exists():
        if read_json(path) != actual:
            raise ValueError("source/config/package registration changed; use a new --name instead of mutating old runs")
    else: atomic_json(path, actual)
    return digest(actual)


def supervise(args, cfg, directory):
    import torch
    from .study import run_study
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    stop_path = directory / "stop.request"
    signal.signal(signal.SIGTERM, lambda *_: stop_path.touch())
    signal.signal(signal.SIGINT, lambda *_: stop_path.touch())
    awake = None
    last_update = 0.
    last_console = 0.
    peak_rss = 0.
    started = time.monotonic()
    def memory():
        nonlocal peak_rss
        result = subprocess.run(["ps", "-axo", "rss=,command="], capture_output=True, text=True)
        total = 0
        for line in result.stdout.splitlines():
            parts = line.split(None, 1)
            if len(parts) == 2 and ("blackout_v9" in parts[1] or "v9_experiments.py" in parts[1] or str(project_path('artifacts/builds/BlackOut.app', root=ROOT)) in parts[1]):
                total += int(parts[0])*1024
        peak_rss = max(peak_rss, total/1024**3)
        return total/1024**3
    def progress(**data):
        nonlocal last_update, last_console
        now = time.monotonic()
        if now-last_update < 2: return
        current = memory()
        if shutil.disk_usage(ROOT).free < cfg["disk_stop_gib"]*1024**3:
            stop_path.touch()
            data["stop_reason"] = "disk reserve reached"
        atomic_json(directory / "status.json", {"state": "stopping" if stop_path.exists() else "running",
                    "pid": os.getpid(), "elapsed_seconds": now-started, "updated_at": time.time(),
                    "process_rss_gib": current, "peak_process_rss_gib": peak_rss, **data})
        last_update = now
        if now-last_console >= 10:
            print(__import__("json").dumps({"phase": data.get("phase"), "run": data.get("run"),
                  "step": data.get("step"), "active_workers": len(data.get("active", [])),
                  "elapsed_seconds": round(now-started, 1)}, ensure_ascii=False), flush=True)
            last_console = now
    with os.fdopen(args.supervisor_fd, "a"), os.fdopen(args.machine_fd, "a"):
        try:
            if shutil.which("caffeinate"):
                awake = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())], stdin=subprocess.DEVNULL)
            clean_orphans(directory)
            reg = prepare_registration(directory, cfg)
            atomic_json(directory / "supervisor.json", {"pid": os.getpid(), "birth": process_birth(os.getpid()), "token": args.start_token})
            progress(phase="starting")
            result = run_study(directory, cfg, reg, stop_path.exists, progress, memory)
            atomic_json(directory / "status.json", {"state": "complete", "submission": result["submission"],
                        "elapsed_seconds": time.monotonic()-started, "updated_at": time.time()})
        except InterruptedError:
            atomic_json(directory / "status.json", {"state": "stopped", "resume_command": "bash run_v9_fast.sh --resume", "updated_at": time.time()})
        except BaseException as exc:
            atomic_json(directory / "status.json", {"state": "failed", "error": str(exc), "updated_at": time.time()})
            raise
        finally:
            clean_orphans(directory)
            if awake is not None:
                awake.terminate(); awake.wait(timeout=5)
            (directory / "supervisor.json").unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="V9 original-environment Attention experiments (detached by default)")
    group = parser.add_mutually_exclusive_group()
    for name in ("check", "status", "stop", "resume", "benchmark"):
        group.add_argument("--"+name, action="store_true")
    parser.add_argument("--name")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--supervisor-fd", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--machine-fd", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--start-token", help=argparse.SUPPRESS)
    args = parser.parse_args()
    cfg = config(args.config.resolve())
    directory = directory_for(cfg, args.name)
    if args.status:
        print(__import__("json").dumps(status(directory), ensure_ascii=False, indent=2)); return
    if args.stop:
        if not is_active(directory): print("v9 is not running"); return
        (directory / "stop.request").touch()
        print("v9 stop requested; committed checkpoint is retained. Use --status to confirm shutdown."); return
    if args.check:
        print(__import__("json").dumps(check(cfg), ensure_ascii=False, indent=2)); return
    if args.benchmark:
        from .benchmark import benchmark_inference
        report = benchmark_inference()
        atomic_json(project_path('logs/v9/reports/inference_benchmark.json', root=ROOT), report)
        print(__import__("json").dumps(report, ensure_ascii=False, indent=2)); return
    if args.supervisor_fd is not None:
        supervise(args, cfg, directory); return
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "experiment.lock").open("a") as lock, (project_path('logs/v9/machine.lock', root=ROOT)).open("a") as machine:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: print("v9 is already running; use --status"); return
        try: fcntl.flock(machine, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError("another v9 supervisor owns the global worker budget")
        existing = read_json(directory / "status.json") if (directory / "status.json").exists() else {}
        if existing.get("state") == "complete":
            print("v9 already completed; see " + str(directory / "summary.json")); return
        if existing and not args.resume:
            raise RuntimeError("existing experiment found; use the same command with --resume")
        check(cfg)
        prepare_registration(directory, cfg)
        (directory / "stop.request").unlink(missing_ok=True)
        token = uuid.uuid4().hex
        command = [sys.executable, "-u", str(project_path('code/v9/scripts/v9_experiments.py', root=ROOT)), "--config", str(args.config.resolve()),
                   "--name", directory.name, "--supervisor-fd", str(lock.fileno()), "--machine-fd", str(machine.fileno()),
                   "--start-token", token]
        with (directory / "console.log").open("ab", buffering=0) as log:
            process = subprocess.Popen(command, cwd=ROOT, env=child_environment(), stdin=subprocess.DEVNULL,
                stdout=log, stderr=log, start_new_session=True, pass_fds=(lock.fileno(), machine.fileno()))
        for _ in range(200):
            if process.poll() is not None: raise RuntimeError("background startup failed; inspect " + str(directory / "console.log"))
            ready = directory / "supervisor.json"
            if ready.exists() and read_json(ready).get("token") == token: break
            time.sleep(.1)
        else: raise RuntimeError("background startup pending; inspect --status before retry")
        print(f"v9 started in background (PID {process.pid}).\nLogs: {directory}\nStatus: bash {project_path('code/v9/run_v9_fast.sh', root=ROOT)} --status")


if __name__ == "__main__":
    main()
