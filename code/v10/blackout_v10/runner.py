"""Detached v10 supervisor. Starting this CLI does not require shell '&' or nohup."""

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
    return project_path(Path("logs/v10") / name, root=ROOT)


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
    from .telemetry import estimate
    registration_path = directory / "registration.json"
    eta = estimate(directory, read_json(registration_path)["config"]) if registration_path.exists() else None
    return {"eta": eta, "active": is_active(directory), "directory": str(directory),
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
    from .value import Critic
    from .validation import validate_algorithms
    algorithms = validate_algorithms()
    if shutil.disk_usage(ROOT).free < cfg["disk_start_gib"]*1024**3:
        raise RuntimeError("insufficient disk reserve to start")
    with tempfile.TemporaryDirectory(prefix="v10-preflight-") as temp:
        torch.manual_seed(909)
        policy = MyPolicy().eval()
        bundle = export_policy(policy, Path(temp) / "submission")
        exported = verify_export(bundle, policy, fixture())
    report = {"passed": True, "original": original, "submission_fixture": exported,
              "unity_started": False, "training_started": False, "algorithms": algorithms,
              "parameters": {"actor": sum(p.numel() for p in MyPolicy().parameters()),
                  **{a: sum(p.numel() for p in Critic(a).parameters()) for a in ("C0", "C1")}},
              "main_target_steps": len(cfg["arms"])*len(cfg["seeds"])*cfg["training_steps"],
              "pilot_budget": len(cfg["arms"])*cfg["pilot_steps"], "confirmation_target_steps": len(cfg["arms"])*len(cfg["confirmation_seeds"])*cfg["training_steps"],
              "workers_initial": cfg["workers"], "workers_cap": cfg["worker_cap"],
              "mps_available_in_this_process": torch.backends.mps.is_available(),
              "contract_limitations": ["self ID absent", "official scoring unverified", "seed application unverified"]}
    atomic_json(project_path('logs/v10/validation/current_layout/preflight.json', root=ROOT), report)
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
            if len(parts) == 2 and ("blackout_v10" in parts[1] or "v10_experiments.py" in parts[1] or str(project_path('artifacts/builds/BlackOut.app', root=ROOT)) in parts[1]):
                total += int(parts[0])*1024
        peak_rss = max(peak_rss, total/1024**3)
        return total/1024**3
    def progress(**data):
        nonlocal last_update, last_console
        now = time.monotonic()
        if now-last_update < 2: return
        current = memory()
        from .telemetry import memory_pressure
        pressure = memory_pressure()
        data["memory_pressure"] = pressure
        if pressure["critical"]:
            stop_path.touch()
            data["stop_reason"] = "critical system memory pressure"
        if current > cfg["memory_soft_gib"]:
            stop_path.touch()
            data["stop_reason"] = "memory soft limit reached"
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
            atomic_json(directory / "supervisor.json", {"pid": os.getpid(), "birth": process_birth(os.getpid()), "token": args.start_token})
            progress(phase="preflight")
            check(cfg)
            reg = prepare_registration(directory, cfg)
            result = run_study(directory, cfg, reg, stop_path.exists, progress, memory)
            atomic_json(directory / "status.json", {"state": "complete", "submission": result["submission"],
                        "elapsed_seconds": time.monotonic()-started, "updated_at": time.time()})
        except InterruptedError:
            atomic_json(directory / "status.json", {"state": "stopped", "resume_command": f"bash {ROOT}/code/v10/run_v10_fast.sh --name {directory.name} --resume", "updated_at": time.time()})
        except BaseException as exc:
            atomic_json(directory / "status.json", {"state": "failed", "error": str(exc), "updated_at": time.time()})
            raise
        finally:
            clean_orphans(directory)
            if awake is not None:
                awake.terminate(); awake.wait(timeout=5)
            (directory / "supervisor.json").unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="V10 original-environment Attention experiments (detached by default)")
    group = parser.add_mutually_exclusive_group()
    for name in ("check", "status", "stop", "resume", "benchmark", "prepare"):
        group.add_argument("--"+name, action="store_true")
    parser.add_argument("--name")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"))
    parser.add_argument("--actor-device", choices=("auto", "cpu", "mps"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--supervisor-fd", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--machine-fd", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--start-token", help=argparse.SUPPRESS)
    args = parser.parse_args()
    cfg = config(args.config.resolve())
    if args.device: cfg["device"] = args.device
    if args.actor_device: cfg["actor_device"] = args.actor_device
    directory = directory_for(cfg, args.name)
    if args.status:
        print(__import__("json").dumps(status(directory), ensure_ascii=False, indent=2)); return
    if args.stop:
        if not is_active(directory): print("v10 is not running"); return
        (directory / "stop.request").touch()
        print("v10 stop requested; committed checkpoint is retained. Use --status to confirm shutdown."); return
    if args.check:
        print(__import__("json").dumps(check(cfg), ensure_ascii=False, indent=2)); return
    if args.benchmark:
        import torch
        torch.set_num_threads(1)
        from .benchmark import benchmark_inference, choose_learner
        report = {"inference": benchmark_inference(), "learner": choose_learner(cfg["device"], cfg["ppo"]["minibatch"])}
        atomic_json(project_path('logs/v10/reports/inference_benchmark.json', root=ROOT), report)
        print(__import__("json").dumps(report, ensure_ascii=False, indent=2)); return
    if args.supervisor_fd is not None:
        supervise(args, cfg, directory); return
    directory.mkdir(parents=True, exist_ok=True)
    shared_lock = ROOT / "logs/runtime/compute.lock"
    shared_lock.parent.mkdir(parents=True, exist_ok=True)
    with (directory / "experiment.lock").open("a") as lock, shared_lock.open("a") as machine:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: print(f"v10 is already running; use --name {directory.name} --status"); return
        try: fcntl.flock(machine, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError("another experiment/viewer owns the global worker budget")
        existing = read_json(directory / "status.json") if (directory / "status.json").exists() else {}
        if existing.get("state") == "complete":
            print("v10 already completed; see " + str(directory / "summary.json")); return
        if existing and not args.resume:
            raise RuntimeError("existing experiment found; use the same command with --resume")
        from .runtime import check_legacy_processes
        check_legacy_processes()
        if args.prepare:
            print("registration: " + prepare_registration(directory, cfg)); return
        (directory / "stop.request").unlink(missing_ok=True)
        token = uuid.uuid4().hex
        command = [sys.executable, "-u", str(project_path('code/v10/scripts/v10_experiments.py', root=ROOT)), "--config", str(args.config.resolve()),
                   "--name", directory.name, "--supervisor-fd", str(lock.fileno()), "--machine-fd", str(machine.fileno()),
                   "--start-token", token]
        command += ["--device", cfg["device"], "--actor-device", cfg["actor_device"]]
        atomic_json(directory / "status.json", {"state": "starting", "updated_at": time.time()})
        with (directory / "console.log").open("ab", buffering=0) as log:
            process = subprocess.Popen(command, cwd=ROOT, env=child_environment(), stdin=subprocess.DEVNULL,
                stdout=log, stderr=log, start_new_session=True, pass_fds=(lock.fileno(), machine.fileno()))
        for _ in range(200):
            if process.poll() is not None: raise RuntimeError("background startup failed; inspect " + str(directory / "console.log"))
            ready = directory / "supervisor.json"
            if ready.exists() and read_json(ready).get("token") == token: break
            time.sleep(.1)
        else: raise RuntimeError("background startup pending; inspect --status before retry")
        print(f"v10 started in background (PID {process.pid}).\nLogs: {directory}\nStatus: bash {project_path('code/v10/run_v10_fast.sh', root=ROOT)} --name {directory.name} --status")


if __name__ == "__main__":
    main()
