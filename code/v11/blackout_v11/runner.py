"""Detached v11 supervisor with a global compute lock and resumable stages."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from .contracts import CONFIG, config, registration, verify_original
from .io import ROOT, atomic_json, digest, read_json
from .runtime import child_environment, clean_orphans, process_birth, process_memory


def directory_for(cfg, name=None):
    name = cfg["experiment"] if name is None else name
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("invalid experiment name")
    return ROOT / "logs/v11" / name


def is_active(directory):
    path = Path(directory) / "experiment.lock"
    if not path.exists():
        return False
    with path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


def status(directory):
    from .telemetry import estimate
    directory = Path(directory)
    path = directory / "status.json"
    record = read_json(path) if path.exists() else {"state": "not_started"}
    registered = directory / "registration.json"
    profile = directory / "runtime_profile.json"
    profile_summary = None
    if profile.exists():
        measured = read_json(profile)
        profile_summary = dict(path=str(profile), signature=measured["signature"],
            actor_device=measured["actor"]["selected_actor_device"],
            planner_device=measured["actor"]["selected_planner_device"],
            learner_device=measured["learner"]["selected_device"],
            learner_threads=measured["learner"].get("selected_threads", 1),
            actor_threads=measured["actor"]["selected_actor_threads"],
            minibatch_sequences=measured["learner"]["selected_batch_sequences"],
            workers=measured["workers"]["selected_workers"])
    return dict(active=is_active(directory), directory=str(directory), status=record,
                eta=estimate(directory, read_json(registered)["config"]) if registered.exists() else None,
                runtime_profile=profile_summary,
                runs={str(p.parent.relative_to(directory)): read_json(p)
                      for p in directory.glob("**/progress.json")})


def check(cfg, force=True):
    import torch
    from .benchmark import fixture
    from .export import export_policy, verify_export
    from .policy import MyPolicy
    from .validation import validate_algorithms
    torch.set_num_threads(1)
    original = verify_original()
    if shutil.disk_usage(ROOT).free < cfg.get("disk_start_gib", 25) * 1024**3:
        raise RuntimeError("insufficient disk reserve to start")
    signature = digest(dict(preflight_version=1, registration=registration(cfg)))
    report_path = ROOT / "logs/v11/reports/preflight.json"
    if not force and report_path.exists():
        cached = read_json(report_path)
        if cached.get("signature") == signature and cached.get("passed"):
            return dict(cached, cached=True, original=original)
    algorithms = validate_algorithms()
    with tempfile.TemporaryDirectory(prefix="v11-preflight-") as temp:
        with torch.random.fork_rng():
            torch.manual_seed(909)
            policy = MyPolicy().eval()
            policy.configure("F", "greedy")
        bundle = export_policy(policy, Path(temp) / "submission")
        exported = verify_export(bundle, policy, fixture())
    if signature != digest(dict(preflight_version=1, registration=registration(cfg))):
        raise RuntimeError("source/config changed during preflight; retry after edits finish")
    report = dict(passed=True, signature=signature, cached=False, original=original, algorithms=algorithms,
                  submission_fixture=exported, unity_started=False, training_started=False,
                  mps_available_in_this_process=torch.backends.mps.is_available(),
                  actor_parameters=sum(p.numel() for p in policy.parameters()),
                  main_target_steps=len(cfg["arms"]) * len(cfg["seeds"]) * cfg["training_steps"],
                  pilot_steps=len(cfg["arms"]) * cfg["pilot_steps"],
                  confirmation_target_steps=2 * len(cfg["confirmation_seeds"]) * cfg["training_steps"],
                  contract="canonical full-team rows through provided run_match/load_checkpoint")
    atomic_json(report_path, report)
    return report


def prepare_registration(directory, cfg):
    actual = registration(cfg)
    path = Path(directory) / "registration.json"
    if path.exists():
        if read_json(path) != actual:
            from .revision import validated_revision
            revised = validated_revision(directory, cfg, actual)
            if revised is not None:
                return revised["training_registration_sha256"]
            raise ValueError("source/config/package registration changed; use a new --name")
    else:
        atomic_json(path, actual)
    return digest(actual)


def finish_behavior_report(directory):
    """Reporting failures must not replace the study's completion/error state."""
    from .behavior_reports import write_report
    try:
        report = dict(state="complete", **write_report(directory), updated_at=time.time())
    except Exception as exc:
        report = dict(state="failed", error=str(exc), updated_at=time.time())
        print("Behavior report failed: " + str(exc), flush=True)
    try:
        atomic_json(Path(directory) / "behavior_report_status.json", report)
    except OSError as exc:
        print("Cannot save behavior report status: " + str(exc), flush=True)
    return report


def supervise(args, cfg, directory):
    import torch
    from .benchmark import apply_profile, calibrate, smoke
    from .telemetry import memory_pressure, mps_memory
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    operation = "benchmark" if args.benchmark else "smoke" if args.smoke else "study"
    stop_path = directory / "stop.request"
    signal.signal(signal.SIGTERM, lambda *_: stop_path.touch())
    signal.signal(signal.SIGINT, lambda *_: stop_path.touch())
    awake = None
    last_update = last_console = 0.
    peak = 0.
    started = time.monotonic()
    phase_state = {}
    def memory():
        nonlocal peak
        current = process_memory()["rss_gib"]
        peak = max(peak, current)
        return current
    def progress(**data):
        nonlocal last_update, last_console
        phase_state.update(data)
        now = time.monotonic()
        if now-last_update < 2:
            return
        current = memory()
        pressure = memory_pressure()
        metal = mps_memory()
        child_mps = sum(row.get("mps_driver_bytes", 0) for row in data.get("active", []))
        # RSS and Metal can overlap in unified memory; retain both measurements
        # and use their sum as a conservative soft-limit estimate.
        conservative = current + (metal["driver_bytes"] + child_mps) / 1024**3
        reason = None
        if pressure["critical"]:
            reason = "critical system memory pressure"
        elif conservative > cfg.get("memory_soft_gib", 32):
            reason = "unified memory soft limit reached"
        elif shutil.disk_usage(ROOT).free < cfg.get("disk_stop_gib", 25)*1024**3:
            reason = "disk reserve reached"
        if reason:
            stop_path.touch()
            phase_state["stop_reason"] = reason
        record = dict(phase_state)
        record.update(
            state="stopping" if stop_path.exists() else "running", operation=operation,
            pid=os.getpid(), elapsed_seconds=now-started, updated_at=time.time(),
            process_rss_gib=current, peak_process_rss_gib=peak,
            mps_memory=metal, conservative_unified_gib=conservative,
            memory_pressure=pressure)
        atomic_json(directory / "status.json", record)
        last_update = now
        if now-last_console >= 10:
            print(json.dumps(dict(operation=operation, phase=phase_state.get("phase"),
                                  run=phase_state.get("run"), step=phase_state.get("step"),
                                  active_workers=len(data.get("active", [])),
                                  elapsed_seconds=round(now-started, 1)), ensure_ascii=False), flush=True)
            last_console = now
    with os.fdopen(args.supervisor_fd, "a"), os.fdopen(args.machine_fd, "a"):
        try:
            if shutil.which("caffeinate"):
                awake = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())], stdin=subprocess.DEVNULL)
            clean_orphans(directory)
            atomic_json(directory / "supervisor.json", dict(pid=os.getpid(), birth=process_birth(os.getpid()),
                                                            token=args.start_token, operation=operation))
            progress(phase="preflight")
            registered = prepare_registration(directory, cfg)
            check(cfg, force=False)
            if stop_path.exists():
                raise InterruptedError("stopped after preflight")
            if operation == "smoke":
                result = smoke(directory, cfg, stop_path.exists, progress)
                atomic_json(directory / "smoke_result.json", result)
            else:
                from .revision import resumed_runtime_profile
                profile = resumed_runtime_profile(directory, cfg)
                if profile is None:
                    profile = calibrate(directory, cfg, stop_path.exists, progress)
                else:
                    progress(phase="calibration_preserved", profile=str(directory / "runtime_profile.json"))
                resolved = apply_profile(cfg, profile)
                resolved["behavior_output_root"] = str(directory / "behavior")
                atomic_json(directory / "resolved_config.json", resolved)
                if operation == "benchmark":
                    result = dict(runtime_profile=str(directory / "runtime_profile.json"))
                else:
                    torch.set_num_threads(resolved.get("resolved_learner_threads", 1))
                    if resolved.get("ablation"):
                        from .ablation import run_ablation
                        result = run_ablation(directory, resolved, registered, stop_path.exists, progress, memory)
                    else:
                        from .study import run_study
                        result = run_study(directory, resolved, registered, stop_path.exists, progress, memory)
            atomic_json(directory / "status.json", dict(state="complete", operation=operation, result=result,
                        submission=result.get("submission"), elapsed_seconds=time.monotonic()-started,
                        updated_at=time.time()))
        except InterruptedError as exc:
            atomic_json(directory / "status.json", dict(state="stopped", operation=operation, reason=str(exc),
                        resume_command=f"bash {ROOT}/code/v11/run_v11_fast.sh --name {directory.name} --resume",
                        updated_at=time.time()))
        except BaseException as exc:
            atomic_json(directory / "status.json", dict(state="failed", operation=operation, error=str(exc),
                                                        updated_at=time.time()))
            raise
        finally:
            clean_orphans(directory)
            if awake is not None:
                awake.terminate()
                try:
                    awake.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    awake.kill(); awake.wait()
            (directory / "supervisor.json").unlink(missing_ok=True)
            if operation == "study":
                finish_behavior_report(directory)


def main():
    parser = argparse.ArgumentParser(description="V11 experiments, hidden Unity and detached supervisor by default")
    group = parser.add_mutually_exclusive_group()
    for name in ("check", "status", "stop", "resume", "benchmark", "prepare", "smoke", "report"):
        group.add_argument("--" + name, action="store_true")
    parser.add_argument("--name")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), help="learner device; auto measures CPU/MPS")
    parser.add_argument("--actor-device", choices=("auto", "cpu", "mps"))
    parser.add_argument("--planner-device", choices=("auto", "cpu", "mps"))
    parser.add_argument("--ablation", choices=("no_gru", "vectorcritic", "blindcritic", "max2", "max5", "goal10", "goal50"),
                        help="optional paired F experiment; use a separate --name")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--supervisor-fd", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--machine-fd", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--start-token", help=argparse.SUPPRESS)
    args = parser.parse_args()
    cfg = config(args.config.resolve())
    for key in ("device", "actor_device", "planner_device"):
        value = getattr(args, key)
        if value:
            cfg[key] = value
    directory = directory_for(cfg, args.name)
    if args.report:
        if not directory.is_dir():
            parser.error("study does not exist: " + str(directory))
        report = finish_behavior_report(directory)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if report["state"] != "complete":
            raise SystemExit(1)
        return
    if args.ablation:
        cfg["ablation"] = args.ablation
    elif args.resume and (directory / "registration.json").exists():
        previous_ablation = read_json(directory / "registration.json")["config"].get("ablation")
        if previous_ablation:
            cfg["ablation"] = previous_ablation
    if args.status:
        print(json.dumps(status(directory), ensure_ascii=False, indent=2)); return
    if args.stop:
        if not is_active(directory):
            print("v11 is not running"); return
        (directory / "stop.request").touch()
        print("v11 stop requested; committed checkpoints are retained. Use --status to confirm shutdown."); return
    if args.check:
        print(json.dumps(check(cfg), ensure_ascii=False, indent=2)); return
    if args.supervisor_fd is not None:
        supervise(args, cfg, directory); return
    directory.mkdir(parents=True, exist_ok=True)
    shared_lock = ROOT / "logs/runtime/compute.lock"
    shared_lock.parent.mkdir(parents=True, exist_ok=True)
    with (directory / "experiment.lock").open("a") as lock, shared_lock.open("a") as machine:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"v11 is already running; use --name {directory.name} --status"); return
        try:
            fcntl.flock(machine, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another experiment/viewer owns the global worker budget")
        existing = read_json(directory / "status.json") if (directory / "status.json").exists() else {}
        if args.resume and existing.get("operation") in ("smoke", "benchmark"):
            setattr(args, existing["operation"], True)
        if existing.get("operation") == "study" and existing.get("state") == "complete":
            print("v11 study already completed; see " + str(directory / "summary.json")); return
        if existing.get("operation") == "study" and not args.resume:
            raise RuntimeError("existing study found; use --resume with the same source/config")
        from .runtime import check_legacy_processes
        check_legacy_processes()
        if args.prepare:
            print("registration: " + prepare_registration(directory, cfg)); return
        # Refuse mismatched source/config before acknowledging a background start.
        prepare_registration(directory, cfg)
        (directory / "stop.request").unlink(missing_ok=True)
        token = uuid.uuid4().hex
        operation = "benchmark" if args.benchmark else "smoke" if args.smoke else "study"
        command = [sys.executable, "-u", str(ROOT / "code/v11/scripts/v11_experiments.py"),
                   "--config", str(args.config.resolve()), "--name", directory.name,
                   "--supervisor-fd", str(lock.fileno()), "--machine-fd", str(machine.fileno()),
                   "--start-token", token, "--device", cfg.get("device", "auto"),
                   "--actor-device", cfg.get("actor_device", "auto")]
        if "planner_device" in cfg:
            command += ["--planner-device", cfg["planner_device"]]
        if cfg.get("ablation"):
            command += ["--ablation", cfg["ablation"]]
        if operation != "study":
            command.append("--" + operation)
        atomic_json(directory / "status.json", dict(state="starting", operation=operation, updated_at=time.time()))
        with (directory / "console.log").open("ab", buffering=0) as log:
            process = subprocess.Popen(command, cwd=ROOT, env=child_environment(), stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=log, start_new_session=True,
                                       pass_fds=(lock.fileno(), machine.fileno()))
        for _ in range(200):
            if process.poll() is not None:
                raise RuntimeError("background startup failed; inspect " + str(directory / "console.log"))
            ready = directory / "supervisor.json"
            if ready.exists() and read_json(ready).get("token") == token:
                break
            time.sleep(.1)
        else:
            raise RuntimeError("background startup pending; inspect --status before retry")
        print(f"v11 {operation} started in background (PID {process.pid}).\nLogs: {directory}\n"
              f"Status: bash {ROOT}/code/v11/run_v11_fast.sh --name {directory.name} --status")


if __name__ == "__main__":
    main()
