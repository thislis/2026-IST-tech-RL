"""Measure actual v11 workloads; select devices independently and freeze results."""
import copy
import gc
import importlib.metadata
from pathlib import Path
import statistics
import shutil
import time
import torch
from .io import ROOT, atomic_json, digest, read_json
from .telemetry import hardware_identity, mps_memory


def fixture():
    """Canonical team observation with traversable terrain and real candidates."""
    vector = torch.zeros(5, 96, dtype=torch.float32)
    entities = vector[:, :90].reshape(5, 10, 9)
    positions = torch.tensor([[.12, .15], [.16, .20], [.20, .15], [.25, .20], [.20, .25],
                              [.85, .80], [.80, .85], [.75, .80], [.80, .75], [.75, .75]])
    entities[:, :, :2] = positions
    entities[:, :, 2] = torch.tensor([1.] * 5 + [-1.] * 5)
    entities[:, :, 3] = 1.
    vector[:, 90] = 1.
    vector[:, 95] = 1.
    graphic = torch.zeros(5, 11, 96, 96, dtype=torch.float32)
    graphic[:, 0] = 1.
    for channel, row, col in [(2, 20, 3), (3, 3, 20), (6, 12, 8), (6, 10, 15),
                               (6, 16, 13), (7, 8, 18), (1, 8, 10), (1, 9, 10)]:
        graphic[:, 0, row*4:(row+1)*4, col*4:(col+1)*4] = 0.
        graphic[:, channel, row*4:(row+1)*4, col*4:(col+1)*4] = 1.
    return vector, graphic


def sync(device):
    if str(device).startswith("mps"):
        torch.mps.synchronize()


def _devices(requested="auto"):
    if requested not in ("auto", "cpu", "mps"):
        raise ValueError("invalid device request")
    candidates = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    if requested != "auto":
        if requested not in candidates:
            raise RuntimeError("requested MPS device is unavailable in this process")
        return [requested]
    return candidates


def benchmark_inference(repetitions=20, requested="auto", planner_requested="auto", progress=lambda **_: None):
    from .policy import MyPolicy
    prior_threads = torch.get_num_threads()
    vector, graphic = fixture()
    with torch.random.fork_rng():
        torch.manual_seed(901)
        reference = MyPolicy().eval()
        reference.configure("F", "greedy")
    reports = []
    try:
        for threads in (1, 2):
            torch.set_num_threads(threads)
            for actor_device in _devices(requested):
                for planner_device in _devices(planner_requested):
                    progress(phase="calibrate_actor", actor_device=actor_device,
                             planner_device=planner_device, actor_threads=threads)
                    candidate = copy.deepcopy(reference).to(actor_device)
                    candidate.configure_devices(planner_device=planner_device)
                    durations = []
                    error = None
                    try:
                        with torch.inference_mode():
                            for index in range(repetitions + 5):
                                v = vector.clone()
                                v[:, 95] -= index / 21000.
                                sync(actor_device); sync(planner_device)
                                start = time.perf_counter()
                                action = candidate(v.to(actor_device), graphic.to(actor_device)).cpu()
                                sync(actor_device); sync(planner_device)
                                elapsed = time.perf_counter() - start
                                if not torch.isfinite(action).all() or action.abs().max() > 1.000001:
                                    raise ValueError("invalid actor output")
                                if index >= 5:
                                    durations.append(elapsed)
                        reports.append(dict(valid=True, actor_device=actor_device, planner_device=planner_device,
                                            threads=threads, seconds=statistics.mean(durations),
                                            p50=statistics.median(durations),
                                            p95=sorted(durations)[min(len(durations)-1, int(.95*len(durations)))],
                                            includes_transfer=True, includes_planner=True,
                                            mps_memory=mps_memory()))
                    except (RuntimeError, ValueError, AssertionError) as exc:
                        error = str(exc)
                        reports.append(dict(valid=False, actor_device=actor_device,
                                            planner_device=planner_device, threads=threads, error=error))
                    finally:
                        del candidate
                        if torch.backends.mps.is_available():
                            torch.mps.empty_cache()
        valid = [row for row in reports if row["valid"]]
        if not valid:
            raise RuntimeError("no valid actor/planner combination: " + str(reports))
        selected = min(valid, key=lambda row: row["seconds"])
        return dict(selected_actor_device=selected["actor_device"],
                    selected_planner_device=selected["planner_device"],
                    selected_actor_threads=selected["threads"], measurements=reports,
                    unity_started=False, persisted_updates=0)
    finally:
        torch.set_num_threads(prior_threads)


def choose_learner(requested="auto", candidates=(4, 8, 16, 32, 64), progress=lambda **_: None,
                   memory_limit_gib=32, cpu_threads=(1, 4)):
    from .recurrent_ppo import benchmark_update
    from .runtime import process_memory
    reports = []
    prior_threads = torch.get_num_threads()
    try:
        for device in _devices(requested):
            for threads in cpu_threads if device == "cpu" else (1,):
                torch.set_num_threads(threads)
                for sequences in candidates:
                    progress(phase="calibrate_learner", learner_device=device,
                             learner_threads=threads, batch_sequences=sequences)
                    try:
                        row = benchmark_update(device, batch_sequences=sequences, steps=4)
                        row.update(device=device, batch_sequences=sequences, threads=threads)
                        row["valid"] = bool(row.get("finite_gradients", False) and row.get("parameters_changed", False))
                        rss = max(process_memory()["rss_bytes"], row.get("peak_rss_bytes", 0))
                        metal = max(row.get("peak_mps_driver_bytes", 0), row.get("memory_allocated") or 0)
                        row["conservative_peak_memory_gib"] = (rss + metal) / 1024**3
                        row["memory_limit_gib"] = memory_limit_gib
                        if row["conservative_peak_memory_gib"] > memory_limit_gib:
                            row["valid"] = False
                            row["error"] = "measured recurrent sequence exceeds unified-memory soft limit"
                        elif not row["valid"]:
                            row["error"] = "nonfinite gradients or optimizer did not change parameters"
                        if not row.get("events_per_second"):
                            row["events_per_second"] = sequences * 4 / row["seconds"]
                        reports.append(row)
                    except (RuntimeError, ValueError, AssertionError) as exc:
                        reports.append(dict(device=device, batch_sequences=sequences, threads=threads,
                                            valid=False, error=str(exc)))
                    finally:
                        gc.collect()
                        if torch.backends.mps.is_available():
                            torch.mps.empty_cache()
    finally:
        torch.set_num_threads(prior_threads)
    valid = [row for row in reports if row["valid"]]
    if not valid:
        raise RuntimeError("no valid learner device: " + str(reports))
    selected = max(valid, key=lambda row: row["events_per_second"])
    return dict(selected_device=selected["device"],
                selected_threads=selected["threads"],
                selected_batch_sequences=selected["batch_sequences"], probes=reports,
                synthetic_only=True, persisted_updates=0)


def _probe_tasks(cfg, count, round_id):
    opponent = cfg.get("benchmark_opponent")
    if opponent is None:
        from .opponents import benchmark_opponent
        opponent = benchmark_opponent()
    return [dict(id=f"probe-{round_id}-{index:02d}", kind="probe", config=cfg,
                 side=index % 2, requested_seed=810000 + index // 2,
                 action_seed=910000 + index, opponent=opponent,
                 probe_seconds=cfg.get("benchmark_probe_seconds", cfg.get("benchmark_seconds", 12)),
                 probe_steps=cfg.get("benchmark_probe_steps", cfg.get("benchmark_steps", 256)), audit_windows=True)
            for index in range(count)]


def benchmark_workers(directory, cfg, stop=lambda: False, progress=lambda **_: None):
    from .runtime import run_tasks, process_memory
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    reports = []
    for count in cfg.get("benchmark_workers", [2, 4, 8]):
        if count > cfg["worker_cap"]:
            continue
        if stop():
            raise InterruptedError("stopped during worker calibration")
        # Match the learner's complete-wave admission rule, including lossless
        # float graphic fallback. Fast workers must also fit a real train wave.
        required = count * cfg.get("max_episode_steps", 22000) * (11*96*96*4 + 16384)
        reserve = cfg.get("disk_stop_gib", 25) * 1024**3
        free = shutil.disk_usage(directory).free
        if free < required + reserve:
            reports.append(dict(workers=count, valid=False, skipped=True, env_steps=0,
                                training_steps=0, required_wave_bytes=required,
                                reserve_bytes=reserve, free_bytes=free,
                                failures=["insufficient disk for a complete lossless training wave"]))
            atomic_json(directory / "worker_probe_progress.json", reports)
            continue
        probe_cfg = dict(cfg, workers=count)
        peak = 0.
        def measured_progress(**data):
            nonlocal peak
            peak = max(peak, process_memory()["rss_gib"])
            progress(phase="calibrate_workers", workers=count, **data)
        started = time.perf_counter()
        tasks = _probe_tasks(probe_cfg, count, str(count))
        rows = run_tasks(tasks, directory / f"workers-{count}-{time.time_ns()}", probe_cfg,
                         stop, measured_progress)
        if stop():
            raise InterruptedError("stopped during worker calibration")
        seconds = time.perf_counter() - started
        valid = len(rows) == count and all(row["valid"] for row in rows)
        steps = sum(row.get("attempt_steps", 0) for row in rows if row["valid"])
        valid = valid and steps > 0 and peak <= cfg.get("memory_soft_gib", 32)
        reports.append(dict(workers=count, valid=valid, env_steps=steps, seconds=seconds,
                            env_steps_per_second=steps/max(seconds, 1e-9), peak_rss_gib=peak,
                            failures=[row["error"] for row in rows if not row["valid"]],
                            required_wave_bytes=required, reserve_bytes=reserve, free_bytes=free,
                            training_steps=0, natural_games=sum(bool(row.get("result", {}).get("natural_terminal"))
                                                               for row in rows if row.get("result"))))
        atomic_json(directory / "worker_probe_progress.json", reports)
    valid = [row for row in reports if row["valid"]]
    if not valid:
        raise RuntimeError("all real-environment worker probes failed: " + str(reports))
    best = max(row["env_steps_per_second"] for row in valid)
    selected = min((row for row in valid if row["env_steps_per_second"] >= best * .95),
                   key=lambda row: row["workers"])
    return dict(selected_workers=selected["workers"], probes=reports,
                selection="maximum valid env steps/s; within 5 percent prefer fewer workers")


def apply_profile(cfg, profile):
    cfg = copy.deepcopy(cfg)
    cfg.update(resolved_actor_device=profile["actor"]["selected_actor_device"],
               resolved_planner_device=profile["actor"]["selected_planner_device"],
               resolved_learner_device=profile["learner"]["selected_device"],
               resolved_learner_threads=profile["learner"].get("selected_threads", 1),
               actor_threads=profile["actor"]["selected_actor_threads"],
               workers=profile["workers"]["selected_workers"])
    cfg["ppo"]["minibatch_sequences"] = profile["learner"]["selected_batch_sequences"]
    cfg["runtime_profile_sha256"] = digest(profile)
    return cfg


def calibrate(directory, cfg, stop=lambda: False, progress=lambda **_: None):
    from .contracts import registration
    directory = Path(directory)
    identity = hardware_identity()
    registered = registration(cfg)
    signature = digest(dict(hardware=identity, sources=registered.get("sources"),
                            packages=registered.get("packages"), config=cfg))
    path = directory / "runtime_profile.json"
    cache = ROOT / "logs/v11/runtime_profiles" / (signature + ".json")
    for candidate in (path, cache):
        if candidate.exists():
            profile = read_json(candidate)
            if profile.get("signature") != signature:
                if candidate == path:
                    raise ValueError("registered runtime profile changed; use a new experiment name")
                continue
            atomic_json(path, profile)
            progress(phase="calibration_cached", profile=str(path))
            return profile
    actor = benchmark_inference(cfg.get("benchmark_actor_repetitions", 24),
                                cfg.get("actor_device", "auto"), cfg.get("planner_device", "auto"), progress)
    learner = choose_learner(cfg.get("device", "auto"),
                             cfg.get("benchmark_sequence_batches", cfg.get("benchmark_sequences", [4, 8, 16, 32, 64])),
                             progress, cfg.get("memory_soft_gib", 32), cfg.get("benchmark_learner_threads", [1, 4]))
    probe_cfg = copy.deepcopy(cfg)
    probe_cfg["behavior_output_root"] = str(directory / "behavior")
    probe_cfg.update(resolved_actor_device=actor["selected_actor_device"],
                     resolved_planner_device=actor["selected_planner_device"],
                     actor_threads=actor["selected_actor_threads"])
    workers = benchmark_workers(directory / "calibration", probe_cfg, stop, progress)
    profile = dict(schema="blackout.v11.runtime.1", signature=signature, hardware=identity,
                   actor=actor, learner=learner, workers=workers, frozen=True,
                   measured_at=time.time(), training_steps=0)
    atomic_json(path, profile)
    atomic_json(cache, profile)
    return profile


def smoke(directory, cfg, stop=lambda: False, progress=lambda **_: None):
    from .runtime import run_tasks
    smoke_cfg = dict(cfg, workers=1, actor_threads=1, resolved_actor_device="cpu",
                     resolved_planner_device="cpu", benchmark_probe_seconds=10, benchmark_probe_steps=128,
                     behavior_output_root=str(Path(directory) / "behavior"))
    rows = run_tasks(_probe_tasks(smoke_cfg, 1, "smoke"), Path(directory) / f"smoke-{time.time_ns()}",
                     smoke_cfg, stop, progress)
    if stop():
        raise InterruptedError("smoke stopped")
    if len(rows) != 1 or not rows[0]["valid"]:
        raise RuntimeError("hidden environment smoke failed: " + str(rows))
    learner_probe = choose_learner(cfg.get("device", "auto"), candidates=(1,), progress=progress,
                                   memory_limit_gib=cfg.get("memory_soft_gib", 32), cpu_threads=(1,))
    learner = next(row for row in learner_probe["probes"]
                   if row["valid"] and row["device"] == learner_probe["selected_device"])
    return dict(passed=True, hidden_environment=rows[0]["result"], learner_update=learner,
                learner_probe=learner_probe,
                persisted_training_steps=0, full_study_started=False)


class WorkerTuner:
    """A measured worker count is fixed throughout a registered comparison."""
    def __init__(self, initial=4, cap=8, state=None):
        if not 1 <= initial <= cap <= 8:
            raise ValueError("invalid worker budget")
        if state and state["selected"] != initial:
            raise ValueError("registered workers changed")
        self.selected = initial
        self.samples = {} if state is None else state.get("samples", {})

    def observe(self, workers, steps, seconds, memory_gib, limit=32):
        if workers != self.selected:
            raise ValueError("worker count changed after calibration")
        self.samples.setdefault(str(workers), []).append(steps/max(seconds, 1e-9))
        self.samples[str(workers)] = self.samples[str(workers)][-16:]
        if memory_gib > limit:
            raise RuntimeError("memory soft limit exceeded")
        return self.selected

    def state(self):
        return dict(selected=self.selected, samples=self.samples, frozen=True,
                    method="measured real-environment throughput")
