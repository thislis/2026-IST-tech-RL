"""Complete-wave PPO with atomic recovery and no invalid-prefix updates."""
import gc
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from .benchmark import WorkerTuner
from .io import state_hash, append, atomic_json, atomic_torch, cpu_state, file_hash, load_checkpoint, read_json, save_checkpoint
from .schedule import task_spec
from .targets import commit_targets
from .policy import MyPolicy
from .value import Critic
from .ppo import optimizers, update
from .runtime import run_tasks
from .trajectory import WaveBatch


def discard_staging(path):
    """Keep a bounded diagnostic tail; never retain multi-GB unused trajectories."""
    path = Path(path)
    for folder in path.glob("*/trajectory"):
        if not folder.is_dir(): continue
        meta = folder / "committed.json"
        if not meta.exists(): meta = folder / "validated.json"
        count = read_json(meta)["length"] if meta.exists() else 0
        invalid = folder.parent / "invalid.json"
        if invalid.exists(): count = read_json(invalid).get("steps", count)
        if count:
            sample = {}
            for key in ("vectors", "maps", "actions", "values", "potentials"):
                f = folder / (key + ".npy")
                if f.exists():
                    a = np.load(f, mmap_mode="r", allow_pickle=False)
                    sample[key] = np.asarray(a[max(0, count-250):count]).copy()
                    del a
            if sample: np.savez_compressed(folder.parent / "tail.npz", **sample)
        shutil.rmtree(folder)


def invalid_gate(valid, invalid, consecutive, cfg):
    return (consecutive >= cfg["invalid_consecutive_limit"] or
            invalid/max(1, valid+invalid) > cfg["invalid_rate_limit"])


def recover_log(path, offset):
    if not path.exists():
        if offset: raise ValueError("committed training log missing")
        return
    if path.stat().st_size < offset: raise ValueError("committed training log shortened")
    with path.open("rb") as stream:
        stream.seek(offset); tail = stream.read()
    if tail:
        (path.parent / f"uncommitted-training-{time.time_ns()}.jsonl").write_bytes(tail)
        with path.open("r+b") as stream: stream.truncate(offset)


def train_run(directory, cfg, arm, seed, registration_sha, device, *, pilot=False,
              stop=lambda: False, progress=lambda **_: None, evaluate=None, memory=lambda: 0.):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    policy = MyPolicy().to(device)
    initial_actor = cpu_state(policy)
    with torch.random.fork_rng():
        torch.manual_seed(seed+100000)
        critic = Critic(arm).to(device)
    policy.configure(arm, cfg["decoder"])
    actor_opt, critic_opt = optimizers(policy, critic, cfg["ppo"])
    rng = np.random.default_rng(seed+90000)
    budget = cfg["pilot_steps"] if pilot else cfg["training_steps"]
    state = {"step": 0, "wave": 0, "attempts": 0, "valid_episodes": 0, "invalid_episodes": 0,
             "invalid_steps": 0, "attempted_steps": 0, "discarded_steps": 0, "unmeasured_crash_attempts": 0,
             "consecutive_invalid": 0, "interrupted_episodes": 0, "exposure": {},
             "next_eval": cfg["evaluate_every"],
             "pending_wave": None, "behavior": {"pickup_proxy": 0, "delivery_proxy": 0,
                    "raid_proxy": 0, "respawn_proxy": 0, "visible_positive_score_delta": 0., "moving_steps": 0.}}

    tuner = WorkerTuner(cfg["workers"], cfg["worker_cap"])
    log = directory / "training.jsonl"
    last_path = None
    if (directory / "checkpoints/latest.json").exists():
        saved, last_path = load_checkpoint(directory / "checkpoints")
        if saved["registration_sha256"] != registration_sha or saved["arm"] != arm or saved["seed"] != seed:
            raise ValueError("resume provenance mismatch")
        policy.load_state_dict(saved["policy_state"]); critic.load_state_dict(saved["critic_state"])
        actor_opt.load_state_dict(saved["actor_optimizer"]); critic_opt.load_state_dict(saved["critic_optimizer"])
        state = saved["state"]
        rng.bit_generator.state = saved["rng"]
        torch.set_rng_state(saved["torch_rng"])

        tuner = WorkerTuner(cfg["workers"], cfg["worker_cap"], saved["tuner"])
        recover_log(log, saved["log_offset"])
        if state["pending_wave"]:
            pending = Path(state["pending_wave"])
            records = [read_json(p) for p in pending.glob("*/attempt.json")]
            recorded_ids = {r["id"] for r in records}
            for row in records:
                steps = row["attempt_steps"]
                state["attempted_steps"] += steps
                state["discarded_steps"] += steps
                if row["interrupted"]: state["interrupted_episodes"] += 1
                elif row["valid"]: state["valid_episodes"] += 1
                else:
                    state["invalid_episodes"] += 1
                    state["invalid_steps"] += steps
                    state["consecutive_invalid"] += 1
            for p in pending.glob("*/task.json"):
                if p.parent.name in recorded_ids: continue
                heartbeat = p.parent / "heartbeat.json"
                steps = read_json(heartbeat).get("steps", 0) if heartbeat.exists() else 0
                state["attempted_steps"] += steps
                state["discarded_steps"] += steps
                state["interrupted_episodes"] += 1
                state["unmeasured_crash_attempts"] += 1
            append(directory / "recovery.jsonl", {"event": "abandoned_uncommitted_wave", "path": str(pending),
                "completed_results_discarded": len(records), "physics_restored": False, "optimizer_unchanged": True})
            discard_staging(pending)
            state["pending_wave"] = None

    def save():
        nonlocal last_path
        payload = dict(step=state["step"], state=state, policy_state=cpu_state(policy), critic_state=cpu_state(critic),
            actor_optimizer=actor_opt.state_dict(), critic_optimizer=critic_opt.state_dict(), rng=rng.bit_generator.state,
            torch_rng=torch.get_rng_state(), tuner=tuner.state(), arm=arm, seed=seed,
            initial_actor_sha256=state_hash(initial_actor), critic_arm=arm,
            registration_sha256=registration_sha, log_offset=log.stat().st_size if log.exists() else 0,
            environment_state_restored=False, objective=cfg["objective"], decoder=cfg["decoder"])
        last_path = save_checkpoint(directory / "checkpoints", payload)
        atomic_json(directory / "progress.json", {"step": state["step"], "budget": budget, "state": state,
                    "checkpoint": str(last_path), "device": device, "workers": tuner.selected})
        return last_path

    if last_path is None: save()
    if invalid_gate(state["valid_episodes"], state["invalid_episodes"], state["consecutive_invalid"], cfg):
        raise RuntimeError("recorded invalid-episode gate remains exceeded; inspect diagnostics before a new registered run")
    try:
        while state["step"] < budget and not stop():
            if shutil.disk_usage(directory).free < cfg["disk_stop_gib"]*1024**3:
                raise RuntimeError("disk reserve reached; last committed checkpoint retained")
            telemetry_bytes = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file() and p.suffix in (".log", ".json", ".jsonl"))
            if telemetry_bytes > cfg["telemetry_budget_gib"]*1024**3:
                raise RuntimeError("run telemetry budget exceeded; retained logs were not deleted")
            wave = directory / "waves" / f"wave-{state['wave']:06d}-{time.time_ns()}"
            wave.mkdir(parents=True)
            behavior_path = wave / "behavior.pt"
            atomic_torch(behavior_path, {"policy_state": cpu_state(policy), "critic_state": cpu_state(critic), "critic_arm": arm})
            behavior_hash = file_hash(behavior_path)
            worker_count = tuner.selected
            # Reserve the worst-case float fallback, not just one-hot map storage.
            worst_wave = worker_count * cfg["max_episode_steps"] * (11*96*96*4 + 96*96 + 5*96*4)
            if shutil.disk_usage(directory).free < worst_wave + cfg["disk_stop_gib"]*1024**3:
                raise RuntimeError("insufficient disk for a lossless worst-case wave plus reserve")
            tasks = []
            for i in range(worker_count):
                serial = state["attempts"]+i
                if serial >= cfg["max_attempts"]: raise RuntimeError("registered attempt cap exceeded")
                tasks.append({"id": f"episode-{serial:08d}", "kind": "collect", "config": cfg,
                              **task_spec(seed, serial), "policy_path": str(behavior_path),
                              "policy_sha256": behavior_hash, "audit_windows": True})
            state["attempts"] += len(tasks)
            state["pending_wave"] = str(wave)
            save()  # Crash rollback boundary, including consumed task RNG.
            started = time.monotonic()
            peak_memory, last_memory_sample = memory(), started
            def collect_progress(**kw):
                nonlocal peak_memory, last_memory_sample
                if time.monotonic()-last_memory_sample >= 5:
                    peak_memory = max(peak_memory, memory())
                    last_memory_sample = time.monotonic()
                progress(phase="collect", run=directory.name, step=state["step"], budget=budget, **kw)
            results = run_tasks(tasks, wave, dict(cfg, workers=worker_count), stop,
                                collect_progress)
            collection_seconds = time.monotonic()-started
            paths = []
            for row in results:
                append(directory / "episodes.jsonl", row)
                state["attempted_steps"] += row["attempt_steps"]
                if not row.get("step_count_complete", True): state["unmeasured_crash_attempts"] += 1
                if row["interrupted"]:
                    state["interrupted_episodes"] += 1
                    continue
                if not row["valid"]:
                    state["invalid_episodes"] += 1
                    state["consecutive_invalid"] += 1
                    state["invalid_steps"] += row["attempt_steps"]
                else:
                    state["valid_episodes"] += 1
                    state["consecutive_invalid"] = 0
                    paths.append(Path(row["directory"]) / "trajectory")
            if stop():
                # Entire current wave stays outside the optimizer on user stop.
                state["pending_wave"] = None
                state["discarded_steps"] += sum(row["attempt_steps"] for row in results)
                save(); discard_staging(wave)
                break
            if invalid_gate(state["valid_episodes"], state["invalid_episodes"], state["consecutive_invalid"], cfg):
                state["pending_wave"] = None
                state["discarded_steps"] += sum(row["attempt_steps"] for row in results)
                save(); discard_staging(wave)
                raise RuntimeError("invalid episode gate exceeded; wave was not used by PPO")
            target_report = commit_targets(paths, behavior_path, behavior_hash, cfg, device,
                progress=lambda **kw: progress(run=directory.name, step=state["step"], budget=budget, **kw), stop=stop)
            batch = WaveBatch(paths)
            metrics = update(policy, critic, actor_opt, critic_opt, batch, cfg["ppo"], rng, arm,
                             lambda **kw: progress(run=directory.name, step=state["step"], budget=budget, **kw))
            new_steps = len(batch)
            for episode in batch.episodes:
                b = episode.meta["behavior"]
                for k in state["behavior"]:
                    state["behavior"][k] += b["moving_step_fraction"]*episode.length if k == "moving_steps" else b[k]
                key = f"side{episode.meta['side']}:{episode.meta['opponent']['kind']}"
                exposure = state["exposure"].setdefault(key, {"episodes": 0, "env_steps": 0})
                exposure["episodes"] += 1; exposure["env_steps"] += episode.length
            state["discarded_steps"] += sum(row["attempt_steps"] for row in results if not row["valid"])
            state["step"] += new_steps
            state["wave"] += 1
            state["pending_wave"] = None
            tuner.observe(worker_count, new_steps, collection_seconds, max(peak_memory, memory()), cfg["memory_soft_gib"])
            append(log, {"step": state["step"], "wave": state["wave"], "valid_steps": new_steps,
                "collection_seconds": collection_seconds, "collection_steps_per_second": new_steps/collection_seconds,
                "end_to_end_steps_per_second": new_steps/(time.monotonic()-started), "ppo": metrics,
                "targets": target_report, "tuner": tuner.state(), "episode_returns": [e.meta["train_return"] for e in batch.episodes],
                "raw_team_returns": [e.meta["raw_team_totals"] for e in batch.episodes]})
            save()
            del batch
            gc.collect(); discard_staging(wave)
            if evaluate is not None and state["step"] >= state["next_eval"] and state["step"] < budget and not stop():
                evaluate(policy, state["step"])
                state["next_eval"] = state["step"]+cfg["evaluate_every"]
                save()
        complete = state["step"] >= budget
        path = save()
        report = {"complete": complete, "step": state["step"], "budget": budget,
                  "overrun": max(0, state["step"]-budget), "checkpoint": str(path),
                  "behavior": state["behavior"], "valid_episodes": state["valid_episodes"],
                  "invalid_episodes": state["invalid_episodes"], "objective": cfg["objective"]}
        report.update(attempted_steps=state["attempted_steps"], discarded_steps=state["discarded_steps"],
                      unmeasured_crash_attempts=state["unmeasured_crash_attempts"], exposure=state["exposure"])
        if pilot:
            report["pilot_passed"] = bool(complete and state["wave"] > 0)
            report["gate_scope"] = "technical; poor behavioral results are retained, not filtered"
        atomic_json(directory / "result.json", report)
        return policy.cpu().eval(), report
    except BaseException:
        # Do not serialize potentially half-updated weights after an exception.
        # The last atomic save remains the only resumable state.
        raise
