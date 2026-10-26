"""Complete frozen waves, recurrent PPO, atomic optimizer/RNG/scheduler resume."""
import gc
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from .critic import Critic
from .io import append, atomic_json, atomic_torch, cpu_state, file_hash, load_checkpoint, read_json, save_checkpoint
from .opponents import training_catalog
from .policy import MyPolicy
from .recurrent_ppo import UpdateRejected, optimizers, update
from .runtime import run_tasks
from .scheduler import DeficitScheduler
from .targets import commit_targets
from .trajectory import WaveBatch


ACCOUNTING_VERSION = 2


def interruption_evidence(row):
    """A supervisor stop flag alone must not conceal a simultaneous failure."""
    if not row.get("interrupted"):
        return None
    error = str(row.get("error") or "")
    if "watchdog" in error or "console budget" in error:
        return None
    sources, kinds = [], []
    for name in ("error.json", "invalid.json"):
        path = Path(row["directory"])/name
        if path.exists():
            record = read_json(path)
            kind = record.get("type")
            if kind and kind != "InterruptedError":
                return None
            kinds.append(kind)
            sources.append({"path": str(path), "sha256": file_hash(path), "type": kind})
    if row.get("valid") and (row.get("result") or {}).get("natural_terminal"):
        return {"kind": "completed_after_stop", "sources": sources}
    if "InterruptedError" in kinds:
        return {"kind": "worker_acknowledged_stop", "sources": sources}
    if error in ("stop grace expired", "worker exit -15") and not kinds:
        return {"kind": "supervisor_stop_termination", "sources": []}
    return None


def account_episode(state, row, *, discard_complete=False):
    """Count each attempted step once and keep cancellations out of error rates."""
    steps = int(row["attempt_steps"])
    state["attempted_steps"] += steps
    if not row.get("step_count_complete", True):
        state["unmeasured_crash_attempts"] += 1
    evidence = interruption_evidence(row)
    if evidence is not None:
        state["interrupted_episodes"] += 1
        state["interrupted_steps"] += steps
        state["interrupted_completed_episodes"] += int(bool(row.get("valid")))
        state["discarded_steps"] += steps
        # An interruption neither extends nor erases a real failure streak.
        return "interrupted"
    if row.get("valid") and not row.get("interrupted"):
        state["valid_episodes"] += 1
        state["consecutive_invalid"] = 0
        if discard_complete:
            state["discarded_steps"] += steps
        return "valid"
    state["invalid_episodes"] += 1
    state["consecutive_invalid"] += 1
    state["discarded_steps"] += steps
    return "invalid"


def repair_interruption_accounting(directory, original, checkpoint=None):
    """Read-only evidence inspection; return counter-only changes for next save.

    Existing checkpoint and ledger files are never rewritten. The caller writes
    a recovery receipt and later creates a normal new checkpoint on user resume.
    """
    directory = Path(directory)
    state = dict(original)
    for key in ("interrupted_episodes", "interrupted_steps", "interrupted_completed_episodes"):
        state.setdefault(key, 0)
    if state.get("episode_accounting_version", 1) >= ACCOUNTING_VERSION:
        return state, None
    receipt = dict(event="interruption_accounting_migration", version=ACCOUNTING_VERSION,
                   checkpoint=None if checkpoint is None else str(checkpoint), applied=False,
                   weights_optimizer_rng_unchanged=True, attempted_and_training_steps_unchanged=True)
    ledger = directory/"episodes.jsonl"
    if not ledger.exists():
        if state["valid_episodes"] or state["invalid_episodes"]:
            receipt["reason"] = "episode ledger missing; existing failure counters retained"
            return state, receipt
        state["episode_accounting_version"] = ACCOUNTING_VERSION
        receipt.update(applied=True, reason="empty initial checkpoint", reclassified=[])
        return state, receipt
    abandoned = set()
    recovery = directory/"recovery.jsonl"
    if recovery.exists():
        for line in recovery.read_text().splitlines():
            record = json.loads(line)
            if record.get("event") == "uncommitted_wave_discarded":
                abandoned.add(str(Path(record["path"]).resolve()))
    if state.get("pending_wave"):
        abandoned.add(str(Path(state["pending_wave"]).resolve()))
    rows, seen = [], {}
    with ledger.open() as stream:
        for line in stream:
            row = json.loads(line)
            wave = str(Path(row["directory"]).resolve().parent)
            if wave in abandoned:
                continue
            key = (row["id"], str(Path(row["directory"]).resolve()))
            if key in seen:
                if seen[key] != row:
                    receipt["reason"] = "conflicting duplicate episode evidence; counters retained"
                    return state, receipt
                continue
            seen[key] = row
            rows.append(row)
    valid, invalid, consecutive = 0, 0, 0
    for row in rows:
        if row.get("valid") and not row.get("interrupted"):
            valid += 1
            consecutive = 0
        else:
            invalid += 1
            consecutive += 1
    expected = dict(valid_episodes=valid, invalid_episodes=invalid, consecutive_invalid=consecutive)
    receipt["ledger_sha256"] = file_hash(ledger)
    if any(state[key] != value for key, value in expected.items()):
        receipt.update(reason="ledger does not exactly reconcile with legacy counters; counters retained",
                       ledger_counters=expected, checkpoint_counters={k: state[k] for k in expected})
        return state, receipt
    corrected, genuine_streak, double_counted = [], 0, 0
    for row in rows:
        evidence = interruption_evidence(row)
        if evidence is not None:
            corrected.append(dict(id=row["id"], directory=row["directory"], steps=int(row["attempt_steps"]),
                                  valid=bool(row.get("valid")), evidence=evidence))
            # Legacy stopped-wave logic added valid+interrupted steps twice.
            # Correct only the uncommitted final wave proven by its index.
            wave = Path(row["directory"]).parent.name
            parts = wave.split("-")
            final_uncommitted = len(parts) >= 3 and parts[0] == "wave" and parts[1].isdigit() and int(parts[1]) == state["wave"]
            if row.get("valid") and final_uncommitted and not state.get("pending_wave"):
                double_counted += int(row["attempt_steps"])
        elif row.get("valid") and not row.get("interrupted"):
            genuine_streak = 0
        else:
            genuine_streak += 1
    if double_counted > state["discarded_steps"]:
        receipt["reason"] = "discarded-step evidence inconsistent; counters retained"
        return state, receipt
    keys = ("invalid_episodes", "consecutive_invalid", "interrupted_episodes", "interrupted_steps",
            "interrupted_completed_episodes", "discarded_steps")
    receipt["before"] = {k: state[k] for k in keys}
    state["invalid_episodes"] -= len(corrected)
    state["consecutive_invalid"] = genuine_streak
    state["interrupted_episodes"] += len(corrected)
    state["interrupted_steps"] += sum(r["steps"] for r in corrected)
    state["interrupted_completed_episodes"] += sum(r["valid"] for r in corrected)
    state["discarded_steps"] -= double_counted
    state["episode_accounting_version"] = ACCOUNTING_VERSION
    receipt.update(applied=True, reclassified=corrected, corrected_double_counted_steps=double_counted,
                   after={k: state[k] for k in keys}, reason="exact legacy ledger reconciliation")
    return state, receipt


def discard_staging(path):
    """Keep compact attempt/episode ledgers, reclaim full update replay."""
    for folder in Path(path).glob("*/trajectory"):
        if folder.is_dir():
            shutil.rmtree(folder)


def _rng_state():
    state = {"torch": torch.get_rng_state()}
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def _restore_rng(state):
    torch.set_rng_state(state["torch"])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])


def recover_log(path, offset):
    if not path.exists():
        if offset:
            raise ValueError("committed log missing")
        return
    if path.stat().st_size < offset:
        raise ValueError("committed log shortened")
    with path.open("rb") as stream:
        stream.seek(offset)
        tail = stream.read()
    if tail:
        (path.parent/f"uncommitted-{time.time_ns()}.jsonl").write_bytes(tail)
        with path.open("r+b") as stream:
            stream.truncate(offset)


def train_run(directory, cfg, arm, seed, registration_sha, device, *, pilot=False,
              stop=lambda: False, progress=lambda **_: None, memory=lambda: 0., demonstrations=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    device = str(cfg.get("resolved_learner_device", device))
    if cfg.get("resolved_learner_threads"):
        torch.set_num_threads(int(cfg["resolved_learner_threads"]))
    torch.manual_seed(seed)
    policy = MyPolicy().to(device)
    policy.configure(arm, cfg.get("decoder", "sample"), max_interventions=cfg.get("max_interventions", 1),
                     goal_period=cfg.get("goal_period", 25), residual_period=cfg.get("residual_period", 4),
                     use_memory=cfg.get("use_memory", True))
    with torch.random.fork_rng():
        torch.manual_seed(seed+100000)
        critic = Critic(cfg.get("critic_arm", "C1")).to(device)
    ppo_cfg = dict(cfg.get("ppo", {}))
    if "resolved_minibatch_sequences" in cfg:
        ppo_cfg["minibatch_sequences"] = cfg["resolved_minibatch_sequences"]
    actor_opt, critic_opt = optimizers(policy, critic, ppo_cfg)
    rng = np.random.default_rng(seed+90000)
    budget = int(cfg["pilot_steps"] if pilot else cfg["training_steps"])
    catalog = cfg.get("training_opponents") or training_catalog()
    scheduler = DeficitScheduler(catalog)
    state = {"step": 0, "trained_steps": 0, "decisions": 0, "wave": 0, "updates": 0,
             "rollbacks": 0, "consecutive_rollbacks": 0, "attempts": 0, "valid_episodes": 0,
             "invalid_episodes": 0, "attempted_steps": 0, "discarded_steps": 0,
             "interrupted_episodes": 0, "interrupted_steps": 0, "interrupted_completed_episodes": 0,
             "episode_accounting_version": ACCOUNTING_VERSION,
             "consecutive_invalid": 0, "unmeasured_crash_attempts": 0, "pending_wave": None,
             "history": [], "teacher_initialization": None,
             "behavior": {"pickup_proxy": 0, "delivery_proxy": 0, "respawn_proxy": 0,
                          "visible_positive_score_delta": 0., "moving_steps": 0., "stuck_steps": 0.}}
    log = directory/"training.jsonl"
    initial_path = directory/"initial.pt"
    last_path = None
    if (directory/"checkpoints/latest.json").exists():
        saved, last_path = load_checkpoint(directory/"checkpoints")
        if saved["registration_sha256"] != registration_sha or saved["arm"] != arm or saved["seed"] != seed:
            raise ValueError("resume provenance mismatch")
        policy.load_state_dict(saved["policy_state"])
        critic.load_state_dict(saved["critic_state"])
        actor_opt.load_state_dict(saved["actor_optimizer"])
        critic_opt.load_state_dict(saved["critic_optimizer"])
        state, accounting_receipt = repair_interruption_accounting(directory, saved["state"], last_path)
        if accounting_receipt is not None:
            append(directory/"recovery.jsonl", accounting_receipt)
        scheduler = DeficitScheduler(catalog, saved["scheduler"])
        rng.bit_generator.state = saved["rng"]
        _restore_rng(saved["torch_rng"])
        recover_log(log, saved["log_offset"])
        if state["pending_wave"]:
            pending = Path(state["pending_wave"])
            for task_path in pending.glob("*/task.json"):
                path = task_path.parent
                if (path/"attempt.json").exists():
                    row = read_json(path/"attempt.json")
                    account_episode(state, row, discard_complete=True)
                else:
                    steps = int(read_json(path/"heartbeat.json").get("steps", 0)) if (path/"heartbeat.json").exists() else 0
                    state["unmeasured_crash_attempts"] += 1
                    state["attempted_steps"] += steps
                    state["discarded_steps"] += steps
                    state["invalid_episodes"] += 1
                    state["consecutive_invalid"] += 1
            append(directory/"recovery.jsonl", {"event": "uncommitted_wave_discarded", "path": str(pending),
                    "physics_restored": False, "optimizer_restored": True})
            discard_staging(pending)
            state["pending_wave"] = None
    else:
        if demonstrations:
            from .demonstrations import initialize_from_demonstrations
            state["teacher_initialization"] = initialize_from_demonstrations(policy, demonstrations, cfg, device, progress, stop)
            # Teacher warm-up has its own optimizer; PPO begins with fresh moments.
            actor_opt, critic_opt = optimizers(policy, critic, ppo_cfg)
        atomic_torch(initial_path, {"policy_state": cpu_state(policy), "arm": arm, "decoder": cfg.get("decoder", "sample")})
    initial_spec = {"kind": "history", "path": str(initial_path), "sha256": file_hash(initial_path),
                    "family": "self_play", "name": "frozen_initial_v11"}

    def save():
        nonlocal last_path
        payload = {"step": state["step"], "state": state, "policy_state": cpu_state(policy),
                   "critic_state": cpu_state(critic), "critic_arm": critic.arm,
                   "actor_optimizer": actor_opt.state_dict(), "critic_optimizer": critic_opt.state_dict(),
                   "scheduler": scheduler.state_dict(), "rng": rng.bit_generator.state, "torch_rng": _rng_state(),
                   "arm": arm, "seed": seed, "registration_sha256": registration_sha,
                   "log_offset": log.stat().st_size if log.exists() else 0,
                   "objective": cfg.get("objective", "provided_runner_v1"), "decoder": cfg.get("decoder", "sample"),
                   "environment_state_restored": False, "learner_device": device}
        last_path = save_checkpoint(directory/"checkpoints", payload)
        atomic_json(directory/"progress.json", {"step": state["step"], "budget": budget, "state": state,
                    "checkpoint": str(last_path), "learner_device": device, "workers": cfg["workers"],
                    "exposure": scheduler.report()})
        atomic_json(directory/"exposure.json", scheduler.report())
        # Historical opponents live in independent snapshots; retain latest two resume commits.
        checkpoints = sorted((directory/"checkpoints").glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in checkpoints[3:]:
            if old != last_path:
                old.unlink()
        return last_path

    save()
    while state["step"] < budget and not stop():
        if memory() > cfg.get("memory_soft_gib", 32):
            raise RuntimeError("memory soft limit reached before new wave")
        workers = int(cfg["workers"])
        # Worst case every primitive step is a decision and every graphic nonbinary.
        required = workers*cfg.get("max_episode_steps", 22000)*(11*96*96*4+16384)
        if shutil.disk_usage(directory).free < required+cfg.get("disk_stop_gib", 25)*1024**3:
            raise RuntimeError("insufficient space for worst-case lossless wave and disk reserve")
        wave = directory/"waves"/f"wave-{state['wave']:06d}-{time.time_ns()}"
        wave.mkdir(parents=True)
        behavior_path = wave/"behavior.pt"
        atomic_torch(behavior_path, {"policy_state": cpu_state(policy), "critic_state": cpu_state(critic),
                                   "arm": arm, "critic_arm": critic.arm, "decoder": cfg.get("decoder", "sample")})
        behavior_hash = file_hash(behavior_path)
        tasks, reservations = [], {}
        for i in range(workers):
            serial = state["attempts"]+i
            if serial >= cfg.get("max_attempts", 100000):
                raise RuntimeError("registered attempt cap reached")
            spec = scheduler.next(seed, state["history"] or [initial_spec], reservations)
            tasks.append(dict(id=f"episode-{serial:08d}", kind="collect", config=cfg, arm=arm,
                              policy_path=str(behavior_path), policy_sha256=behavior_hash,
                              audit_windows=True, **spec))
        state["attempts"] += len(tasks)
        state["pending_wave"] = str(wave)
        save()
        started = time.monotonic()
        def report_progress(**kw):
            progress(run=directory.name, step=state["step"], budget=budget, **kw)
        results = run_tasks(tasks, wave, cfg, stop, report_progress)
        collection_seconds = time.monotonic()-started
        paths = []
        valid_wave_steps = 0
        for row in results:
            append(directory/"episodes.jsonl", row)
            category = account_episode(state, row)
            if category == "valid":
                paths.append(Path(row["directory"])/"trajectory")
                valid_wave_steps += int(row["attempt_steps"])
        invalid_rate = state["invalid_episodes"]/max(1, state["valid_episodes"]+state["invalid_episodes"])
        if stop() or not paths or state["consecutive_invalid"] >= cfg.get("invalid_consecutive_limit", 3) or invalid_rate > cfg.get("invalid_rate_limit", .01):
            state["discarded_steps"] += valid_wave_steps
            state["pending_wave"] = None
            save()
            discard_staging(wave)
            if stop():
                break
            raise RuntimeError("invalid complete-episode gate exceeded; no wave update")
        target_report = commit_targets(paths, behavior_path, behavior_hash, cfg, device, report_progress, stop)
        batch = WaveBatch(paths)
        try:
            metrics = update(policy, critic, actor_opt, critic_opt, batch, ppo_cfg, rng, arm, report_progress, stop)
        except UpdateRejected as exc:
            # Restore every trainable/RNG component from the pre-wave commit;
            # retain spent steps and allocation history instead of hiding cost.
            saved, _ = load_checkpoint(directory/"checkpoints")
            policy.load_state_dict(saved["policy_state"])
            critic.load_state_dict(saved["critic_state"])
            actor_opt.load_state_dict(saved["actor_optimizer"])
            critic_opt.load_state_dict(saved["critic_optimizer"])
            rng.bit_generator.state = saved["rng"]
            _restore_rng(saved["torch_rng"])
            for episode in batch.episodes:
                scheduler.observe(episode.meta["family"], episode.meta["side"], episode.env_steps)
            state["step"] += batch.env_steps
            state["discarded_steps"] += batch.env_steps
            state["wave"] += 1
            state["rollbacks"] += 1
            state["consecutive_rollbacks"] += 1
            state["pending_wave"] = None
            append(log, {"step": state["step"], "wave": state["wave"], "rolled_back": True,
                         "spent_env_steps": batch.env_steps, "reason": exc.report})
            save()
            del batch
            gc.collect()
            discard_staging(wave)
            if state["consecutive_rollbacks"] >= cfg.get("rollbacks", 3):
                raise RuntimeError("consecutive PPO rollback limit reached; spent budget retained") from exc
            continue
        for episode in batch.episodes:
            scheduler.observe(episode.meta["family"], episode.meta["side"], episode.env_steps)
            b = episode.meta["behavior"]
            for key in state["behavior"]:
                if key == "moving_steps":
                    value = b["moving_step_fraction"]*episode.env_steps
                elif key == "stuck_steps":
                    value = b["stuck_team_step_fraction"]*episode.env_steps
                else:
                    value = b.get(key, 0)
                state["behavior"][key] += value
        state["step"] += batch.env_steps
        state["trained_steps"] += batch.env_steps
        state["decisions"] += len(batch)
        state["wave"] += 1
        state["updates"] += 1
        state["consecutive_rollbacks"] = 0
        state["pending_wave"] = None
        capacity = int(cfg.get("history_capacity", 8))
        if capacity:
            snapshot = directory/"history"/f"wave-{state['wave']:06d}.pt"
            atomic_torch(snapshot, {"policy_state": cpu_state(policy), "arm": arm, "decoder": cfg.get("decoder", "sample")})
            state["history"].append({"kind": "history", "path": str(snapshot), "sha256": file_hash(snapshot),
                                     "family": "self_play", "name": f"{arm}_s{seed}_wave{state['wave']}"})
            removed = state["history"][:-capacity]
            state["history"] = state["history"][-capacity:]
        else:
            removed = []
        append(log, {"step": state["step"], "decisions": state["decisions"], "wave": state["wave"],
                     "valid_env_steps": batch.env_steps, "collection_seconds": collection_seconds,
                     "collection_steps_per_second": batch.env_steps/max(collection_seconds, 1e-9),
                     "end_to_end_steps_per_second": batch.env_steps/max(time.monotonic()-started, 1e-9),
                     "ppo": metrics, "targets": target_report, "exposure": scheduler.report(),
                     "episode_returns": [e.meta["train_return"] for e in batch.episodes]})
        save()
        for item in removed:
            Path(item["path"]).unlink(missing_ok=True)
        del batch
        gc.collect()
        discard_staging(wave)
    report = {"complete": state["step"] >= budget, "step": state["step"], "env_steps": state["step"],
              "decisions": state["decisions"], "trained_env_steps": state["trained_steps"],
              "updates": state["updates"], "rollbacks": state["rollbacks"],
              "technical_passed": state["updates"] > 0 and state["invalid_episodes"] == 0,
              "budget": budget, "overrun": max(0, state["step"]-budget),
              "checkpoint": str(save()), "behavior": state["behavior"], "valid_episodes": state["valid_episodes"],
              "invalid_episodes": state["invalid_episodes"], "attempted_steps": state["attempted_steps"],
              "interrupted_episodes": state["interrupted_episodes"], "interrupted_steps": state["interrupted_steps"],
              "interrupted_completed_episodes": state["interrupted_completed_episodes"],
              "discarded_steps": state["discarded_steps"], "exposure": scheduler.report(),
              "objective": cfg.get("objective", "provided_runner_v1"), "learner_device": device,
              "teacher_initialization": state["teacher_initialization"]}
    if pilot:
        report["pilot_passed"] = bool(report["complete"] and report["technical_passed"])
        report["gate_scope"] = "technical; study applies F behavior gates, controls retain equal budgets"
    atomic_json(directory/"result.json", report)
    return policy.cpu().eval(), report
