"""Plan gates, three-arm study, independent confirmation and frozen submission."""
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from .benchmark import benchmark_inference, choose_learner, fixture
from .contracts import verify_original
from .export import export_policy, verify_export
from .io import ROOT, atomic_json, atomic_torch, file_hash, read_json
from .policy import MyPolicy
from .value import Critic
from .runtime import run_tasks
from .statistics import summarize, run_interval
from .training import discard_staging, train_run


def ensure_export(policy, path):
    path = Path(path)
    if not path.exists(): export_policy(policy, path)
    return verify_export(path, policy, fixture())


def evaluate_bundle(bundle, directory, cfg, *, replicates, offset=41000, stop=lambda: False,
                    progress=lambda **_: None, audit_windows=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lock = {p.name: file_hash(p) for p in Path(bundle).iterdir()}
    if (directory / "model_lock.json").exists() and read_json(directory / "model_lock.json") != lock:
        raise ValueError("evaluation model changed")
    atomic_json(directory / "model_lock.json", lock)
    cells = directory / "cells"
    cells.mkdir(exist_ok=True)
    tasks = []
    for fi, family in enumerate(cfg["evaluation_families"]):
        for replicate in range(replicates):
            for side in (0, 1):
                key = f"{family}-{replicate:03d}-{side}"
                if (cells / (key+".json")).exists(): continue
                tasks.append({"id": key, "kind": "evaluate", "config": cfg, "bundle": str(bundle),
                    "bundle_hashes": lock, "side": side, "replicate": replicate,
                    "action_seed": offset+fi*1000+replicate*2+side,
                    "opponent": {"kind": family}, "audit_windows": audit_windows})
    attempts = directory / "attempts" / str(time.time_ns())
    results = run_tasks(tasks, attempts, cfg, stop,
        lambda **kw: progress(phase="evaluation", evaluation=str(directory), **kw)) if tasks and not stop() else []
    invalid = 0
    for row in results:
        if row["valid"]:
            atomic_json(cells / (row["id"]+".json"), row["result"])
        elif not row["interrupted"]:
            invalid += 1
    if stop(): raise InterruptedError("evaluation stopped; completed cells retained")
    if invalid: raise RuntimeError(f"{invalid} invalid evaluation attempts; inspect {attempts}")
    rows = [read_json(cells / f"{f}-{r:03d}-{s}.json") for f in cfg["evaluation_families"]
            for r in range(replicates) for s in (0, 1)]
    # Include previous failed attempts in the denominator on resumed evaluation.
    all_invalid = sum(not read_json(p)["valid"] and not read_json(p)["interrupted"]
                      for p in (directory / "attempts").glob("*/*/attempt.json"))
    report = summarize(rows, all_invalid)
    if report["invalid_rate"] > cfg["invalid_rate_limit"]:
        raise RuntimeError("cumulative evaluation invalid-rate gate exceeded")
    report.update(bundle=str(bundle), bundle_hashes=lock, episodes=rows, action_rng_offset=offset)
    atomic_json(directory / "result.json", report)
    return report


def lifecycle(directory, cfg, reg_sha, stop, progress):
    directory = Path(directory)
    if (directory / "result.json").exists(): return read_json(directory / "result.json")
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(900)
    policy = MyPolicy(); policy.configure("A")
    behavior = directory / "behavior.pt"
    if not behavior.exists(): atomic_torch(behavior, {"policy_state": policy.state_dict(), "critic_state": Critic().state_dict()})
    checksum = file_hash(behavior)
    tasks = []
    for i in range(cfg["lifecycle_games"]):
        key = f"lifecycle-{i:03d}"
        if (directory / "cells" / (key+".json")).exists(): continue
        tasks.append({"id": key, "kind": "collect", "config": cfg, "side": i % 2,
                      "requested_seed": 90000+i, "action_seed": 70000+i,
                      "policy_path": str(behavior), "policy_sha256": checksum,
                      "opponent": {"kind": "noop" if i < cfg["lifecycle_games"]//2 else "target"},
                      "audit_windows": True})
    attempts = directory / "attempts" / str(time.time_ns())
    results = run_tasks(tasks, attempts, cfg, stop,
                        lambda **kw: progress(phase="lifecycle", **kw)) if tasks and not stop() else []
    errors = []
    for row in results:
        if row["valid"]: atomic_json(directory / "cells" / (row["id"]+".json"), row["result"])
        elif not row["interrupted"]: errors.append(row["error"])
    discard_staging(attempts)
    if stop(): raise InterruptedError("lifecycle stopped")
    if errors: raise RuntimeError("original lifecycle audit failed: " + str(errors))
    rows = [read_json(directory / "cells" / f"lifecycle-{i:03d}.json") for i in range(cfg["lifecycle_games"])]
    report = {"passed": True, "games": len(rows), "env_steps": sum(r["length"] for r in rows),
              "max_length": max(r["length"] for r in rows), "fresh_environment_per_episode": True,
              "graphics_and_hidden_window_verified": all(r["window_audits"] and all(a["verified"] and not a["onscreen_owned_windows"] for a in r["window_audits"]) for r in rows),
              "initial_hashes_by_side": {str(s): len({r["initial_observation_sha256"] for r in rows if r["side"] == s}) for s in (0, 1)},
              "registration_sha256": reg_sha, "game_outcome_verified": False}
    if not report["graphics_and_hidden_window_verified"]: raise RuntimeError("lifecycle window audit incomplete")
    atomic_json(directory / "result.json", report)
    return report


def run_study(directory, cfg, registration_sha, stop, progress, memory):
    directory = Path(directory)
    verify_original()
    progress(phase="offline_device_probe")
    runtime_path = directory / "runtime.json"
    if runtime_path.exists(): runtime = read_json(runtime_path)
    else:
        inference = benchmark_inference()
        learner = choose_learner(cfg["device"], cfg["ppo"]["minibatch"])
        runtime = {"inference": inference, "learner": learner}
        atomic_json(runtime_path, runtime)
    device = runtime["learner"]["selected_device"]
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("registered learner MPS is unavailable in this session")
    cfg = dict(cfg, actor_threads=runtime["inference"]["selected_actor_threads"])
    lifecycle(directory / "lifecycle", cfg, registration_sha, stop, progress)
    if stop(): raise InterruptedError("stopped before pilot")
    progress(phase="pilot")
    _, pilot = train_run(directory / "pilot", cfg, "A", 11, registration_sha, device, pilot=True,
                         stop=stop, progress=progress, memory=memory)
    if stop(): raise InterruptedError("pilot stopped")
    if not pilot["pilot_passed"]:
        raise RuntimeError("pilot gate failed: pickup/delivery/movement evidence insufficient; long training was not started. See pilot/result.json")
    # A new random Attention baseline is directly compatible with the same inputs.
    # The retained v8 submission adds an actual historical submission reference.
    baseline_directory = directory / "baseline"
    baseline_directory.mkdir(exist_ok=True)
    torch.manual_seed(11)
    initial = MyPolicy().eval()
    initial_bundle = baseline_directory / "initial_bundle"
    ensure_export(initial, initial_bundle)
    baseline = evaluate_bundle(initial_bundle, baseline_directory / "initial_eval", cfg,
                 replicates=cfg["dev_replicates"], stop=stop, progress=progress)
    v8_summary = ROOT / "logs/v8/provided_competition_v1/summary.json"
    historical = None
    if v8_summary.exists():
        old_bundle = ROOT / read_json(v8_summary)["selected"]["bundle"]
        if old_bundle.is_dir():
            historical = evaluate_bundle(old_bundle, baseline_directory / "v8_eval", cfg,
                         replicates=cfg["dev_replicates"], stop=stop, progress=progress)
    results = {}
    for arm in cfg["arms"]:
        results[arm] = []
        for seed in cfg["seeds"]:
            name = f"{arm}_s{seed}"
            run = directory / "runs" / name
            def intermediate(policy, step, run=run):
                bundle = run / "exports" / f"step-{step}"
                ensure_export(policy, bundle)
                evaluate_bundle(bundle, run / "evaluation" / f"step-{step}", cfg, replicates=2,
                                stop=stop, progress=progress)
            policy, training = train_run(run, cfg, arm, seed, registration_sha, device,
                stop=stop, progress=progress, evaluate=intermediate, memory=memory)
            if stop(): raise InterruptedError("training stopped")
            bundle = run / "exports" / "final"
            verification = ensure_export(policy, bundle)
            atomic_json(run / "export_verification.json", verification)
            report = evaluate_bundle(bundle, run / "evaluation/final", cfg,
                         replicates=cfg["dev_replicates"], stop=stop, progress=progress)
            results[arm].append({"run": name, "training": training, "evaluation": report, "bundle": str(bundle)})
    best_arm = max(cfg["arms"], key=lambda a: (np.mean([r["evaluation"]["macro_win_rate"] for r in results[a]]),
                                               np.mean([r["evaluation"]["worst_family"] for r in results[a]])))
    selected = max(results[best_arm], key=lambda r: (r["evaluation"]["macro_win_rate"], r["evaluation"]["worst_family"]))
    selection = {"arm": best_arm, "run": selected["run"], "bundle": selected["bundle"],
                 "rule": "arm mean macro win rate, then worst-family; seed dev macro then worst-family; ordered ties",
                 "test_used_for_selection": False}
    atomic_json(directory / "selection.json", selection)
    confirmation = []
    for seed in cfg["confirmation_seeds"]:
        run = directory / "confirmation" / f"{best_arm}_s{seed}"
        policy, training = train_run(run, cfg, best_arm, seed, registration_sha, device,
                                     stop=stop, progress=progress, memory=memory)
        if stop(): raise InterruptedError("confirmation stopped")
        bundle = run / "exports/final"
        ensure_export(policy, bundle)
        evaluation = evaluate_bundle(bundle, run / "evaluation", cfg, replicates=cfg["dev_replicates"],
                                      stop=stop, progress=progress, offset=61000)
        confirmation.append({"seed": seed, "training": training, "evaluation": evaluation})
    # Test RNG repetitions are held out from selection, not claimed as new maps.
    test = evaluate_bundle(selected["bundle"], directory / "frozen_test", cfg,
                           replicates=cfg["test_replicates"], offset=81000, stop=stop, progress=progress)
    frozen_baseline = evaluate_bundle(initial_bundle, directory / "frozen_baseline_test", cfg,
                           replicates=cfg["test_replicates"], offset=81000, stop=stop, progress=progress)
    checkpoint_sha = file_hash(Path(selected["bundle"]) / "checkpoint.pt")
    destination = ROOT / "submission/v9" / checkpoint_sha
    if not destination.exists(): shutil.copytree(selected["bundle"], destination)
    submission = verify_export(destination)
    summary = {"complete": True, "registration_sha256": registration_sha, "selected": selection,
               "submission": str(destination), "submission_verification": submission,
               "results": results, "confirmation": confirmation, "initial_baseline": baseline,
               "historical_v8_baseline": historical, "frozen_test": test, "frozen_baseline_test": frozen_baseline,
               "by_arm": {a: run_interval([r["evaluation"] for r in rows]) for a, rows in results.items()},
               "local_runner_improved": test["macro_win_rate"] > frozen_baseline["macro_win_rate"],
               "strategy_coordination_certified": False, "self_id_available": False,
               "game_outcome_verified": False, "official_server_certified": False,
               "external_submission_sent": False, "format_ready": True,
               "main_training_steps": sum(r["training"]["step"] for rows in results.values() for r in rows),
               "confirmation_steps": sum(r["training"]["step"] for r in confirmation),
               "pilot_steps": pilot["step"]}
    atomic_json(directory / "summary.json", summary)
    return summary
