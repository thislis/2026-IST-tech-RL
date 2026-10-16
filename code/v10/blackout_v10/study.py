"""Plan gates, three-arm study, independent confirmation and frozen submission."""

from project_paths import project_root, project_path

from pathlib import Path
import shutil
import time
import numpy as np
import torch
from .benchmark import benchmark_inference, choose_learner, benchmark_values, fixture
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
                    progress=lambda **_: None, audit_windows=True):
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
    if report["invalid"] > 0:
        raise RuntimeError("cumulative evaluation invalid-rate gate exceeded")
    report.update(bundle=str(bundle), bundle_hashes=lock, episodes=rows, action_rng_offset=offset)
    atomic_json(directory / "result.json", report)
    return report


def lifecycle(directory, cfg, reg_sha, stop, progress):
    directory = Path(directory)
    if (directory / "result.json").exists(): return read_json(directory / "result.json")
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(900)
    policy = MyPolicy(); policy.configure("C1")
    behavior = directory / "behavior.pt"
    if not behavior.exists(): atomic_torch(behavior, {"policy_state": policy.state_dict(), "critic_state": Critic("C1").state_dict(), "critic_arm": "C1"})
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
    from .io import digest, state_hash
    from .schedule import task_plan
    from .value_diagnostic import run_diagnostic
    from .telemetry import hardware_identity
    directory = Path(directory); verify_original()
    progress(phase='offline_device_probe')
    runtime_path = directory/'runtime_profile.json'
    hardware = hardware_identity()
    runtime_key = digest(dict(registration=registration_sha, hardware=hardware))
    if runtime_path.exists():
        runtime=read_json(runtime_path)
        if runtime['profile_key'] != runtime_key: raise ValueError('runtime profile changed; new registration required')
    else:
        inference=benchmark_inference()
        learner=choose_learner(cfg['device'],cfg['ppo']['minibatch'])
        candidates=[r for r in inference['measurements'] if 'error' not in r and
                    (cfg['actor_device']=='auto' or r['device']==cfg['actor_device'])]
        if not candidates: raise RuntimeError('requested actor device is unavailable')
        # CPU default avoids small-batch multi-process MPS contention; explicit MPS is validated below.
        if cfg['actor_device']=='auto': candidates=[r for r in candidates if r['device']=='cpu']
        actor_choice=min(candidates,key=lambda r:r['p95'])
        if actor_choice['device']=='mps' and not any(r['device']=='mps' and r['valid'] for r in learner['probes']):
            raise RuntimeError('MPS actor requested but device parity failed')
        values=benchmark_values(learner["selected_device"])
        runtime=dict(profile_key=runtime_key,hardware=hardware,inference=inference,learner=learner,actor=actor_choice,values=values)
        atomic_json(runtime_path,runtime)
    device=runtime['learner']['selected_device']
    if (directory/'device_fallback.json').exists():
        if cfg['device']!='auto': raise ValueError('device fallback requires original auto registration')
        device='cpu'
    def execute_training(*args, **kwargs):
        nonlocal device
        try:
            return train_run(*args, device=device, **kwargs)
        except RuntimeError as exc:
            message=str(exc).lower()
            if cfg['device']!='auto' or device!='mps' or not any(k in message for k in ('out of memory','mps backend','not implemented for mps')):
                raise
            atomic_json(directory/'device_fallback.json',dict(previous_device=device,selected_device='cpu',
                reason=str(exc),boundary='last committed checkpoint; uncommitted wave discarded',
                comparison_hardware_changed=True))
            device='cpu'
        # Leave the except block so the failed MPS traceback releases its model tensors.
        import gc
        gc.collect();torch.mps.empty_cache()
        return train_run(*args,device=device,**kwargs)
    cfg=dict(cfg, actor_threads=runtime['actor']['threads'],resolved_actor_device=runtime['actor']['device'],
             value_batch=runtime['values']['selected_batch'])
    plan=task_plan(cfg['seeds']+cfg['confirmation_seeds'],cfg['max_attempts'])
    lock=directory/'task_plan.json'
    if lock.exists() and read_json(lock)!=plan: raise ValueError('task schedule changed')
    atomic_json(lock,plan)
    lifecycle(directory/'lifecycle',cfg,registration_sha,stop,progress)
    if stop(): raise InterruptedError('stopped before diagnostic')
    diagnostic=run_diagnostic(directory/'value_diagnostic',cfg,device,stop,progress)
    pilots={}
    for arm in cfg['arms']:
        run=directory/'pilot'/arm
        policy, report=execute_training(run,cfg,arm,11,registration_sha,pilot=True,stop=stop,progress=progress,memory=memory)
        if stop(): raise InterruptedError('pilot stopped')
        if not report['pilot_passed']: raise RuntimeError('pilot technical gate failed')
        bundle=run/'exports/final'; ensure_export(policy,bundle)
        pilot_eval=evaluate_bundle(bundle,run/'evaluation',cfg,replicates=cfg['pilot_replicates'],offset=31000,stop=stop,progress=progress)
        pilots[arm]=dict(training=report,evaluation=pilot_eval)
    results={arm:[] for arm in cfg['arms']}; confirmation={arm:[] for arm in cfg['arms']}; baselines={}
    seeds=cfg['seeds']+cfg['confirmation_seeds']
    for index, seed in enumerate(seeds):
        if stop(): raise InterruptedError('study stopped')
        with torch.random.fork_rng():
            torch.manual_seed(seed); initial=MyPolicy().eval()
        initial_hash=state_hash(initial.state_dict())
        baseline=directory/'baseline'/f's{seed}'; initial_bundle=baseline/'bundle'
        ensure_export(initial,initial_bundle)
        baselines[str(seed)]=dict(bundle=str(initial_bundle),actor_sha256=initial_hash,
            evaluation=evaluate_bundle(initial_bundle,baseline/'evaluation',cfg,replicates=cfg['baseline_replicates'],
                                       offset=21000+seed*100,stop=stop,progress=progress))
        is_confirmation=seed in cfg['confirmation_seeds']
        target=confirmation if is_confirmation else results
        arms=cfg['arms'] if index%2==0 else list(reversed(cfg['arms']))
        for arm in arms:
            name=f'{arm}_s{seed}'; run=directory/('confirmation' if is_confirmation else 'runs')/name
            policy, training=execute_training(run,cfg,arm,seed,registration_sha,stop=stop,progress=progress,memory=memory)
            if stop(): raise InterruptedError('training stopped')
            from .io import load_checkpoint
            payload,_=load_checkpoint(run/'checkpoints')
            if payload['initial_actor_sha256'] != initial_hash: raise ValueError('paired actor initialization mismatch')
            bundle=run/'exports/final'; verification=ensure_export(policy,bundle)
            atomic_json(run/'export_verification.json',verification)
            evaluation=evaluate_bundle(bundle,run/'evaluation/final',cfg,replicates=cfg['dev_replicates'],
                offset=(61000 if is_confirmation else 41000)+seed*100,stop=stop,progress=progress)
            target[arm].append(dict(run=name,seed=seed,training=training,evaluation=evaluation,bundle=str(bundle)))
        # Freeze P4 selection before confirmation, never let confirmation or test change it.
        if index==len(cfg['seeds'])-1:
            selected=sorted(results['C1'],key=lambda r:(-r['evaluation']['macro_win_rate'],r['seed']))[0]
            selection=dict(arm='C1',run=selected['run'],seed=selected['seed'],step=selected['training']['step'],bundle=selected['bundle'],
                checkpoint_sha256=file_hash(Path(selected['bundle'])/'checkpoint.pt'),
                rule='P4 C1 final dev macro win rate; ascending seed breaks ties',test_used_for_selection=False)
            path=directory/'selection_lock.json'
            if path.exists() and read_json(path)!=selection: raise ValueError('selection lock changed')
            atomic_json(path,selection)
    selection=read_json(directory/'selection_lock.json'); seed=selection['seed']
    c0=next(r for r in results['C0'] if r['seed']==seed)
    bundles={'C1':selection['bundle'],'C0':c0['bundle'],'initial':baselines[str(seed)]['bundle']}
    tests={key:evaluate_bundle(bundle,directory/'frozen_test'/key,cfg,replicates=cfg['test_replicates'],
                              offset=181000,stop=stop,progress=progress) for key,bundle in bundles.items()}
    destination=ROOT/'artifacts/submission/v10'/selection['checkpoint_sha256']
    if not destination.exists(): shutil.copytree(selection['bundle'],destination)
    verification=verify_export(destination)
    paired=[]
    for seed in seeds:
        source=confirmation if seed in cfg['confirmation_seeds'] else results
        a={arm:next(r for r in source[arm] if r['seed']==seed) for arm in cfg['arms']}
        paired.append(dict(seed=seed,delta=a['C1']['evaluation']['macro_win_rate']-a['C0']['evaluation']['macro_win_rate'],
            baseline_delta=a['C1']['evaluation']['macro_win_rate']-baselines[str(seed)]['evaluation']['macro_win_rate'],
            nonraider_improved=any(a['C1']['evaluation']['families'][f]['win_rate']>a['C0']['evaluation']['families'][f]['win_rate']
                                  for f in cfg['evaluation_families'] if f!='raider')))
    deltas=np.asarray([r['delta'] for r in paired]); rng=np.random.default_rng(930)
    samples=deltas[rng.integers(len(deltas),size=(2000,len(deltas)))].mean(1)
    interval=np.quantile(samples,[.025,.975]).tolist()
    behavioral=any(r['training']['behavior']['delivery_proxy']>0 for r in results['C1']+confirmation['C1'])
    improved=bool(len(paired)>=5 and deltas.mean()>=.05 and sum(deltas>0)>=4 and interval[0]>0 and
                  np.mean([r['baseline_delta'] for r in paired])>0 and any(r['nonraider_improved'] for r in paired) and behavioral)
    improved = bool(improved and not (directory/'device_fallback.json').exists() and
        tests['C1']['macro_win_rate'] > max(tests['C0']['macro_win_rate'], tests['initial']['macro_win_rate']))
    summary=dict(complete=True,registration_sha256=registration_sha,selected=selection,submission=str(destination),
        submission_verification=verification,results=results,confirmation=confirmation,initial_baselines=baselines,
        pilots=pilots,value_diagnostic=diagnostic,frozen_tests=tests,paired_seeds=paired,
        paired_delta_interval=interval,interval_scope='exploratory few-seed paired bootstrap; not new-map generalization',
        local_runner_improved=improved,map_information_causality_verified=False,blind_ablation_required=True,
        device_fallback=(read_json(directory/"device_fallback.json") if (directory/"device_fallback.json").exists() else None),
        research_candidate=not improved,format_ready=True,experiment_ready=True,watch_ready=False,
        strategy_coordination_certified=False,self_id_available=False,game_outcome_verified=False,
        official_server_certified=False,external_submission_sent=False,
        main_training_steps=sum(r['training']['step'] for rows in results.values() for r in rows),
        confirmation_steps=sum(r['training']['step'] for rows in confirmation.values() for r in rows),
        pilot_steps=sum(r['training']['step'] for r in pilots.values()))
    atomic_json(directory/'summary.json',summary)
    return summary
