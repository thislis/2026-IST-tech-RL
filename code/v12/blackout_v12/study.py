"""Documented comparison stages with concurrent, bounded experiment jobs."""
import copy
from pathlib import Path
import subprocess
import sys
import time
from .io import ROOT, atomic_json, atomic_torch, digest, file_hash, read_json
from .runtime import child_environment, process_options, managed_popen
from .execution import compatible_manifest, schedule_allocation


def plan(config):
    learned = len(config['arms'])
    main_seeds = len(config['seeds'])
    additional = len(config['confirmation_seeds'])
    return dict(schema='v12.plan.1', scope=config.get('scope','available'),
                stages=['Q0_Q1_Q2_and_frozen_baselines', 'public_task_data', 'pilot',
                        'independent_main', 'dev_decoder_and_candidate_selection',
                        'additional_seed_diagnostics', 'confirmation', 'held_out_test', 'export'],
                data_steps=config['data_steps'], pilot_steps=learned*config['pilot_steps'],
                main_steps=learned*main_seeds*config['training_steps'],
                additional_seed_steps=additional*config['training_steps'],
                total_max_learning_steps=config['data_steps']+learned*config['pilot_steps']+
                    (learned*main_seeds+additional)*config['training_steps'],
                pilot_weights_reused=False, full_reference_evaluation_games=2760,
                decoder_dev_additional_games='sample/MAP comparisons are recorded separately',
                incomplete_waves_used_for_ppo=False, actual_exposure_includes_natural_wave_overshoot=True,
                adoption_requires_frozen_references=True,
                missing_reference_opponents=config.get('missing_opponent_ids',[]))


def run_jobs(tasks, config, stop=lambda:False, progress=lambda **_:None):
    """Separate experiment processes overlap CPU collection with CUDA updates."""
    pending, active, completed = list(tasks), {}, {}
    training = all(task['kind'] in ('train','data') for task in tasks)
    limit = int(config.get('parallel_train_runs',config.get('parallel_runs',1)) if training
                else config.get('parallel_runs',1))
    total_workers = int(config.get('workers',limit*int(config.get('workers_per_run',1))))
    try:
        while pending or active:
            if stop():
                raise InterruptedError('study stopped before new job dispatch')
            while pending and len(active)<limit:
                free = total_workers-sum(item[1]['workers'] for item in active.values())
                if free < 1: break
                slots = min(limit-len(active),len(pending),free)
                allocated = schedule_allocation(pending,dict(config,workers=free,
                    parallel_runs=slots,parallel_train_runs=slots))
                task = allocated[0]
                pending.pop(0)
                output=Path(task['output']); output.mkdir(parents=True,exist_ok=True)
                task['config'] = dict(task['config'],worker_allocation_file=str(output/'worker_allocation.json'),
                                      evaluation_worker_capacity=total_workers)
                task_path=output/'job.json'; result_path=output/'job_result.json'
                if task_path.exists() and not compatible_manifest(read_json(task_path),task):
                    raise ValueError('Job configuration changed; choose a new experiment name: '+str(output))
                if not task_path.exists(): atomic_json(task_path,task)
                if result_path.exists():
                    completed[task['id']]=read_json(result_path)
                    continue
                dispatch_path=output/'dispatch.json'
                atomic_json(dispatch_path,task)
                atomic_json(output/'worker_allocation.json',dict(workers=task['workers']))
                log=(output/'console.log').open('ab',buffering=0)
                process=managed_popen([sys.executable,'-u','-m','blackout_v12.job','--task',str(dispatch_path)],
                    cwd=ROOT,env=child_environment(),stdin=subprocess.DEVNULL,stdout=log,stderr=log,**process_options())
                log.close()
                active[task['id']]=(process,task,time.monotonic())
            for name,(process,task,started) in list(active.items()):
                code=process.poll()
                if code is None: continue
                del active[name]
                path=Path(task['output'])/'job_result.json'
                if code or not path.exists():
                    error=Path(task['output'])/'job_error.json'
                    details=read_json(error) if error.exists() else dict(exit_code=code)
                    raise RuntimeError(f'Experiment job {name} failed: {details}')
                completed[name]=read_json(path)
            # Transfer slots released by finished evaluations to remaining
            # pools. Their bounded submit loop observes the new target; live
            # matches and immutable match identities are never changed.
            if not pending and active:
                free=total_workers-sum(item[1]['workers'] for item in active.values())
                evaluations=[item for item in active.values() if item[1]['kind']=='evaluate']
                for index,(_,task,_) in enumerate(evaluations):
                    if free<1: break
                    extra=max(1,free//(len(evaluations)-index))
                    task['workers']+=extra; free-=extra
                    atomic_json(Path(task['output'])/'worker_allocation.json',dict(workers=task['workers']))
            progress(active_jobs=list(active),pending_jobs=len(pending),completed_jobs=len(completed),
                     active_worker_limits={name:item[1]['workers'] for name,item in active.items()})
            if active: time.sleep(.25)
        if stop(): raise InterruptedError('study stopped')
        return completed
    except BaseException:
        if config.get('stop_file'):
            Path(config['stop_file']).touch()
        # Episode loops see the stop marker and close their own Unity instance.
        deadline=time.monotonic()+config.get('stop_grace_seconds',20)
        while active and time.monotonic()<deadline:
            active={k:v for k,v in active.items() if v[0].poll() is None}
            if active: time.sleep(.2)
        for process,_,_ in active.values():
            # taskkill /T is restricted to the child PID we created, not an image name.
            if sys.platform=='win32':
                subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True,**process_options())
            else:
                process.terminate()
        raise


def _checkpoint(output, config, arm):
    import torch
    from .policy import MyPolicy
    path=Path(output)/'initial.pt'
    if not path.exists():
        with torch.random.fork_rng():
            torch.manual_seed(config['initial_seed'])
            policy=MyPolicy(config,arm=arm)
        atomic_torch(path,dict(scorer_state={k:v.detach().cpu() for k,v in policy.scorer.state_dict().items()},
                              config=config,arm=arm,seed=config['initial_seed']))
    return str(path)


def run_study(directory, config, stop=lambda:False, progress=lambda **_:None):
    from .opponents import dev_catalog, confirmation_catalog
    from .evaluation import summarize, paired_macro_difference
    directory=Path(directory)
    atomic_json(directory/'plan.json',plan(config))
    catalog=config['opponents']; ids={s['id'] for s in catalog}
    dev=[s for s in dev_catalog() if s['id'] in ids]
    confirmation=[s for s in confirmation_catalog() if s['id'] in ids]
    workers=int(config['workers_per_run'])
    common=dict(config=config,workers=workers)
    def training(stage,arm,seed,steps,initial):
        name=f'{stage}_{arm}_s{seed}'
        return dict(common,id=name,kind='train',output=str(directory/'runs'/name),arm=arm,
                    seed=seed,steps=steps,initial_checkpoint=initial)
    def evaluation(stage,name,checkpoint,arm,opponents,pairs,split='dev',decoder='map'):
        return dict(common,id=f'{stage}_{name}_{decoder}',kind='evaluate',
                    output=str(directory/'evaluation'/stage/f'{name}_{decoder}'),checkpoint=checkpoint,
                    arm=arm,opponents=opponents,pairs=pairs,split=split,decoder=decoder)
    def launch(stage,tasks):
        progress(phase=stage)
        return run_jobs(tasks,config,stop,progress)

    initial={arm:_checkpoint(directory/'initial'/arm,config,arm) for arm in ['Q0','Q1','Q2']}
    baseline_jobs=[evaluation('baseline',arm,initial[arm],arm,dev,config['dev_pairs']) for arm in initial]
    for spec in catalog:
        if spec['id'].startswith(('v4-','v6-')):
            arm=spec['id'].split('-')[0]
            baseline_jobs.append(evaluation('baseline',arm,spec,arm,[s for s in dev if s['id']!=spec['id']],config['dev_pairs']))
    baselines=launch('P2_baseline',baseline_jobs)
    atomic_json(directory/'baseline_summary.json',baselines)
    for arm in initial:
        report=baselines[f'baseline_{arm}_map']
        if not report['completeness']:
            raise RuntimeError(f'{arm} technical baseline failures; inspect the recorded evaluation')
    # A zero-collection policy must not consume the main learning budget.
    q2=baselines['baseline_Q2_map']
    basic=[v for k,v in q2['opponents'].items() if k in {'scripted-noop','scripted-random','scripted-collector'}]
    if basic and all(v['raw']==0 for v in basic):
        raise RuntimeError('Q2 lost every basic-opponent game; repair base control before learning')
    data=launch('P3_data',[dict(common,id='data',kind='data',output=str(directory/'runs'/'data'),
        steps=config['data_steps'],seed=config['initial_seed'])])['data']
    warm=data['checkpoint']
    pilot=launch('P3_pilot',[training('pilot',arm,config['initial_seed'],config['pilot_steps'],warm) for arm in config['arms']])
    atomic_json(directory/'pilot_summary.json',pilot)
    main=launch('P3_main',[training('main',arm,seed,config['training_steps'],warm)
                          for arm in config['arms'] for seed in config['seeds']])
    dev_jobs=[]
    for name,result in main.items():
        for decoder in config.get('evaluation_decoders',['map','sample']):
            dev_jobs.append(evaluation('main_dev',name,result['checkpoint'],result['arm'],dev,config['dev_pairs'],decoder=decoder))
    learned_dev=launch('P4_dev',dev_jobs)
    # Select an arm/decoder using all development seeds; seed 11 is fixed beforehand.
    options=[dict(arm='Q2',decoder='map',raw=q2['opponent_macro_raw'],score=q2['opponent_macro_score'],checkpoint=initial['Q2'])]
    for arm in config['arms']:
        for decoder in config.get('evaluation_decoders',['map','sample']):
            reports=[learned_dev[f'main_dev_main_{arm}_s{seed}_{decoder}'] for seed in config['seeds']]
            if not all(r['completeness'] for r in reports): continue
            options.append(dict(arm=arm,decoder=decoder,
                raw=sum(r['opponent_macro_raw'] for r in reports)/len(reports),
                score=sum(r['opponent_macro_score'] for r in reports)/len(reports),
                checkpoint=main[f'main_{arm}_s{config["initial_seed"]}']['checkpoint']))
    eligible=[o for o in options if o['score'] >= q2['opponent_macro_score']-config['adoption']['noninferiority_margin']]
    selected=dict(max(eligible,key=lambda o:(o['raw'],o['score'],o['arm']=='Q2',o['decoder']=='map')))
    selected.update(seed=config['initial_seed'],selection_split='dev',test_used=False,
                    source_checkpoint_sha256=file_hash(selected['checkpoint']),options=options,
                    seed_selection='initial_seed fixed before training; held-out results never change selection')
    lock=directory/'selection_lock.json'
    if lock.exists() and read_json(lock)!=selected: raise ValueError('Frozen selection changed')
    atomic_json(lock,selected)
    arm,decoder=selected['arm'],selected['decoder']
    if arm=='Q2':
        confirmation_models=[('Q2',selected['checkpoint'])]
        atomic_json(directory/'additional_seed_summary.json',dict(skipped=True,reason='Q2 has no learned training-seed variance'))
    else:
        extra=launch('P4_additional_seeds',[training('additional',arm,seed,config['training_steps'],warm) for seed in config['confirmation_seeds']])
        confirmation_models=[(f'{arm}_s{seed}',main[f'main_{arm}_s{seed}']['checkpoint']) for seed in config['seeds']]
        confirmation_models.extend((f'{arm}_s{seed}',extra[f'additional_{arm}_s{seed}']['checkpoint']) for seed in config['confirmation_seeds'])
    confirmed=launch('P4_confirmation',[evaluation('confirmation',name,path,arm,confirmation,config['confirmation_pairs'],'confirmation',decoder)
                    for name,path in confirmation_models])
    test_pairs={s['id']:config['strong_test_pairs'] if s['id'].startswith(('v4-','v6-')) else config['test_pairs'] for s in catalog}
    final_jobs=[evaluation('test','selected',selected['checkpoint'],arm,catalog,test_pairs,'test',decoder)]
    reference=next((s for s in catalog if s['id'].startswith('v4-')),None)
    if reference:
        opponents=[s for s in catalog if s['id']!=reference['id']]
        final_jobs.append(evaluation('test','frozen_v4',reference,'v4',opponents,test_pairs,'test'))
    final=launch('P4_test',final_jobs)
    candidate=final[f'test_selected_{decoder}']
    comparisons={}
    if reference:
        baseline=final['test_frozen_v4_map']
        candidate_rows=read_json(candidate['rows_path']); baseline_rows=read_json(baseline['rows_path'])
        comparisons={m:paired_macro_difference(candidate_rows,baseline_rows,m) for m in ('raw_points','score_points')}
    from .export import export_checkpoint
    destination=ROOT/'artifacts/submission/v12'/directory.name
    submission=export_checkpoint(selected['checkpoint'],destination,arm=arm,decoder=decoder)
    report=dict(schema='v12.study_result.1',complete=True,scope=config.get('scope'),selection=selected,
                submission=str(submission),confirmation=confirmed,final=final,comparisons=comparisons,
                missing_references=config.get('missing_opponent_ids',[]),
                judgment='unconfirmed',game_strength_claim=False,
                limitation='Frozen-reference noninferiority and full ONNX sequence certification require separate evidence.',
                plan=plan(config))
    atomic_json(directory/'summary.json',report)
    return report
