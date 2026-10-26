"""Opt-in paired F controls; never extends the registered default study budget."""
import copy
from pathlib import Path
from .io import ROOT, atomic_json, read_json


def run_ablation(directory,cfg,registration_sha,stop=lambda:False,progress=lambda **_:None,memory=lambda:0.):
    from .training import train_run
    from .demonstrations import prepare_demonstrations
    from .evaluation import evaluate_bundle
    from .opponents import dev_catalog
    from .study import _bundle, _map_audit
    settings=read_json(ROOT/'code/v11/configs/ablations.json')
    variant=next((a for a in settings['arms'] if a['name']==cfg['ablation']),None)
    if variant is None: raise ValueError('unsupported ablation '+cfg['ablation'])
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    manifest,teacher=prepare_demonstrations(directory/'demonstrations',cfg,registration_sha,stop,progress)
    reports={};budgets={}
    for label,overrides in (('F_control',{}),(cfg['ablation'],variant['overrides'])):
        local=copy.deepcopy(cfg);local.update(overrides)
        local['training_steps']=settings['training_steps']
        reports[label]={}
        for seed in settings['seeds']:
            if stop(): raise InterruptedError('ablation stopped')
            run=directory/'runs'/f'{label}_s{seed}'
            progress(phase='ablation',run=run.name)
            identity=registration_sha+':'+label
            policy,training=train_run(run,local,'F',seed,identity,local.get('resolved_learner_device','cpu'),
                                      stop=stop,progress=progress,memory=memory,demonstrations=manifest)
            if stop(): raise InterruptedError('ablation stopped before completion')
            if not training['complete'] or not training['technical_passed']:
                raise RuntimeError('ablation technical gate failed: '+run.name)
            bundle=_bundle(policy,run/'exports/final','F','greedy')
            result=evaluate_bundle(bundle,run/'evaluation/dev',local,split='dev',opponents=dev_catalog(),
                                   stop=stop,progress=progress)
            reports[label][str(seed)]=dict(training=training,evaluation=result,bundle=str(bundle))
        budgets[label]=len(settings['seeds'])*settings['training_steps']
    _map_audit(directory,dict(dev=[r['evaluation'] for runs in reports.values() for r in runs.values()]),
               strict=cfg.get('strict_map_splits',True))
    report=dict(complete=True,ablation=cfg['ablation'],paired_control='F_control',
        budgets=budgets,teacher=teacher,results=reports,decoder='greedy',
        frozen_test_opened=False,submission=None,performance_goal_achieved=False)
    atomic_json(directory/'summary.json',report)
    return report
