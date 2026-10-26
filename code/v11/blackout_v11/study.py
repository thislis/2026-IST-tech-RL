"""Registered P0–P6 study, fixed budgets and selection before held-out results."""
import copy
from pathlib import Path
import time
import torch
from .io import ROOT, atomic_json, digest, file_hash, read_json
from .opponents import dev_catalog, frozen_catalog, reference_spec
from .evaluation import evaluate_bundle, load_rows
from .statistics import selection_key, final_gates
from .export import ensure_export, export_onnx, verify_export
from .evaluation_schedule import record_evaluation_schedule, confirmation_catalog


class StudyGateError(RuntimeError):
    pass


def _gate(directory,name,passed,*,blocking=True,**evidence):
    report=dict(gate=name,passed=bool(passed),blocking=bool(blocking),**evidence)
    atomic_json(Path(directory)/'gates'/(name+'.json'),report)
    if not passed and blocking:
        atomic_json(Path(directory)/'summary.json',dict(complete=False,stopped_at_gate=name,
                    evidence=report,performance_goal_achieved=False))
        raise StudyGateError(name+' failed; inspect gates/'+name+'.json')
    return report


def _policy(arm,seed=11,decoder='greedy'):
    from .policy import MyPolicy
    with torch.random.fork_rng():
        torch.manual_seed(seed); p=MyPolicy().eval();p.configure(arm,decoder)
    return p


def _bundle(policy,path,arm,decoder):
    p=copy.deepcopy(policy).cpu().eval()
    p.configure(arm,decoder,max_interventions=policy.max_interventions,
                goal_period=policy.goal_period,residual_period=policy.residual_period,use_memory=policy.use_memory)
    return ensure_export(p,path)


def _rate(report,prefix,default=0.):
    values=[v['score_rate'] for k,v in report['opponents'].items() if k.startswith(prefix)]
    return sum(values)/len(values) if values else default


def _common_macro(a,b,scripted=False):
    common=set(a['opponents'])&set(b['opponents'])
    if scripted: common={k for k in common if k.startswith('scripted-')}
    if not common: raise ValueError('no common opponents')
    return (sum(a['opponents'][k]['score_rate'] for k in common)/len(common),
            sum(b['opponents'][k]['score_rate'] for k in common)/len(common))


def _map_audit(directory,reports,strict=True):
    """Never relabel a repeated actual map as held-out because seeds differ."""
    partitions={}; details={}
    for split,group in reports.items():
        hashes=set()
        for report in group:
            hashes.update(r['initial_map_hash'] for r in load_rows(report))
        partitions[split]=hashes
    # Training records and demonstrations both belong to the learning partition.
    train=set()
    for path in Path(directory).glob('**/episodes.jsonl'):
        import json
        for line in path.read_text().splitlines():
            row=json.loads(line); meta=row.get('result') or row.get('metadata') or row
            h=meta.get('initial_map_hash') or row.get('initial_map_hash')
            if h: train.add(h)
    partitions['train']=train
    names=list(partitions)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            common=partitions[a]&partitions[b]
            if common: details[a+'__'+b]=dict(count=len(common),hashes=sorted(common)[:20])
    report=dict(counts={k:len(v) for k,v in partitions.items()},overlaps=details,
                unseen_map_generalization_supported=not details and bool(train),strict=strict)
    atomic_json(Path(directory)/'map_split_audit.json',report)
    if strict and details:
        _gate(directory,'map_split',False,**report)
    return report


def run_study(directory,cfg,registration_sha,stop=lambda:False,progress=lambda **_:None,memory=lambda:0.):
    from .training import train_run
    from .demonstrations import prepare_demonstrations
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    device=cfg.get('resolved_learner_device','cpu')
    plan=dict(schema='v11.study.2',stages=['P0','P1','P2','P3','P4','P5','P6'],
        early_performance_policy='G1_G2_diagnostic_only',
        arms=cfg['arms'],seeds=cfg['seeds'],confirmation_seeds=cfg['confirmation_seeds'],
        teacher_steps=cfg['teacher_steps'],pilot_steps=cfg['pilot_steps']*len(cfg['arms']),
        main_steps=cfg['training_steps']*len(cfg['arms'])*len(cfg['seeds']),
        confirmation_steps=cfg['training_steps']*2*len(cfg['confirmation_seeds']),
        evaluation_outside_training_budget=True,optional_ablations=False,
        pilot_weights_reused=False,selection_seeds=cfg['seeds'],economic_baseline='v4-29794421')
    plan['total_learning_steps']=sum(plan[k] for k in ('teacher_steps','pilot_steps','main_steps','confirmation_steps'))
    if (directory/'task_plan.json').exists() and read_json(directory/'task_plan.json')!=plan:
        raise ValueError('registered study plan changed')
    atomic_json(directory/'task_plan.json',plan)
    evaluation_schedule,evaluation_schedule_sha=record_evaluation_schedule(directory,cfg)
    def phase(name,**extra):
        if stop(): raise InterruptedError('stopped before '+name)
        progress(phase=name,**extra)
        atomic_json(directory/'stage.json',dict(stage=name,updated_at=time.time(),**extra))
    def evaluate(bundle,path,split='dev',pool=None,replicates=None,candidate_spec=None,
                 replicates_by_opponent=None,evaluation_scope=None):
        return evaluate_bundle(bundle,path,cfg,split=split,opponents=pool,replicates=replicates,
                               candidate_spec=candidate_spec,replicates_by_opponent=replicates_by_opponent,
                               evaluation_scope=evaluation_scope,stop=stop,progress=progress)
    phase('P0_contracts')
    baseline=_policy('P'); base_bundle=ensure_export(baseline,directory/'baseline/P')
    g0=directory/'gates/G0.json'
    if g0.exists() and not read_json(g0).get('passed'):
        raise StudyGateError('recorded G0 failure; repair and use a new registered study')
    if not g0.exists():
        loader=verify_export(base_bundle,baseline)
        full=_policy('F');onnx=export_onnx(full,directory/'development_onnx/initial')
        _gate(directory,'G0',loader['passed'] and onnx['passed'],loader=loader,onnx=onnx)

    phase('P1_lifecycle')
    # Ten opponents x both sides; completed games also supply hidden-window evidence.
    lifecycle_pool=dev_catalog()[:10]
    lifecycle=evaluate(base_bundle,directory/'lifecycle',split='baseline',pool=lifecycle_pool,
                       replicates=max(1,cfg['lifecycle_games']//20))
    phase('P1_planner_baseline')
    base_dev=evaluate(base_bundle,directory/'evaluation/baseline/P',split='baseline',pool=dev_catalog(),
                      replicates=cfg['baseline_replicates'])
    reference=reference_spec()
    reference_dev=evaluate(None,directory/'evaluation/baseline/reference',split='baseline',pool=dev_catalog(),
                           candidate_spec=reference,replicates=cfg['baseline_replicates'])
    own,ref=_common_macro(base_dev,reference_dev)
    _gate(directory,'G1',own>=ref-.05 and _rate(base_dev,'scripted-noop')>=.9
          and _rate(base_dev,'scripted-random')>=.9 and _rate(base_dev,'scripted-collector')>=.8,
          blocking=False,planner_macro=own,reference_macro=ref,planner=base_dev,
          comparison_scope='untrained_fixed_planner_vs_trained_reference',
          continue_to='P2_teacher')

    phase('P2_teacher')
    demonstration_path,teacher_report=prepare_demonstrations(directory/'demonstrations',cfg,registration_sha,
                                                             stop=stop,progress=progress)
    atomic_json(directory/'teacher_summary.json',teacher_report)
    phase('P3_pilots')
    pilots={};decoders={}
    for arm in cfg['arms']:
        phase('P3_pilot',run=arm+'_s11')
        run=directory/'pilot'/f'{arm}_s11'
        policy,training=train_run(run,cfg,arm,11,registration_sha,device,pilot=True,
            stop=stop,progress=progress,memory=memory,demonstrations=demonstration_path)
        if stop(): raise InterruptedError('pilot stopped before completion')
        _gate(directory,'technical_pilot_'+arm,training['complete'] and training['technical_passed'],training=training)
        evaluations={}
        for decoder in ('greedy','sample'):
            bundle=_bundle(policy,run/'exports'/decoder,arm,decoder)
            evaluations[decoder]=evaluate(bundle,run/'evaluation'/decoder,pool=dev_catalog(),
                                           replicates=cfg['pilot_replicates'])
        decoder=max(('greedy','sample'),key=lambda d:selection_key(evaluations[d]))
        decoders[arm]=decoder;pilots[arm]=dict(training=training,decoder=decoder,evaluation=evaluations[decoder])
    atomic_json(directory/'pilot_summary.json',pilots)
    f=pilots['F']['evaluation'];f_rows=load_rows(f)
    deposits={str(s):sum(r.get('delivery_proxy',0) for r in f_rows if r['side']==s) for s in (0,1)}
    _gate(directory,'technical_pilot_evaluation',f['invalid']==0,evaluation=f)
    _gate(directory,'G2',min(deposits.values())>0
          and _rate(f,'scripted-collector')>=_rate(base_dev,'scripted-collector')-.05
          and _rate(f,'scripted-random')>=.9,blocking=False,deliveries_by_side=deposits,evaluation=f,
          controls_retained=cfg['arms'],comparison_scope='pilot_not_full_training_budget',
          continue_to='P4_main')

    phase('P4_main')
    main={arm:{} for arm in cfg['arms']}
    for seed in cfg['seeds']:
        for arm in cfg['arms']:
            run=directory/'runs'/f'{arm}_s{seed}';phase('P4_main',run=run.name)
            policy,training=train_run(run,cfg,arm,seed,registration_sha,device,stop=stop,progress=progress,
                                      memory=memory,demonstrations=demonstration_path)
            if stop(): raise InterruptedError('main run stopped before completion')
            _gate(directory,'technical_'+run.name,training['complete'] and training['technical_passed'],training=training)
            bundle=_bundle(policy,run/'exports/final',arm,decoders[arm])
            report=evaluate(bundle,run/'evaluation/dev',pool=dev_catalog(),
                            replicates=evaluation_schedule['main_dev']['map_pairs_per_opponent'],
                            evaluation_scope='main_dev_15_opponents_5_map_pairs_selection')
            main[arm][seed]=dict(training=training,evaluation=report,bundle=str(bundle),
                                 parameters=sum(p.numel() for p in policy.parameters()))
    atomic_json(directory/'main_summary.json',main)
    dev_reports=[r['evaluation'] for runs in main.values() for r in runs.values()]
    _map_audit(directory,dict(dev=dev_reports),strict=cfg.get('strict_map_splits',True))
    def arm_key(arm):
        keys=[selection_key(r['evaluation'],r['parameters']) for r in main[arm].values()]
        return tuple(sum(k[i] for k in keys)/len(keys) for i in range(len(keys[0])))
    winner=max(cfg['arms'],key=arm_key)
    comparison=max([a for a in ('R','G','D') if a!=winner],key=arm_key)
    seed=max(cfg['seeds'],key=lambda s:(*selection_key(main[winner][s]['evaluation'],main[winner][s]['parameters']),-s))
    selected=main[winner][seed]
    strong=_rate(selected['evaluation'],('v4-','v6-'))
    candidate_scripted,baseline_scripted=_common_macro(selected['evaluation'],base_dev,scripted=True)
    _gate(directory,'G3',strong>=.55 and candidate_scripted>=baseline_scripted-.05,
          winner=winner,comparison=comparison,seed=seed,strong_score_rate=strong,
          scripted_score_rate=candidate_scripted,baseline_scripted_rate=baseline_scripted)
    selection=dict(arm=winner,comparison_arm=comparison,seed=seed,decoder=decoders[winner],
        bundle=selected['bundle'],bundle_hashes={n:file_hash(Path(selected['bundle'])/n) for n in ('policy.py','checkpoint.pt')},
        selection_seeds=cfg['seeds'],confirmation_seeds_not_selection_candidates=cfg['confirmation_seeds'],
        objective=cfg['objective'],registration_sha256=registration_sha,
        evaluation_schedule_sha256=evaluation_schedule_sha,
        dev_map_pairs_per_opponent=evaluation_schedule['main_dev']['map_pairs_per_opponent'],
        held_out_results_used_for_selection=False)
    lock=directory/'selection_lock.json'
    if lock.exists() and read_json(lock)!=selection: raise ValueError('selection lock changed')
    atomic_json(lock,selection)

    phase('P5_confirmation_training')
    for arm in (winner,comparison):
        for seed2 in cfg['confirmation_seeds']:
            run=directory/'runs'/f'{arm}_s{seed2}';phase('P5_confirmation_training',run=run.name)
            policy,training=train_run(run,cfg,arm,seed2,registration_sha,device,stop=stop,progress=progress,
                                      memory=memory,demonstrations=demonstration_path)
            if stop(): raise InterruptedError('confirmation training stopped before completion')
            _gate(directory,'technical_'+run.name,training['complete'] and training['technical_passed'],training=training)
            bundle=_bundle(policy,run/'exports/final',arm,decoders[arm])
            main[arm][seed2]=dict(training=training,bundle=str(bundle))
    # All four additional runs above retain the original 2,097,152-step budget.
    # The reduced five-opponent confirmation is a reproducibility diagnostic,
    # not a substitute population for the original full-league G4/G5/G6 tests.
    phase('P5_confirmation')
    confirmation={}
    confirmation_plan=evaluation_schedule['confirmation']
    for arm in (winner,comparison):
        for seed2 in cfg['seeds']+cfg['confirmation_seeds']:
            name=f'{arm}_s{seed2}';phase('P5_confirmation',run=name)
            report=evaluate(main[arm][seed2]['bundle'],directory/'evaluation/confirmation'/name,
                            split='confirmation',pool=confirmation_catalog(),
                            replicates=confirmation_plan['map_pairs_per_opponent'],
                            evaluation_scope='five_opponent_reproducibility_diagnostic_only')
            confirmation[name]=report
    _gate(directory,'technical_confirmation',len(confirmation)==confirmation_plan['candidates'] and all(
        report['complete'] and report['n']==confirmation_plan['games_per_candidate']
        for report in confirmation.values()),
        expected_candidates=confirmation_plan['candidates'],actual_candidates=len(confirmation),
        expected_games=confirmation_plan['total_games'],
        actual_games=sum(r['n'] for r in confirmation.values()))
    reproduction={str(s):dict(
        opponent_macro=confirmation[f'{winner}_s{s}']['opponent_macro'],
        strong_score_rate=_rate(confirmation[f'{winner}_s{s}'],('v4-','v6-')),
        worst_side=confirmation[f'{winner}_s{s}']['worst_side'],
        full_league_performance_assessed=False)
        for s in cfg['seeds']+cfg['confirmation_seeds']}
    diagnostic=dict(gate='confirmation',blocking=False,diagnostic_only=True,
                    technical_passed=True,performance_passed=None,
                    full_league_G4_G5_G6_applied=False,selection_allowed=False,
                    scope=confirmation_plan,reports=confirmation,seed_reproduction=reproduction,
                    test_opened=False,continue_to='P6_frozen_test')
    atomic_json(directory/'gates/confirmation.json',diagnostic)
    atomic_json(directory/'evaluation/confirmation/diagnostics.json',diagnostic)
    _map_audit(directory,dict(dev=dev_reports,confirmation=list(confirmation.values())),
               strict=cfg.get('strict_map_splits',True))

    phase('P6_frozen_test')
    # Freeze the one submission candidate before opening any final result. The
    # reference gets exactly its common opponent/map/side cells (self excluded).
    test_plan=evaluation_schedule['frozen_test']
    selected_name=f'{winner}_s{seed}'
    for name,expected in selection['bundle_hashes'].items():
        if file_hash(Path(selected['bundle'])/name)!=expected:
            raise ValueError('locked submission checkpoint changed before final test')
    baseline_report=evaluate(None,directory/'evaluation/test/reference',split='test',
                             candidate_spec=reference,replicates=10,
                             replicates_by_opponent=test_plan['map_pairs_by_opponent'],
                             evaluation_scope='frozen_reference_common_cells_self_excluded')
    for name,expected in selection['bundle_hashes'].items():
        if file_hash(Path(selected['bundle'])/name)!=expected:
            raise ValueError('locked submission checkpoint changed during reference evaluation')
    final_evaluation=evaluate(selected['bundle'],directory/'evaluation/test'/selected_name,split='test',
                              replicates=10,replicates_by_opponent=test_plan['map_pairs_by_opponent'],
                              evaluation_scope='locked_submission_full_league_strong_opponent_oversampling')
    _gate(directory,'technical_final_evaluation',
          baseline_report['complete'] and final_evaluation['complete']
          and baseline_report['n']==test_plan['reference_games']
          and final_evaluation['n']==test_plan['submission_games'],
          expected_submission_games=test_plan['submission_games'],actual_submission_games=final_evaluation['n'],
          expected_reference_games=test_plan['reference_games'],actual_reference_games=baseline_report['n'])
    selected_gates=final_gates(final_evaluation,load_rows(final_evaluation),load_rows(baseline_report))
    reports_by_split=dict(confirmation=confirmation,test={selected_name:final_evaluation})
    _map_audit(directory,dict(dev=dev_reports,confirmation=list(confirmation.values()),
                             test=[final_evaluation,baseline_report]),
               strict=cfg.get('strict_map_splits',True))
    atomic_json(directory/'evaluation/test/gates.json',{selected_name:selected_gates})

    phase('P6_packaging')
    for name,expected in selection['bundle_hashes'].items():
        if file_hash(Path(selected['bundle'])/name)!=expected:
            raise ValueError('tested submission checkpoint changed before packaging')
    from .policy import MyPolicy
    final=MyPolicy().eval()
    final.load_state_dict(torch.load(Path(selected['bundle'])/'checkpoint.pt',map_location='cpu',weights_only=True)['policy_state'])
    identity=digest(selection['bundle_hashes'])
    submission=ensure_export(final,ROOT/'artifacts/submission/v11'/identity)
    loader=verify_export(submission,final)
    # Deterministic development ONNX is labelled separately if sampled submission won.
    onnx=export_onnx(final,directory/'development_onnx/final')
    final_report=dict(complete=True,submission=str(submission),selection=selection,
        planned_learning_steps=plan['total_learning_steps'],teacher=teacher_report,pilots=pilots,
        confirmation=reports_by_split['confirmation'],frozen_test=reports_by_split['test'],
        frozen_reference=baseline_report,evaluation_schedule=evaluation_schedule,
        evaluation_schedule_sha256=evaluation_schedule_sha,
        evaluation_revision=str(directory/'evaluation_revision.json') if (directory/'evaluation_revision.json').exists() else None,
        final_gates=selected_gates,seed_reproduction=reproduction,
        confirmation_scope='five_opponent_reproducibility_diagnostic_only',
        performance_goal_scope='locked_submission_final_test_only',
        all_seed_full_league_performance_assessed=False,
        performance_goal_achieved=all(g['passed'] for g in selected_gates.values()),
        G7=dict(two_file_loader=loader,development_onnx=onnx,official_runtime_certified=False),
        runtime_profile=str(directory/'runtime_profile.json'),official_server_certified=False)
    atomic_json(directory/'summary.json',final_report)
    return final_report
