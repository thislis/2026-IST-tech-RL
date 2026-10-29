"""Explicit, audited runtime-only migration of a stopped pre-training study.

The original registration and valid match files are never rewritten. A new
source/resource revision is activated only after all migration checks pass.
"""
import copy
from pathlib import Path
import shutil
from .execution import semantic_config
from .io import ROOT, atomic_json, digest, file_hash, read_json
from .runtime import FileLease


REVIEWED_EXISTING_CHANGES = frozenset({
    'code/v12/blackout_v12/runner.py', 'code/v12/blackout_v12/study.py',
    'code/v12/blackout_v12/runtime.py', 'code/v12/blackout_v12/benchmark.py',
    'code/v12/blackout_v12/training.py', 'code/v12/blackout_v12/evaluation.py',
    'code/v12/blackout_v12/collector.py', 'code/v12/configs/default.json',
})
REVIEWED_NEW_SOURCES = frozenset({
    'code/v12/blackout_v12/execution.py', 'code/v12/blackout_v12/resume.py',
    'code/v12/blackout_v12/throughput_probe.py',
    'code/v12/tests/test_resume.py', 'code/v12/tests/test_resume_workers.py',
})


def effective_registration(directory):
    directory=Path(directory)
    pointer=directory/'resume_preparation.json'
    if not pointer.exists():
        return read_json(directory/'registration.json')
    record=read_json(pointer)
    if record.get('state')!='ready':
        raise ValueError('runtime migration is not committed')
    revision=(directory/record['revision']).resolve()
    if not revision.is_relative_to(directory.resolve()/'runtime_revisions'):
        raise ValueError('invalid runtime revision location')
    registration=read_json(revision/'registration.json')
    if digest(registration)!=record['registration_sha256']:
        raise ValueError('prepared runtime registration changed')
    if file_hash(directory/'registration.json')!=record['original_registration_sha256']:
        raise ValueError('original experiment registration changed')
    return registration


def _sources(previous, current):
    old,new=previous['source_sha256'],current['source_sha256']
    changed=sorted(path for path in old if new.get(path)!=old[path])
    added=sorted(set(new)-set(old))
    disallowed=[path for path in changed if path not in REVIEWED_EXISTING_CHANGES]
    disallowed += [path for path in added if path not in REVIEWED_NEW_SOURCES]
    if disallowed:
        raise ValueError('Unreviewed source changes cannot reuse matches: '+', '.join(disallowed))
    for key in ('packages','python','executable','feature_schema','state_schema','objective'):
        if previous[key]!=current[key]:
            raise ValueError('Runtime migration changed protected identity: '+key)
    return dict(changed=changed,added=added)


def inspect_matches(directory):
    from .evaluation import interrupted_row
    directory=Path(directory)
    valid,interrupted,failed={},{},{}
    for path in sorted((directory/'evaluation').glob('**/cells/*.json')):
        row=read_json(path)
        relative=path.relative_to(directory).as_posix()
        target=valid if row.get('valid') else interrupted if interrupted_row(row) else failed
        target[relative]=file_hash(path)
    return dict(valid=valid,interrupted=interrupted,failed=failed)


def prepare(directory, workers, evidence):
    """Prepare only. Does not clear the stop marker or spawn any process."""
    from .runner import registration
    directory=Path(directory).resolve()
    workers=int(workers)
    if workers<1: raise ValueError('positive worker count required')
    if not directory.is_relative_to((ROOT/'logs/v12').resolve()):
        raise ValueError('migration must stay inside logs/v12')
    with FileLease(directory/'launch.lock',blocking=False), \
         FileLease(directory/'experiment.lock',blocking=False), \
         FileLease(ROOT/'logs/runtime/compute.lock',blocking=False):
        if read_json(directory/'status.json').get('state') not in ('stopped','failed','interrupted'):
            raise ValueError('stop the study before preparing a runtime revision')
        if list(directory.glob('**/latest.pt')) or (directory/'selection_lock.json').exists():
            raise ValueError('this migration is scoped to the current pre-training baseline study')
        if (directory/'resume_preparation.json').exists():
            raise ValueError('a runtime revision is already prepared; verify it rather than overwrite')
        original=read_json(directory/'registration.json')
        resolved=read_json(directory/'resolved_config.json')
        profile=read_json(directory/'runtime_profile.json')
        cfg=copy.deepcopy(original['config'])
        wave_size=int(resolved.get('rollout_episodes_per_wave',resolved['workers_per_run']))
        cfg.update(workers=workers,worker_cap=max(workers,int(cfg.get('worker_cap',8))),
                   cpu_budget_basis='logical',parallel_runs=min(2,workers),
                   parallel_train_runs=max(1,workers//wave_size),rollout_episodes_per_wave=wave_size)
        evidence=Path(evidence).resolve()
        if not evidence.is_file(): raise FileNotFoundError(evidence)
        measurements=read_json(evidence)
        cfg.update(tune_on_resume=bool(measurements.get('measurement_pending',False)),
                   resume_tuning_candidates=[4,6,8],resume_tuning_seconds=15.)
        # Pin the already measured learning batch/device; this is a scheduling
        # migration, not a change to the PPO optimizer or training budget.
        if semantic_config(original['config'])!=semantic_config(cfg):
            raise ValueError('runtime migration would alter the scientific configuration')
        actual=registration(cfg)
        changes=_sources(original,actual)
        if int(measurements['selected_workers'])!=workers:
            raise ValueError('worker selection must match the throughput evidence')
        before=inspect_matches(directory)
        revision=directory/'runtime_revisions'/'0001'
        if revision.exists(): raise FileExistsError('unfinished migration directory: '+str(revision))
        revision.mkdir(parents=True)
        for name in ('registration.json','runtime_profile.json','resolved_config.json','status.json'):
            shutil.copyfile(directory/name,revision/('previous_'+name))
        shutil.copyfile(evidence,revision/'throughput_evidence.json')
        atomic_json(revision/'registration.json',actual)
        updated=copy.deepcopy(profile)
        updated['selected'].update(workers=workers,workers_per_run=max(1,workers//min(2,workers)),
            parallel_runs=min(2,workers),parallel_train_runs=max(1,workers//wave_size),
            measured_worker_cap=workers)
        updated.update(resource_tuning_evidence=str(revision/'throughput_evidence.json'),
                       resource_tuning_sha256=file_hash(evidence),
                       selection_method=('actual Unity measurement deferred until user resume' if cfg['tune_on_resume'] else
                                         'actual bounded Unity throughput; scientific config and PPO batch unchanged'))
        atomic_json(revision/'runtime_profile.json',updated)
        affected={}
        for relative in before['interrupted']:
            cell=directory/relative
            affected[cell.parent.parent]=True
        archive_paths=[directory/relative for relative in before['interrupted']]
        for folder in affected:
            archive_paths.extend(path for name in ('job_result.json','job_error.json','rows.json','summary.json')
                                 if (path:=folder/name).is_file())
        atomic_json(revision/'migration_plan.json',dict(valid_files=before['valid'],
                    interrupted_files=before['interrupted'],genuine_failures=before['failed'],
                    archive=[p.relative_to(directory).as_posix() for p in archive_paths],
                    source_changes=changes))
        # Preserve old cancellation records verbatim outside the resumable
        # cells namespace. Valid matches and genuine failures stay untouched.
        for path in archive_paths:
            destination=revision/'archived'/path.relative_to(directory)
            source=path.resolve(); destination=destination.resolve()
            if not source.is_relative_to(directory) or not destination.is_relative_to(revision):
                raise ValueError('archive path escaped the experiment')
            destination.parent.mkdir(parents=True,exist_ok=True)
            path.replace(destination)
        after=inspect_matches(directory)
        if before['valid']!=after['valid'] or before['failed']!=after['failed']:
            raise ValueError('migration changed a completed match or genuine failure')
        record=dict(schema='v12.resume_preparation.1',state='ready',
            revision=revision.relative_to(directory).as_posix(),registration_sha256=digest(actual),
            original_registration_sha256=file_hash(directory/'registration.json'),
            preserved_valid_matches=len(before['valid']),retry_interrupted_matches=len(before['interrupted']),
            retained_genuine_failures=len(before['failed']),workers=workers,
            rollout_episodes_per_wave=wave_size,parallel_train_runs=cfg['parallel_train_runs'],
            measurement_pending=cfg['tune_on_resume'],
            main_experiment_started=False,source_changes=changes)
        atomic_json(directory/'resume_preparation.json',record)
        return record


def check(directory):
    """Read-only readiness check: no source rewrite, stop clearing, or launch."""
    from .runner import registration
    directory=Path(directory)
    expected=effective_registration(directory)
    if registration(expected['config'])!=expected:
        raise ValueError('source/config/packages changed after resume preparation')
    pointer=read_json(directory/'resume_preparation.json')
    revision=directory/pointer['revision']
    plan=read_json(revision/'migration_plan.json')
    actual=inspect_matches(directory)
    if any(actual['valid'].get(path)!=value for path,value in plan['valid_files'].items()):
        raise ValueError('a preserved completed match changed')
    return dict(ready=True,state=read_json(directory/'status.json')['state'],
                preserved_valid_matches=len(plan['valid_files']),
                retry_interrupted_matches=pointer['retry_interrupted_matches'],
                workers=pointer['workers'],rollout_episodes_per_wave=pointer['rollout_episodes_per_wave'],
                parallel_train_runs=pointer['parallel_train_runs'],
                measurement_pending=pointer.get('measurement_pending',False),experiment_started=False)


def activate_tuning(directory, evidence):
    """Called by the user's resume launcher after its bounded probe completed.

    The caller holds launch.lock. No policy/optimizer/batch/seed setting can
    change here, and no study process is started by this function.
    """
    from .runner import registration
    directory=Path(directory)
    previous=effective_registration(directory)
    if registration(previous['config'])!=previous:
        raise ValueError('source changed while tuning; refuse to resume')
    report=read_json(evidence)
    workers=int(report['selected_workers'])
    if (workers not in previous['config']['resume_tuning_candidates']
            or not report.get('throughput_only') or report.get('evaluation_evidence')):
        raise ValueError('invalid isolated Unity throughput evidence')
    with FileLease(directory/'experiment.lock',blocking=False), \
         FileLease(ROOT/'logs/runtime/compute.lock',blocking=False):
        old_pointer=read_json(directory/'resume_preparation.json')
        if not old_pointer.get('measurement_pending'):
            raise ValueError('this runtime revision was already measured')
        old_revision=directory/old_pointer['revision']
        cfg=copy.deepcopy(previous['config'])
        wave=cfg['rollout_episodes_per_wave']
        cfg.update(workers=workers,worker_cap=max(workers,cfg['worker_cap']),
                   parallel_runs=min(2,workers),parallel_train_runs=max(1,workers//wave),tune_on_resume=False)
        if semantic_config(cfg)!=semantic_config(previous['config']):
            raise ValueError('tuning changed learning or match identity')
        number=max(int(p.name) for p in (directory/'runtime_revisions').iterdir() if p.name.isdigit())+1
        revision=directory/'runtime_revisions'/f'{number:04d}'
        revision.mkdir()
        actual=registration(cfg)
        profile=read_json(old_revision/'runtime_profile.json')
        profile['selected'].update(workers=workers,workers_per_run=max(1,workers//cfg['parallel_runs']),
            parallel_runs=cfg['parallel_runs'],parallel_train_runs=cfg['parallel_train_runs'],measured_worker_cap=workers)
        profile.update(resource_tuning_evidence=str(Path(evidence).resolve()),
                       resource_tuning_sha256=file_hash(evidence),selection_method=report['selection_rule'])
        atomic_json(revision/'registration.json',actual)
        atomic_json(revision/'runtime_profile.json',profile)
        shutil.copyfile(old_revision/'migration_plan.json',revision/'migration_plan.json')
        shutil.copyfile(evidence,revision/'throughput_evidence.json')
        atomic_json(revision/'previous_resume_preparation.json',old_pointer)
        record=dict(old_pointer,revision=revision.relative_to(directory).as_posix(),
                    registration_sha256=digest(actual),workers=workers,
                    parallel_train_runs=cfg['parallel_train_runs'],measurement_pending=False,
                    selected_by=report['selection_rule'])
        atomic_json(directory/'resume_preparation.json',record)
        return cfg


def prepare_scale(directory, evidence):
    """Audit an explicitly requested data-stage scheduling revision, then prepare.

    Completed data and evaluations stay intact. Larger fixed-policy collection
    batches can change future adaptive opponent/side scheduling; PPO waves,
    learning budgets, seeds and model definitions remain protected.
    """
    from .runner import registration
    directory=Path(directory).resolve()
    if not directory.is_relative_to((ROOT/'logs/v12').resolve()):
        raise ValueError('runtime revision must stay inside logs/v12')
    evidence=Path(evidence).resolve()
    report=read_json(evidence)
    workers=int(report['selected_workers'])
    if (not report.get('throughput_only') or report.get('evaluation_evidence')
            or workers<2 or workers%2
            or workers not in [row['workers'] for row in report['candidates']]
            or report.get('source_file_sha256')!=file_hash(Path(__file__).with_name('throughput_probe.py'))):
        raise ValueError('require measured, current, isolated throughput evidence')
    with FileLease(directory/'launch.lock',blocking=False), \
         FileLease(directory/'experiment.lock',blocking=False), \
         FileLease(ROOT/'logs/runtime/compute.lock',blocking=False):
        if read_json(directory/'status.json').get('state') not in ('stopped','interrupted'):
            raise ValueError('stop study before changing resources')
        checkpoints=list(directory.glob('**/latest.pt'))
        if any(path!=directory/'runs/data/latest.pt' for path in checkpoints) or (directory/'selection_lock.json').exists():
            raise ValueError('this reviewed revision is limited to the fixed-policy data stage')
        old_pointer=read_json(directory/'resume_preparation.json')
        old_revision=directory/old_pointer['revision']
        previous=effective_registration(directory)
        number=max(int(p.name) for p in (directory/'runtime_revisions').iterdir() if p.name.isdigit())+1
        cfg=copy.deepcopy(previous['config'])
        cfg.update(workers=workers,worker_cap=workers,parallel_runs=2,
                   parallel_train_runs=max(1,workers//cfg['rollout_episodes_per_wave']),
                   data_collection_workers=workers,cpu_reserve=0,memory_reserve_gib=4,
                   tune_on_resume=False,execution_revision=f'runtime_revisions/{number:04d}')
        if semantic_config(cfg)!=semantic_config(previous['config']):
            raise ValueError('scaling changed a protected experiment setting')
        actual=registration(cfg)
        changed=sorted(path for path in previous['source_sha256']
                       if actual['source_sha256'].get(path)!=previous['source_sha256'][path])
        added=sorted(set(actual['source_sha256'])-set(previous['source_sha256']))
        allowed_changed={'code/v12/blackout_v12/'+name+'.py' for name in
                         ('training','execution','demonstrations','throughput_probe','resume','runner')}
        allowed_added={'code/v12/tests/'+name+'.py' for name in
                       ('test_data_parallel','test_probe_scaling','test_scale_resume')}
        if set(changed)-allowed_changed or set(added)-allowed_added:
            raise ValueError('unreviewed scaling source changes: '+str(dict(changed=changed,added=added)))
        for key in ('packages','python','executable','feature_schema','state_schema','objective'):
            if actual[key]!=previous[key]:
                raise ValueError('protected experiment identity changed: '+key)
        before=inspect_matches(directory)
        if before['interrupted']:
            raise ValueError('finish separate interrupted evaluation migration first')
        artifacts=ROOT/'artifacts/data/v12'/directory.relative_to(ROOT/'logs/v12')
        preserved={path.relative_to(ROOT).as_posix():file_hash(path)
                   for path in sorted(artifacts.glob('**/*.pt'))}
        preserved.update({path.relative_to(ROOT).as_posix():file_hash(path) for path in checkpoints})
        revision=directory/'runtime_revisions'/f'{number:04d}'
        revision.mkdir()
        atomic_json(revision/'previous_resume_preparation.json',old_pointer)
        for name in ('status.json','resolved_config.json'):
            shutil.copyfile(directory/name,revision/('previous_'+name))
        if checkpoints:
            shutil.copyfile(checkpoints[0],revision/'previous_data_checkpoint.pt')
        shutil.copyfile(evidence,revision/'throughput_evidence.json')
        atomic_json(revision/'registration.json',actual)
        profile=read_json(old_revision/'runtime_profile.json')
        profile['selected'].update(workers=workers,workers_per_run=max(1,workers//2),
            parallel_runs=2,parallel_train_runs=cfg['parallel_train_runs'],
            measured_worker_cap=workers,data_collection_workers=workers)
        profile.update(resource_tuning_evidence=str(revision/'throughput_evidence.json'),
                       resource_tuning_sha256=file_hash(evidence),selection_method=report['selection_rule'])
        atomic_json(revision/'runtime_profile.json',profile)
        atomic_json(revision/'migration_plan.json',dict(valid_files=before['valid'],
            interrupted_files={},genuine_failures=before['failed'],preserved_artifacts=preserved,
            source_changes=dict(changed=changed,added=added),archive=[],
            data_schedule_change='Finish immutable partial wave first; subsequent fixed-Q2 batches use available workers. '
                'Continuous episode serial seeds retained; adaptive opponent/side decisions may differ from two-episode batches. '
                'No Q2 optimizer update and no change to PPO episode wave size.'))
        if inspect_matches(directory)!=before or any(file_hash(ROOT/path)!=value for path,value in preserved.items()):
            raise ValueError('completed evidence changed during runtime preparation')
        record=dict(old_pointer,revision=revision.relative_to(directory).as_posix(),
            registration_sha256=digest(actual),workers=workers,data_collection_workers=workers,
            parallel_train_runs=cfg['parallel_train_runs'],measurement_pending=False,
            preserved_valid_matches=len(before['valid']),retry_interrupted_matches=0,
            selected_by=report['selection_rule'],main_experiment_started=False,
            source_changes=dict(changed=changed,added=added))
        atomic_json(directory/'resume_preparation.json',record)
        return record
