"""Explicit, non-launching continuation with reduced dev cost and immutable origin.

The source run is never rewritten. Training state is copied to a new run;
only evaluation scheduling/model-selection metadata and provenance change.
"""
import argparse
import copy
from pathlib import Path
import re
import shutil
import sqlite3
import time
import torch
from .io import ROOT,read_json,atomic_json,atomic_torch,digest,file_hash,source_identity
from .runtime import FileLease
from .evaluation import schedule,cached_matches,summarize,seed_cache
from .study import run_configuration,preflight


EVALUATION_FILES={
    'code/v13/blackout_v13/study.py',
    'code/v13/blackout_v13/evaluation.py',
    'code/v13/blackout_v13/migrate_evaluation.py',
    'code/v13/scripts/prepare_reduced_evaluation.py',
    'code/v13/configs/default.json',
    'code/v13/tests/test_evaluation_continuation.py',
}


def validate_source_changes(old,new):
    changed=sorted(k for k in old.keys()|new.keys() if old.get(k)!=new.get(k))
    if set(changed)-EVALUATION_FILES:
        raise ValueError('Non-evaluation source changes cannot be continued: '+str(sorted(set(changed)-EVALUATION_FILES)))
    return changed


def continuation_configuration(original,name,origin,teacher_source):
    config=copy.deepcopy(original)
    config.update(name=name,evaluation_interval=2097152,intermediate_dev_pairs=2,
                  continuation_origin=origin,teacher_source=teacher_source)
    if config['dev_pairs']!=5 or config['confirmation_pairs']!=20:
        raise ValueError('This migration preserves the registered 220/1720-game final evaluations')
    allowed={'name','evaluation_interval','intermediate_dev_pairs','continuation_origin','teacher_source'}
    if any(config.get(k)!=original.get(k) for k in config.keys()|original.keys() if k not in allowed):
        raise ValueError('Training configuration changed')
    return config


def retarget_checkpoint(payload,config,identity,source_path,checksum):
    result=dict(payload)
    result['config']=copy.deepcopy(config)
    for key in ('episode_start','worker_rng_states'):
        if key in payload['config']:result['config'][key]=payload['config'][key]
    result['identity']=identity
    result['progress']=copy.deepcopy(payload['progress'])
    result['continuation']=dict(source_checkpoint=str(source_path),source_sha256=checksum,
                                source_identity=payload['identity'],source_version=payload['version'])
    return result


def best_common_screen(config,directory,checkpoints):
    tasks=schedule(config['evaluation_opponents'],config['intermediate_dev_pairs'],config['dev_seed_base'])
    candidates=[]
    for version,path in sorted(checkpoints.items()):
        rows=cached_matches(tasks,[directory/'dev'/str(version)])
        if len(rows)==len(tasks):
            report=summarize(rows)
            candidates.append(dict(version=version,path=path,score=report['macro_score_rate'],report=report))
    if not candidates:raise ValueError('No checkpoint has a complete common screening set')
    # Prefer the earlier version on a tie, matching the strict > update rule.
    return max(candidates,key=lambda x:(x['score'],-x['version'])),candidates


def prepare(source_name,target_name):
    from agent_league.new_game_runtime import runtime_spec
    for name in (source_name,target_name):
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',name):raise ValueError('invalid run name')
    if source_name==target_name:raise ValueError('Continuation requires a new run name')
    source=ROOT/'logs/v13'/source_name;target=ROOT/'logs/v13'/target_name
    old_artifacts=ROOT/'artifacts/checkpoints/v13'/source_name
    artifacts=ROOT/'artifacts/checkpoints/v13'/target_name
    if not source.is_dir():raise FileNotFoundError(source)
    if target.exists() or artifacts.exists():raise FileExistsError('Target already exists; refusing to overwrite continuation')
    with FileLease(source/'run.lock',blocking=False),FileLease(ROOT/'logs/v13/hardware.lock',blocking=False):
        state=read_json(source/'status.json')
        if state['phase'] not in ('stopped','failed'):raise ValueError('Stop the source run before preparing continuation')
        original=read_json(source/'request.json');old_manifest=read_json(source/'manifest.json')
        if digest({k:v for k,v in old_manifest.items() if k!='identity'})!=old_manifest['identity']:
            raise ValueError('Origin manifest digest mismatch')
        if original!=old_manifest['config']:raise ValueError('Origin request/manifest mismatch')
        current_sources=source_identity();changed=validate_source_changes(old_manifest['sources'],current_sources)
        runtime=runtime_spec()
        if runtime!=old_manifest['runtime']:raise ValueError('Runtime changed')
        data=ROOT/'artifacts/data/v13'/source_name/'teacher';meta=read_json(data/'dataset.json')
        if meta['identity']!=old_manifest['identity'] or meta['steps']<original['teacher_steps']:
            raise ValueError('Complete compatible teacher dataset is required')
        origin=dict(run=source_name,identity=old_manifest['identity'],manifest_sha256=file_hash(source/'manifest.json'))
        teacher=dict(directory=data.relative_to(ROOT).as_posix(),identity=meta['identity'],
                     manifest_sha256=file_hash(data/'dataset.json'))
        config=continuation_configuration(original,target_name,origin,teacher)
        preflight(config,full_hash=True)
        resources=read_json(source/'resources/selected.json')
        manifest=dict(config=config,sources=current_sources,runtime=runtime);identity=digest(manifest)
        plans=[];source_files={}
        def registered(path):
            checksum=file_hash(path);source_files[str(path.relative_to(ROOT))]=checksum
            return checksum
        def read_checkpoint(path,expected):
            registered(path);payload=torch.load(path,map_location='cpu',weights_only=True)
            if payload['identity']!=expected:raise ValueError('Checkpoint identity mismatch: '+str(path))
            return payload
        # Validate everything before writing the continuation directory.
        for arm in original['arms']:
            tag=f'pilot_{arm}_s{original["seeds"][0]}'
            if not (source/'runs'/tag/'result.json').exists():raise ValueError('Complete pilots required')
        for directory in sorted((source/'runs').iterdir()):
            tag=directory.name
            match=re.fullmatch(r'(pilot|main)_(B[1-6])_s(\d+)',tag)
            if not match:continue
            kind,arm,seed=match.groups();seed=int(seed)
            old_config=run_configuration(original,arm,seed,resources)
            expected=digest(dict(study=old_manifest['identity'],config=old_config,tag=tag))
            new_config=run_configuration(config,arm,seed,resources)
            new_identity=digest(dict(study=identity,config=new_config,tag=tag))
            old_folder=old_artifacts/tag
            latest=read_checkpoint(old_folder/'latest.pt',expected)
            best=read_checkpoint(old_folder/'best.pt',expected)
            if kind=='main' and latest['progress']['completed']:
                raise ValueError('Completed main runs need separate full-score import; not supported by this migration')
            selection=None;candidates=[]
            best_source=old_folder/'best.pt'
            if kind=='main':
                checkpoints={}
                for path in [*sorted((old_folder/'selfplay').glob('*.pt')),old_folder/'best.pt',old_folder/'latest.pt']:
                    p=read_checkpoint(path,expected);checkpoints[p['version']]=path
                selection,candidates=best_common_screen(config,directory,checkpoints)
                best_source=selection['path'];best=read_checkpoint(best_source,expected)
            new_latest=retarget_checkpoint(latest,new_config,new_identity,old_folder/'latest.pt',registered(old_folder/'latest.pt'))
            new_best=retarget_checkpoint(best,new_config,new_identity,best_source,registered(best_source))
            if selection is not None:
                progress=new_latest['progress'];progress['best_score']=selection['score']
                progress['next_eval']=(progress['env_steps']//config['evaluation_interval']+1)*config['evaluation_interval']
                progress['selection_protocol']='common_88_then_selected_220'
                new_best['progress']['best_score']=selection['score']
            result=None
            if kind=='pilot':
                if not latest['progress']['completed']:raise ValueError('Incomplete pilot checkpoint')
                registered(directory/'result.json');result=read_json(directory/'result.json')
                result['checkpoint']=str(artifacts/tag/'best.pt')
                result['continuation_origin']=str(directory/'result.json')
            plans.append(dict(tag=tag,latest=new_latest,best=new_best,result=result,selection=selection,candidates=candidates))
        for path in (source/'resources/selected.json',source/'map_splits.sqlite3',data/'dataset.json'):
            registered(path)
        target.mkdir(parents=True);artifacts.mkdir(parents=True)
        atomic_json(target/'request.json',config)
        atomic_json(target/'manifest.json',dict(identity=identity,**manifest))
        shutil.copytree(source/'resources',target/'resources')
        # SQLite backup yields a consistent standalone database, including WAL content.
        old_db=sqlite3.connect('file:'+str((source/'map_splits.sqlite3').resolve()).replace('\\','/')+'?mode=ro',uri=True)
        new_db=sqlite3.connect(target/'map_splits.sqlite3')
        try:old_db.backup(new_db)
        finally:new_db.close();old_db.close()
        screening=schedule(config['evaluation_opponents'],config['intermediate_dev_pairs'],config['dev_seed_base'])
        imported=[];pilots=[]
        for plan in plans:
            tag=plan['tag'];directory=target/'runs'/tag;directory.mkdir(parents=True)
            atomic_torch(artifacts/tag/'latest.pt',plan['latest']);atomic_torch(artifacts/tag/'best.pt',plan['best'])
            if plan['result']:
                atomic_json(directory/'result.json',plan['result']);pilots.append(plan['result'])
            else:
                # Keep completed and partial old matches as immutable full-set caches.
                for old_eval in (source/'runs'/tag/'dev').iterdir():
                    shutil.copytree(old_eval,directory/'dev'/old_eval.name/'imported_full')
                    seed_cache(screening,[old_eval],directory/'dev'/old_eval.name/'screen')
            imported.append(dict(run=tag,version=plan['latest']['version'],env_steps=plan['latest']['progress']['env_steps'],
                                 next_eval=plan['latest']['progress']['next_eval'],
                                 best_version=plan['best']['version'],screening_score=plan['selection']['score'] if plan['selection'] else None,
                                 checkpoint_sha256=file_hash(artifacts/tag/'latest.pt')))
        atomic_json(target/'pilot_results.json',pilots)
        receipt=dict(schema='blackout.v13.evaluation-continuation.1',time=time.time(),source=origin,
                     target_identity=identity,changed_source_files=changed,source_files=source_files,imported=imported,
                     evaluation_interval=config['evaluation_interval'],intermediate_games=88,final_dev_games=220,
                     confirmation_games=1720,training_changed=False,experiment_started=False,
                     note='Full old evaluations are re-scored on the exact common two paired seeds; source files remain unchanged.')
        atomic_json(target/'continuation.json',receipt)
        atomic_json(target/'status.json',dict(phase='prepared',time=time.time(),game_started=False,
                    source_run=source_name,resume_command=f'powershell -NoProfile -ExecutionPolicy Bypass -File .\\code\\v13\\run_v13.ps1 --name {target_name}',
                    imported=imported))
        return receipt


def main():
    parser=argparse.ArgumentParser(description='Prepare a reduced-evaluation continuation without starting games or training')
    parser.add_argument('--source',required=True);parser.add_argument('--name',required=True)
    args=parser.parse_args();receipt=prepare(args.source,args.name)
    import json
    print(json.dumps({k:receipt[k] for k in ('target_identity','imported','training_changed','experiment_started')},indent=2))


if __name__=='__main__':main()
