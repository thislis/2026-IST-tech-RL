"""Default launch detaches. Check/status/dry-run never launch Unity or training."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from .io import ROOT,read_json,atomic_json
from .runtime import FileLease,child_environment,process_options


def main():
    parser=argparse.ArgumentParser(description='v13 background study; resumes committed work automatically')
    parser.add_argument('--name',default='hierarchical_v13')
    parser.add_argument('--config')
    parser.add_argument('--arm',action='append',choices=['B1','B2','B3','B4','B5','B6'])
    parser.add_argument('--scope',choices=['full','pilot'],default='full')
    parser.add_argument('--resume',action='store_true',help='automatic by default')
    group=parser.add_mutually_exclusive_group()
    for flag in ('check','dry-run','status','stop','foreground','resume-check'):
        group.add_argument('--'+flag,action='store_true')
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',args.name):parser.error('name must contain only letters, digits, - or _')
    directory=(ROOT/'logs/v13'/args.name).resolve()
    if not directory.is_relative_to((ROOT/'logs/v13').resolve()):raise ValueError('run directory escaped workspace')
    if args.status:
        print(json.dumps(read_json(directory/'status.json') if (directory/'status.json').exists() else
                         dict(phase='not_started',game_started=False),ensure_ascii=False,indent=2));return
    if args.stop:
        if not directory.exists():raise SystemExit('Run has not started')
        (directory/'stop.request').write_text('user stop\n',encoding='utf-8');print('Stop requested; latest committed checkpoint is preserved.');return
    from .study import configuration,preflight,Study
    # A launch request is immutable; status/resume need not restate arm/scope flags.
    request=directory/'request.json'
    if request.exists():
        saved=read_json(request)
        if args.config or args.arm or args.scope!='full':
            proposed=configuration(args.config,args.name,args.arm,args.scope)
            if proposed!=saved:raise ValueError('Existing run options differ; choose a new name')
        cfg=saved
    else:cfg=configuration(args.config,args.name,args.arm,args.scope)
    if args.check or args.resume_check:
        result=preflight(cfg,full_hash=True)
        if args.resume_check and (directory/'manifest.json').exists():
            from .io import digest,source_identity
            from agent_league.new_game_runtime import runtime_spec
            identity=digest(dict(config=cfg,sources=source_identity(),runtime=runtime_spec()))
            if identity!=read_json(directory/'manifest.json')['identity']:raise ValueError('Resume identity changed')
            result['resume_identity_matches']=True
        print(json.dumps(result,ensure_ascii=False,indent=2));return
    if args.dry_run:
        total=cfg['teacher_steps']+len(cfg['arms'])*cfg['pilot_steps']
        if cfg['scope']=='full':total+=min(cfg['finalist_count'],len(cfg['arms']))*len(cfg['seeds'])*sum(cfg[k] for k in ('worker_steps','manager_steps','alternating_steps'))
        print(json.dumps(dict(config=cfg,estimated_training_env_steps=total,directory=str(directory),
            stages=['full ONNX gate','hardware probes','teacher dataset','BC / pilots','finalists','confirmation / export'],
            game_started=False,background=True),ensure_ascii=False,indent=2));return
    directory.mkdir(parents=True,exist_ok=True)
    if args.foreground:
        with FileLease(directory/'run.lock',blocking=False), FileLease(ROOT/'logs/v13/hardware.lock',blocking=False):
            atomic_json(request,cfg)
            Study(cfg,directory).run()
        return
    with FileLease(directory/'launch.lock',blocking=False):
        try:
            with FileLease(directory/'run.lock',blocking=False):pass
        except (OSError,RuntimeError):
            print('Already running: '+str(directory/'status.json'));return
        preflight(cfg,full_hash=False)
        atomic_json(request,cfg);(directory/'stop.request').unlink(missing_ok=True)
        command=[sys.executable,str(ROOT/'code/v13/scripts/v13_experiments.py'),'--name',args.name,'--foreground']
        environment=child_environment()
        with (directory/'console.log').open('ab',buffering=0) as log:
            process=subprocess.Popen(command,cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=log,
                                     **process_options(detached=True))
        # Keep launch.lock until the child owns run.lock to close the double-launch race.
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            if process.poll() is not None:
                raise RuntimeError('Background startup failed; inspect '+str(directory/'console.log'))
            try:
                with FileLease(directory/'run.lock',blocking=False):pass
            except (OSError,RuntimeError):break
            time.sleep(.1)
        print(json.dumps(dict(launched=True,pid=process.pid,background=True,log=str(directory/'console.log'),
                             status=str(directory/'status.json')),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
