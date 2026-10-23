import argparse
import fcntl
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from .common import BASE,ROOT,DEFAULTS,atomic,read,digest,sources,schedule,verify_assets


def parser():
    p=argparse.ArgumentParser(description='모든 구조·설정 대표의 진영 교대 풀리그; 학습 없음')
    p.add_argument('--name',default='all_agents_v1');p.add_argument('--rounds',type=int,default=5)
    p.add_argument('--workers',type=int);p.add_argument('--models',help='comma-separated IDs; omitted = all discovered representatives')
    group=p.add_mutually_exclusive_group()
    for flag in ('list','check','benchmark','smoke','status','stop','resume','report','prepare'):group.add_argument('--'+flag,action='store_true')
    p.add_argument('--trace',metavar='MATCH_ID');p.add_argument('--step',type=int,default=0)
    p.add_argument('--supervisor',action='store_true',help=argparse.SUPPRESS)
    return p


def pick_roster(args):
    from .registry import discover
    roster=discover()
    if args.models:
        ids={x.strip() for x in args.models.split(',')};available={r['id'] for r in roster['models']}
        if ids-available:raise ValueError('unknown model IDs: '+str(sorted(ids-available)))
        roster['excluded'] += [dict(path=r['path'] or r['id'],reason='explicit --models filter') for r in roster['models'] if r['id'] not in ids]
        roster['models']=[r for r in roster['models'] if r['id'] in ids]
    if len(roster['models'])<2:raise ValueError('at least two models required')
    return roster


def trace_query(directory,args):
    import numpy as np
    record=read(directory/'cells'/(args.trace+'.json'));trace=Path(record['directory'])/'trace';index=read(trace/'index.json')
    chunk=next((c for c in index['chunks'] if c['first']<=args.step<c['first']+c['count']),None)
    if chunk is None:raise ValueError('step outside saved trace')
    from .trace import load_chunk
    data=load_chunk(trace/chunk['path'])
    i=args.step-chunk['first'];row={k:data[k][i].tolist() for k in data}
    import json
    print(json.dumps(row,ensure_ascii=False,indent=2,allow_nan=True))


def supervise(directory,args):
    from .runtime import preflight,run_league,resource_sample,execute
    from .benchmark import benchmark
    from .report import generate
    from blackout_v10.contracts import verify_original
    from blackout_v10.runtime import check_legacy_processes,clean_orphans
    lockpath=ROOT/'logs/runtime/compute.lock';lockpath.parent.mkdir(parents=True,exist_ok=True)
    with lockpath.open('a') as compute,(directory/'run.lock').open('a') as own:
        fcntl.flock(compute,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        check_legacy_processes();verify_original()
        stopped=False
        validation_deadline=None
        def stop_signal(*_):
            nonlocal stopped
            stopped=True
        signal.signal(signal.SIGINT,stop_signal);signal.signal(signal.SIGTERM,stop_signal)
        def stop():return stopped or (directory/'STOP').exists() or (validation_deadline is not None and time.monotonic()>validation_deadline)
        atomic(directory/'supervisor.json',dict(pid=os.getpid(),started_at=time.time()))
        try:
            if (directory/'manifest.json').exists():
                manifest=read(directory/'manifest.json')
                if manifest.get('experiment_kind')=='smoke' and args.resume:validation_deadline=time.monotonic()+1200
                if not args.resume:raise FileExistsError('existing experiment: use --resume or a new --name')
                if manifest['sources']!=sources():raise ValueError('source changed since registration; use new experiment name')
                for spec in manifest['roster']['models']:verify_assets(spec)
            else:
                request=read(directory/'request.json');roster=request['roster'];cfg=request['config']
                if request['sources']!=sources():raise ValueError('source changed after preparation')
                atomic(directory/'status.json',dict(state='preflight',updated_at=time.time()))
                preflight(roster,directory/'checks',stop)
                if args.check:
                    atomic(directory/'status.json',dict(state='checked',updated_at=time.time(),full_league_started=False));return
                if stop():raise InterruptedError('stopped before probes')
                if args.smoke:validation_deadline=time.monotonic()+1200
                atomic(directory/'status.json',dict(state='benchmark',updated_at=time.time()))
                bench=benchmark(roster,directory/'probes'/str(time.time_ns()),cfg,stop,args.workers)
                atomic(directory/'benchmark.json',bench);cfg.update(bench['selected'])
                if args.benchmark:
                    atomic(directory/'status.json',dict(state='benchmarked',updated_at=time.time(),full_league_started=False));return
                tasks=schedule(roster['models'],cfg)
                if args.smoke:
                    legacy=next((m for m in roster['models'] if m['id'].startswith('v6-')),roster['models'][0])
                    submission=next((m for m in roster['models'] if m['id']=='v10-C1'),roster['models'][-1])
                    if legacy['id']==submission['id']:submission=next(m for m in roster['models'] if m['id']!=legacy['id'])
                    tasks=schedule([legacy,submission],dict(cfg,rounds=1))
                if request['sources']!=sources():raise ValueError('source changed during preparation; use a new name')
                manifest=dict(schema='blackout.league.v1',roster=roster,config=cfg,tasks=tasks,sources=request['sources'],
                    created_at=time.time(),experiment_kind='smoke' if args.smoke else 'league',benchmark=bench,
                    estimated_trace_gib=len(tasks)*cfg['max_steps']*bench['bytes_per_step']/1024**3,
                    provisional_max_steps_hours=len(tasks)*cfg['max_steps']/bench['steps_per_second']/3600)
                atomic(directory/'manifest.json',manifest);generate(directory)
                resource=resource_sample()
                if resource['free_gib']<cfg['reserve_gib']+manifest['estimated_trace_gib']:
                    raise RuntimeError('estimated trace space exceeds free space minus reserve; see manifest.json')
                print(f"등록 완료: {len(tasks)}경기, 예상 trace 상한 {manifest['estimated_trace_gib']:.1f} GiB",flush=True)
            run_league(directory,manifest,stop)
        except BaseException as exc:
            atomic(directory/'status.json',dict(state='stopped' if stop() else 'failed',error=str(exc),updated_at=time.time()))
            if (directory/'manifest.json').exists():generate(directory)
            raise


def main():
    args=parser().parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.name):raise ValueError('name: letters/digits/_/- only')
    if args.rounds<1 or args.workers is not None and not 1<=args.workers<=8:raise ValueError('rounds>=1, workers=1..8')
    directory=BASE/args.name
    if args.trace:trace_query(directory,args);return
    if args.status:
        print((directory/'status.json').read_text() if (directory/'status.json').exists() else 'No status yet')
        if (directory/'checks/result.json').exists():
            c=read(directory/'checks/result.json');print(f"모델 검사 {c['checked']}/{c['expected']}")
        probes=sorted(directory.glob('probes/*/status.json'),key=lambda p:p.stat().st_mtime)
        if probes:print('최근 probe: '+probes[-1].read_text())
        return
    if args.stop:
        if not directory.exists():raise FileNotFoundError(directory)
        atomic(directory/'STOP',dict(requested_at=time.time()));print('중단 요청 기록. 완료 경기와 부분 로그를 보존합니다.');return
    if args.report:
        from .report import generate
        generate(directory);print(directory/'report.md');return
    if args.list:
        roster=pick_roster(args)
        for m in roster['models']:print(f"{m['id']}\t{m['kind']}\t{m['step']}\t{m['label']}")
        n=len(roster['models']);print(f'{n} models, {args.rounds*n*(n-1)} games; {len(roster["excluded"])} excluded/aliases');return
    if args.supervisor:supervise(directory,args);return
    if args.resume:
        if not (directory/'request.json').exists():raise FileNotFoundError('no registered experiment')
        with (directory/'run.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        (directory/'STOP').unlink(missing_ok=True)
    else:
        roster=pick_roster(args);cfg=dict(DEFAULTS,rounds=args.rounds)
        directory.mkdir(parents=True,exist_ok=False)
        atomic(directory/'request.json',dict(roster=roster,config=cfg,sources=sources()))
        atomic(directory/'status.json',dict(state='prepared',full_league_started=False,updated_at=time.time()))
    if args.prepare:print(directory/'request.json');return
    if args.check:
        # Offline checks run in foreground; --check never starts Unity.
        from .runtime import preflight
        preflight(read(directory/'request.json')['roster'],directory/'checks')
        atomic(directory/'status.json',dict(state='checked',updated_at=time.time(),full_league_started=False));return
    argv=[sys.executable,str(ROOT/'code/run.py'),'-m','agent_league.cli',*sys.argv[1:],'--supervisor']
    env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONHASHSEED='0')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):env[key]='1'
    with (directory/'console.log').open('ab',buffering=0) as log:
        p=subprocess.Popen(argv,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    output='benchmark.json' if args.benchmark else 'report.md'
    print(f'PID {p.pid}\n상태: bash code/shared/run_agent_league.sh --name {args.name} --status\n로그: {directory}/console.log\n산출물: {directory}/{output}')

if __name__=='__main__':main()
