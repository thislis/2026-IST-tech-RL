"""Bounded background scheduler, complete dev accounting and two-file export."""

from project_paths import project_root, project_path

import argparse
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import numpy as np
from . import study
from .exporting import export_policy,verify_export
from blackout_rl.v8.checkpoints import CheckpointStore,atomic_json
from blackout_rl.v8.contracts import file_hash,digest


def check():
    r=study.validate()
    if shutil.disk_usage(study.ROOT).free<r['config']['disk_reserve_bytes']:raise RuntimeError('insufficient disk reserve')
    print(dict(state='prepared',workers=r['config']['workers'],training_steps=2*len(r['config']['seeds'])*r['config']['training']['steps'],
               dev_games=sum(len(j.get('replicates',[]))*2 for j in study.tasks(r)),original_game=True,unity_started=False,live_throughput_verified=False),flush=True)
    return r


def status():
    active=False;path=study.DIRECTORY/'experiment.lock'
    if path.exists():
        with path.open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:active=True
    return dict(active=active,status=study.read(study.DIRECTORY/'status.json') if (study.DIRECTORY/'status.json').exists() else {'state':'not_started'},
                runs={p.parent.name:study.read(p) for p in (study.DIRECTORY/'runs').glob('*/progress.json')})


def aggregate(r):
    groups={};audit=[];initial_states={}
    for job in study.tasks(r):
        if not study.completed(job):raise ValueError('missing complete job: '+job['id'])
        done=study.read(study.DIRECTORY/'jobs'/job['id']/'done.json');audit.append(done)
        if job['kind']=='train':continue
        result=study.read(project_path(done['artifact'], root=study.ROOT))
        model_lock=study.read((project_path(done['artifact'], root=study.ROOT)).with_name('model_lock.json'))
        lock=model_lock['lock']
        if (digest(lock)!=model_lock['sha256'] or result['lock_sha256']!=model_lock['sha256'] or
            lock['sources']!=r['sources'] or lock['job']!=job or lock['registration_sha256']!=file_hash(study.REGISTRATION) or
            any(x['lock_sha256']!=model_lock['sha256'] for x in result['episodes'])):
            raise ValueError('evaluation model lock mismatch')
        cells={(x['replicate'],x['side']) for x in result['episodes']}
        expected={(n,s) for n in job['replicates'] for s in (0,1)}
        if cells!=expected or len(cells)!=len(result['episodes']) or any(x['status']!='valid' for x in result['episodes']):raise ValueError('incomplete or duplicate evaluation cells')
        group=groups.setdefault((job['run'],job['step']),{})
        for x in result['episodes']:
            key=(x['replicate'],x['side'])
            if key in group:raise ValueError('duplicate aggregate cell')
            group[key]=int(x['winner']==0)
            initial_states.setdefault(f"{job['run']}:{job['step']}:side{x['side']}",set()).add(x['initial_observation_sha256'])
    def cube(runs,step):
        return np.array([[[groups[(run,step)][(i,s)] for s in (0,1)] for i in range(r['config']['evaluation_replicates'])] for run in runs],float)
    rng=np.random.default_rng(929);analyses={};by_run={}
    for step in r['config']['endpoints']:
        a=cube([f'c1_s{s}' for s in r['config']['seeds']],step);b=cube([f'flat_s{s}' for s in r['config']['seeds']],step)
        planner=cube(['planner'],0)
        def interval(x,y,fixed=False):
            # Requested seed != verified actual map. Never pair unrelated map draws.
            values=[]
            for _ in range(r['config']['bootstrap_replicates']):
                xi=rng.integers(len(x),size=len(x));yi=np.array([0]) if fixed else rng.integers(len(y),size=len(y))
                xm=np.stack([x[k,rng.integers(x.shape[1],size=x.shape[1])] for k in xi])
                ym=np.stack([y[k,rng.integers(y.shape[1],size=y.shape[1])] for k in yi])
                values.append(xm.mean()-ym.mean())
            return dict(delta=float(x.mean()-y.mean()),interval=np.quantile(values,[.025,.975]).tolist(),
                        method='independent run and episode-block bootstrap; sides retained; maps not paired',replicates=len(values))
        analyses[str(step)]=dict(c1_mean=float(a.mean()),flat_mean=float(b.mean()),planner_mean=float(planner.mean()),
            c1_vs_flat=interval(a,b),c1_vs_planner=interval(a,planner,True),flat_vs_planner=interval(b,planner,True),
            cube=dict(c1=a.tolist(),flat=b.tolist(),planner=planner.tolist()))
        for arm,data in (('c1',a),('flat',b)):
            for seed,score in zip(r['config']['seeds'],data.mean((1,2))):by_run[(f'{arm}_s{seed}',step)]=float(score)
    endpoint=r['config']['endpoints'][-1];final=analyses[str(endpoint)]
    arm='c1' if final['c1_mean']>=final['flat_mean'] else 'flat'
    run=max((f'{arm}_s{s}' for s in r['config']['seeds']),key=lambda name:by_run[(name,endpoint)])
    h=study.read(study.DIRECTORY/'runs'/run/'checkpoint_index'/f'{endpoint}.json')['sha256']
    payload,_=CheckpointStore(study.DIRECTORY/'runs'/run/'checkpoints').load(h)
    bundle=project_path(Path('submission/v8') / h, root=study.ROOT)
    if not bundle.exists():export_policy(payload,bundle)
    # Never accept a different local policy source under the same selected checkpoint.
    if file_hash(bundle/'policy.py')!=file_hash(Path(__file__).with_name('policy.py')):raise ValueError('existing submission source changed')
    report=verify_export(bundle,payload['policy_state'],study.DIRECTORY/'runs'/run/'policy_inputs.pt')
    atomic_json(study.DIRECTORY/'submission_verification.json',report)
    summary=dict(complete=True,registration_sha256=file_hash(study.REGISTRATION),training_steps=2*len(r['config']['seeds'])*r['config']['training']['steps'],dev_games=sum(len(j.get('replicates',[]))*2 for j in study.tasks(r)),
        analyses=analyses,selected=dict(run=run,checkpoint_sha256=h,bundle=str(bundle.relative_to(study.ROOT))),
        selection='final endpoint arm mean, ties C1; highest seed-run dev W/N within arm, ties earliest seed',
        metric='provided_run_match_reward_total_W_over_N',held_out_test=False,confirmation_performed=False,
        distinct_initial_observations={k:len(v) for k,v in initial_states.items()},
        uncertainty_limit='descriptive local-run intervals; actual map sampling/independence and server scoring unverified',
        selection_adjusted_evidence=False,official_server_certified=False,external_submission_sent=False,inputs=audit)
    atomic_json(study.DIRECTORY/'summary.json',summary)


def cleanup_unity(path):
    records=path/'unity_pids.json'
    if not records.exists():return
    for row in study.read(records):
        # Kill only the still-owned original player. PID start time prevents reuse.
        result=subprocess.run(['ps','-p',str(row['pid']),'-o','command='],capture_output=True,text=True)
        text=result.stdout.strip()
        if str(project_path('artifacts/builds/BlackOut.app', root=study.ROOT)) not in text:continue
        try:
            birth=subprocess.run(['ps','-p',str(row['pid']),'-o','lstart='],capture_output=True,text=True).stdout.strip()
            if birth and birth==row['birth']:os.kill(row['pid'],signal.SIGTERM)
        except (ValueError,ProcessLookupError):pass


def supervise(fd):
    with os.fdopen(fd,'a'):
        r=check();cfg=r['config'];atomic_json(study.DIRECTORY/'supervisor.json',dict(pid=os.getpid()))
        stop_path=study.DIRECTORY/'stop.request';active={};failure=None;stop_at=None
        awake=subprocess.Popen(['caffeinate','-i','-w',str(os.getpid())]) if shutil.which('caffeinate') else None
        signal.signal(signal.SIGTERM,lambda *_:stop_path.touch());signal.signal(signal.SIGINT,lambda *_:stop_path.touch())
        all_jobs=study.tasks(r);done={j['id'] for j in all_jobs if study.completed(j)};pending=[j for j in all_jobs if j['id'] not in done]
        try:
            while pending or active:
                if shutil.disk_usage(study.ROOT).free<cfg['disk_reserve_bytes']:failure='disk reserve reached';stop_path.touch()
                if stop_path.exists() and stop_at is None:stop_at=time.monotonic()
                while len(active)<cfg['workers'] and not stop_path.exists():
                    job=next((j for j in pending if all(k in done for k in j['depends'])),None)
                    if job is None:break
                    slot=next(i for i in range(cfg['workers']) if i not in active);pending.remove(job);path=study.attempt(job)
                    with (path/'console.log').open('ab',buffering=0) as log:
                        process=subprocess.Popen([sys.executable,'-u',str(project_path('code/v8/scripts/v8_worker.py', root=study.ROOT)),'--job',job['id'],'--attempt',str(path),'--slot',str(slot)],
                            cwd=study.ROOT,stdout=log,stderr=log,stdin=subprocess.DEVNULL,start_new_session=True,pass_fds=(fd,),env=dict(os.environ,PYTHONHASHSEED='0'))
                    active[slot]=(process,job,path)
                for slot,(process,job,path) in list(active.items()):
                    code=process.poll();p=path/'progress.json'
                    recent=p.stat().st_mtime if p.exists() else (path/'console.log').stat().st_mtime
                    if code is None and (time.time()-recent>cfg['stall_seconds'] or stop_at is not None and time.monotonic()-stop_at>cfg['stop_grace_seconds']):
                        cleanup_unity(path);process.kill();code=process.wait();failure='worker watchdog: '+job['id']
                    if code is None:continue
                    cleanup_unity(path);del active[slot]
                    if code or not study.completed(job) and not stop_path.exists():failure=f'{job["id"]} failed; see {path}';stop_path.touch()
                    elif study.completed(job):done.add(job['id'])
                atomic_json(study.DIRECTORY/'status.json',dict(state='stopping' if stop_path.exists() else 'running',completed=len(done),pending=len(pending),active=[j['id'] for _,j,_ in active.values()],error=failure))
                if stop_path.exists() and not active:break
                if not active and pending and not any(all(k in done for k in j['depends']) for j in pending):raise RuntimeError('unresolved dependencies')
                if active:time.sleep(2)
            if failure:raise RuntimeError(failure)
            if pending or stop_path.exists():atomic_json(study.DIRECTORY/'status.json',dict(state='stopped',completed=len(done)));return
            aggregate(r);atomic_json(study.DIRECTORY/'status.json',dict(state='complete',completed=len(done)))
        except BaseException as exc:
            stop_path.touch();atomic_json(study.DIRECTORY/'status.json',dict(state='failed',error=str(exc)));raise
        finally:
            for process,job,path in active.values():
                process.terminate()
                try:process.wait(timeout=cfg['stop_grace_seconds'])
                except subprocess.TimeoutExpired:cleanup_unity(path);process.kill();process.wait()
                cleanup_unity(path)
            if awake is not None:awake.terminate();awake.wait()
            (study.DIRECTORY/'supervisor.json').unlink(missing_ok=True)


def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group()
    for flag in ('check','status','stop','prepare','aggregate-only'):g.add_argument('--'+flag,action='store_true')
    p.add_argument('--supervisor-fd',type=int,help=argparse.SUPPRESS);args=p.parse_args();os.chdir(study.ROOT)
    if args.status:print(__import__('json').dumps(status(),indent=2));return
    if args.prepare:study.prepare();print(study.REGISTRATION);return
    if args.check:check();return
    if args.stop:study.DIRECTORY.mkdir(parents=True,exist_ok=True);(study.DIRECTORY/'stop.request').touch();return
    if args.supervisor_fd is not None:supervise(args.supervisor_fd);return
    study.DIRECTORY.mkdir(parents=True,exist_ok=True)
    with (study.DIRECTORY/'experiment.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:print('v8 competition experiment already running');return
        r=check()
        if args.aggregate_only:aggregate(r);return
        (study.DIRECTORY/'stop.request').unlink(missing_ok=True)
        with (study.DIRECTORY/'console.log').open('ab',buffering=0) as log:
            process=subprocess.Popen([sys.executable,'-u',str(project_path('code/v8/scripts/v8_experiments.py', root=study.ROOT)),'--supervisor-fd',str(lock.fileno())],
                cwd=study.ROOT,stdout=log,stderr=log,stdin=subprocess.DEVNULL,start_new_session=True,pass_fds=(lock.fileno(),))
        for _ in range(150):
            if process.poll() is not None:raise RuntimeError('startup failed; inspect console.log')
            pid_path=study.DIRECTORY/'supervisor.json'
            if pid_path.exists() and study.read(pid_path)['pid']==process.pid:break
            time.sleep(.1)
        else:raise RuntimeError('background startup pending; inspect --status before retry')
        print(f'v8 background PID {process.pid}; logs: {study.DIRECTORY}')


if __name__=='__main__':main()
