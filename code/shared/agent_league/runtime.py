"""Supervision uses the existing audited worker process-group scheduler."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from .common import ROOT,atomic,read,digest


def resource_sample():
    # Count supervisor + descendants, including Unity, rather than only Python RSS.
    p=subprocess.run(['ps','-axo','pid=,ppid=,rss='],capture_output=True,text=True,check=True)
    rows=[tuple(map(int,l.split())) for l in p.stdout.splitlines() if len(l.split())==3]
    owned={os.getpid()}
    for _ in range(20):
        expanded=owned|{pid for pid,parent,rss in rows if parent in owned}
        if expanded==owned:break
        owned=expanded
    from blackout_v10.telemetry import memory_pressure
    return dict(rss_gib=sum(rss for pid,parent,rss in rows if pid in owned)/1024**2,
                free_gib=shutil.disk_usage(ROOT).free/1024**3,pressure=memory_pressure())


def scheduler_config(cfg):
    return dict(workers=cfg['workers'],episode_wall_seconds=cfg['episode_seconds'],stall_seconds=cfg['stall_seconds'],stop_grace_seconds=10)


def execute(tasks,directory,cfg,stop=lambda:False,progress=lambda **k:None):
    from blackout_v10.runtime import run_tasks
    return run_tasks(tasks,directory,scheduler_config(cfg),stop,progress,worker_module='agent_league.worker')


def preflight(roster,directory,stop=lambda:False):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);rows=[]
    for i,spec in enumerate(roster['models']):
        if stop():raise InterruptedError('preflight stopped')
        folder=directory/spec['id'];folder.mkdir(exist_ok=True)
        atomic(folder/'spec.json',spec)
        with (folder/'console.log').open('w') as stream:
            cmd=[sys.executable,str(ROOT/'code/run.py'),'-m','agent_league.check',str(folder/'spec.json'),str(folder/'result.json')]
            p=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            started=time.monotonic()
            try:
                while p.poll() is None:
                    if stop() or time.monotonic()-started>240:
                        p.terminate();p.wait(timeout=10);raise RuntimeError('adapter check stopped/timed out: '+spec['id'])
                    time.sleep(.1)
            finally:
                if p.poll() is None:p.kill();p.wait()
        row=read(folder/'result.json') if (folder/'result.json').exists() else dict(passed=False,error='checker failed')
        rows.append(dict(row, id=spec['id']));atomic(directory/'result.json',dict(passed=all(x['passed'] for x in rows),checked=len(rows),expected=len(roster['models']),models=rows))
        print(f"검사 {i+1}/{len(roster['models'])}: {spec['id']} {'PASS' if row['passed'] else 'FAIL'}",flush=True)
    if not all(x['passed'] for x in rows):raise RuntimeError('model preflight failed; inspect '+str(directory/'result.json'))
    return rows


def recover_completed(directory, manifest):
    """Promote natural-terminal results after a supervisor crash before cell commit."""
    expected={t['id']:t for t in manifest['tasks']}
    models={m['id']:m for m in manifest['roster']['models']}
    for result_path in sorted((directory/'attempts').glob('*/*/result.json')):
        folder=result_path.parent;task_path=folder/'task.json'
        if not task_path.exists() or (folder/'error.json').exists():continue
        task=read(task_path);base=expected.get(task.get('id'))
        if base is None or (directory/'cells'/(base['id']+'.json')).exists():continue
        if any(task.get(k)!=v for k,v in base.items()):continue
        if task.get('config')!=manifest['config']:continue
        if task.get('models')!=[models[base['a']],models[base['b']]]:continue
        if manifest.get('sources') and task.get('expected_sources')!=digest(manifest['sources']):continue
        envelope=read(result_path);result=envelope.get('result',{})
        if envelope.get('task_sha256')!=digest(task):continue
        if not result.get('valid') or not result.get('natural_terminal') or result.get('probe'):continue
        index=folder/'trace/index.json'
        if not index.exists() or not read(index).get('complete'):continue
        if (folder/'attempt.json').exists():
            row=read(folder/'attempt.json')
            if not row.get('valid'):continue
        else:row=dict(id=base['id'],task=task,result=result,directory=str(folder),valid=True,error=None)
        row['task']={k:v for k,v in task.items() if k!='models'}
        row.update(manifest_sha256=digest(manifest),recovered_after_supervisor_exit=True)
        atomic(directory/'cells'/(base['id']+'.json'),row)


def run_league(directory,manifest,stop):
    from .report import generate
    from blackout_v10.runtime import clean_orphans
    directory=Path(directory);clean_orphans(directory/'attempts');recover_completed(directory,manifest)
    cfg=manifest['config'];model_by_id={m['id']:m for m in manifest['roster']['models']}
    start=time.monotonic();last_status=0.;last_report=0.;peak=0.;last_resources=None;completed=[]
    for p in (directory/'cells').glob('*.json'):
        row=read(p)
        if row.get('manifest_sha256')!=digest(manifest):raise ValueError('completed cell belongs to changed manifest')
        completed.append(row)
    complete_ids={r['task']['id'] for r in completed};remaining=[t for t in manifest['tasks'] if t['id'] not in complete_ids]
    def progress(**data):
        nonlocal last_status,peak,last_resources
        if time.monotonic()-last_status<3:return
        last_status=time.monotonic();last_resources=resource_sample();peak=max(peak,last_resources['rss_gib'])
        if last_resources['rss_gib']>cfg['memory_gib'] or last_resources['free_gib']<cfg['reserve_gib'] or last_resources['pressure'].get('critical'):
            raise RuntimeError('resource reserve reached; completed cells preserved, resume after resolving pressure')
        rate=sum(r['result']['wall_seconds'] for r in completed)/len(completed) if completed else None
        atomic(directory/'status.json',dict(state='running',updated_at=time.time(),completed=len(completed),expected=len(manifest['tasks']),
            elapsed_this_session=time.monotonic()-start,eta_seconds=(len(manifest['tasks'])-len(completed))*rate/cfg['workers'] if rate else None,
            peak_rss_gib=peak,resources=last_resources,worker_progress=data))
    # Small bounded waves permit durable commits/reports and resource rechecks.
    for offset in range(0,len(remaining),cfg['workers']):
        if stop():break
        batch=remaining[offset:offset+cfg['workers']]
        for attempt in range(cfg['retries']+1):
            if not batch or stop():break
            tasks=[dict(t,config=cfg,expected_sources=digest(manifest['sources']) if 'sources' in manifest else None,models=[model_by_id[t['a']],model_by_id[t['b']]]) for t in batch]
            path=directory/'attempts'/str(time.time_ns());error=None
            try:out=execute(tasks,path,cfg,stop,progress)
            except BaseException as exc:
                error=exc
                out=[read(p) for p in path.glob('*/attempt.json')]
            retry=[]
            for row in out:
                t=next(x for x in batch if x['id']==row['id'])
                if row['valid'] and row['result']['natural_terminal'] and not row['result'].get('probe'):
                    row['task']={k:v for k,v in row['task'].items() if k!='models'}
                    row['manifest_sha256']=digest(manifest);atomic(directory/'cells'/(t['id']+'.json'),row)
                    completed.append(row);(directory/'failures'/(t['id']+'.json')).unlink(missing_ok=True)
                else:
                    atomic(directory/'failures'/(t['id']+'.json'),dict(task=t,attempt=row,attempt_number=attempt+1));retry.append(t)
            batch=retry
            if error is not None:raise error
        progress()
        if time.monotonic()-last_report>=300:
            generate(directory,detailed=False);last_report=time.monotonic()
    state='complete' if len(completed)==len(manifest['tasks']) else 'stopped' if stop() else 'incomplete'
    atomic(directory/'status.json',dict(state=state,updated_at=time.time(),completed=len(completed),expected=len(manifest['tasks']),
        elapsed_this_session=time.monotonic()-start,peak_rss_gib=peak,resources=last_resources))
    generate(directory,detailed=True)
