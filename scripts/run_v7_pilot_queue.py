#!/usr/bin/env python3
"""Sequential pilot manager: per-run locks and failure records, no shared checkpoints."""
import argparse,fcntl,json,os,subprocess,sys,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from blackout_rl.v7_registry import load_config,check_config,path
from blackout_rl.connectome.graph_artifact import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--queue',required=True);p.add_argument('--check',action='store_true');a=p.parse_args()
    queue=json.loads(path(a.queue).read_text());directory=path(queue['log_dir'])
    for job in queue['jobs']:check_config(load_config(job['config']))
    if a.check:print(json.dumps(dict(status='ready',jobs=len(queue['jobs']))));return
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        (directory/'queue.pid').write_text(str(os.getpid()))
        stopped=False
        child=None
        def stop(signum,frame):
            nonlocal stopped
            stopped=True
            if child is not None and child.poll() is None: child.send_signal(signal.SIGTERM)
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        try:
            for job in queue['jobs']:
                if stopped: break
                cfg=load_config(job['config']);run=path(f'logs/v7/{cfg["experiment_id"]}/{job["seed"]}')
                if (run/'status.json').exists() and json.loads((run/'status.json').read_text()).get('status')=='budget_complete':continue
                script='train_v7_1.py' if cfg['mode']=='residual_graph' else 'run_v7_2.py'
                command=[sys.executable,'-u',str(ROOT/'scripts'/script),'--config',str(path(job['config'])),'--seed',str(job['seed'])]
                if (run/'checkpoints/latest.pt').exists():command.append('--resume')
                run.mkdir(parents=True,exist_ok=True)
                with (run/'console.log').open('ab',buffering=0) as output:
                    child=subprocess.Popen(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                    atomic_json(directory/'status.json',dict(status='running',job=job,child_pid=child.pid,run=str(run)))
                    code=child.wait()
                if stopped:
                    atomic_json(directory/'status.json',dict(status='stopped',job=job));return
                if code:
                    atomic_json(directory/'status.json',dict(status='failed',job=job,exit_code=code,run=str(run)))
                    raise RuntimeError(f'pilot failed: {run}; inspect console.log')
            atomic_json(directory/'status.json',dict(status='complete'))
        finally:(directory/'queue.pid').unlink(missing_ok=True)

if __name__=='__main__':main()
