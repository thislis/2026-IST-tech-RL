#!/usr/bin/env python3
"""Detached v7 launcher, with lifetime run lock and explicit startup status."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from blackout_rl.v7_registry import load_config,check_config,path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--seed',type=int,default=11)
    p.add_argument('--run-dir');p.add_argument('--resume',action='store_true');p.add_argument('--check',action='store_true')
    a=p.parse_args();cfg=load_config(a.config);check_config(cfg)
    run=path(a.run_dir or f'logs/v7/{cfg["experiment_id"]}/{a.seed}')
    script='train_v7_1.py' if cfg['mode']=='residual_graph' else 'run_v7_2.py'
    command=[str(ROOT/'.venv/bin/python'),'-u',str(ROOT/'scripts'/script),'--config',str(path(a.config)), '--seed',str(a.seed),'--run-dir',str(run)]
    if a.resume:command.append('--resume')
    if a.check:
        return subprocess.call(command+['--check'],cwd=ROOT)
    run.mkdir(parents=True,exist_ok=True)
    with (run/'launch.lock').open('a') as launcher:
        fcntl.flock(launcher,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (run/'run.lock').open('a') as lock:
            try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise RuntimeError('this run is already active')
        if not a.resume and (run/'resolved_config.json').exists():raise FileExistsError('existing run: use --resume or another --run-dir')
        with (run/'console.log').open('ab',buffering=0) as output:
            child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,
                start_new_session=True,env={**os.environ,'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':str(cfg['runtime']['torch_threads'])})
        for _ in range(300):
            if child.poll() is not None:raise RuntimeError(f'child exited during startup; inspect {run}/console.log')
            pidfile=run/'training.pid'
            if pidfile.is_file() and pidfile.read_text().strip()==str(child.pid):
                print(json.dumps(dict(status='spawned',pid=child.pid,log=str(run/'console.log'),note='status.json confirms Unity readiness')));return 0
            time.sleep(.1)
        print(json.dumps(dict(status='starting',pid=child.pid,log=str(run/'console.log'))));return 0

if __name__=='__main__':raise SystemExit(main())
