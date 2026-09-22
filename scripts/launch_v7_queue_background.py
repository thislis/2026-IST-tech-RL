#!/usr/bin/env python3
"""Start one sequential pilot queue detached from the terminal."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from blackout_rl.v7_registry import path

def main():
    p=argparse.ArgumentParser();p.add_argument('--queue',required=True);p.add_argument('--check',action='store_true');a=p.parse_args()
    command=[str(ROOT/'.venv/bin/python'),'-u',str(ROOT/'scripts/run_v7_pilot_queue.py'),'--queue',str(path(a.queue))]
    subprocess.run(command+['--check'],check=True,cwd=ROOT)
    if a.check:return
    cfg=json.loads(path(a.queue).read_text());directory=path(cfg['log_dir']);directory.mkdir(parents=True,exist_ok=True)
    with (directory/'launch.lock').open('a') as launch:
        fcntl.flock(launch,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (directory/'queue.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise RuntimeError('queue already running')
        with (directory/'console.log').open('ab',buffering=0) as output:
            child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(status='spawned',pid=child.pid,status_file=str(directory/'status.json'))))

if __name__=='__main__':main()
