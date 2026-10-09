#!/usr/bin/env python3
"""Start one sequential pilot queue detached from the terminal."""

# Locate shared modules from the physical versioned layout.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents
                    if (p / "code/shared/project_paths.py").is_file())
_layout_sys.path.insert(0, str(_layout_root / "code/shared"))
from project_paths import activate as _activate_layout
_activate_layout()
from project_paths import project_root, project_path

import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
ROOT=project_root();sys.path.insert(0,str(ROOT))
from blackout_rl.v7_registry import path

def main():
    p=argparse.ArgumentParser();p.add_argument('--queue',required=True);p.add_argument('--check',action='store_true');a=p.parse_args()
    command=[str(project_path('.venv/bin/python', root=ROOT)),'-u',str(project_path('code/v7/scripts/run_v7_pilot_queue.py', root=ROOT)),'--queue',str(path(a.queue))]
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
