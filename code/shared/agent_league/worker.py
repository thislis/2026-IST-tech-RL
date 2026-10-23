import argparse
from pathlib import Path
import os
import signal
import time
import traceback
from .common import read,atomic,digest,sources

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--task',required=True);args=parser.parse_args()
    path=Path(args.task);task=read(path);stopped=False
    def request_stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    def heartbeat(**data):atomic(path.parent/'heartbeat.json',dict(time=time.time(),pid=os.getpid(),**data))
    try:
        heartbeat(phase='starting',steps=0)
        if task.get('expected_sources') and digest(sources())!=task['expected_sources']:raise ValueError('source changed before match')
        import torch
        torch.set_num_threads(task['config']['threads']);torch.set_num_interop_threads(1)
        from .match import play
        from blackout_v10.contracts import make_env
        result=play(task,path.parent,lambda:make_env(task['config']['time_scale']),heartbeat=heartbeat,stop=lambda:stopped)
        if task.get('expected_sources') and digest(sources())!=task['expected_sources']:raise ValueError('source changed during match')
        result['trace_bytes']=sum(p.stat().st_size for p in (path.parent/'trace').rglob('*') if p.is_file())
        atomic(path.parent/'result.json',dict(task_sha256=digest(task),result=result))
    except BaseException as exc:
        atomic(path.parent/'error.json',dict(error=str(exc),type=type(exc).__name__,traceback=traceback.format_exc()))
        raise

if __name__=='__main__':main()
