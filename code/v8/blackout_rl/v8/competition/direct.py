"""Explicit single-job commands using the same registration/locks as the queue."""
import argparse
import fcntl
from . import study
from .worker import execute


def main(kind):
    p=argparse.ArgumentParser(description='Original-game v8 competition job; --check never starts Unity.')
    p.add_argument('--run',choices=[f'{a}_s{s}' for a in ('c1','flat') for s in (11,22,33)])
    p.add_argument('--job');p.add_argument('--check',action='store_true');args=p.parse_args()
    r=study.validate()
    if args.check:print('original-game registration verified; Unity not started');return
    job_id=args.run+'-train' if kind=='train' and args.run else args.job
    job=next((j for j in study.tasks(r) if j['id']==job_id and j['kind']==kind),None)
    if job is None:p.error('specify a registered --run (training) or --job (evaluation)')
    study.DIRECTORY.mkdir(parents=True,exist_ok=True)
    with (study.DIRECTORY/'experiment.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if study.completed(job):print('registered job already complete');return
        jobs={j['id']:j for j in study.tasks(r)}
        if not all(study.completed(jobs[name]) for name in job['depends']):raise RuntimeError('training dependency incomplete')
        (study.DIRECTORY/'stop.request').unlink(missing_ok=True)
        execute(job,study.attempt(job),0,r)
