"""One original-API experiment worker; never changes the environment class."""
import argparse
import os
import signal
import subprocess
import time
import torch
from . import study
from .training import create_policy,ActorCritic,Collector,train
from .exporting import export_policy,verify_export
from .evaluation import evaluate
from blackout_rl.v8.checkpoints import CheckpointStore,atomic_json
from blackout_rl.v8.contracts import file_hash,digest
from blackout_rl.v8.provided_environment import make_original_env


def execute(job,directory,slot,r):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    registration_sha=file_hash(study.REGISTRATION)
    cfg=study.run_config(r,'c1_s11' if job['run']=='planner' else job['run'])
    stop=lambda:(study.DIRECTORY/'stop.request').exists()
    signal.signal(signal.SIGTERM,lambda *_:(study.DIRECTORY/'stop.request').touch())
    signal.signal(signal.SIGINT,lambda *_:(study.DIRECTORY/'stop.request').touch())
    last=0.
    def progress(step):
        nonlocal last
        if time.monotonic()-last>5:
            atomic_json(directory/'progress.json',dict(step=step,time=time.time()));last=time.monotonic()
    envs=[]
    def factory():
        env=make_original_env(time_scale=r['config']['time_scale'],worker_id=slot,base_port=27000)
        # Read-only PID bookkeeping; no method replacement or private writes.
        process=env._unity_env._process
        if process is not None:
            birth=subprocess.check_output(['ps','-p',str(process.pid),'-o','lstart='],text=True).strip()
            envs.append(dict(pid=process.pid,birth=birth))
            atomic_json(directory/'unity_pids.json',envs)
        return env
    if job['kind']=='train':
        run=study.DIRECTORY/'runs'/job['run'];torch.manual_seed(cfg['seed'])
        model=ActorCritic(create_policy(cfg['arm'],cfg['seed'],cfg['opponent']['path']))
        collector=Collector(model,cfg,cfg['seed'],factory,run,progress=progress)
        h=train(model,collector,cfg,run,registration_sha,stop)
        if study.read(run/'status.json')['state']!='complete':return
        artifact=run/'checkpoints'/(h+'.pt')
    else:
        bundle=None;h=None
        if job['run']!='planner':
            run=study.DIRECTORY/'runs'/job['run']
            h=study.read(run/'checkpoint_index'/f"{job['step']}.json")['sha256']
            payload,_=CheckpointStore(run/'checkpoints').load(h)
            if payload['registration_sha256']!=registration_sha or payload['global_step']!=job['step'] or payload['config_sha256']!=digest(cfg):
                raise ValueError('foreign evaluation checkpoint/config/endpoint')
            bundle=export_policy(payload,directory/'submission')
            atomic_json(directory/'loader_verification.json',verify_export(bundle,payload['policy_state'],run/'policy_inputs.pt'))
        lock=dict(job=job,registration_sha256=registration_sha,checkpoint_sha256=h,config=cfg,sources=r['sources'],
                  build='builds/BlackOut.app',metric='provided_run_match_reward_total_W_over_N')
        evaluate(bundle,cfg,job['replicates'],factory,directory/'evaluation',planner=job['run']=='planner',progress=progress,stop=stop,lock=lock)
        artifact=directory/'evaluation/result.json'
    atomic_json(directory.parent/'done.json',dict(registration_sha256=registration_sha,artifact=str(artifact.relative_to(study.ROOT)),artifact_sha256=file_hash(artifact)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--attempt',required=True,type=__import__('pathlib').Path);p.add_argument('--slot',type=int,required=True);args=p.parse_args()
    os.chdir(study.ROOT);r=study.validate()
    if not 0<=args.slot<r['config']['workers']:raise ValueError('worker slot out of range')
    job=next(j for j in study.tasks(r) if j['id']==args.job)
    try:execute(job,args.attempt,args.slot,r)
    except BaseException as exc:
        atomic_json(args.attempt/'failure.json',dict(error=type(exc).__name__,detail=str(exc)))
        if isinstance(exc,InterruptedError) and (study.DIRECTORY/'stop.request').exists():return
        raise


if __name__=='__main__':main()
