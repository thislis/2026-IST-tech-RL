#!/usr/bin/env python3
"""Parallel continuation of the stopped main study; no change to scientific budgets.

Original configs, sources and completed checkpoints remain registered. An explicit
execution-runtime lineage records the compatible transport/MPS overlay at save.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.frozen_pilot_runtime import sha256
from scripts.connectome_main import validate_registration as validate_original, QUEUES

DIRECTORY = ROOT/'logs/v7/main_study/accelerated'
REGISTRATION = ROOT/'reports/v7/acceleration_registration.json'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    temporary.replace(path)


def validate_runtime():
    validate_original()
    registration = read(REGISTRATION)
    for package, expected in registration['base_packages'].items():
        if importlib.metadata.version(package) != expected:
            raise ValueError(f'base dependency changed: {package}')
    for name, digest in registration['files'].items():
        if sha256(ROOT/name) != digest:
            raise ValueError(f'acceleration runtime changed: {name}')
    return registration


def jobs():
    from blackout_rl.v7_registry import load_config
    result = []
    for suite, queue in QUEUES.items():
        for row in read(queue)['jobs']:
            cfg = load_config(ROOT/row['config'])
            result.append(dict(index=len(result), suite=suite, config=row['config'], seed=row['seed'],
                               run=str(ROOT/'logs/v7'/cfg['experiment_id']/str(row['seed'])),
                               experiment_id=cfg['experiment_id']))
    return result


def inspect_job(job, *, validate=False, cache=None):
    job = {key:job[key] for key in ('index','suite','config','seed','run','experiment_id')}
    import torch
    from blackout_rl.v7_registry import load_config, check_config, make_model, source_fingerprint
    from blackout_rl.connectome.graph_artifact import digest
    from scripts.pilot_evaluation import completed_evaluation
    cfg = load_config(ROOT/job['config'])
    run = Path(job['run'])
    status = read(run/'status.json') if (run/'status.json').exists() else {}
    complete = status.get('status') == 'budget_complete'
    checkpoint = run/'checkpoints/latest.pt'
    metadata = run/'acceleration_runtime.json'
    if metadata.exists() and read(metadata)['runtime_sha256'] != sha256(REGISTRATION):
        raise ValueError(f'different execution runtime already registered: {run}')
    if (run/'resolved_config.json').exists() and read(run/'resolved_config.json') != cfg:
        raise ValueError(f'configuration mismatch: {run}')
    if (complete or (run/'resolved_config.json').exists()) and not checkpoint.exists():
        raise ValueError(f'run has no resumable checkpoint: {run}')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True) if checkpoint.exists() else None
    step = payload['global_step'] if payload is not None else 0
    budget = cfg['training']['max_environment_steps']
    if payload is not None:
        if (payload['config_sha256'] != digest(cfg) or payload['sources'] != source_fingerprint()
                or payload['seed'] != job['seed'] or payload['schema'] != 'blackout.v7.checkpoint.v1'
                or not 0 <= step <= budget):
            raise ValueError(f'checkpoint provenance mismatch: {run}')
        for event in payload.get('lineage', []):
            if event.get('event') == 'execution_acceleration' and event['runtime_sha256'] != sha256(REGISTRATION):
                raise ValueError(f'checkpoint used another acceleration runtime: {run}')
    if complete and (step != budget or status['environment_steps'] != budget):
        raise ValueError(f'completion/checkpoint mismatch: {run}')
    if not complete and step == budget:
        raise ValueError(f'budget checkpoint lacks completion status: {run}; inspect before resuming')
    if validate:
        if cache is None:cache = {}
        if job['config'] not in cache:
            graph, splits = check_config(cfg)
            with torch.random.fork_rng():model = make_model(cfg,graph)
            cache[job['config']] = model
        if payload is not None:
            cache[job['config']].load_state_dict(payload['model'],strict=True)
    maps = read(ROOT/cfg['evaluation']['split_manifest'])['dev']
    evaluation_job = dict(**job, maps=maps, checkpoint_sha256=sha256(checkpoint) if payload is not None else None)
    existing = completed_evaluation(evaluation_job) if complete else None
    return dict(**evaluation_job, step=step, budget=budget, complete=complete,
                phase='done' if existing else 'evaluate' if complete else 'train',
                result=str(existing[0]) if existing else None)


def preflight():
    registration = validate_runtime()
    import torch
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable. Run the prepared shell command in the macOS terminal; do not silently change the registered backend.')
    cache = {}
    result = [inspect_job(job,validate=True,cache=cache) for job in jobs()]
    print(json.dumps(dict(status='ready',workers=registration['workers'],mps_workers=registration['mps_workers'],
                          completed_training=sum(j['complete'] for j in result),
                          completed_evaluations=sum(j['phase']=='done' for j in result),
                          remaining_steps=sum(j['budget']-j['step'] for j in result),
                          runtime_sha256=sha256(REGISTRATION)),ensure_ascii=False),flush=True)
    return result


def execute_job(index, phase):
    registration = validate_runtime()
    from scripts.connectome_fast_runtime import install
    install()
    job = jobs()[index]
    from blackout_rl.v7_registry import load_config, check_config, make_model
    from blackout_rl.connectome.graph_artifact import atomic_json
    import blackout_rl.v7_training as training
    cfg = load_config(ROOT/job['config'])
    run = Path(job['run'])
    state = inspect_job(job)
    if state['phase'] == 'done' or (phase=='train' and state['complete']):return
    if phase == 'evaluate' and not state['complete']:
        raise ValueError('cannot evaluate incomplete training')
    if job['suite']=='v7-2':
        from scripts.connectome_metal import install as install_metal
        install_metal()
    # Preserve time_scale, simulation dt, graph, action repeat, observation and PPO semantics.
    original_env = training.ContractBlackOutEnv
    engine = registration.get('engine',{})
    class FastEnv(original_env):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            if engine:self._engine_channel.set_configuration_parameters(**engine)
    training.ContractBlackOutEnv = FastEnv
    import blackout_rl.env as env_module
    env_module.ContractBlackOutEnv = FastEnv
    runtime = dict(event='execution_acceleration',runtime_sha256=sha256(REGISTRATION),
                   created_at_utc=datetime.now(timezone.utc).isoformat(),
                   parent_checkpoint_sha256=state['checkpoint_sha256'],resumed_environment_steps=state['step'],
                   protobuf='4.25.9-upb',rpc_handoff='in_process_queue',
                   brain_backend='MPS_CSR' if job['suite']=='v7-2' else 'CPU',
                   engine=engine,scientific_config_unchanged=True)
    run.mkdir(parents=True,exist_ok=True)
    # This phase lock also excludes a simultaneous accelerated evaluator of the same run.
    with (run/'acceleration.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not (run/'acceleration_runtime.json').exists():
            atomic_json(run/'acceleration_runtime.json',runtime)
        if phase=='train':
            original_save = training.save_checkpoint
            def save(filename,model,optimizer,collector,*args):
                if not any(e.get('event')=='execution_acceleration' and e.get('runtime_sha256')==runtime['runtime_sha256'] for e in collector.lineage):
                    collector.lineage.append(runtime)
                original_save(filename,model,optimizer,collector,*args)
            training.save_checkpoint = save
            graph,splits = check_config(cfg)
            training.train(cfg,graph,splits,seed=job['seed'],run=run,resume=state['checkpoint_sha256'] is not None)
        else:
            # The original evaluator keeps paired map ordering and immutable checkpoint copies.
            import runpy
            with (run/'run.lock').open('a') as train_lock:
                fcntl.flock(train_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                sys.argv=['evaluate_v7.py','--run',str(run),'--split','dev','--opponent','target']
                runpy.run_path(str(ROOT/'scripts/evaluate_v7.py'),run_name='__main__')


def choose(pending, active, capacity, mps_capacity):
    """Independent run scheduling; never mixes rollouts or RNG streams."""
    if len(active)>=capacity:return None
    direct = sum(j['suite']=='v7-2' for j in active.values())
    for job in sorted(pending,key=lambda j:(j['phase']!='train',j['suite']!='v7-2',j['index'])):
        if job['suite']=='v7-2' and direct>=mps_capacity:continue
        return job
    return None


def supervisor(lock_fds):
    registration = validate_runtime()
    active = {}
    children = {}
    stopped = False
    failed = None
    caffeinate = None
    def stop(signum=None,frame=None):
        nonlocal stopped
        stopped=True
        for process in children.values():
            if process.poll() is None:process.send_signal(signal.SIGINT)
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    with ExitStack() as stack:
        for fd in lock_fds:stack.enter_context(os.fdopen(fd,'a'))
        (DIRECTORY/'experiment.pid').write_text(str(os.getpid()))
        try:
            caffeinate=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())],stdin=subprocess.DEVNULL)
            pending=preflight()
            done=[j for j in pending if j['phase']=='done']
            pending=[j for j in pending if j['phase']!='done']
            while pending or active:
                if (DIRECTORY/'stop.request').exists() and not stopped:stop()
                if not stopped:
                    while True:
                        job=choose(pending,active,registration['workers'],registration['mps_workers'])
                        if job is None:break
                        pending.remove(job)
                        log=DIRECTORY/f'{job["index"]:02d}_{job["phase"]}.log'
                        with log.open('ab',buffering=0) as output:
                            environment=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                                             OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',PYTORCH_MPS_FAST_MATH='0',
                                             PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='upb')
                            process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),
                                '--job',str(job['index']),'--phase',job['phase']],cwd=ROOT,
                                stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,env=environment)
                        active[process.pid]=dict(job,log=str(log));children[process.pid]=process
                        if stopped:process.send_signal(signal.SIGINT)
                for pid in list(active):
                    code=children[pid].poll()
                    if code is None:continue
                    job=active.pop(pid);children.pop(pid)
                    if stopped:continue
                    if code:
                        failed=dict(job=job,exit_code=code);stop();break
                    state=inspect_job(job)
                    if (job['phase']=='train' and not state['complete']) or (job['phase']=='evaluate' and state['phase']!='done'):
                        failed=dict(job=job,error='child exited before registered phase completed');stop();break
                    (done if state['phase']=='done' else pending).append(state)
                write(DIRECTORY/'status.json',dict(status='stopping' if stopped else 'running',pid=os.getpid(),
                      workers=registration['workers'],active=list(active.values()),pending=len(pending),evaluated=len(done),failed=failed))
                if stopped and not active:break
                time.sleep(1)
            summaries=[]
            for job in done:
                result=read(job['result'])
                summaries.append(dict(experiment_id=job['experiment_id'],seed=job['seed'],result=job['result'],
                    win_rate=result['win_rate'],draw_rate=result['draw_rate'],paired_map_bootstrap_ci=result['paired_map_bootstrap_ci']))
            complete=not stopped and len(done)==23
            write(DIRECTORY/'summary.json',dict(complete=complete,runs=summaries,runtime_sha256=sha256(REGISTRATION)))
            write(DIRECTORY/'status.json',dict(status='failed' if failed else 'stopped' if stopped else 'complete',
                  pid=os.getpid(),evaluated=len(done),failed=failed,active=[]))
        except BaseException as exc:
            stop()
            for process in children.values():process.wait()
            write(DIRECTORY/'status.json',dict(status='failed',error=str(exc),active=[]))
            raise
        finally:
            if caffeinate is not None:
                caffeinate.terminate();caffeinate.wait()
            (DIRECTORY/'experiment.pid').unlink(missing_ok=True)
            (DIRECTORY/'stop.request').unlink(missing_ok=True)


def status():
    active=False
    file=DIRECTORY/'experiment.lock'
    if file.exists():
        with file.open() as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:active=True
    details=read(DIRECTORY/'status.json') if (DIRECTORY/'status.json').exists() else {'status':'not_started'}
    for job in details.get('active',[]):
        path=Path(job['run'])/'status.json'
        if path.exists():job['training_status']=read(path)
    print(json.dumps(dict(active=active,details=details),ensure_ascii=False,indent=2))
    return active


def main():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group()
    group.add_argument('--check',action='store_true')
    group.add_argument('--status',action='store_true')
    group.add_argument('--stop',action='store_true')
    p.add_argument('--job',type=int,help=argparse.SUPPRESS)
    p.add_argument('--phase',choices=('train','evaluate'),help=argparse.SUPPRESS)
    p.add_argument('--lock-fds',help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.job is not None:
        execute_job(a.job,a.phase);return
    if a.lock_fds:
        supervisor([int(x) for x in a.lock_fds.split(',')]);return
    if a.status or a.stop:
        if status() and a.stop:
            (DIRECTORY/'stop.request').write_text('stop\n')
            print('정상 종료 요청: 현재 학습 저장과 자식 프로세스 종료를 기다립니다.')
        return
    if a.check:
        preflight();return
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        locks=[]
        # Share the original manager and per-suite locks to exclude old launchers.
        paths=[DIRECTORY/'experiment.lock',DIRECTORY.parent/'experiment.lock']
        for suite,queue in QUEUES.items():
            paths += [ROOT/read(queue)['log_dir']/'queue.lock',
                      DIRECTORY.parent/f'{suite.replace("-","_")}_dev_evaluation/evaluation.lock']
        for path in paths:
            path.parent.mkdir(parents=True,exist_ok=True)
            lock=stack.enter_context(path.open('a'))
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                print(f'이미 실행 중인 작업이 있습니다: {path}');return
            locks.append(lock)
        preflight()
        (DIRECTORY/'stop.request').unlink(missing_ok=True)
        fds=tuple(lock.fileno() for lock in locks)
        with (DIRECTORY/'console.log').open('ab',buffering=0) as output:
            child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--lock-fds',','.join(map(str,fds))],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=fds)
        for _ in range(100):
            if child.poll() is not None:raise RuntimeError(f'startup failed: {DIRECTORY}/console.log')
            pidfile=DIRECTORY/'experiment.pid'
            if pidfile.exists() and pidfile.read_text()==str(child.pid):break
            time.sleep(.1)
        print(f'가속 본실험 재개: PID {child.pid}\n상태: {DIRECTORY}/status.json\n로그: {DIRECTORY}/console.log')


if __name__=='__main__':main()
