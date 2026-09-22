#!/usr/bin/env python3
"""Evaluate completed v7-1 pilots sequentially, detached by default.

This orchestration helper lives outside the frozen training-source file set.
The original evaluator and its checkpoint provenance checks remain unchanged.
"""
import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.frozen_pilot_runtime import resolve
CODE_ROOT = ROOT
DIRECTORY = ROOT / 'logs/v7/pilot_v7_1_dev_evaluation'
QUEUE = ROOT / 'configs/v7/pilot_v7_1_queue.json'
PIPELINE = False


def configure(suite, pipeline=False, *, queue=None, output_dir=None, current_runtime=False):
    global CODE_ROOT, DIRECTORY, QUEUE, PIPELINE
    if (queue is None) != (output_dir is None):
        raise ValueError('custom queue and output directory must be supplied together')
    CODE_ROOT = resolve() if suite == 'v7-1' and not current_runtime else ROOT
    sys.path.insert(0, str(CODE_ROOT))
    DIRECTORY = ROOT / ('logs/v7/pilot_v7_1_dev_evaluation' if suite == 'v7-1'
                        else 'logs/v7/pilot_v7_2_dev_evaluation')
    QUEUE = ROOT / ('configs/v7/pilot_v7_1_queue.json' if suite == 'v7-1'
                    else 'configs/v7/pilot_v7_2_recovery_queue.json')
    if queue is not None:
        QUEUE = ROOT / queue
        DIRECTORY = ROOT / output_dir
    PIPELINE = pipeline


def read_json(path):
    return json.loads(path.read_text())


def completed_evaluation(job):
    """Reuse only full paired dev evaluations of this exact checkpoint."""
    expected = {(seed, team) for seed in job['maps'] for team in (0, 1)}
    for file in sorted(Path(job['run']).glob('eval_dev_target_*.json'), reverse=True):
        if file.name.endswith('.progress.json'):
            continue
        try:
            result = read_json(file)
            rows = result['episodes']
            pairs = [(r['seed'], r['team']) for r in rows]
            valid = (result['schema'] == 'blackout.v7.evaluation.v1'
                     and result['checkpoint_sha256'] == job['checkpoint_sha256']
                     and result['training_seed'] == job['seed']
                     and result['split'] == 'dev' and result['opponent'] == 'target'
                     and len(pairs) == len(expected) and set(pairs) == expected
                     and all(r['winner'] in (-1, 0, 1) for r in rows))
            if valid:
                return file, result
        except (ValueError, KeyError, TypeError):
            continue
    return None


def preflight(allow_incomplete=False):
    import torch
    from blackout_rl.v7_registry import load_config, check_config, make_model, source_fingerprint
    from blackout_rl.connectome.graph_artifact import digest, sha256
    current = source_fingerprint()
    cache = {}
    jobs = []
    for job in read_json(QUEUE)['jobs']:
        configured = load_config(ROOT / job['config'])
        run = ROOT / 'logs/v7' / configured['experiment_id'] / str(job['seed'])
        cfg = read_json(run / 'resolved_config.json') if (run / 'resolved_config.json').exists() else configured
        if cfg != configured:
            raise ValueError(f'configuration changed since training: {run}')
        status = read_json(run / 'status.json') if (run / 'status.json').exists() else {}
        complete = status.get('status') == 'budget_complete'
        if not complete and not allow_incomplete:
            raise ValueError(f'pilot incomplete: {run}')
        if (run / 'run.lock').exists():
            with (run / 'run.lock').open('r') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checkpoint = run / 'checkpoints/latest.pt'
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True) if checkpoint.exists() else None
        if payload is None and (complete or (run / 'resolved_config.json').exists()):
            raise ValueError(f'existing run has no resumable checkpoint: {run}')
        budget = cfg['training']['max_environment_steps']
        if payload is not None and (payload['schema'] != 'blackout.v7.checkpoint.v1'
                or payload['config_sha256'] != digest(cfg) or payload['sources'] != current
                or payload['seed'] != job['seed']
                or not 0 <= payload['global_step'] <= budget
                or (complete and (payload['global_step'] != budget
                                  or status['environment_steps'] != payload['global_step']))
                or (not complete and payload['global_step'] == budget)):
            raise ValueError(f'checkpoint/config/source/budget mismatch: {run}')
        key = digest(cfg)
        if key not in cache:
            graph, splits = check_config(cfg)
            with torch.random.fork_rng():
                model = make_model(cfg, graph)
            cache[key] = model, splits
        model, splits = cache[key]
        if payload is not None:
            model.load_state_dict(payload['model'], strict=True)
        jobs.append(dict(run=str(run), seed=job['seed'],
                         checkpoint_sha256=sha256(checkpoint) if payload is not None else None,
                         maps=splits['dev'], experiment_id=cfg['experiment_id'], complete=complete,
                         config=str(ROOT / job['config']),
                         script='train_v7_1.py' if cfg['mode'] == 'residual_graph' else 'run_v7_2.py'))
    return jobs


def worker(lock_fd):
    from blackout_rl.connectome.graph_artifact import atomic_json
    with os.fdopen(lock_fd, 'a'):
        stopped = False
        child = None
        def stop(signum, frame):
            nonlocal stopped
            stopped = True
            if child is not None and child.poll() is None:
                child.send_signal(signal.SIGINT)  # Let the evaluator close Unity in finally.
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        (DIRECTORY / 'evaluation.pid').write_text(str(os.getpid()))
        atomic_json(DIRECTORY / 'status.json', dict(status='starting', pid=os.getpid()))
        try:
            if PIPELINE:
                for job in preflight(allow_incomplete=True):
                    if stopped:
                        break
                    if job['complete']:
                        print(f'학습 완료: {job["run"]} (건너뜀)', flush=True)
                        continue
                    command = [sys.executable, '-u', str(CODE_ROOT / 'scripts' / job['script']),
                               '--config', job['config'], '--seed', str(job['seed']), '--run-dir', job['run']]
                    if job['checkpoint_sha256'] is not None:
                        command.append('--resume')
                    atomic_json(DIRECTORY / 'status.json', dict(status='training', pid=os.getpid(), run=job['run']))
                    child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL)
                    if stopped:
                        child.send_signal(signal.SIGINT)
                    code = child.wait()
                    child = None
                    if stopped:
                        break
                    if code:
                        raise RuntimeError(f'training failed (exit={code}): {job["run"]}')
                if stopped:
                    atomic_json(DIRECTORY / 'status.json', dict(status='stopped', pid=os.getpid()))
                    return
            jobs = preflight()
            summaries = []
            for index, job in enumerate(jobs):
                if stopped:
                    break
                existing = completed_evaluation(job)
                if existing is None:
                    command = [sys.executable, '-u', str(CODE_ROOT / 'scripts/evaluate_v7.py'),
                               '--run', job['run'], '--split', 'dev', '--opponent', 'target']
                    atomic_json(DIRECTORY / 'status.json', dict(status='running', pid=os.getpid(),
                                job=index + 1, jobs=len(jobs), run=job['run'], completed=len(summaries)))
                    print(f'[{index + 1}/{len(jobs)}] {job["run"]}', flush=True)
                    child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL)
                    # Cover a stop arriving while Popen was creating the child.
                    if stopped:
                        child.send_signal(signal.SIGINT)
                    code = child.wait()
                    child = None
                    if stopped:
                        break
                    if code:
                        raise RuntimeError(f'evaluation failed (exit={code}): {job["run"]}')
                    existing = completed_evaluation(job)
                    if existing is None:
                        raise RuntimeError(f'no complete paired evaluation result: {job["run"]}')
                file, result = existing
                rows = result['episodes']
                summaries.append(dict(experiment_id=job['experiment_id'], seed=job['seed'],
                                      result=str(file), episodes=len(rows),
                                      wins=sum(r['winner'] == r['team'] for r in rows),
                                      draws=sum(r['winner'] == -1 for r in rows),
                                      win_rate=result['win_rate'], draw_rate=result['draw_rate'],
                                      paired_map_bootstrap_ci=result['paired_map_bootstrap_ci']))
                atomic_json(DIRECTORY / 'summary.json', dict(split='dev', opponent='target', runs=summaries,
                            complete=len(summaries) == len(jobs),
                            note='per-run results; not an across-training-seed significance test'))
            atomic_json(DIRECTORY / 'status.json', dict(status='stopped' if stopped else 'complete',
                        pid=os.getpid(), completed=len(summaries), jobs=len(jobs)))
        except BaseException as exc:
            atomic_json(DIRECTORY / 'status.json', dict(status='failed', pid=os.getpid(), error=str(exc)))
            raise
        finally:
            (DIRECTORY / 'evaluation.pid').unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=('v7-1', 'v7-2'), default='v7-1')
    parser.add_argument('--pipeline', action='store_true', help='finish/resume registered training before dev evaluation')
    parser.add_argument('--foreground', action='store_true', help='wait for completion (used by combined runner)')
    parser.add_argument('--queue', help='registered custom training queue')
    parser.add_argument('--output-dir', help='custom evaluation status and summary directory')
    parser.add_argument('--current-runtime', action='store_true', help='use current sources for a newly registered study')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--check', action='store_true', help='read-only validation; no Unity or new outputs')
    group.add_argument('--status', action='store_true', help='show latest evaluation status')
    parser.add_argument('--worker-lock-fd', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    configure(args.suite, args.pipeline, queue=args.queue, output_dir=args.output_dir,
              current_runtime=args.current_runtime)
    if args.worker_lock_fd is not None:
        worker(args.worker_lock_fd)
        return
    if args.status:
        file = DIRECTORY / 'status.json'
        print(file.read_text() if file.exists() else 'Evaluation has not been started.')
        return
    jobs = preflight(allow_incomplete=args.pipeline)
    training = sum(not job['complete'] for job in jobs)
    remaining = sum(not job['complete'] or completed_evaluation(job) is None for job in jobs)
    print(f'평가 코드: {CODE_ROOT}', flush=True)
    print(f'{args.suite} 검사 통과: {len(jobs)}개 실험, 남은 학습 {training}개, 남은 평가 {remaining}개 × dev 30맵 × A/B 2진영.', flush=True)
    if args.check or (not remaining and not args.foreground):
        return
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        lock = stack.enter_context((DIRECTORY / 'evaluation.lock').open('a'))
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('development evaluation is already running')
        if args.pipeline:
            queue_directory = ROOT / read_json(QUEUE)['log_dir']
            queue_directory.mkdir(parents=True, exist_ok=True)
            queue_lock = stack.enter_context((queue_directory / 'queue.lock').open('a'))
            fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.foreground:
            worker(os.dup(lock.fileno()))
            if read_json(DIRECTORY / 'status.json')['status'] != 'complete':
                raise SystemExit(130)
            return
        # The inherited descriptor holds the lifetime lock continuously, including startup.
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--suite', args.suite,
                   '--worker-lock-fd', str(lock.fileno())]
        if args.queue:
            command.extend(['--queue', args.queue, '--output-dir', args.output_dir])
        if args.current_runtime:
            command.append('--current-runtime')
        inherited = [lock.fileno()]
        if args.pipeline:
            command.append('--pipeline')
            inherited.append(queue_lock.fileno())
        with (DIRECTORY / 'console.log').open('ab', buffering=0) as output:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=output,
                                     stderr=subprocess.STDOUT, start_new_session=True, pass_fds=tuple(inherited))
        for _ in range(100):
            if child.poll() is not None:
                raise RuntimeError(f'evaluation startup failed; inspect {DIRECTORY}/console.log')
            pidfile = DIRECTORY / 'evaluation.pid'
            if pidfile.is_file() and pidfile.read_text().strip() == str(child.pid):
                print(f'백그라운드 평가 시작: PID {child.pid}\n상태: {DIRECTORY}/status.json\n로그: {DIRECTORY}/console.log')
                return
            time.sleep(.1)
        print(f'백그라운드 프로세스 생성: PID {child.pid}; {DIRECTORY}/console.log 에서 시작 상태를 확인하세요.')


if __name__ == '__main__':
    main()
