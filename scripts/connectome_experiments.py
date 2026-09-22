#!/usr/bin/env python3
"""One-command background runner for the registered v7 pilots and dev evaluations.

Each suite runs in a separate interpreter so its registered source revision is
used for checkpoint validation, training, and evaluation.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'logs/v7/pilot_experiments'
SUITES = ('v7-1', 'v7-2')
SUITE_DIRECTORIES = {suite: ROOT / f'logs/v7/pilot_{suite.replace("-", "_")}_dev_evaluation'
                     for suite in SUITES}
SCOPE = 'registered pilots; dev target evaluation; not final test or main study'
ENTRYPOINT = Path(__file__).resolve()


def write_json(name, value):
    target = DIRECTORY / name
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(target)


def command(suite):
    return [sys.executable, '-u', str(ROOT / 'scripts/pilot_evaluation.py'),
            '--suite', suite, '--pipeline']


def show_status():
    lockfile = DIRECTORY / 'experiment.lock'
    active = False
    if lockfile.exists():
        with lockfile.open('r') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                active = True
    result = dict(active=active)
    for name, directory in [('combined', DIRECTORY)] + [(suite, SUITE_DIRECTORIES[suite]) for suite in SUITES]:
        path = directory / 'status.json'
        result[name] = json.loads(path.read_text()) if path.exists() else {'status': 'not_started'}
    print(json.dumps(result, ensure_ascii=False, indent=2))


def worker(lock_fd):
    with os.fdopen(lock_fd, 'a'):
        child = None
        stopped = False

        def stop(signum, frame):
            nonlocal stopped
            stopped = True
            if child is not None and child.poll() is None:
                child.send_signal(signal.SIGTERM)

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        (DIRECTORY / 'experiment.pid').write_text(str(os.getpid()))
        summaries = {}
        write_json('status.json', dict(status='starting', pid=os.getpid()))
        write_json('summary.json', dict(complete=False, suites=summaries))
        try:
            for suite in SUITES:
                if stopped:
                    break
                write_json('status.json', dict(status='running', suite=suite, pid=os.getpid()))
                print(f'실행: {suite} 학습 확인/재개 → dev 평가', flush=True)
                child = subprocess.Popen(command(suite) + ['--foreground'], cwd=ROOT,
                                         stdin=subprocess.DEVNULL)
                if stopped:
                    child.send_signal(signal.SIGTERM)
                code = child.wait()
                child = None
                if stopped:
                    break
                if code:
                    raise RuntimeError(f'{suite} failed (exit={code}); inspect console.log')
                summary = SUITE_DIRECTORIES[suite] / 'summary.json'
                result = json.loads(summary.read_text())
                if not result['complete']:
                    raise RuntimeError(f'{suite} did not produce a complete summary')
                summaries[suite] = result
                write_json('summary.json', dict(complete=len(summaries) == len(SUITES), suites=summaries,
                           scope=SCOPE))
            write_json('status.json', dict(status='stopped' if stopped else 'complete',
                       pid=os.getpid(), completed_suites=list(summaries)))
        except BaseException as exc:
            write_json('status.json', dict(status='failed', pid=os.getpid(), error=str(exc)))
            raise
        finally:
            (DIRECTORY / 'experiment.pid').unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--check', action='store_true', help='validate both suites without starting Unity')
    group.add_argument('--status', action='store_true', help='show combined and per-suite status')
    parser.add_argument('--worker-lock-fd', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_lock_fd is not None:
        worker(args.worker_lock_fd)
        return
    if args.status:
        show_status()
        return
    if args.check:
        for suite in SUITES:
            subprocess.run(command(suite) + ['--check'], cwd=ROOT, check=True)
        return
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    with (DIRECTORY / 'experiment.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('v7 통합 실험이 이미 실행 중입니다. --status로 진행 상태를 확인하세요.')
            return
        for suite in SUITES:
            subprocess.run(command(suite) + ['--check'], cwd=ROOT, check=True)
        with (DIRECTORY / 'console.log').open('ab', buffering=0) as output:
            child = subprocess.Popen([sys.executable, '-u', str(ENTRYPOINT),
                                      '--worker-lock-fd', str(lock.fileno())], cwd=ROOT,
                                     stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                     start_new_session=True, pass_fds=(lock.fileno(),))
        for _ in range(100):
            if child.poll() is not None:
                raise RuntimeError(f'startup failed; inspect {DIRECTORY}/console.log')
            pidfile = DIRECTORY / 'experiment.pid'
            if pidfile.exists() and pidfile.read_text().strip() == str(child.pid):
                print(f'v7 백그라운드 실험 시작: PID {child.pid}\n로그: {DIRECTORY}/console.log\n상태: {DIRECTORY}/status.json')
                return
            time.sleep(.1)
        print(f'프로세스 생성: PID {child.pid}; {DIRECTORY}/console.log에서 시작 상태를 확인하세요.')


if __name__ == '__main__':
    main()
