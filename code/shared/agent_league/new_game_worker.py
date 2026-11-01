"""One bounded newest-game league match with isolated API and native Job ownership."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

from .common import digest
from blackout_v12.io import atomic_json as atomic, read_json as read, transient_file_access
from blackout_v12.runtime import (configure_worker, managed_popen, process_options,
                                 process_tree_guard)


class _ManagedSubprocess:
    """Only ML-Agents' Unity launch goes through the worker's native Job guard."""
    def Popen(self, *args, **kwargs):
        if os.name == 'nt':
            kwargs['creationflags'] = kwargs.get('creationflags', 0) | process_options()['creationflags']
        return managed_popen(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(subprocess, name)


def publish_heartbeat(path, **data):
    try:
        atomic(path, dict(time=time.time(), pid=os.getpid(), **data))
    except PermissionError as exc:
        if not transient_file_access(exc):
            raise
        # Telemetry is retried at the next heartbeat. Persistent failures still
        # expire the supervisor's last-successful-heartbeat watchdog.
        print(f'Heartbeat temporarily unavailable; retrying next update: {exc}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True)
    parser.add_argument('--slot', required=True, type=int)
    args = parser.parse_args()
    path = Path(args.task)
    task = read(path)
    stopped = False
    def request_stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    def heartbeat(**data):
        publish_heartbeat(path.parent / 'heartbeat.json', **data)
    heartbeat(phase='starting', steps=0)
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        from .new_game_runtime import configure_runtime, make_env
        configure_runtime()
        configure_worker()
        if (task['config']['threads'] != 1 or task['config']['device'] != 'cpu'
                or task.get('audit_windows') is not False
                or not 0 <= args.slot < task['config']['workers'] <= 16
                or task['config'].get('visible_slot') != 0):
            raise ValueError('Windows league requires CPU, one thread, and the native window adapter')
        from .new_game_match import play
        from mlagents_envs import env_utils
        environments = []
        def environment():
            original = env_utils.subprocess
            env_utils.subprocess = _ManagedSubprocess()
            try:
                env = make_env(task, background=args.slot != 0, worker_id=args.slot)
                environments.append(env)
                return env
            finally:
                env_utils.subprocess = original
        with process_tree_guard():
            result = play(task, path.parent, environment, heartbeat=heartbeat, stop=lambda: stopped)
        result['trace_bytes'] = sum(p.stat().st_size for p in (path.parent / 'trace').rglob('*') if p.is_file())
        result['execution_adapter'] = 'agent_league.new_game_runtime.make_env'
        result['windows_window_audit'] = 'slot 0 visible rendered Unity; remaining slots rendered background'
        result['native_window_audit'] = environments[0].native_window_audit
        result['execution'] = dict(slot=args.slot, background=args.slot != 0,
                                   no_graphics=False, time_scale=task['config']['time_scale']
                                   if args.slot else task['config']['visible_time_scale'])
        atomic(path.parent / 'result.json', dict(task_sha256=digest(task), result=result))
    except BaseException as exc:
        atomic(path.parent / 'error.json', dict(error=str(exc), type=type(exc).__name__, traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
