"""One bounded Windows league match with its own Unity Job ownership."""
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
        configure_worker()
        if task['config']['threads'] != 1 or task['config']['device'] != 'cpu' or task.get('audit_windows') is not False:
            raise ValueError('Windows league requires CPU, one thread, and the native window adapter')
        from .match import play
        from blackout_v12.env import make_env
        from mlagents_envs import env_utils
        def environment():
            original = env_utils.subprocess
            env_utils.subprocess = _ManagedSubprocess()
            try:
                return make_env(task, background=True, worker_id=args.slot)
            finally:
                env_utils.subprocess = original
        with process_tree_guard():
            result = play(task, path.parent, environment, heartbeat=heartbeat, stop=lambda: stopped)
        result['trace_bytes'] = sum(p.stat().st_size for p in (path.parent / 'trace').rglob('*') if p.is_file())
        result['execution_adapter'] = 'blackout_v12.env.make_env(background=True)'
        result['windows_window_audit'] = 'native hidden process + background; macOS CoreGraphics audit disabled'
        atomic(path.parent / 'result.json', dict(task_sha256=digest(task), result=result))
    except BaseException as exc:
        atomic(path.parent / 'error.json', dict(error=str(exc), type=type(exc).__name__, traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
