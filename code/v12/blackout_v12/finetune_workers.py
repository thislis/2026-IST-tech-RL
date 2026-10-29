"""Persistent, pinned CPU actors; only slot zero displays its training game.

Unity is reused between natural episodes and frozen-policy waves. Policy and
opponent construction, task manifests, RNG seeding and cache validation remain
in collector.run_episode, so an idle actor cannot retain an old policy.
"""
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import atexit
import copy
import multiprocessing
import os
from pathlib import Path
import random
import subprocess

from .collector import run_episode
from .env import executable, make_env
from .runtime import configure_worker, managed_popen, process_tree_guard


MAX_WORKERS = 16
_ENV = None
_ENV_KEY = None
_SLOT = None
_VISIBLE = False
_STOP = None
_GUARD = None


class _ManagedSubprocess:
    """Route Unity launches through the worker's process tree guard."""
    Popen = staticmethod(managed_popen)

    def __getattr__(self, name):
        return getattr(subprocess, name)


def _check_worker_stop():
    if _STOP is not None and _STOP.is_set():
        raise InterruptedError('persistent collection stopped; partial episode excluded')


class _EpisodeLease:
    """The collector closes each episode lease, never its reusable Unity player."""
    def __init__(self, env):
        self.env = env

    def reset(self, *args, **kwargs):
        _check_worker_stop()
        return self.env.reset(*args, **kwargs)

    def step(self, *args, **kwargs):
        _check_worker_stop()
        return self.env.step(*args, **kwargs)

    def close(self):
        pass


def _destroy_environment():
    global _ENV, _ENV_KEY
    env, _ENV, _ENV_KEY = _ENV, None, None
    if env is not None:
        env.close()


def _close_worker():
    global _GUARD
    try:
        _destroy_environment()
    finally:
        guard, _GUARD = _GUARD, None
        if guard is not None:
            guard.__exit__(None, None, None)


def _initialize_worker(slot, visible, stop):
    global _SLOT, _VISIBLE, _STOP, _GUARD
    _SLOT, _VISIBLE, _STOP = int(slot), bool(visible), stop
    # Workers never need CUDA, including when their supervisor owns a CUDA learner.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    configure_worker()
    _GUARD = process_tree_guard()
    _GUARD.__enter__()
    # Explicit cleanup tasks are the normal path; atexit covers interpreter exit.
    atexit.register(_close_worker)


def _environment(task):
    global _ENV, _ENV_KEY
    _check_worker_stop()
    config = task['config']
    key = (str(executable(config).resolve()), float(config.get('time_scale', 50)))
    if _ENV is not None and key != _ENV_KEY:
        _destroy_environment()
    if _ENV is None:
        from mlagents_envs import env_utils
        original = env_utils.subprocess
        # Patch this module reference only: managed_popen itself uses the real
        # subprocess module. Each dedicated actor constructs one game at a time.
        env_utils.subprocess = _ManagedSubprocess()
        try:
            _ENV = make_env(task, background=not _VISIBLE, worker_id=_SLOT)
            _ENV_KEY = key
        finally:
            env_utils.subprocess = original
    return _EpisodeLease(_ENV)


def _run_task(task):
    _check_worker_stop()
    seed = int(task['action_seed'])
    random.seed(seed)
    import numpy as np
    np.random.seed(seed % (2**32))
    try:
        # The task itself stays unchanged, including its immutable manifest hash.
        return run_episode(task, env_factory=lambda:_environment(task), actor_threads=1)
    except BaseException:
        # Never reuse a game after a failed/partial transition or failed reset.
        try:
            _destroy_environment()
        except Exception:
            # Keep the original invalid-transition error. The worker Job still
            # owns the game, and pool shutdown closes that handle in all cases.
            pass
        raise


def _stopped(tasks):
    return any(task['config'].get('stop_file')
               and Path(task['config']['stop_file']).exists() for task in tasks)


class PersistentEpisodePool:
    """At most 16 actors, with one deterministic visible training slot.

    Keep this context open across optimizer updates; close it before evaluation.
    Each slot owns a dedicated one-process executor, preventing the visible
    role from moving between workers when a new wave is dispatched.
    """
    def __init__(self, workers=16, visible=True):
        if isinstance(workers, bool) or int(workers) != workers or not 1 <= int(workers) <= MAX_WORKERS:
            raise ValueError(f'workers must be an integer between 1 and {MAX_WORKERS}')
        self.workers = int(workers)
        self.visible = bool(visible)
        self._context = multiprocessing.get_context('spawn')
        self._stop = self._context.Event()
        self._pools = []
        self._started = set()
        self._closed = False
        self._running = False

    def __enter__(self):
        if self._closed:
            raise RuntimeError('persistent episode pool is closed')
        return self

    def _ensure_pools(self):
        if not self._pools:
            try:
                for slot in range(self.workers):
                    self._pools.append(ProcessPoolExecutor(
                        max_workers=1, mp_context=self._context,
                        initializer=_initialize_worker,
                        initargs=(slot, self.visible and slot == 0, self._stop)))
            except BaseException:
                self.close()
                raise

    def run(self, tasks):
        if self._closed:
            raise RuntimeError('persistent episode pool is closed')
        if self._running:
            raise RuntimeError('a persistent episode wave is already running')
        # Copy before submission: ProcessPoolExecutor serializes asynchronously.
        tasks = copy.deepcopy(list(tasks))
        if not tasks:
            return []
        if len({task['id'] for task in tasks}) != len(tasks):
            raise ValueError('duplicate episode ids in persistent wave')
        if len({str(Path(task['directory']).resolve()) for task in tasks}) != len(tasks):
            raise ValueError('duplicate episode directories in persistent wave')
        if len({task.get('policy_sha256') for task in tasks}) != 1:
            raise ValueError('mixed behavior snapshots in persistent wave')
        if _stopped(tasks):
            raise InterruptedError('collection stopped before new episode')
        self._ensure_pools()
        self._running = True
        # On a partial-wave resume, give slot zero actual remaining work before
        # cached validations, so the displayed game belongs to training.
        scheduled = sorted(tasks, key=lambda task:(Path(task['directory'])/'episode.pt').exists())
        jobs, results, pending = {}, [], iter(scheduled)

        def dispatch(slot):
            if _stopped(tasks):
                raise InterruptedError('collection stopped before new episode')
            task = next(pending, None)
            if task is not None:
                self._started.add(slot)
                jobs[self._pools[slot].submit(_run_task, task)] = (slot, task)

        try:
            for slot in range(min(self.workers, len(tasks))):
                dispatch(slot)
            while jobs:
                if _stopped(tasks):
                    raise InterruptedError('collection stopped before optimizer commit')
                done, _ = wait(jobs, timeout=.25, return_when=FIRST_COMPLETED)
                # Check every completed result before dispatching replacement work.
                idle = []
                for future in done:
                    slot, task = jobs.pop(future)
                    row = future.result()
                    if (row.get('episode_id') != task['id']
                            or row.get('policy_sha256') != task.get('policy_sha256')
                            or not row.get('natural_terminal')):
                        raise ValueError('persistent actor returned an invalid behavior episode')
                    results.append(row)
                    idle.append(slot)
                for slot in sorted(idle):
                    dispatch(slot)
            if _stopped(tasks):
                raise InterruptedError('collection stopped before optimizer commit')
            return sorted(results, key=lambda row:row['episode_id'])
        except BaseException:
            self._stop.set()
            for future in jobs:
                future.cancel()
            self.close()
            raise
        finally:
            self._running = False

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        cleanup = []
        for slot in sorted(self._started):
            try:
                cleanup.append(self._pools[slot].submit(_close_worker))
            except RuntimeError:
                # A broken worker already closed its Windows Job handle.
                pass
        try:
            for future in cleanup:
                try:
                    future.result()
                except Exception:
                    # The worker's guard is also closed by OS process teardown.
                    pass
        finally:
            for pool in self._pools:
                pool.shutdown(wait=True, cancel_futures=True)
            self._pools.clear()

    def __exit__(self, *_):
        self.close()
