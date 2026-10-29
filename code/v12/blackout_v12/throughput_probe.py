"""Short, isolated Unity throughput samples; never training/evaluation evidence.

Example: python code/run.py -m blackout_v12.throughput_probe --study joint_trip_v12
The study must already be stopped. Only this probe's output directory is written.
"""
import argparse
import copy
import ctypes
import json
import itertools
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

from .io import ROOT, atomic_json, digest, file_hash, read_json
from .runtime import (FileLease, child_environment, configure_worker, managed_popen,
                      process_options, process_tree_guard)

GIB = 1024**3
MEMORY_RESERVE_BYTES = 4 * GIB
INITIAL_WORKER_BYTES = GIB


def _memory_check(workers, reports=()):
    """Budget Python + Unity together, keeping desktop applications usable."""
    try:
        import psutil
        available = psutil.virtual_memory().available
        logical = psutil.cpu_count(logical=True) or os.cpu_count() or 1
        provider = 'psutil.virtual_memory.available'
    except ImportError:
        if os.name != 'nt':
            raise RuntimeError('Install psutil to measure available memory on this platform')
        class MemoryStatus(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong),
                        *[(name, ctypes.c_ulonglong) for name in
                          ('total', 'available', 'page_total', 'page_available',
                           'virtual_total', 'virtual_available', 'extended_available')]]
        status = MemoryStatus(); status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise ctypes.WinError()
        available = status.available
        logical = os.cpu_count() or 1
        provider = 'Windows.GlobalMemoryStatusEx.available_physical'
    measured = [row['resources'].get('sum_process_lifetime_peak_rss_bytes', 0) / row['workers']
                for row in reports]
    per_worker = max([INITIAL_WORKER_BYTES] + [int(value * 1.2) for value in measured])
    memory_cap = max(0, int((available - MEMORY_RESERVE_BYTES) // per_worker))
    safe_max = memory_cap
    safe_max -= safe_max % 2
    return dict(available_bytes=available, memory_provider=provider, reserve_bytes=MEMORY_RESERVE_BYTES,
                estimated_worker_bytes=per_worker, logical_cores=logical,
                safe_max_workers=safe_max, requested_workers=workers,
                allowed=workers <= safe_max)


def _templates(study, probe_arm):
    """Read immutable tasks; Q2 uses the most recent data collector manifest."""
    templates, sources = [], []
    arms = ('Q0', 'Q1') if probe_arm is None else (probe_arm,)
    for arm in arms:
        paths = []
        if arm == 'Q2':
            paths = sorted((ROOT/'artifacts/data/v12'/study.name/'runs/data').glob('wave_*/plan.json'),
                           reverse=True)
        paths.append(study/f'evaluation/baseline/{arm}_map/plan.json')
        for plan_path in paths:
            plan = read_json(plan_path)
            eligible = [task for task in plan['tasks'] if task.get('arm') == arm and
                        task['opponent']['family'] == 'scripted']
            if eligible:
                task = next((task for task in eligible
                             if task['opponent']['id'] == 'scripted-collector'), eligible[0])
                templates.append(copy.deepcopy(task))
                sources.append(dict(path=str(plan_path), sha256=file_hash(plan_path),
                                    task_id=task['id'], task_sha256=digest(task)))
                break
        else:
            raise ValueError(f'No immutable scripted-opponent task for {arm}')
    comparable = lambda t: (t['requested_seed'], t['action_seed'], t['side'], t['opponent']['id'],
                           t.get('decoder', 'map'))
    if len({comparable(task) for task in templates}) != 1:
        raise ValueError('Probe templates must use identical maps, seeds, opponent and side')
    return templates, sources, comparable(templates[0])


def _process_resources(handle):
    """Read cumulative CPU and current/peak RSS without optional dependencies."""
    if os.name != 'nt':
        return None
    from ctypes import wintypes
    class Memory(ctypes.Structure):
        _fields_ = [('size', wintypes.DWORD), ('faults', wintypes.DWORD),
                    *[(key, ctypes.c_size_t) for key in
                      ('peak_rss', 'rss', 'peak_pool', 'pool', 'peak_nonpaged',
                       'nonpaged', 'page', 'peak_page')]]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(ctypes.c_ulonglong)] * 4
    cpu = [ctypes.c_ulonglong() for _ in range(4)]
    if not kernel.GetProcessTimes(handle, *(ctypes.byref(v) for v in cpu)):
        raise ctypes.WinError(ctypes.get_last_error())
    memory = Memory(); memory.size = ctypes.sizeof(memory)
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Memory), wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(memory), memory.size):
        raise ctypes.WinError(ctypes.get_last_error())
    return dict(cpu_seconds=(cpu[2].value + cpu[3].value) / 1e7,
                rss_bytes=memory.rss, lifetime_peak_rss_bytes=memory.peak_rss)


def _resources(env):
    if os.name != 'nt':
        return dict(python_cpu_seconds=time.process_time())
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    unity = env._unity_env._process
    return dict(python=_process_resources(kernel.GetCurrentProcess()),
                unity=_process_resources(int(unity._handle)))


def _latencies(values):
    if not values:
        return None
    import numpy as np
    return dict(p50_ms=float(np.quantile(values, .5) * 1000),
                p95_ms=float(np.quantile(values, .95) * 1000),
                mean_ms=statistics.mean(values) * 1000)


def _worker(task_path):
    """Own one Unity environment, warming then resetting before the barrier."""
    import numpy as np
    import torch
    from blackout_v11.trajectory import canonical_names, pack
    from blackout_v11.evaluation import initial_map_hash, observation_hash
    from .env import make_env
    from .opponents import make_opponent
    from .policy import MyPolicy
    from .critic import critic_features
    from .rewards import potential
    from .trips import TripLedger
    configure_worker()
    record = read_json(task_path)
    task = record['task']; directory = Path(task_path).parent
    seed = int(task['action_seed'])
    torch.manual_seed(seed)
    policy = MyPolicy(task['config'], arm=task['arm']).eval()
    if task.get('checkpoint'):
        if file_hash(task['checkpoint']) != task['policy_sha256']:
            raise ValueError('Probe checkpoint differs from immutable source task')
        policy.scorer.load_state_dict(torch.load(task['checkpoint'], map_location='cpu',
                                                 weights_only=True)['scorer_state'])
    own = canonical_names(task['side']); other = canonical_names(1-task['side'])
    expected = set(own + other)
    opponent = make_opponent(task['opponent'], seed + 1)
    collector = record.get('collector_workload', False)
    ledger = TripLedger(task['side']); collected_events = []; episode_step = 0
    totals = np.zeros(2, np.float64)
    env = None
    started = time.perf_counter()
    try:
        env = make_env(task)
        obs, _ = env.reset(seed=task['requested_seed'])
        initial_startup = time.perf_counter() - started
        initial_hashes = (initial_map_hash(obs), observation_hash(obs))

        def restart_episode():
            nonlocal ledger, collected_events, episode_step, totals
            fresh, _ = env.reset(seed=task['requested_seed'])
            torch.manual_seed(seed); policy.reset()
            ledger = TripLedger(task['side']); collected_events = []; episode_step = 0
            totals = np.zeros(2, np.float64)
            fresh_opponent = make_opponent(task['opponent'], seed + 1)
            if (initial_map_hash(fresh), observation_hash(fresh)) != initial_hashes:
                raise ValueError('Probe episode reset changed the fixed initial Unity observations')
            return fresh, fresh_opponent

        def step(current):
            nonlocal episode_step
            vector, graphic = pack(current, own)
            mark = time.perf_counter()
            with torch.inference_mode():
                result = policy.step(torch.from_numpy(vector),
                    torch.from_numpy(graphic)[None].expand(5, -1, -1, -1),
                    deterministic=task.get('decoder', 'map') == 'map')
            policy_seconds = time.perf_counter() - mark
            collector_seconds = 0.
            if collector:
                mark = time.perf_counter()
                if result['event']:
                    decision = {key:value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
                                for key,value in result['decision'].items()}
                    decision['old_log_prob'] = torch.as_tensor(decision.get('old_log_prob', decision['log_prob'])).float().reshape(())
                    decision['temperature'] = torch.as_tensor(decision['temperature']).float().reshape(())
                    decision['critic_features'] = critic_features(decision['features'], decision['legal'])
                    decision.update(step=episode_step, potential=potential(vector), on_policy=False)
                    collected_events.append(decision)
                    ledger.decision(episode_step, decision)
                collector_seconds += time.perf_counter() - mark
            actions = dict(zip(own, result['action'].cpu().numpy().astype(np.float32)))
            mark = time.perf_counter()
            actions.update(opponent.act({name:current[name] for name in other}, other))
            opponent_seconds = time.perf_counter() - mark
            mark = time.perf_counter()
            nxt, rewards, terms, truncs, _ = env.step(actions)
            env_seconds = time.perf_counter() - mark
            if any(set(x) != expected for x in (rewards, terms, truncs)):
                raise ValueError('Incomplete transition in throughput sample')
            if any(truncs.values()) or len(set(map(bool, terms.values()))) != 1:
                raise ValueError('Truncated/partial transition in throughput sample')
            terminal = all(terms.values())
            if not terminal and set(nxt) != expected:
                raise ValueError('Missing nonterminal observations in throughput sample')
            if collector:
                mark = time.perf_counter()
                episode_step += 1
                for side in (0, 1):
                    totals[side] += sum(float(rewards[name]) for name in canonical_names(side))
                if not np.isfinite(totals).all():
                    raise FloatingPointError('Nonfinite collector reward')
                nv = pack(nxt, own)[0] if set(nxt) == expected else None
                clock_reset = nv is not None and float(nv[0,95]) > float(vector[0,95])+1e-3
                if clock_reset and not terminal:
                    raise ValueError('Collector clock reset without terminal')
                ledger.observe(vector, nv, episode_step, terminal, nv is not None and not clock_reset, graphic)
                collector_seconds += time.perf_counter() - mark
            return nxt, terminal, policy_seconds, opponent_seconds, env_seconds, result['event'], collector_seconds

        warmup_completed_episodes = 0
        for _ in range(record['warmup_steps']):
            obs, terminal, *_ = step(obs)
            if terminal:
                warmup_completed_episodes += 1
                obs, opponent = restart_episode()
        # Measured trajectories always start at the same initial state, independently
        # of environment startup speed and warmup duration.
        obs, opponent = restart_episode()
        ready = dict(pid=os.getpid(), unity_pid=env._unity_env._process.pid,
                     arm=task['arm'], initial_map_hash=initial_hashes[0],
                     initial_observation_sha256=initial_hashes[1],
                     warmup_completed_probe_episodes=warmup_completed_episodes,
                     initial_startup_seconds=initial_startup,
                     startup_warmup_reset_seconds=time.perf_counter()-started)
        atomic_json(directory/'ready.json', ready)
        barrier_path = directory.parent/'go.json'
        timeout = time.perf_counter() + 180
        while not barrier_path.exists():
            if time.perf_counter() > timeout:
                raise TimeoutError('Probe start barrier timed out')
            time.sleep(.02)
        barrier = read_json(barrier_path)
        while time.perf_counter() < barrier['start']:
            time.sleep(min(.01, max(0, barrier['start'] - time.perf_counter())))
        resource_before = _resources(env)
        sample_start = time.perf_counter()
        loops, policies, opponents, envs, resets, collectors = [], [], [], [], [], []
        events = 0; completed_episodes = 0
        while time.perf_counter() < barrier['deadline']:
            mark = time.perf_counter()
            obs, terminal, policy_s, opponent_s, env_s, event, collector_s = step(obs)
            policies.append(policy_s); opponents.append(opponent_s); envs.append(env_s)
            collectors.append(collector_s)
            loops.append(time.perf_counter() - mark)
            events += int(event)
            if terminal:
                completed_episodes += 1
                if time.perf_counter() < barrier['deadline']:
                    mark = time.perf_counter()
                    obs, opponent = restart_episode()
                    resets.append(time.perf_counter()-mark)
        sample_end = time.perf_counter()
        result = dict(schema='v12.throughput_worker.1', **ready,
                      provenance='throughput-only; partial trajectories discarded',
                      training_started=False, evaluation_evidence=False, trajectory_saved=False,
                      env_steps=len(loops), decision_events=events,
                      terminal_reached=completed_episodes > 0,
                      completed_probe_episodes=completed_episodes, measured_episode_resets=len(resets),
                      episode_reset_seconds=sum(resets), episode_reset_latency=_latencies(resets),
                      sample_start=sample_start, sample_end=sample_end,
                      sample_seconds=sample_end-sample_start,
                      policy_latency=_latencies(policies), opponent_latency=_latencies(opponents),
                      collector_workload=collector, collector_latency=_latencies(collectors),
                      unity_step_latency=_latencies(envs), loop_latency=_latencies(loops),
                      resources_before=resource_before, resources_after=_resources(env))
        atomic_json(directory/'result.json', result)
    except BaseException as exc:
        atomic_json(directory/'error.json', dict(error=str(exc), traceback=traceback.format_exc()))
        raise
    finally:
        if env is not None:
            env.close()


def _gpu_snapshot():
    try:
        run = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw',
                              '--format=csv,noheader,nounits'], capture_output=True, text=True,
                             timeout=4, **process_options())
        return dict(time=time.time(), csv=run.stdout.strip(), error=run.stderr.strip(),
                    returncode=run.returncode,
                    columns=['gpu_percent', 'memory_controller_percent', 'used_mib', 'total_mib', 'power_watts'])
    except (OSError, subprocess.TimeoutExpired) as exc:
        return dict(time=time.time(), error=str(exc))


def _candidate(output, templates, workers, seconds, warmup, collector_workload=False):
    directory = output/f'workers_{workers}'; directory.mkdir()
    processes, logs = [], []
    started = time.perf_counter()
    try:
        for index in range(workers):
            worker_dir = directory/f'worker_{index:02d}'; worker_dir.mkdir()
            task = copy.deepcopy(templates[index % len(templates)])
            # These fields only control study output/stop handling, never decisions.
            for key in ('stop_file', 'map_split_registry'):
                task['config'].pop(key, None)
            task_path = worker_dir/'task.json'
            atomic_json(task_path, dict(task=task, warmup_steps=warmup,
                                       collector_workload=collector_workload))
            log = (worker_dir/'console.log').open('ab', buffering=0); logs.append(log)
            command = [sys.executable, '-u', str(ROOT/'code/run.py'), '-m',
                       'blackout_v12.throughput_probe', '--worker', '--task', str(task_path)]
            processes.append(managed_popen(command, cwd=ROOT, env=child_environment(),
                stdin=subprocess.DEVNULL, stdout=log, stderr=log, **process_options()))
        ready = []
        while True:
            errors = list(directory.glob('worker_*/error.json'))
            if errors:
                raise RuntimeError('Worker failed: ' + str(read_json(errors[0])))
            for process in processes:
                if process.poll() is not None:
                    raise RuntimeError('Worker exited before sample barrier; inspect ' + str(directory))
            paths = sorted(directory.glob('worker_*/ready.json'))
            if len(paths) == workers:
                ready = [read_json(path) for path in paths]
                break
            if time.perf_counter()-started > 180:
                raise TimeoutError('Unity initialization/warmup exceeded probe startup limit')
            time.sleep(.1)
        if len({row['initial_map_hash'] for row in ready}) != 1 or len({row['initial_observation_sha256'] for row in ready}) != 1:
            raise ValueError('Identical seeds did not produce identical initial Unity observations')
        barrier_start = time.perf_counter()+.5
        barrier = dict(start=barrier_start, deadline=barrier_start+seconds, seconds=seconds)
        atomic_json(directory/'go.json', barrier)
        startup_seconds = time.perf_counter()-started
        gpu = []
        while time.perf_counter() < barrier['deadline']:
            gpu.append(_gpu_snapshot())
            errors = list(directory.glob('worker_*/error.json'))
            if errors:
                raise RuntimeError('Sample worker failed: ' + str(read_json(errors[0])))
            time.sleep(min(2., max(0, barrier['deadline']-time.perf_counter())))
        for process in processes:
            process.wait(timeout=30)
            if process.returncode != 0:
                raise RuntimeError('Worker failed after sample; inspect ' + str(directory))
        rows = [read_json(path) for path in sorted(directory.glob('worker_*/result.json'))]
        if len(rows) != workers:
            raise RuntimeError('Missing worker report after the throughput window')
        start_skew = max(row['sample_start'] for row in rows)-min(row['sample_start'] for row in rows)
        actual_seconds = max(row['sample_end'] for row in rows)-min(row['sample_start'] for row in rows)
        steps = sum(row['env_steps'] for row in rows)
        resources = {}
        if os.name == 'nt':
            cpu = sum(row['resources_after'][kind]['cpu_seconds']-row['resources_before'][kind]['cpu_seconds']
                      for row in rows for kind in ('python', 'unity'))
            resources = dict(aggregate_cpu_seconds=cpu, average_logical_cores_used=cpu/actual_seconds,
                rss_at_end_bytes=sum(row['resources_after'][kind]['rss_bytes'] for row in rows for kind in ('python', 'unity')),
                sum_process_lifetime_peak_rss_bytes=sum(row['resources_after'][kind]['lifetime_peak_rss_bytes']
                    for row in rows for kind in ('python', 'unity')))
        report = dict(schema='v12.throughput_candidate.1', workers=workers,
            env_steps=steps, aggregate_env_steps_per_second=steps/actual_seconds,
            completed_probe_episodes=sum(row['completed_probe_episodes'] for row in rows),
            episode_reset_seconds=sum(row['episode_reset_seconds'] for row in rows),
            configured_sample_seconds=seconds, sample_wall_seconds=actual_seconds,
            barrier_start_skew_ms=start_skew*1000, startup_warmup_reset_seconds=startup_seconds,
            total_end_to_end_seconds=time.perf_counter()-started,
            loop_p50_ms_median=statistics.median(row['loop_latency']['p50_ms'] for row in rows),
            loop_p95_ms_median=statistics.median(row['loop_latency']['p95_ms'] for row in rows),
            policy_p95_ms_median=statistics.median(row['policy_latency']['p95_ms'] for row in rows),
            unity_step_p95_ms_median=statistics.median(row['unity_step_latency']['p95_ms'] for row in rows),
            resources=resources, gpu_samples=gpu, workers_detail=rows,
            initial_map_hash=ready[0]['initial_map_hash'],
            initial_observation_sha256=ready[0]['initial_observation_sha256'])
        atomic_json(directory/'summary.json', report)
        print(json.dumps({key:value for key,value in report.items()
                          if key not in ('workers_detail', 'gpu_samples')}, ensure_ascii=False), flush=True)
        return report
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait(timeout=10)
        for log in logs:
            log.close()


def select_workers(reports):
    """Maximize measured throughput; prefer more workers only within a 1% tie."""
    if not reports or any(not math.isfinite(row['aggregate_env_steps_per_second']) or
                          row['aggregate_env_steps_per_second'] <= 0 for row in reports):
        raise ValueError('Selection requires positive finite throughput measurements')
    best = max(reports, key=lambda row:row['aggregate_env_steps_per_second'])
    selected = max((row for row in reports
                    if row['aggregate_env_steps_per_second'] >=
                    .99*best['aggregate_env_steps_per_second']), key=lambda row:row['workers'])
    return dict(measured_best_workers=best['workers'],
        measured_best_env_steps_per_second=best['aggregate_env_steps_per_second'],
        selected_workers=selected['workers'],
        selected_env_steps_per_second=selected['aggregate_env_steps_per_second'],
        selection_rule='Highest aggregate env_steps/s; prefer more workers only within 1% of measured maximum')


def run(study='joint_trip_v12', workers=(4, 6, 8), seconds=12., warmup_steps=20, output=None,
        scale_up=False, max_workers=None, probe_arm=None):
    """Explicitly run the short probes and return their summary.json Path.

    Importing this module never launches Unity. The caller must explicitly invoke
    this function or the CLI; it neither modifies nor resumes the primary study.
    """
    args = argparse.Namespace(study=study, workers=list(workers), seconds=seconds,
                              warmup_steps=warmup_steps, output=None if output is None else Path(output))
    if max_workers is not None and (max_workers < 8 or max_workers % 2):
        raise ValueError('max_workers must be even and at least 8')
    if scale_up:
        args.workers = [8]
    if not args.workers or not 0 < args.seconds <= 20 or any(n < 2 or n % 2 for n in args.workers):
        raise ValueError('Use positive even worker counts and at most 20 active sample seconds')
    if not 0 <= args.warmup_steps <= 50:
        raise ValueError('Warmup is limited to 0..50 Unity steps')
    from .runner import is_active
    study = ROOT/'logs/v12'/args.study
    if study.parent != ROOT/'logs/v12' or not study.is_dir():
        raise ValueError('Choose an existing v12 study name')
    if is_active(study):
        raise RuntimeError('Stop the study and confirm shutdown before probing')
    if probe_arm not in (None, 'Q0', 'Q1', 'Q2'):
        raise ValueError('probe_arm must be Q0, Q1 or Q2')
    templates, sources, sample_config = _templates(study, probe_arm)
    collector_workload = probe_arm == 'Q2'
    output = args.output or ROOT/'logs/v12'/('resource_tuning_'+time.strftime('%Y%m%d_%H%M%S'))
    output = output.resolve()
    if output.parent != (ROOT/'logs/v12').resolve() or not output.name.startswith('resource_tuning_'):
        raise ValueError('Probe output must be logs/v12/resource_tuning_<id>')
    output.mkdir(exist_ok=False)
    provenance = dict(schema='v12.resource_tuning.1', purpose='actual Unity throughput only',
        synthetic_only=False, throughput_only=True, training_started=False,
        evaluation_evidence=False, primary_study_resumed=False, partial_trajectories_discarded=True,
        study=str(study), sources=sources, source_file_sha256=file_hash(__file__),
        v12_source_sha256={path.relative_to(ROOT).as_posix():file_hash(path)
                          for path in sorted(Path(__file__).parent.glob('*.py'))},
        workers=args.workers, seconds=args.seconds, warmup_steps=args.warmup_steps,
        arms=[task['arm'] for task in templates], actor_threads=1, actor_device='cpu',
        scale_up=scale_up, maximum_workers=max_workers, memory_reserve_gib=4,
        plateau_stop_rule='Stop after two consecutive candidates below 102% of prior measured best, or one throughput drop over 20%',
        sample_config=sample_config,
        measured_workload=('Q2 data policy/scripted opponent with collector event copies, critic features, rewards and trip ledger; no disk trajectories'
                           if collector_workload else 'Exact evaluation policy/opponent action loop; reward/ledger aggregation excluded'),
        terminal_handling='Restart same-seed episodes; reset/hash-check costs included in aggregate throughput',
        limitation='Single short same-seed workload sweep; excludes natural-terminal serialization and optimizer; not a global optimum or match-quality measurement')
    atomic_json(output/'provenance.json', provenance)
    with FileLease(ROOT/'logs/runtime/compute.lock', blocking=False):
        if is_active(study):
            raise RuntimeError('Study became active before compute lease acquisition')
        reports = []; checks = []; consecutive_plateaus = 0
        stop_reason = 'maximum_requested_workers_reached' if scale_up else 'all_requested_candidates_measured'
        candidates = (itertools.count(8, 2) if max_workers is None else range(8, max_workers+1, 2)) if scale_up else args.workers
        for workers in candidates:
            check = _memory_check(workers, reports); checks.append(check)
            atomic_json(output/'memory_checks.json', checks)
            if not check['allowed']:
                stop_reason = 'safe_resource_bound: next candidate would consume the reserved desktop RAM'
                if not reports:
                    raise RuntimeError(f'Insufficient safe capacity for {workers} workers: {check}')
                break
            prior_best = max((row['aggregate_env_steps_per_second'] for row in reports), default=0.)
            # Each candidate closes its own Job, preventing failed Unity children
            # from leaking into the next candidate or surviving this probe.
            with process_tree_guard():
                row = _candidate(output, templates, workers, args.seconds, args.warmup_steps,
                                 collector_workload=collector_workload)
            row['improvement_over_prior_best_fraction'] = (row['aggregate_env_steps_per_second']/prior_best-1
                                                          if prior_best else None)
            row['memory_check_before'] = check
            reports.append(row)
            consecutive_plateaus = (consecutive_plateaus+1 if prior_best and
                                    row['aggregate_env_steps_per_second'] < 1.02*prior_best else 0)
            if scale_up and prior_best and row['aggregate_env_steps_per_second'] < .8*prior_best:
                stop_reason = 'severe_throughput_regression_over_20_percent'
                break
            if scale_up and consecutive_plateaus >= 2:
                stop_reason = 'two_consecutive_candidates_improved_less_than_2_percent'
                break
        if (len({row['initial_map_hash'] for row in reports}) != 1 or
                len({row['initial_observation_sha256'] for row in reports}) != 1):
            raise ValueError('Worker candidates used different initial observations')
        report = dict(provenance, candidates=[{key:value for key,value in row.items()
            if key not in ('workers_detail', 'gpu_samples')} for row in reports],
            memory_checks=checks, consecutive_plateaus=consecutive_plateaus, stop_reason=stop_reason,
            **select_workers(reports))
        atomic_json(output/'summary.json', report)
    print(json.dumps(dict(output=str(output), summary_path=str(output/'summary.json'),
        **select_workers(reports)), ensure_ascii=False), flush=True)
    return output/'summary.json'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', default='joint_trip_v12')
    parser.add_argument('--workers', nargs='+', type=int, default=[4, 6, 8])
    parser.add_argument('--seconds', type=float, default=12.)
    parser.add_argument('--scale-up', action='store_true', help='Adaptively measure 8,10,... workers')
    parser.add_argument('--max-workers', type=int, help='Optional explicit even scale-up ceiling')
    parser.add_argument('--probe-arm', choices=('Q0', 'Q1', 'Q2'),
                        help='Q2 measures the current data collector workload against a scripted opponent')
    parser.add_argument('--warmup-steps', type=int, default=20)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--task', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        return _worker(args.task)
    else:
        summary_path = run(args.study, args.workers, args.seconds, args.warmup_steps, args.output,
                           scale_up=args.scale_up, max_workers=args.max_workers, probe_arm=args.probe_arm)
        summary = read_json(summary_path)
        return dict(output=str(summary_path.parent), summary_path=str(summary_path),
                    selected_workers=summary['selected_workers'])


if __name__ == '__main__':
    main()
