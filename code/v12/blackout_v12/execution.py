"""Separate scheduling resources from the immutable learning/match contract."""
import copy


RUNTIME_KEYS = frozenset({
    'workers', 'workers_per_run', 'worker_cap', 'parallel_runs', 'parallel_train_runs',
    'actor_device', 'planner_device', 'learner_device', 'device', 'actor_threads',
    'learner_threads', 'cpu_reserve', 'cpu_budget_basis', 'memory_reserve_gib',
    'worker_memory_gib', 'disk_reserve_gib', 'stop_grace_seconds', 'stop_file',
    'benchmark_repeats', 'benchmark_batches', 'benchmark_cpu_threads',
    'benchmark_worker_seconds', 'measured_worker_cap', 'measured_fastest_actor_device',
    'resource_profile_sha256', 'execution_revision',
    'worker_allocation_file', 'evaluation_worker_capacity', 'data_collection_workers',
    'tune_on_resume', 'resume_tuning_candidates', 'resume_tuning_seconds',
})


def semantic_config(config):
    """Keep PPO minibatch/epochs/seeds/wave size; ignore only execution resources.

    The original implementation used workers_per_run as the episode wave
    size. Preserve that interpretation for old resolved configurations.
    """
    result = {key:copy.deepcopy(value) for key,value in config.items()
              if key not in RUNTIME_KEYS}
    result['rollout_episodes_per_wave'] = int(config.get(
        'rollout_episodes_per_wave', config.get('workers_per_run', 2)))
    return result


def _identity(value, config=False):
    if isinstance(value, dict):
        source = semantic_config(value) if config else value
        return {key:_identity(item, config=(key=='config')) for key,item in source.items()
                if config or key not in ('workers', 'execution_revision')}
    if isinstance(value, (list,tuple)):
        return [_identity(item) for item in value]
    return value


def compatible_manifest(previous, current):
    return _identity(previous) == _identity(current)


def schedule_allocation(tasks, config):
    """Allocate every available slot without changing per-policy wave size.

    Evaluations and fixed-policy data episodes can use the full allocation.
    Learning jobs cannot use more workers than their fixed episode wave, so
    additional independent arms/seeds may run alongside them instead.
    """
    if not tasks:
        return []
    total = int(config['workers'])
    if total < 1:
        raise ValueError('positive global worker budget required')
    training = all(task['kind'] in ('train','data') for task in tasks)
    wave_size = int(config.get('rollout_episodes_per_wave',2))
    parallel = int(config.get('parallel_train_runs',total) if training else config.get('parallel_runs',2))
    count = min(len(tasks), max(1,parallel), total)
    quotient, remainder = divmod(total, count)
    result = []
    for index, task in enumerate(tasks[:count]):
        assigned = quotient + int(index < remainder)
        if task['kind'] == 'train':
            assigned = min(assigned,wave_size)
        result.append(dict(task, workers=assigned))
    uncapped = [task for task in result if task['kind'] != 'train']
    if uncapped:
        quotient, remainder = divmod(total-sum(task['workers'] for task in result),len(uncapped))
        for index, task in enumerate(uncapped):
            task['workers'] += quotient + int(index < remainder)
    return result
