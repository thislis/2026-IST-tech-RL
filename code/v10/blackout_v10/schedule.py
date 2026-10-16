"""Counter-based tasks independent of PPO permutations, results, and critic RNG."""
import numpy as np
FAMILIES = ('noop', 'random', 'collector', 'rush', 'raider', 'defender', 'target')
PROBABILITIES = (.10, .10, .15, .15, .15, .15, .20)


def task_spec(seed, serial):
    rng = np.random.default_rng(np.random.SeedSequence([seed, serial, 1010]))
    return dict(side=serial % 2, requested_seed=int(rng.integers(1000, 40000)),
                action_seed=int(rng.integers(1, 2**31)),
                opponent={'kind': str(rng.choice(FAMILIES, p=PROBABILITIES))})


def task_plan(seeds, maximum):
    return {'schema': 'v10.counter-task.v1', 'seeds': list(seeds), 'maximum_attempts_per_run': maximum,
            'generator': 'numpy.default_rng(SeedSequence([seed,serial,1010]))',
            'probabilities': dict(zip(FAMILIES, PROBABILITIES)), 'history': False,
            'prefix': {str(s): [task_spec(s, i) for i in range(32)] for s in seeds},
            'identical_environment_trajectory_claimed': False}
