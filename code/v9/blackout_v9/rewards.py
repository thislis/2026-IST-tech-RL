"""Training-only bounded rewards. Never mutates an environment reward."""
import numpy as np

OBJECTIVE = "provided_runner_v1"


def potential(vector):
    vector = np.asarray(vector)
    if vector.shape[-1] != 96 or not np.isfinite(vector).all():
        raise ValueError("invalid potential observation")
    return float(.25 * np.clip(.7 * (vector[..., 93] - vector[..., 94]).mean(), -1, 1))


def terminal_outcome(team_totals, side):
    totals = np.asarray(team_totals, dtype=np.float64)
    if totals.shape != (2,) or not np.isfinite(totals).all():
        raise ValueError("nonfinite/incomplete raw team totals")
    return float(np.sign(totals[side] - totals[1-side]))


def shape_episode(potentials, outcome, gamma=1.0):
    """T pre-action potentials; verified natural termination is mandatory."""
    phi = np.asarray(potentials, dtype=np.float64)
    if phi.ndim != 1 or not len(phi) or not np.isfinite(phi).all() or np.abs(phi).max() > .250001:
        raise ValueError("invalid episode potential")
    if outcome not in (-1, 0, 1) or gamma not in (1.0, .99995):
        raise ValueError("invalid objective/gamma")
    following = np.append(phi[1:], 0.)
    rewards = gamma * following - phi
    rewards[-1] += outcome
    discounted = float(np.dot(gamma ** np.arange(len(phi)), rewards))
    expected = gamma ** (len(phi)-1) * outcome - phi[0]
    if abs(discounted-expected) > 1e-8:
        raise ArithmeticError("potential telescope failed")
    if gamma == 1 and abs(discounted) > 1.250001:
        raise ArithmeticError("episode return bound failed")
    return rewards.astype(np.float32)


def gae_episode(rewards, values, gamma=1., lam=.9975):
    """One complete, naturally terminated episode. Never crosses episodes."""
    rewards, values = np.asarray(rewards), np.asarray(values)
    if rewards.shape != values.shape or rewards.ndim != 1 or not len(rewards):
        raise ValueError("invalid GAE inputs")
    advantages = np.empty_like(rewards, dtype=np.float32)
    carry, next_value = 0., 0.
    for i in range(len(rewards)-1, -1, -1):
        carry = float(rewards[i]) + gamma * next_value - float(values[i]) + gamma * lam * carry
        advantages[i] = carry
        next_value = float(values[i])
    return advantages, advantages + values
