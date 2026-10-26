"""Bounded potential shaping and variable-duration, complete-episode GAE."""
import numpy as np

OBJECTIVE = "provided_runner_v1"


def potential(vector):
    vector = np.asarray(vector)
    if vector.shape[-1] != 96 or not np.isfinite(vector).all():
        raise ValueError("invalid potential observation")
    return float(.25 * np.clip((vector[..., 93] - vector[..., 94]).mean(), -1, 1))


def terminal_outcome(team_totals, side):
    totals = np.asarray(team_totals, dtype=np.float64)
    if totals.shape != (2,) or not np.isfinite(totals).all() or side not in (0, 1):
        raise ValueError("invalid runner outcome")
    return float(np.sign(totals[side] - totals[1-side]))


def shape_episode(potentials, durations, outcome, gamma=1.):
    """Each reward covers exactly one interval until the next decision/terminal.

    Potential differences telescope over primitive steps, including variable
    interval lengths. Terminal outcome is paid on the final primitive step.
    """
    phi = np.asarray(potentials, np.float64)
    k = np.asarray(durations, np.int64)
    if (phi.ndim != 1 or not len(phi) or k.shape != phi.shape or np.any(k < 1)
            or not np.isfinite(phi).all() or np.abs(phi).max() > .250001):
        raise ValueError("invalid event potential/duration")
    if outcome not in (-1, 0, 1) or not 0 < gamma <= 1:
        raise ValueError("invalid outcome/gamma")
    rewards = gamma ** k * np.append(phi[1:], 0.) - phi
    rewards[-1] += gamma ** (k[-1]-1) * outcome
    weights = gamma ** np.r_[0, np.cumsum(k[:-1])]
    expected = gamma ** (k.sum()-1) * outcome - phi[0]
    if abs(float(np.dot(weights, rewards))-expected) > 1e-8:
        raise ArithmeticError("potential telescope failed")
    return rewards.astype(np.float32)


def gae_episode(rewards, values, durations=None, gamma=1., lam=.9975):
    r, v = np.asarray(rewards, np.float64), np.asarray(values, np.float64)
    k = np.ones(len(r), np.int64) if durations is None else np.asarray(durations, np.int64)
    if r.ndim != 1 or not len(r) or v.shape != r.shape or k.shape != r.shape or np.any(k < 1):
        raise ValueError("invalid complete event sequence")
    if not np.isfinite(r).all() or not np.isfinite(v).all():
        raise ValueError("nonfinite return input")
    advantages = np.empty(len(r), np.float32)
    carry, next_value = 0., 0.
    for i in range(len(r)-1, -1, -1):
        delta = r[i] + gamma ** int(k[i]) * next_value - v[i]
        carry = delta + (gamma * lam) ** int(k[i]) * carry
        advantages[i], next_value = carry, v[i]
    return advantages, (advantages.astype(np.float64) + v).astype(np.float32)
