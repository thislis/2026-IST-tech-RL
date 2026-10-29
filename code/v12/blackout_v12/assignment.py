"""Exact capacity-constrained assignments on at most five units.

A task of capacity two remains one task: subsets, never cloned task slots,
define the Gibbs support. Invalid intermediate states carry a separate mask,
so autograd never differentiates logsumexp of an entirely infinite row.
Leading batch dimensions are supported by every probability operation.
"""
from functools import lru_cache
import torch


@lru_cache(maxsize=32)
def _topology(units, device_string):
    # Cached indices can first be requested by an inference-mode actor and
    # later reused by an autograd learner in the same process. Always create
    # ordinary tensors: inference tensors cannot be saved for backward.
    with torch.inference_mode(False):
        device = torch.device(device_string)
        subsets = torch.arange(1 << units, device=device)
        bits = ((subsets[:, None] >> torch.arange(units, device=device)) & 1).bool()
        contained = (subsets[:, None] & subsets[None, :]) == subsets[None, :]
        previous = subsets[:, None] ^ subsets[None, :]
        return bits, bits.sum(-1), contained, previous


def _inputs(scores, capacities, legal, temperature):
    if scores.ndim < 2 or not 1 <= scores.shape[-2] <= 5:
        raise ValueError('scores must have shape [..., units<=5, tasks]')
    if not scores.is_floating_point():
        raise TypeError('scores must be floating point')
    shape = scores.shape[:-2]
    units, tasks = scores.shape[-2:]
    if tasks < 1:
        raise ValueError('at least one task is required')
    legal = torch.as_tensor(legal, device=scores.device, dtype=torch.bool)
    caps = torch.as_tensor(capacities, device=scores.device, dtype=torch.long)
    legal = torch.broadcast_to(legal, scores.shape).reshape(-1, units, tasks)
    caps = torch.broadcast_to(caps, (*shape, tasks)).reshape(-1, tasks)
    tau = torch.as_tensor(temperature, device=scores.device, dtype=scores.dtype)
    tau = torch.broadcast_to(tau, shape).reshape(-1, 1, 1)
    if not torch.isfinite(scores).all() or not torch.isfinite(tau).all() or (tau <= 0).any():
        raise ValueError('finite scores and positive finite temperature are required')
    if (caps < 0).any() or (caps > units).any():
        raise ValueError('capacities must lie between zero and the team size')
    return scores.reshape(-1, units, tasks) / tau, caps, legal, shape


def _reduce(values, valid, maximum=False):
    any_valid = valid.any(-1)
    masked = values.masked_fill(~valid, -torch.inf)
    peak = masked.max(-1).values
    if maximum:
        return torch.where(any_valid, peak, 0.), any_valid
    peak = torch.where(any_valid, peak, 0.)
    mass = torch.exp(masked - peak.unsqueeze(-1)).sum(-1)
    value = peak + mass.clamp_min(torch.finfo(values.dtype).tiny).log()
    return torch.where(any_valid, value, 0.), any_valid


def _tables(scores, capacities, legal, temperature=1.0, maximum=False):
    scaled, caps, legal, shape = _inputs(scores, capacities, legal, temperature)
    batch, units, tasks = scaled.shape
    bits, count, contained, previous = _topology(units, str(scores.device))
    weights = torch.einsum('bum,au->bma', scaled, bits.to(scaled.dtype))
    # b,m,a: every member of subset a must be eligible for task m.
    subset_legal = ~((~legal.transpose(1, 2)).unsqueeze(2) & bits[None, None]).any(-1)
    subset_legal = subset_legal & (count[None, None] <= caps[:, :, None])
    value = scaled.new_zeros((batch, 1 << units))
    valid = torch.zeros_like(value, dtype=torch.bool)
    valid[:, 0] = True
    history, validity = [value], [valid]
    for task in range(tasks):
        candidates = value[:, previous] + weights[:, task, None, :]
        ok = contained[None] & valid[:, previous] & subset_legal[:, task, None, :]
        value, valid = _reduce(candidates, ok, maximum)
        history.append(value)
        validity.append(valid)
    if not valid[:, -1].all():
        raise ValueError('no complete legal assignment; include private holds or repair locked capacity')
    return history, validity, weights, subset_legal, shape


def log_partition(scores, capacities, legal, temperature=1.0):
    history, _, _, _, shape = _tables(scores, capacities, legal, temperature)
    return history[-1][:, -1].reshape(shape)


def _backtrack(scores, capacities, legal, temperature, sample, generator=None):
    history, validity, weights, subset_legal, shape = _tables(
        scores, capacities, legal, temperature, maximum=not sample)
    units, tasks = scores.shape[-2:]
    bits, _, contained, previous = _topology(units, str(scores.device))
    batch = weights.shape[0]
    state = torch.full((batch,), (1 << units) - 1, device=scores.device, dtype=torch.long)
    assignments = torch.full((batch, units), -1, device=scores.device, dtype=torch.long)
    row = torch.arange(batch, device=scores.device)
    for task in range(tasks - 1, -1, -1):
        pred = previous[state]
        values = history[task].gather(1, pred) + weights[:, task]
        valid = contained[state] & validity[task].gather(1, pred) & subset_legal[:, task]
        values = values.masked_fill(~valid, -torch.inf)
        subset = (torch.multinomial(values.softmax(-1), 1, generator=generator).squeeze(1)
                  if sample else values.argmax(-1))
        assignments = torch.where(bits[subset], task, assignments)
        state = pred[row, subset]
    return assignments.reshape(*shape, units)


@torch.no_grad()
def map_assignment(scores, capacities, legal, temperature=1.0, unit_ids=None, task_ids=None):
    """Stable MAP; optional physical IDs canonicalize exact ties before solving."""
    if unit_ids is None and task_ids is None:
        return _backtrack(scores, capacities, legal, temperature, False)
    units, tasks = scores.shape[-2:]
    u = sorted(range(units), key=lambda i: unit_ids[i]) if unit_ids is not None else list(range(units))
    t = sorted(range(tasks), key=lambda i: task_ids[i]) if task_ids is not None else list(range(tasks))
    ui = torch.tensor(u, device=scores.device)
    ti = torch.tensor(t, device=scores.device)
    eligibility = torch.broadcast_to(torch.as_tensor(legal, device=scores.device), scores.shape)
    result = _backtrack(scores.index_select(-2, ui).index_select(-1, ti),
                        torch.as_tensor(capacities, device=scores.device).index_select(-1, ti),
                        eligibility.index_select(-2, ui).index_select(-1, ti), temperature, False)
    return ti[result].index_select(-1, ui.argsort())


@torch.no_grad()
def sample_assignment(scores, capacities, legal, temperature=1.0, generator=None):
    return _backtrack(scores, capacities, legal, temperature, True, generator)


def joint_log_prob(scores, assignment, capacities, legal, temperature=1.0):
    scaled, caps, mask, shape = _inputs(scores, capacities, legal, temperature)
    units, tasks = scores.shape[-2:]
    assignment = torch.as_tensor(assignment, device=scores.device, dtype=torch.long)
    assignment = torch.broadcast_to(assignment, (*shape, units)).reshape(-1, units)
    if (assignment < 0).any() or (assignment >= tasks).any():
        raise ValueError('assignment contains an out-of-range task')
    chosen = assignment.unsqueeze(-1)
    eligible = mask.gather(-1, chosen).squeeze(-1).all(-1)
    usage = torch.nn.functional.one_hot(assignment, tasks).sum(-2)
    valid = eligible & (usage <= caps).all(-1)
    value = scaled.gather(-1, chosen).sum((-2, -1))
    probability = value.reshape(shape) - log_partition(scores, capacities, legal, temperature)
    return probability.masked_fill(~valid.reshape(shape), -torch.inf)


def marginals(scores, capacities, legal, temperature=1.0, create_graph=False):
    """Exact edge probabilities from the derivative of the log partition."""
    with torch.enable_grad():
        values = scores if scores.requires_grad else scores.detach().requires_grad_(True)
        partition = log_partition(values, capacities, legal, temperature)
        gradient, = torch.autograd.grad(partition.sum(), values, create_graph=create_graph)
        tau = torch.as_tensor(temperature, device=values.device, dtype=values.dtype)
        return gradient * tau[..., None, None]


def joint_entropy(scores, capacities, legal, temperature=1.0, create_graph=True):
    p = marginals(scores, capacities, legal, temperature, create_graph=create_graph)
    tau = torch.as_tensor(temperature, device=scores.device, dtype=scores.dtype)
    return log_partition(scores, capacities, legal, temperature) - (p * scores).sum((-2, -1)) / tau


def _sequential(scores, capacities, legal, temperature, assignment=None, sample=False, generator=None):
    scaled, caps, mask, shape = _inputs(scores, capacities, legal, temperature)
    units, tasks = scores.shape[-2:]
    caps = caps.clone()
    decisions, log_probs = [], []
    if assignment is not None:
        assignment = torch.as_tensor(assignment, device=scores.device, dtype=torch.long).reshape(-1, units)
    for unit in range(units):
        # Check feasibility for the remaining rows. This matters for commitments:
        # an earlier row cannot consume a later locked unit's only task.
        ok = mask[:, unit] & (caps > 0)
        if unit + 1 < units:
            remaining = caps[:, None, :] - torch.eye(tasks, device=caps.device, dtype=caps.dtype)[None]
            tail = mask[:, None, unit + 1:, :].expand(-1, tasks, -1, -1)
            feasible = _support_feasible(tail.reshape(-1, units-unit-1, tasks),
                                         remaining.reshape(-1, tasks))
            ok &= feasible.reshape(-1, tasks)
        if not ok.any(-1).all():
            raise ValueError('sequential assignment has no feasible completion')
        logits = scaled[:, unit].masked_fill(~ok, -torch.inf)
        choice = (assignment[:, unit] if assignment is not None else
                  torch.multinomial(logits.softmax(-1), 1, generator=generator).squeeze(-1)
                  if sample else logits.argmax(-1))
        if (choice < 0).any() or (choice >= tasks).any():
            raise ValueError('assignment contains an out-of-range task')
        decisions.append(choice)
        log_probs.append(logits.log_softmax(-1).gather(1, choice[:, None]).squeeze(-1))
        caps = caps - torch.nn.functional.one_hot(choice, tasks)
    return torch.stack(decisions, -1).reshape(*shape, units), torch.stack(log_probs, -1).sum(-1).reshape(shape)


def _support_feasible(mask, capacities):
    """Hall's capacity condition on every subset of remaining units (<=16)."""
    units = mask.shape[-2]
    bits, count, _, _ = _topology(units, str(mask.device))
    neighbours = (mask[:, None] & bits[None, :, :, None]).any(-2)
    room = (neighbours * capacities.clamp_min(0)[:, None]).sum(-1)
    return (room >= count[None]).all(-1) & (capacities >= 0).all(-1)


@torch.no_grad()
def sequential_assignment(scores, capacities, legal, temperature=1.0, sample=False, generator=None):
    return _sequential(scores, capacities, legal, temperature, sample=sample, generator=generator)[0]


def sequential_log_prob(scores, assignment, capacities, legal, temperature=1.0):
    return _sequential(scores, capacities, legal, temperature, assignment=assignment)[1]
