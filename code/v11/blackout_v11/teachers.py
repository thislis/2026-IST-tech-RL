"""Observation-only projection of frozen teacher actions into legal v11 heads.

These are directional supervision labels, not privileged teacher goal IDs.
Actual collection/delivery outcomes remain the competence criterion.
"""
import torch
from .policy import own_entities


@torch.no_grad()
def label_choices(policy, vector, graphic, state, teacher_actions, base_action=None):
    del graphic
    n, device = len(vector), vector.device
    own = own_entities(vector)
    roles = state["role"].clone()
    target = teacher_actions.float()
    if "side_swap" in state:
        swap = state["side_swap"].bool()[:, None, None]
        own = own.clone()
        own[..., :2] = torch.where(swap, own[..., :2].flip(-1), own[..., :2])
        target = torch.where(swap, target.flip(-1), target)
        if base_action is not None:
            base_action = torch.where(swap, base_action.flip(-1), base_action)
    magnitude = target.square().sum(-1).sqrt()
    unit_target = target/magnitude.clamp_min(1e-8)[..., None]
    goals, cells_chosen = [], []
    count = state["candidate_cells"].shape[-1]
    for slot in range(5):
        role = roles[:, slot]
        gather = role[:, None, None].expand(-1, 1, count)
        cells = state["candidate_cells"][:, slot].gather(1, gather).squeeze(1)
        intents = state["candidate_intents"][:, slot].gather(1, gather).squeeze(1)
        legal = state["candidate_legal"][:, slot].gather(1, gather).squeeze(1).clone()
        prior = state["candidate_prior"][:, slot].gather(1, gather).squeeze(1)
        used = torch.zeros_like(cells)
        for before in range(slot):
            used += (cells == cells_chosen[before][:, None]).long()
        for later in range(slot+1, 5):
            used += ((cells == state["previous_goal"][:, later, None]) & ~state["goal_event"][:, later, None]).long()
        capacity = torch.where((intents == 0) | (intents == 2), torch.ones_like(cells), torch.full_like(cells, 2))
        legal &= used < capacity
        exhausted = ~legal.any(-1)
        current_xy = (own[:, slot, :2]*24).long().clamp(0, 23)
        current_cell = current_xy[:, 1]*24+current_xy[:, 0]
        cells[:, -1] = torch.where(exhausted, current_cell, cells[:, -1])
        legal[:, -1] |= exhausted
        xy = torch.stack(((cells % 24 + .5)/24, (cells // 24 + .5)/24), -1)
        displacement = xy-own[:, slot, None, :2]
        direction = displacement/displacement.square().sum(-1).sqrt().clamp_min(1e-8)[..., None]
        agreement = (direction*unit_target[:, slot, None]).sum(-1)
        # Geometric feasibility/intent prior dominates; a teacher resolves local
        # ambiguity without turning unknown inventory/hidden goals into inputs.
        rank = prior+2*agreement*(magnitude[:, slot, None] > .05)
        chosen = rank.masked_fill(~legal, -1e9).argmax(-1)
        chosen = torch.where(state["goal_event"][:, slot], chosen, torch.full_like(chosen, count-1))
        goals.append(chosen)
        cell = cells.gather(-1, chosen[:, None]).squeeze(-1)
        cells_chosen.append(torch.where(state["goal_event"][:, slot], cell, state["previous_goal"][:, slot]))
    if base_action is None:
        # Safe teacher warm-up fallback; collector normally supplies actual
        # tensor-planner movement so residual labels use execution semantics.
        base_action = state["previous_action"]
    alternatives = torch.stack([policy._correct(base_action, torch.full((n, 5), i, device=device, dtype=torch.long))
                                for i in range(8)], 2)
    errors = (alternatives-target[:, :, None]).square().sum(-1)
    best_error, correction = errors.min(-1)
    base_error = (base_action-target).square().sum(-1)
    benefit = base_error-best_error
    active = state["residual_event"]
    held = (state["correction_before"] >= 0) & ~active
    gate = held.long()
    remaining = (policy.max_interventions-held.sum(-1)).clamp_min(0)
    for rank in range(5):
        scores = benefit.masked_fill(~active | (gate > 0), -1e9)
        value, slot = scores.max(-1)
        select = (value > .05) & (remaining > 0)
        gate.scatter_(1, slot[:, None], (gate.gather(1, slot[:, None]).squeeze(1) | select.long())[:, None])
        remaining -= select.long()
    residual_gain = benefit.masked_fill(~active, -1e9)
    gain, best_slot = residual_gain.max(-1)
    residual_corr = correction.gather(1, best_slot[:, None]).squeeze(1)
    residual = torch.where(gain > .05, 1+best_slot*8+residual_corr, torch.zeros_like(best_slot))
    direct_error = (policy.directions[None, None]-target[:, :, None]).square().sum(-1)
    direct = torch.where(active, direct_error.argmin(-1), state["direct_before"])
    return dict(role=roles, goal=torch.stack(goals, 1), gate=gate, correction=correction,
                residual=residual, direct=direct)
