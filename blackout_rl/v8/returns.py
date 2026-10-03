"""Raw reward units; bootstrap on truncation, never across a reset trace."""
import torch
from .contracts import ContractError

def gae(reward, value, next_value, terminated, truncated, gamma=.9995, lam=.99):
    if not all(x.shape == reward.shape for x in (value, next_value, terminated, truncated)):
        raise ContractError('return tensor shapes differ')
    if bool((terminated & truncated).any()) or not 0 <= gamma <= 1 or not 0 <= lam <= 1:
        raise ContractError('invalid boundary/discount')
    if not all(bool(torch.isfinite(x).all()) for x in (reward, value, next_value)):
        raise ContractError('nonfinite return input')
    advantage = torch.empty_like(reward)
    carry = torch.zeros_like(reward[0])
    for i in range(len(reward)-1, -1, -1):
        delta = reward[i] + gamma * (~terminated[i]).float() * next_value[i] - value[i]
        carry = delta + gamma * lam * (~(terminated[i] | truncated[i])).float() * carry
        advantage[i] = carry
    return advantage, advantage + value
