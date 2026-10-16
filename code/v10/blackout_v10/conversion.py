"""Exact single-active-expert conversion; no optimizer state migration."""
import torch
from .policy import MyPolicy
from .io import file_hash


def convert_state(state):
    mask = state.get('expert_mask')
    if mask is None or mask.shape != (5,) or mask[0] != 0 or not torch.isneginf(mask[1:]).all():
        raise ValueError('only v9 A/B expert zero can be converted exactly')
    if not bool(state['sample_actions']): raise ValueError('sampling decoder required')
    model = MyPolicy()
    ignored = {'expert_mask', 'experts.weight', 'experts.bias', 'gate.weight', 'gate.bias',
               'auxiliary.weight', 'auxiliary.bias'}
    converted = {k: v.clone() for k, v in state.items() if k not in ignored}
    converted['action_head.weight'] = state['experts.weight'][:9].clone()
    converted['action_head.bias'] = state['experts.bias'][:9].clone()
    model.load_state_dict(converted, strict=True)
    return model.eval()


def convert_checkpoint(path, expected_sha256, source, expected_source_sha256):
    if file_hash(path) != expected_sha256 or file_hash(source) != expected_source_sha256:
        raise ValueError('conversion checkpoint/source provenance mismatch')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('arm') not in ('A', 'B'): raise ValueError('v9 A/B checkpoint required')
    return convert_state(payload['policy_state'])
