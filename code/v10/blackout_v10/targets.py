"""Validated episode -> frozen-critic microbatches -> committed GAE targets."""
import time
from pathlib import Path
import numpy as np
import torch
from .io import atomic_json, file_hash, state_hash
from .trajectory import Episode
from .value import Critic
from .rewards import gae_episode


def commit_targets(paths, snapshot, expected_hash, cfg, device, progress=lambda **_: None, stop=lambda: False):
    if file_hash(snapshot) != expected_hash: raise ValueError('frozen behavior checksum mismatch')
    payload = torch.load(snapshot, map_location='cpu', weights_only=True)
    with torch.random.fork_rng():
        critic = Critic(payload['critic_arm']).to(device).eval()
    critic.load_state_dict(payload['critic_state'])
    critic_hash = state_hash(payload['critic_state'])
    started = time.monotonic(); count = 0
    with torch.inference_mode():
        for path in paths:
            ep = Episode(path, validated=True)
            if ep.meta['policy_sha256'] != expected_hash: raise ValueError('episode behavior mismatch')
            values = np.empty(ep.length, np.float32)
            for start in range(0, ep.length, cfg['value_batch']):
                if stop(): raise InterruptedError('target construction interrupted before optimizer')
                ids = np.arange(start, min(ep.length, start+cfg['value_batch']))
                v = torch.from_numpy(np.array(ep.arrays['vectors'][ids])).to(device)
                g = torch.from_numpy(np.ascontiguousarray(ep.graphic(ids))).to(device)
                values[ids] = critic(v, g).cpu().numpy()
                progress(phase='targets', target_steps=count+int(ids[-1])+1)
            if not np.isfinite(values).all(): raise ArithmeticError('nonfinite frozen value')
            advantages, returns = gae_episode(ep.arrays['rewards'], values, cfg['gamma'], cfg['gae_lambda'])
            for key, value in (('values', values), ('advantages', advantages), ('returns', returns)):
                np.save(Path(path)/(key+'.npy'), value, allow_pickle=False)
            atomic_json(Path(path)/'committed.json', dict(ep.meta, critic_sha256=critic_hash,
                behavior_sha256=expected_hash, targets_complete=True))
            count += ep.length
    return {'steps': count, 'seconds': time.monotonic()-started, 'critic_sha256': critic_hash,
            'behavior_sha256': expected_hash, 'device': str(device), 'value_batch': cfg['value_batch']}
