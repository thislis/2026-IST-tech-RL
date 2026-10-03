"""Frozen historical opponent implementation, including its planner overrides."""
import torch
from blackout_rl.frozen_opponent import FrozenScriptedOpponent
from blackout_rl.policy import DeterministicCheckpointPolicy
from blackout_rl.mappo_v6 import ROLES
from .contracts import ContractError,digest,file_hash

def make_opponent(kind,team,config):
    if kind=='target':
        if file_hash(config['path'])!=config['sha256']:raise ContractError('opponent checkpoint changed')
        with torch.random.fork_rng():policy=DeterministicCheckpointPolicy(config['path'],team=team)
    elif kind=='scripted':policy=FrozenScriptedOpponent(team)
    elif kind=='weak':policy=FrozenScriptedOpponent(team,roles=ROLES,chase_radius_cells=12,policy_id='weak-win70-v1')
    else:raise ContractError('unknown opponent')
    policy.reset()
    identity=digest(dict(kind=kind,checkpoint_sha256=config['sha256'] if kind=='target' else None,
                         source_sha256=config['source_sha256'],revision='historical_full_policy_v1'))
    return policy,identity
