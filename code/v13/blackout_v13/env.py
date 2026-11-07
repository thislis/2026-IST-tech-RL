"""Explicit new-game launch only. Tests inject a fake environment instead."""
import numpy as np
from .io import ROOT


def make_env(config,slot):
    from agent_league.new_game_runtime import configure_runtime,make_env as create
    configure_runtime()
    task=dict(config={**config,'game_runtime':'artifacts/new_game/20261101/runtime.json'})
    return create(task,background=True,worker_id=300+slot)


def pack(observations,side):
    names=[f'unit_{i}' for i in range(side*5,side*5+5)]
    vector=np.stack([observations[n]['vector'] for n in names]).astype(np.float32)
    image=np.stack([observations[n]['graphic'] for n in names]).astype(np.float32)
    if image.shape==(5,96,96,11):image=image.transpose(0,3,1,2)
    return vector,np.ascontiguousarray(image),names


def outcome(infos,side):
    from agent_league.new_game_match import terminal_scores
    scores=terminal_scores(infos)
    return float(np.sign(scores[side]-scores[1-side])),scores
