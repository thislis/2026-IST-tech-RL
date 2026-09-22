import math
import numpy as np
import torch
from torch import nn
from blackout_rl.mappo_v6 import CENTRAL_SIZE

DIRECTIONS = np.array([(0.,0.)] + [(math.cos(i*math.pi/4),math.sin(i*math.pi/4)) for i in range(8)],np.float32)


class ReadoutModel(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.readout = nn.Linear(feature_dim,9)
        self.critic = nn.Sequential(nn.Linear(CENTRAL_SIZE,256),nn.Tanh(),nn.Linear(256,128),nn.Tanh(),nn.Linear(128,1))
        nn.init.zeros_(self.readout.weight); nn.init.zeros_(self.readout.bias)

    def distribution(self, features):
        return torch.distributions.Independent(torch.distributions.Categorical(logits=self.readout(features)),1)


def fixed_decode(rates, names, headings, *, dt=.02, turn_gain=40., turn_limit=2., movement_threshold=.02):
    left = [i for i,n in enumerate(names) if n.startswith(('DNa02','DNg13')) and n.endswith(':L')]
    right = [i for i,n in enumerate(names) if n.startswith(('DNa02','DNg13')) and n.endswith(':R')]
    forward = [i for i,n in enumerate(names) if n.startswith('DNg100')]
    if not left or not right or not forward:
        raise ValueError('fixed decoder needs bilateral steering and a movement pool')
    turn = np.clip(turn_gain*(rates[:,right].mean(-1)-rates[:,left].mean(-1)),-turn_limit,turn_limit)
    headings = (headings+turn*dt) % (2*math.pi)
    action = 1+(np.rint(headings/(math.pi/4)).astype(int)%8)
    action[rates[:,forward].mean(-1) <= movement_threshold] = 0
    return action, headings
