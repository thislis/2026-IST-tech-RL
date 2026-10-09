import math
import torch
from torch import nn
from .distributions import gated, flat
from .contracts import ContractError

class ResidualModel(nn.Module):
    def __init__(self, actor, mode='c1', width=64, initial_q=.1):
        super().__init__()
        if mode not in ('c1', 'flat') or width < 1 or not 0 < initial_q < 1:
            raise ContractError('invalid model configuration')
        if actor.model_config['recurrent_version'] != 'none':
            raise ContractError('v8 cached-feature contract requires frozen nonrecurrent encoder')
        self.actor_model = actor.requires_grad_(False).eval()
        feature_dim = int(actor.model_config['hidden_dim']) + 49
        self.mode = mode
        self.trunk = nn.Sequential(nn.Linear(feature_dim, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh())
        self.correction = nn.Linear(width, 8)
        self.gate = nn.Sequential(nn.Linear(5*width, width), nn.Tanh(), nn.Linear(width, 1))
        self.critic = nn.Sequential(nn.Linear(299, 256), nn.Tanh(), nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 1))
        nn.init.zeros_(self.correction.weight)
        nn.init.constant_(self.correction.bias, 0. if mode == 'c1' else math.log(initial_q/(40*(1-initial_q))))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, math.log(initial_q/(1-initial_q)) if mode == 'c1' else 0.)

    def distribution(self, features, valid):
        hidden = self.trunk(features)
        correction = self.correction(hidden).flatten(-2)
        scalar = self.gate(hidden.flatten(-2)).squeeze(-1)
        return (gated if self.mode == 'c1' else flat)(scalar, correction, valid)

    def train(self, mode=True):
        super().train(mode)
        self.actor_model.eval()
        return self
