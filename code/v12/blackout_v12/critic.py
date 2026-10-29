"""Small training-only critic, independent of the deployed scorer."""
import torch
from torch import nn


def critic_features(features, legal):
    weight = legal.to(features.dtype).unsqueeze(-1)
    return (features * weight).sum(dim=(-3, -2)) / weight.sum(dim=(-3, -2)).clamp_min(1)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(64, 64), nn.Tanh(),
                                 nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, features):
        return self.net(features).squeeze(-1)
