"""The deployable scorer has exactly 6,273 trainable parameters."""
import torch
from torch import nn


class TaskScorer(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.policy_mode = dict(config or {}).get('policy_mode', 'residual')
        if self.policy_mode not in ('residual', 'learned'):
            raise ValueError('policy_mode must be residual or learned')
        self.layers = nn.Sequential(nn.Linear(64, 64), nn.ReLU(),
                                    nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, features, prior=None):
        residual = self.layers(features).squeeze(-1).clamp(-20.0, 20.0)
        return residual if prior is None or self.policy_mode == 'learned' else prior.clamp(-20.0, 20.0) + residual

    def hidden(self, features):
        """Optional training-only auxiliary heads can consume these 32 values."""
        return self.layers[:4](features)
