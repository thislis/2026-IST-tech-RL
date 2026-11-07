"""Raw, weight-independent history. Index 0 is the immediately previous call."""
from typing import List
import torch
from torch import nn


class History(nn.Module):
    def __init__(self, lags=(0, 4, 8, 16, 32, 64)):
        super().__init__()
        if not lags or lags[0] != 0 or max(lags) > 64:
            raise ValueError('lags must start at zero and fit the 64-step buffer')
        self.register_buffer('indices', torch.tensor(lags, dtype=torch.long))

    def forward(self, token: torch.Tensor, past: torch.Tensor):
        all_frames = torch.cat((token.unsqueeze(-2), past), dim=-2)
        return all_frames.index_select(-2, self.indices)


class TemporalEncoder(nn.Module):
    def __init__(self, lags=(0, 4, 8, 16, 32, 64)):
        super().__init__()
        self.history = History(lags)
        self.token = nn.Sequential(nn.Linear(130, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.merge = nn.Sequential(nn.Linear(len(lags)*32, 64), nn.ReLU())

    def forward(self, token: torch.Tensor, past: torch.Tensor):
        lagged = self.history(token, past)
        return self.encode_lagged(lagged)

    def encode_lagged(self, lagged: torch.Tensor):
        token=lagged[...,0,:]
        lag = self.history.indices.to(token.dtype) / 64.
        age = (token[..., 118:119].unsqueeze(-2) - lagged[..., 118:119]).clamp(0., 420.) / 420.
        lag = lag + torch.zeros_like(age[..., 0])
        scaled=lagged.clone()
        scaled[...,118]=scaled[...,118]/420.
        scaled[...,119:121]=scaled[...,119:121]/1000.
        x = torch.cat((scaled.clamp(-10,10), age, lag.unsqueeze(-1)), -1)
        encoded = self.token(x) * lagged[..., 121:122]
        return self.merge(encoded.flatten(-2))


def advance(token, past):
    return torch.cat((token.unsqueeze(-2), past[..., :63, :]), -2)
