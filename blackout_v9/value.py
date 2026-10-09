"""Own-team public-observation critic with output-preserving PopArt."""
import torch
from torch import nn


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(96, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh())
        self.output = nn.Linear(128, 1)
        self.register_buffer("mean", torch.tensor(0.))
        self.register_buffer("scale", torch.tensor(1.))

    def normalized(self, vectors):
        return self.output(self.body(vectors.mean(-2))).squeeze(-1)

    def forward(self, vectors):
        return self.normalized(vectors) * self.scale + self.mean

    @torch.no_grad()
    def update_scale(self, targets, beta=.99):
        new_mean = beta * self.mean + (1-beta) * targets.mean()
        second = beta * (self.scale.square()+self.mean.square()) + (1-beta)*targets.square().mean()
        new_scale = (second-new_mean.square()).clamp_min(1e-4).sqrt()
        self.output.weight.mul_(self.scale / new_scale)
        self.output.bias.copy_((self.scale*self.output.bias + self.mean-new_mean) / new_scale)
        self.mean.copy_(new_mean)
        self.scale.copy_(new_scale)
