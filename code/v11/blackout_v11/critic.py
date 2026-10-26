"""Independent observation/state critic; its gradients never enter the actor."""
import torch
from torch import nn


class Critic(nn.Module):
    def __init__(self, arm="C1"):
        super().__init__()
        if arm not in ("C0", "C1", "C1-blind"):
            raise ValueError("unknown critic arm")
        self.arm = arm
        self.vector = nn.Sequential(nn.Linear(96+15, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh())
        if arm != "C0":
            self.map = nn.Sequential(nn.Conv2d(11, 32, 4, 4), nn.GELU(), nn.Conv2d(32, 32, 3, 2, 1),
                                     nn.GELU(), nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Linear(512, 128), nn.Tanh())
        self.hidden = nn.Sequential(nn.Linear(128 if arm == "C0" else 256, 128), nn.Tanh())
        self.output = nn.Linear(128, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.register_buffer("mean", torch.tensor(0.))
        self.register_buffer("scale", torch.tensor(1.))

    @staticmethod
    def state_features(vectors, state):
        batch = len(vectors)
        parts = []
        for names, scale in ((('previous_goal', 'goal_cell'), 575.), (('previous_role', 'role'), 3.),
                             (('goal_age',), 25.)):
            value = next((state[n] for n in names if state is not None and n in state), None)
            if value is None:
                value = vectors.new_zeros((batch, 5))
            if value.ndim == 3:
                value = value[..., 0]
            parts.append(value.float().reshape(batch, 5) / scale)
        return torch.cat(parts, -1).clamp(-2, 2)

    def normalized(self, vectors, graphic=None, state=None):
        if vectors.ndim != 3 or vectors.shape[1:] != (5, 96):
            raise ValueError("critic requires [N,5,96]")
        vec = self.vector(torch.cat((vectors.mean(1), self.state_features(vectors, state)), -1))
        if self.arm != "C0":
            if graphic is None:
                raise ValueError("map critic requires graphic")
            if self.arm == "C1-blind":
                graphic = torch.zeros_like(graphic)
            vec = torch.cat((vec, self.map(graphic)), -1)
        return self.output(self.hidden(vec)).squeeze(-1)

    def forward(self, vectors, graphic=None, state=None):
        return self.normalized(vectors, graphic, state)*self.scale + self.mean

    @torch.no_grad()
    def update_scale(self, targets, beta=.99):
        if not torch.isfinite(targets).all():
            raise ValueError("nonfinite PopArt targets")
        new_mean = beta*self.mean + (1-beta)*targets.mean()
        second = beta*(self.scale.square()+self.mean.square()) + (1-beta)*targets.square().mean()
        new_scale = (second-new_mean.square()).clamp_min(1e-4).sqrt()
        self.output.weight.mul_(self.scale/new_scale)
        self.output.bias.copy_((self.scale*self.output.bias+self.mean-new_mean)/new_scale)
        self.mean.copy_(new_mean)
        self.scale.copy_(new_scale)
