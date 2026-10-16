"""Independent vector/map critics with output-preserving PopArt."""
import torch
from torch import nn
from .policy import MyPolicy


class Critic(nn.Module):
    def __init__(self, arm='C1'):
        super().__init__()
        if arm not in ('C0', 'C1', 'C1-blind'):
            raise ValueError('unsupported critic arm')
        self.arm = arm
        if arm == 'C0':
            self.body = nn.Sequential(nn.Linear(96, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh())
        else:
            self.encoder = MyPolicy()
            del self.encoder.action_head
            self.body = nn.Sequential(nn.Linear(64, 128), nn.Tanh())
        self.output = nn.Linear(128, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.register_buffer('mean', torch.tensor(0.))
        self.register_buffer('scale', torch.tensor(1.))

    def normalized(self, vectors, graphic=None):
        if vectors.ndim != 3 or vectors.shape[1:] != (5, 96):
            raise ValueError('critic requires complete team vectors')
        if not (torch.equal(vectors[:, :, :90], vectors[:, :1, :90].expand(-1, 5, -1)) and
                torch.equal(vectors[:, :, 93:], vectors[:, :1, 93:].expand(-1, 5, -1))):
            raise ValueError('critic received unshared scene')
        if self.arm == 'C0':
            features = vectors.mean(1)
        else:
            self.encoder.validate(vectors[:, 0], graphic)
            if self.arm == 'C1-blind': graphic = torch.zeros_like(graphic)
            latent = self.encoder.scene(vectors[:, 0], graphic)
            context = torch.cat((vectors[:, :, 90:93].mean(1), vectors[:, 0, 93:]), -1)[:, None]
            query = self.encoder.query(context)
            features = self.encoder.readout_norm(query+self.encoder.readout(query, latent))[:, 0]
        return self.output(self.body(features)).squeeze(-1)

    def forward(self, vectors, graphic=None):
        return self.normalized(vectors, graphic)*self.scale+self.mean

    @torch.no_grad()
    def update_scale(self, targets, beta=.99):
        if not torch.isfinite(targets).all(): raise ValueError('nonfinite PopArt target')
        new_mean = beta*self.mean+(1-beta)*targets.mean()
        second = beta*(self.scale.square()+self.mean.square())+(1-beta)*targets.square().mean()
        new_scale = (second-new_mean.square()).clamp_min(1e-4).sqrt()
        self.output.weight.mul_(self.scale/new_scale)
        self.output.bias.copy_((self.scale*self.output.bias+self.mean-new_mean)/new_scale)
        self.mean.copy_(new_mean); self.scale.copy_(new_scale)
