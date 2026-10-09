"""Standalone competition policy: this file and checkpoint.pt are sufficient.

Only torch/stdlib dependencies. No agent names, row identities, filesystem reads,
environment state, planner override, or hidden episode state.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, width=64, heads=4):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(width, width)
        self.kv = nn.Linear(width, width * 2)
        self.out = nn.Linear(width, width)

    def forward(self, query, context):
        b, nq, width = query.shape
        q = self.q(query).reshape(b, nq, self.heads, width // self.heads).transpose(1, 2)
        k, v = self.kv(context).chunk(2, -1)
        k = k.reshape(b, -1, self.heads, width // self.heads).transpose(1, 2)
        v = v.reshape(b, -1, self.heads, width // self.heads).transpose(1, 2)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        return self.out(x.transpose(1, 2).reshape(b, nq, width))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(64)
        self.attn = Attention()
        self.ffnorm = nn.LayerNorm(64)
        self.ff = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 64))

    def forward(self, x):
        n = self.norm(x)
        x = x + self.attn(n, n)
        return x + self.ff(self.ffnorm(x))


class MyPolicy(nn.Module):
    def __init__(self, vector_size=96, n_channels=11):
        super().__init__()
        if (vector_size, n_channels) != (96, 11):
            raise ValueError("requires vector_size=96 and n_channels=11")
        self.patch = nn.Linear(176, 64)
        self.xy = nn.Linear(2, 64, bias=False)
        self.entity = nn.Sequential(nn.Linear(9, 64), nn.GELU(), nn.Linear(64, 64))
        self.entity_slot = nn.Parameter(torch.randn(10, 64) * .02)
        self.context = nn.Linear(3, 64)
        self.types = nn.Parameter(torch.randn(3, 64) * .02)
        self.latents = nn.Parameter(torch.randn(32, 64) * .02)
        self.input_norm = nn.LayerNorm(64)
        self.cross = Attention()
        self.blocks = nn.Sequential(Block(), Block())
        self.query = nn.Linear(6, 64)
        self.readout = Attention()
        self.readout_norm = nn.LayerNorm(64)
        self.experts = nn.Linear(64, 5 * 9)
        self.gate = nn.Linear(64, 5)
        self.auxiliary = nn.Linear(64, 3)
        grid = torch.linspace(-1, 1, 24)
        yy, xx = torch.meshgrid(grid, grid, indexing="ij")
        self.register_buffer("patch_xy", torch.stack((xx, yy), -1).reshape(576, 2))
        self.register_buffer("expert_mask", torch.tensor([0., -torch.inf, -torch.inf, -torch.inf, -torch.inf]))
        self.register_buffer("sample_actions", torch.tensor(True))
        d = math.sqrt(.5)
        self.register_buffer("directions", torch.tensor([[0, 0], [1, 0], [d, d], [0, 1],
                                                         [-d, d], [-1, 0], [-d, -d], [0, -1], [d, -d]], dtype=torch.float32))
        nn.init.normal_(self.experts.weight, std=.01)
        nn.init.zeros_(self.experts.bias)

    def configure(self, arm="A", decoder="sample"):
        if arm not in ("A", "B", "C") or decoder not in ("sample", "greedy"):
            raise ValueError("invalid arm/decoder")
        self.expert_mask.fill_(0 if arm == "C" else -torch.inf)
        self.expert_mask[0] = 0
        self.sample_actions.fill_(decoder == "sample")

    def scene(self, vector, graphic):
        # Patch flattening preserves all 11 channels and all sub-tile pixels.
        b = len(vector)
        patches = graphic.reshape(b, 11, 24, 4, 24, 4).permute(0, 2, 4, 1, 3, 5).reshape(b, 576, 176)
        maps = self.patch(patches) + self.xy(self.patch_xy) + self.types[0]
        entities = self.entity(vector[:, :90].reshape(b, 10, 9)) + self.entity_slot + self.types[1]
        context = self.context(vector[:, 93:96])[:, None] + self.types[2]
        tokens = self.input_norm(torch.cat((maps, entities, context), 1))
        latent = self.latents[None].expand(b, -1, -1)
        return self.blocks(latent + self.cross(latent, tokens))

    def heads(self, latent, context):
        # context: [scene_count, query_count, 6]. Queries are independent;
        # no attention or identity is inferred along the arbitrary batch axis.
        query = self.query(context)
        features = self.readout_norm(query + self.readout(query, latent))
        expert_logs = self.experts(features).reshape(*features.shape[:-1], 5, 9).log_softmax(-1)
        gate_logs = (self.gate(features) + self.expert_mask).log_softmax(-1)
        logs = torch.logsumexp(gate_logs[..., None] + expert_logs, -2)
        return logs, features, gate_logs.exp()

    def log_probs(self, vector, graphic):
        self.validate(vector, graphic)
        if len(vector) == 0:
            return vector.new_empty((0, 9))
        # Exact common-scene reuse, never cached across optimizer steps/calls.
        shared = (len(vector) <= 40 and
                  torch.equal(vector[:, :90], vector[:1, :90].expand(len(vector), -1)) and
                  torch.equal(vector[:, 93:], vector[:1, 93:].expand(len(vector), -1)) and
                  torch.equal(graphic, graphic[:1].expand_as(graphic)))
        if shared:
            return self.heads(self.scene(vector[:1], graphic[:1]), vector[None, :, 90:])[0][0]
        return self.heads(self.scene(vector, graphic), vector[:, None, 90:])[0][:, 0]

    def team_distribution(self, vector, graphic):
        """Training-only batching of verified shared scenes [T,5,96]."""
        if vector.ndim != 3 or vector.shape[1:] != (5, 96):
            raise ValueError("team vectors must be [T,5,96]")
        if not (torch.equal(vector[:, :, :90], vector[:, :1, :90].expand(-1, 5, -1)) and
                torch.equal(vector[:, :, 93:], vector[:, :1, 93:].expand(-1, 5, -1))):
            raise ValueError("unshared team scene; cannot compress")
        return self.heads(self.scene(vector[:, 0], graphic), vector[:, :, 90:])

    @staticmethod
    def validate(vector, graphic):
        if vector.ndim != 2 or vector.shape[1] != 96 or graphic.shape != (len(vector), 11, 96, 96):
            raise ValueError("expected (B,96), (B,11,96,96)")
        if vector.dtype != torch.float32 or graphic.dtype != torch.float32:
            raise TypeError("float32 observations required")
        if vector.device != graphic.device:
            raise ValueError("observation devices differ")
        if not torch.isfinite(vector).all() or not torch.isfinite(graphic).all():
            raise ValueError("nonfinite observation")

    def choose(self, logs, generator=None):
        if bool(self.sample_actions):
            return torch.multinomial(logs.exp(), 1, generator=generator).squeeze(-1)
        return logs.argmax(-1)

    def forward(self, vector, graphic):
        logs = self.log_probs(vector, graphic)
        if not len(vector):
            return vector.new_empty((0, 2))
        return self.directions[self.choose(logs)].to(torch.float32)


SemanticEntityAttentionPolicy = MyPolicy
