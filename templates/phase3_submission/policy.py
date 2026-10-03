"""Self-contained deterministic BlackOut policy; depends only on torch."""

from __future__ import annotations

from pathlib import Path
import torch
from torch import nn


DIRECTIONS = torch.tensor([
    [0.,0.],[1.,0.],[2**-.5,2**-.5],[0.,1.],[-2**-.5,2**-.5],
    [-1.,0.],[-2**-.5,-2**-.5],[0.,-1.],[2**-.5,-2**-.5],
], dtype=torch.float32)


class EntityEncoder(nn.Module):
    def __init__(self, entity_dim: int, context_dim: int) -> None:
        super().__init__()
        self.entity_encoder=nn.Sequential(nn.Linear(9,entity_dim),nn.ReLU(),nn.Linear(entity_dim,entity_dim),nn.ReLU())
        self.context_encoder=nn.Sequential(nn.Linear(6,context_dim),nn.ReLU())
        self.output_dim=10*entity_dim+context_dim
    def forward(self,x):
        entities=self.entity_encoder(x[:,:90].reshape(-1,10,9)).flatten(1)
        return torch.cat((entities,self.context_encoder(x[:,90:])),dim=-1)


class GraphicEncoder(nn.Module):
    def __init__(self, graphic_dim: int) -> None:
        super().__init__(); self.network=nn.Sequential(
            nn.Conv2d(11,16,5,4,2),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),
            nn.AdaptiveAvgPool2d((4,4)),nn.Flatten(),nn.Linear(512,graphic_dim),nn.ReLU())
    def forward(self,x): return self.network(x)


class ActorCritic(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__(); entity=int(config["entity_dim"]); context=int(config["context_dim"])
        graphic=int(config["graphic_dim"]); hidden=int(config["hidden_dim"]); slot=int(config["slot_embedding_dim"])
        self.vector_encoder=EntityEncoder(entity,context); self.graphic_encoder=GraphicEncoder(graphic)
        self.slot_embedding=nn.Embedding(5,slot)
        self.fusion=nn.Sequential(nn.Linear(self.vector_encoder.output_dim+graphic+slot,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU())
        self.actor=nn.Linear(hidden,9); self.critic=nn.Linear(hidden,1)
    def forward(self,vector,graphic,slot_id):
        latent=self.fusion(torch.cat((self.vector_encoder(vector),self.graphic_encoder(graphic),self.slot_embedding(slot_id)),dim=-1))
        return self.actor(latent)


class Policy(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__(); self.actor_critic=ActorCritic(config)
    def forward(self, vector, graphic):
        if vector.ndim != 2 or vector.shape != (5,96): raise ValueError("vector must be (5,96)")
        if graphic.ndim != 4 or graphic.shape[0] != 5 or graphic.shape[1] != 11: raise ValueError("graphic must be (5,11,H,W)")
        if vector.dtype != torch.float32 or graphic.dtype != torch.float32: raise TypeError("vector and graphic must be float32")
        if vector.device != graphic.device or vector.device != next(self.parameters()).device: raise ValueError("model, vector, and graphic must use the same device")
        slots=torch.arange(5,dtype=torch.int64,device=vector.device)
        logits=self.actor_critic(vector,graphic,slots)
        return DIRECTIONS.to(logits.device)[torch.argmax(logits,dim=-1)]


def load_policy(checkpoint: str | Path, device: str = "cpu") -> Policy:
    if device.startswith("cuda") and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    if device == "mps" and not torch.backends.mps.is_available(): raise RuntimeError("MPS requested but unavailable")
    payload=torch.load(Path(checkpoint),map_location=device,weights_only=True)
    if payload.get("schema_version") != "blackout.checkpoint.v1": raise ValueError("unsupported checkpoint")
    model=Policy(payload["model_config"])
    model.load_state_dict(payload["policy_state"],strict=True)
    return model.to(device).eval()
