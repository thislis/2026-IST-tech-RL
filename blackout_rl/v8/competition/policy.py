"""Standalone submission policy. Runtime dependencies: torch and Python stdlib.

No agent IDs, batch-position identities, reset callbacks, files or environment
access. C1-obs9 replaces the unavailable team-slot contract with a row-wise gate.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class VectorEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.entity_encoder=nn.Sequential(nn.Linear(9,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU())
        self.context_encoder=nn.Sequential(nn.Linear(6,32),nn.ReLU())

    def forward(self,x):
        return torch.cat((self.entity_encoder(x[:,:90].reshape(-1,10,9)).flatten(1),self.context_encoder(x[:,90:])),1)


class GraphicEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_network=nn.Sequential(nn.Conv2d(11,16,5,4,2),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d((4,4)),nn.Flatten())
        self.local_network=nn.Sequential(nn.Conv2d(11,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d((4,4)),nn.Flatten())
        self.projection=nn.Sequential(nn.Linear(1024,128),nn.ReLU())

    def forward(self,x,position):
        theta=x.new_zeros((len(x),2,3))
        theta[:,0,0]=31/(x.shape[-1]-1);theta[:,1,1]=31/(x.shape[-2]-1)
        theta[:,0,2]=position[:,0].clamp(0,1)*2-1;theta[:,1,2]=1-position[:,1].clamp(0,1)*2
        grid=F.affine_grid(theta,(len(x),11,32,32),align_corners=True)
        local=F.grid_sample(x,grid,align_corners=True,padding_mode='zeros')
        return self.projection(torch.cat((self.global_network(x),self.local_network(local)),1))


class FrozenReference(nn.Module):
    """Original checkpoint weights; explicit observation-only input adaptation.

    Self slot is absent from the provided tensors. Use visible ally centroid and
    mean slot embedding, never pretend a batch row supplies the missing identity.
    This is not the historical full opponent/planner policy.
    """
    def __init__(self):
        super().__init__()
        self.vector_encoder=VectorEncoder();self.graphic_encoder=GraphicEncoder()
        self.slot_embedding=nn.Embedding(5,16)
        self.fusion=nn.Sequential(nn.Linear(496,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU())
        self.actor=nn.Linear(128,9);self.critic=nn.Linear(128,1)

    def forward(self,vector,graphic):
        entities=vector[:,:90].reshape(-1,10,9)
        ally=(entities[:,:,2]>0).to(vector.dtype)
        position=(entities[:,:,:2]*ally[:,:,None]).sum(1)/ally.sum(1,keepdim=True).clamp_min(1)
        slots=self.slot_embedding.weight.mean(0).expand(len(vector),-1)
        latent=self.fusion(torch.cat((self.vector_encoder(vector),self.graphic_encoder(graphic,position),slots),1))
        return latent,self.actor(latent)


class MyPolicy(nn.Module):
    def __init__(self,vector_size=96,n_channels=11):
        super().__init__()
        if (vector_size,n_channels)!=(96,11):raise ValueError('requires vector_size=96, n_channels=11')
        self.encoder=FrozenReference().requires_grad_(False)
        self.trunk=nn.Sequential(nn.Linear(177,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh())
        self.correction=nn.Linear(64,8)
        self.gate=nn.Sequential(nn.Linear(64,64),nn.Tanh(),nn.Linear(64,1))
        self.register_buffer('mode_id',torch.tensor(0,dtype=torch.long))
        self.register_buffer('threshold',torch.tensor(.5))
        self.register_buffer('directions',torch.tensor([[0,0],[1,0],[2**-.5,2**-.5],[0,1],[-2**-.5,2**-.5],[-1,0],[-2**-.5,-2**-.5],[0,-1],[2**-.5,-2**-.5]],dtype=torch.float32))
        self.configure('c1')

    def configure(self,mode,initial_q=.1):
        if mode not in ('c1','flat') or not 0<initial_q<1:raise ValueError('invalid policy mode')
        self.mode_id.fill_(mode=='flat')
        nn.init.zeros_(self.correction.weight)
        nn.init.constant_(self.correction.bias,0 if mode=='c1' else math.log(initial_q/(8*(1-initial_q))))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias,math.log(initial_q/(1-initial_q)) if mode=='c1' else 0)

    def features(self,vector,graphic):
        if vector.ndim!=2 or vector.shape[1]!=96 or graphic.shape!=(len(vector),11,96,96):raise ValueError('expected (B,96) and (B,11,96,96)')
        if vector.dtype!=torch.float32 or graphic.dtype!=torch.float32:raise TypeError('float32 required')
        if vector.device!=graphic.device or vector.device!=self.directions.device:raise ValueError('device mismatch')
        if not torch.isfinite(vector).all() or not torch.isfinite(graphic).all():raise ValueError('nonfinite observation')
        with torch.no_grad():latent,logits=self.encoder(vector,graphic)
        features=torch.cat((latent,vector[:,:49]),1)
        baseline=self.directions[logits.argmax(-1)]
        # Only duplicate actions are masked: no guessed self location/geometry.
        valid=~torch.isclose(baseline[:,None,:],self.directions[None,1:,:],atol=1e-5,rtol=1e-5).all(-1)
        return features,baseline,valid

    def log_distribution(self,features,valid):
        hidden=self.trunk(features);correction=self.correction(hidden)
        scalar=self.gate(hidden).squeeze(-1);has=valid.any(-1)
        safe=valid.clone();safe[...,0]|=~has
        if int(self.mode_id)==0:
            keep=torch.where(has,F.logsigmoid(-scalar),torch.zeros_like(scalar))
            conditional=correction.masked_fill(~safe,-torch.inf).log_softmax(-1)
            rest=(F.logsigmoid(scalar)[...,None]+conditional).masked_fill(~valid,-torch.inf)
            return torch.cat((keep[...,None],rest),-1)
        return torch.cat((scalar[...,None],correction.masked_fill(~valid,-torch.inf)),-1).log_softmax(-1)

    def decode(self,log_probs):
        if int(self.mode_id)==1:return log_probs.argmax(-1)
        q=log_probs[...,1:].exp().sum(-1)
        return (log_probs[...,1:].argmax(-1)+1)*(q>self.threshold).long()

    def execute(self,actions,baseline):
        return torch.where((actions==0)[...,None],baseline,self.directions[actions]).to(torch.float32)

    def forward(self,vector,graphic):
        if len(vector)==0:
            if vector.shape!=(0,96) or graphic.shape!=(0,11,96,96):raise ValueError('invalid empty batch')
            return vector.new_empty((0,2))
        features,baseline,valid=self.features(vector,graphic)
        return self.execute(self.decode(self.log_distribution(features,valid)),baseline)

    def train(self,mode=True):
        super().train(mode);self.encoder.eval();return self
