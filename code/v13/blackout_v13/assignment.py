"""Exact Gibbs training probabilities and exportable MAP use the same support."""
import torch
from torch import nn
from blackout_v12.assignment import joint_log_prob, sample_assignment, joint_entropy, log_partition


class Assignment(nn.Module):
    def __init__(self):
        super().__init__()
        subsets=torch.arange(32)
        bits=((subsets[:,None]>>torch.arange(5))&1).to(dtype=torch.bool)
        self.register_buffer('bits',bits)
        self.register_buffer('counts',bits.sum(-1))
        self.register_buffer('contained',(subsets[:,None]&subsets[None])==subsets[None])
        self.register_buffer('previous',subsets[:,None]^subsets[None])

    def forward(self, scores: torch.Tensor, legal: torch.Tensor, caps: torch.Tensor):
        weights=scores.transpose(0,1)@self.bits.float().transpose(0,1)
        allowed=((~legal).float().transpose(0,1)@self.bits.float().transpose(0,1)==0)
        allowed=allowed & (self.counts[None]<=caps[:,None])
        v=torch.full((32,),-1e20,device=scores.device)
        v[0]=0.
        history=[v]
        for j in range(scores.shape[1]):
            candidates=v[self.previous]+weights[j,None]
            ok=self.contained & allowed[j,None]
            v=torch.where(ok,candidates,torch.full_like(candidates,-1e20)).max(-1).values
            history.append(v)
        state=torch.tensor(31,dtype=torch.long,device=scores.device)
        assignment=torch.zeros(5,dtype=torch.long,device=scores.device)
        for j in range(scores.shape[1]-1,-1,-1):
            candidates=history[j][self.previous[state]]+weights[j]
            ok=self.contained[state]&allowed[j]
            subset=torch.where(ok,candidates,torch.full_like(candidates,-1e20)).argmax()
            assignment=torch.where(self.bits[subset],torch.full_like(assignment,j),assignment)
            state=self.previous[state,subset]
        return assignment
