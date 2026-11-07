import torch
from torch import nn
from .history import TemporalEncoder


class Manager(nn.Module):
    def __init__(self,lags=(0,4,8,16,32,64)):
        super().__init__()
        self.temporal=TemporalEncoder(lags)
        self.team=nn.Sequential(nn.Linear(64,32),nn.ReLU())
        self.scorer=nn.Sequential(nn.Linear(160,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))

    def forward(self,features:torch.Tensor,token:torch.Tensor,history:torch.Tensor):
        return self.from_lags(features,self.temporal.history(token,history))

    def from_lags(self,features:torch.Tensor,lagged:torch.Tensor):
        memory=self.temporal.encode_lagged(lagged)
        context=self.team(memory.mean(-2,keepdim=True)).expand_as(memory[...,:32])
        shared=torch.cat((memory,context),-1).unsqueeze(-2).expand(list(features.shape[:-1])+[96])
        return self.scorer(torch.cat((features,shared),-1)).select(-1,0).clamp(-20.,20.)
