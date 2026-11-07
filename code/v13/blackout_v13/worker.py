import math
import torch
from torch import nn
from .history import TemporalEncoder


class Worker(nn.Module):
    def __init__(self,lags=(0,4,8,16,32,64),recurrent=True,directions=32):
        super().__init__()
        self.recurrent=recurrent
        self.temporal=TemporalEncoder(lags)
        self.cnn=nn.Sequential(nn.Conv2d(13,16,3,padding=1),nn.ReLU(),
            nn.Conv2d(16,32,3,stride=2,padding=1),nn.ReLU(),nn.Flatten(),nn.Linear(800,64),nn.ReLU())
        self.fuse=nn.Sequential(nn.Linear(160,128),nn.ReLU())
        self.gru=nn.GRUCell(128,64)
        self.mlp=nn.Sequential(nn.Linear(128,64),nn.ReLU())
        self.head=nn.Linear(64,directions+1)
        angle=torch.arange(directions)*2*math.pi/directions
        self.register_buffer('directions',torch.cat((torch.stack((angle.cos(),angle.sin()),-1),torch.zeros(1,2))))

    def encode(self,local:torch.Tensor,vector:torch.Tensor,token:torch.Tensor,history:torch.Tensor):
        return self.encode_lagged(local,vector,self.temporal.history(token,history))

    def encode_lagged(self,local:torch.Tensor,vector:torch.Tensor,lagged:torch.Tensor):
        memory=self.temporal.encode_lagged(lagged)
        image=self.cnn(local.reshape(-1,13,9,9)).reshape_as(memory)
        return self.fuse(torch.cat((image,vector,memory),-1))

    def step_encoded(self,encoded:torch.Tensor,hidden:torch.Tensor):
        if self.recurrent:
            # Equivalent to GRUCell, avoiding its Python rank-validation branches in ONNX.
            i=torch.nn.functional.linear(encoded,self.gru.weight_ih,self.gru.bias_ih)
            old=torch.nn.functional.linear(hidden,self.gru.weight_hh,self.gru.bias_hh)
            reset=torch.sigmoid(i[...,:64]+old[...,:64])
            update=torch.sigmoid(i[...,64:128]+old[...,64:128])
            candidate=torch.tanh(i[...,128:]+reset*old[...,128:])
            h=(1-update)*candidate+update*hidden
        else:
            h=self.mlp(encoded)
        return self.head(h),h

    def forward(self,local:torch.Tensor,vector:torch.Tensor,token:torch.Tensor,
                history:torch.Tensor,hidden:torch.Tensor):
        return self.step_encoded(self.encode(local,vector,token,history),hidden)
