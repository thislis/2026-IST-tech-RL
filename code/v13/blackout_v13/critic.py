from torch import nn


class Critic(nn.Module):
    """Training-only centralized value baseline; public history only."""
    def __init__(self,lags=6):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(lags*128,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))

    def forward(self,lagged):
        x=lagged.clone();x[...,118]=x[...,118]/420.;x[...,119:121]=x[...,119:121]/1000.
        return self.net(x.clamp(-10,10).mean(-3).flatten(-2)).squeeze(-1)
