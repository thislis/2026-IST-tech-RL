"""Exportable exact positive-weight grid relaxation, with convergence exit.

No Python search/cache is required by the deployed graph. Diagonal movement
requires BOTH orthogonal neighbours; INF remains finite for ONNX reductions.
"""
import math
import torch
from torch import nn


class Geometry(nn.Module):
    def __init__(self):
        super().__init__()
        offsets = [(1,0),(0,1),(-1,0),(0,-1),(1,1),(-1,1),(-1,-1),(1,-1)]
        indices, valid, corner_a, corner_b = [], [], [], []
        for cell in range(576):
            x,y=cell%24,cell//24
            indices.append([min(23,max(0,y+dy))*24+min(23,max(0,x+dx)) for dx,dy in offsets])
            valid.append([0<=x+dx<24 and 0<=y+dy<24 for dx,dy in offsets])
            corner_a.append([y*24+min(23,max(0,x+dx)) for dx,dy in offsets])
            corner_b.append([min(23,max(0,y+dy))*24+x for dx,dy in offsets])
        self.register_buffer('neighbours',torch.tensor(indices))
        self.register_buffer('bounds',torch.tensor(valid))
        self.register_buffer('corner_a',torch.tensor(corner_a))
        self.register_buffer('corner_b',torch.tensor(corner_b))
        self.register_buffer('cost',torch.tensor([1.]*4+[math.sqrt(2)]*4))
        self.register_buffer('cells',torch.arange(576))

    def forward(self, walk: torch.Tensor, seeds: torch.Tensor):
        walk=walk.to(dtype=torch.bool)
        edge = (walk[:, :, None] & walk[:,self.neighbours] & self.bounds[None]
                & walk[:,self.corner_a] & walk[:,self.corner_b])
        valid = seeds.to(dtype=torch.bool) & walk
        distance = torch.where(valid, torch.zeros_like(walk,dtype=torch.float32),
                               torch.full_like(walk,10000.,dtype=torch.float32))
        owner = torch.where(valid,self.cells[None].expand_as(walk),torch.full_like(walk,-1,dtype=torch.long))
        for _ in range(575):
            proposals=torch.where(edge,distance[:,self.neighbours]+self.cost,
                                  torch.full_like(distance[:,self.neighbours],10000.))
            best,index=proposals.min(-1)
            improve=best<distance
            if not bool(improve.any()):
                break
            source=owner[:,self.neighbours].gather(-1,index.unsqueeze(-1)).select(-1,0)
            distance=torch.where(improve,best,distance)
            owner=torch.where(improve,source,owner)
        return distance,owner


def cell_of(position: torch.Tensor):
    xy=(position*24).long().clamp(0,23)
    return xy[...,1]*24+xy[...,0]


def center_of(cell: torch.Tensor):
    return torch.stack(((cell%24).float()+.5,(cell//24).float()+.5),-1)/24.
