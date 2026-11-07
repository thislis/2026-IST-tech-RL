"""Fixed 32-task support built from public geometry, never from learned scores.

10 resource sources, five storage components (deliver/defend per edge),
two role changes, five opponents, five private holds, five private evades.
"""
import torch
from torch import nn
import torch.nn.functional as F
from .geometry import Geometry,cell_of,center_of


class Tasks(nn.Module):
    def __init__(self):
        super().__init__()
        self.geometry=Geometry()
        cells=torch.arange(576)
        self.register_buffer('cells',cells)
        self.register_buffer('identity',torch.eye(5,dtype=torch.bool))
        self.register_buffer('hunter',(cells==275)|(cells==276)|(cells==299)|(cells==300))
        self.register_buffer('carrier',cells==481)
        self.register_buffer('enemy_base',((cells%24)>=21)&((cells//24)<3))

    def walks(self,semantic:torch.Tensor,classes:torch.Tensor):
        base=(~semantic[1].to(dtype=torch.bool)) & (~self.enemy_base)
        walk=base[None].expand(5,576).clone()
        forbidden=(classes==0)[:,None] & (self.hunter|self.carrier)[None]
        return walk & ~forbidden

    def components(self,stores:torch.Tensor):
        labels=torch.where(stores,self.cells,torch.full_like(self.cells,576))
        for _ in range(575):
            neighbours=labels[self.geometry.neighbours[:,:4]]
            neighbours=torch.where(self.geometry.bounds[:,:4],neighbours,torch.full_like(neighbours,576))
            updated=torch.where(stores,torch.minimum(labels,neighbours.min(-1).values),labels)
            if bool((updated==labels).all()):
                break
            labels=updated
        root=stores & (labels==self.cells)
        ids=torch.topk(torch.where(root,self.cells,torch.full_like(self.cells,576)),5,largest=False).values
        return labels,ids

    def forward(self,semantic:torch.Tensor,pos:torch.Tensor,enemies:torch.Tensor,
                classes:torch.Tensor,holding:torch.Tensor,task:torch.Tensor,token:torch.Tensor,
                present:torch.Tensor):
        walk=self.walks(semantic,classes)
        current=cell_of(pos)
        seeds=F.one_hot(current,576).to(dtype=torch.bool)
        walk=walk|seeds  # allow exit from an incidental current shrine cell
        wh=walk | (self.hunter[None]&(~semantic[1].to(dtype=torch.bool))[None])
        wc=walk | (self.carrier[None]&(~semantic[1].to(dtype=torch.bool))[None])
        go,_=self.geometry(torch.cat((walk,wh,wc),0),seeds.repeat(3,1))
        base=go[:5]
        back,owner=self.geometry(walk,semantic[2].to(dtype=torch.bool)[None].expand(5,576))
        resource=(semantic[6:].sum(0)>0)&~semantic[2].to(dtype=torch.bool)&~self.enemy_base&~self.hunter&~self.carrier
        trip=base+back
        rank=trip.min(0).values
        old=task[:,0].long().clamp(0,575)
        continuing=((task[:,2]==0)|(task[:,2]==2)|(task[:,2]==5)) & (task[:,11]>0) & (holding==0)
        reserved=(F.one_hot(old,576).to(dtype=torch.bool)&continuing[:,None]).any(0)
        viable=resource[None] & (classes!=1)[:,None] & (holding==0)[:,None] & present[:,None]
        best=torch.where(viable,trip,torch.full_like(trip,20000.)).argmin(-1)
        preferred=(F.one_hot(best,576).to(dtype=torch.bool)&viable.any(-1)[:,None]).any(0)
        rank=rank-reserved.float()*40000.-preferred.float()*20000.
        rank=torch.where(resource,rank+self.cells.float()*1e-5,1e6+self.cells.float())
        resources=torch.topk(rank,10,largest=False).indices
        labels,store_ids=self.components(semantic[2].to(dtype=torch.bool))
        store_cost=torch.where((labels[None,None,:]==store_ids[None,:,None]) & (store_ids[None,:,None]<576),
            base[:,None,:],torch.full((5,5,576),10000.,device=pos.device))
        store_goal=store_cost.argmin(-1)
        hunter_cost=torch.where(self.hunter[None],go[5:10],torch.full_like(base,10000.))
        hg=hunter_cost.argmin(-1,keepdim=True)
        cg=torch.full_like(hg,481)
        enemy_cells=cell_of(enemies)[None].expand(5,5)
        private=current[None].expand(5,5)
        goals=torch.cat((resources[None].expand(5,10),store_goal,hg,cg,enemy_cells,private,private),-1)
        kinds=torch.zeros((5,32),dtype=torch.long,device=pos.device)
        resource_kind=torch.where(semantic[7:,resources].sum(0)>0,torch.full_like(resources,2),
                                 torch.where(semantic[3,resources]>0,torch.full_like(resources,5),torch.zeros_like(resources)))
        kinds[:,:10]=resource_kind[None]
        kinds[:,10:15]=torch.where(holding[:,None]>0,torch.ones((5,5),dtype=torch.long,device=pos.device),
                                  torch.full((5,5),4,dtype=torch.long,device=pos.device))
        kinds[:,15]=6; kinds[:,16]=7; kinds[:,17:22]=3; kinds[:,22:27]=8; kinds[:,27:]=9
        distance=base.gather(1,goals)
        distance[:,15]=go[5:10].gather(1,hg).select(1,0)
        distance[:,16]=go[10:15,481]
        distance[:,22:]=0.
        returning=torch.zeros_like(distance)
        returning[:,:10]=back[:,resources]
        destinations=goals.clone()
        destinations[:,:10]=owner[:,resources].clamp_min(0)
        caps=torch.ones(32,dtype=torch.long,device=pos.device)
        caps[10:15]=5; caps[17:22]=2
        caps[15]=(2-(classes==1).sum()).clamp(0,2)
        caps[16]=(1-(classes==2).sum()).clamp(0,1)
        legal=(distance<9999.)&(returning<9999.)&present[:,None]
        legal[:,:10]=legal[:,:10]&resource[resources][None]&(holding==0)[:,None]&(classes!=1)[:,None]
        legal[:,10:15]=legal[:,10:15]&(store_ids<576)[None]&((holding>0)|(classes==1))[:,None]
        legal[:,15:17]=legal[:,15:17]&((classes==0)&(holding==0))[:,None]
        legal[:,17:22]=legal[:,17:22]&(classes==1)[:,None]
        identity=self.identity
        legal[:,22:27]=identity
        legal[:,27:]=identity&present[:,None]
        # Missing slots hold their last reservation; do not allocate that source to another unit.
        missing_resource=(~present)&continuing
        collision=(goals[:,:10,None]==old[None,None,:])&missing_resource[None,None,:]
        legal[:,:10]=legal[:,:10]&~collision.any(-1)
        caps[15]=(caps[15]-((~present)&(task[:,2]==6)).sum()).clamp_min(0)
        caps[16]=(caps[16]-((~present)&(task[:,2]==7)).sum()).clamp_min(0)
        legal=legal & (caps[None]>0)
        f=torch.zeros((5,32,64),device=pos.device)
        f[:,:,:3]=F.one_hot(classes,3)[:,None].float()
        f[:,:,3:9]=F.one_hot(holding,6)[:,None].float()
        f[:,:,9]=task[:,17:18]/10.
        f[:,:,10]=token[:,116:117]
        f[:,:,11]=task[:,4:5]/21000.
        f[:,:,12:22]=F.one_hot(kinds,10).float()
        f[:,:,22]=legal.float()
        f[:,:,24]=distance/48.; f[:,:,25]=returning/48.
        speed=task[:,17:18].clamp(.15,15.)
        eta=(distance+returning)/speed+.5
        f[:,:,26]=eta/60.
        same=(task[:,0:1].long()==goals)&(task[:,2:3].long()==kinds)&(task[:,11:12]>0)
        f[:,:,27]=(~same).float()
        f[:,:,28]=task[:,23:24]/10.
        ep=center_of(goals)
        enemy_distance=((ep[:,:,None]-enemies[None,None])**2).sum(-1).sqrt().min(-1).values*24
        f[:,:,31]=enemy_distance/10./60.
        f[:,:,32]=-1.; f[:,:,33]=(enemy_distance/10.-distance/speed)/60.; f[:,:,34]=-1.
        f[:,:,35]=torch.exp(-enemy_distance/3.)
        f[:,:,39]=legal.float()
        f[:,:,40:43]=token[:,None,93:96]
        phase=(1-token[:,95])*420./20.*6.28318530718
        f[:,:,43]=phase.sin()[:,None]; f[:,:,44]=phase.cos()[:,None]
        f[:,:,45:48]=F.one_hot(classes,3).float().mean(0)
        f[:,:,48]=(holding>0).float().mean()
        f[:,:,49]=resource.float().sum()/64.
        f[:,:,50]=legal.float().mean(-1,keepdim=True)
        f[:,:,56]=same.float();f[:,:,57]=(task[:,2:3].long()==kinds).float()
        f[:,:,58]=token[:,117:118]
        f[:,:,59]=(holding!=task[:,7]).float()[:,None]
        alternatives=torch.where(legal,eta,torch.full_like(eta,10000.))
        others=alternatives[None].expand(5,5,32).clone()
        others=others.masked_fill(identity[:,:,None],10000.)
        two=others.topk(2,dim=1,largest=False).values
        f[:,:,60]=two[:,0]/60.;f[:,:,61]=two[:,1]/60.
        f[:,:,62]=1.;f[:,:,63]=legal.float().sum(0)[None]/5.
        f[:,:,30]=(eta-alternatives.min(-1,keepdim=True).values)/60.
        return f.clamp(-1,1),legal,caps,goals,destinations,kinds,walk
