"""One graph from public observations/state to actions/state; actor training shares it."""
from typing import Tuple,Dict
import torch
from torch import nn
import torch.nn.functional as F
from .geometry import cell_of,center_of
from .tasks import Tasks
from .manager import Manager
from .worker import Worker
from .assignment import Assignment


class Policy(nn.Module):
    def __init__(self,config=None):
        super().__init__()
        cfg=dict(config or {})
        self.period=int(cfg.get('event_period',25))
        self.temperature=float(cfg.get('temperature',1.))
        self.planner_worker=cfg.get('worker_mode','neural')=='planner'
        lags=tuple(cfg.get('lags',(0,4,8,16,32,64)))
        self.manager=Manager(tuple(cfg.get('manager_lags',lags)))
        self.worker=Worker(lags,bool(cfg.get('recurrent',True)),int(cfg.get('directions',32)))
        self.tasks=Tasks()
        self.assignment=Assignment()
        yy,xx=torch.meshgrid(torch.arange(-4,5),torch.arange(-4,5),indexing='ij')
        self.register_buffer('crop_x',xx.reshape(-1));self.register_buffer('crop_y',yy.reshape(-1))

    def decode(self,vector:torch.Tensor,graphic:torch.Tensor):
        present=vector.abs().sum(-1)>0
        first=present.long().argmax()
        v=vector.clone()
        e=v[first,:90].reshape(10,9)
        side=e[0,2]<0
        image=graphic[first]
        if bool(side):
            e=torch.cat((e[:,:2].flip(-1),e[:,2:]),-1)
            image=image.transpose(-1,-2).flip((-1,-2))
            e=torch.cat((e[5:],e[:5]),0)
        own=e[:5];enemy=e[5:]
        v[:,:90]=e.reshape(1,90).expand(5,90)
        v[:,93:96]=vector[first,93:96].unsqueeze(0).expand(5,3)
        semantic=image.reshape(11,24,4,24,4).amax((2,4)).flip(-2).reshape(11,576)>.5
        classes=vector[:,90:93].argmax(-1)
        holding=own[:,3:].argmax(-1)
        return v,semantic.float(),own[:,:2],enemy[:,:2],classes,holding,present,side

    def prepare(self,vector:torch.Tensor,graphic:torch.Tensor,history:torch.Tensor,
                hidden:torch.Tensor,task:torch.Tensor,nav:torch.Tensor):
        v,sem,pos,enemies,classes,holding,present,side=self.decode(vector,graphic)
        classes=torch.where(present,classes,task[:,8].long().clamp(0,2))
        t=task.clone()
        now=(1-v[:,95])*420.
        elapsed=now-t[:,6]
        reset=(t[:,11]==0)|(elapsed<-.01)|(t[:,22]!=side.float())
        displacement=((pos-t[:,9:11])**2).sum(-1).sqrt()
        respawn=(displacement>.15)&~reset
        respawn=respawn|((classes==0)&(t[:,8]!=0)&~reset)
        gap=elapsed>1.
        clear=(reset|respawn|gap)&present
        h=torch.where(clear[:,None,None],torch.zeros_like(history),history)
        hidden=torch.where(clear[:,None],torch.zeros_like(hidden),hidden)
        old_goal=t[:,16].long().clamp(0,575)
        old_kind=t[:,2].long()
        gone=((old_kind==0)|(old_kind==2)|(old_kind==5)) & (sem[6:,old_goal].sum(0)==0)&(holding==0)
        event=bool(((t[:,15]-t[:,5]>=self.period)|clear|(holding!=t[:,7])|(classes!=t[:,8])|gone).any())
        token=torch.zeros((5,128),device=v.device)
        token[:,:96]=v
        token[:,96:98]=t[:,13:15]
        token[:,98:100]=center_of(old_goal)
        token[:,100:110]=F.one_hot(old_kind.clamp(0,9),10).float()
        token[:,110:114]=F.one_hot(t[:,3].long().clamp(0,3),4).float()
        token[:,114]=(t[:,4]/21000.).clamp(0,1)
        token[:,115]=t[:,19]/48.
        token[:,116]=((pos-t[:,9:11])**2).sum(-1).sqrt()*24
        token[:,117]=respawn.float();token[:,118]=now
        token[:,119]=t[:,18]+respawn.float();token[:,120]=t[:,12];token[:,121]=present.float()
        speed=torch.where(reset,torch.full_like(now,2.5),t[:,17])
        measured=(displacement*24/elapsed.clamp_min(.0001)).clamp(.15,15.)
        attempted=(t[:,13:15].square().sum(-1)>.25)&(t[:,19]>.75)
        speed=torch.where(((displacement>1e-5)|attempted)&~clear,.9*speed+.1*measured,speed)
        t[:,23]=torch.where(clear|~attempted|(displacement>1e-5),torch.zeros_like(now),
                            t[:,23]+elapsed.clamp(0,1))
        t[:,17]=speed;t[:,18]=token[:,119]
        t[:,0]=torch.where(clear,torch.full_like(now,-1),t[:,0])
        t[:,11]=torch.where(clear,torch.zeros_like(now),t[:,11])
        walk=self.tasks.walks(sem,classes)|F.one_hot(cell_of(pos),576).to(dtype=torch.bool)
        walk=walk|((t[:,2]==6)[:,None]&self.tasks.hunter[None])|((t[:,2]==7)[:,None]&self.tasks.carrier[None])
        if bool((walk.float()!=nav[:,1]).any()):
            event=True
        return v,sem,pos,enemies,classes,holding,present,side,h,hidden,t,token,clear,event

    def candidates(self,sem:torch.Tensor,pos:torch.Tensor,enemies:torch.Tensor,classes:torch.Tensor,
                   holding:torch.Tensor,t:torch.Tensor,token:torch.Tensor,present:torch.Tensor):
        return self.tasks(sem,pos,enemies,classes,holding,t,token,present)

    def assign(self,t:torch.Tensor,choice:torch.Tensor,goals:torch.Tensor,destinations:torch.Tensor,kinds:torch.Tensor):
        t=t.clone()
        index=choice[:,None]
        source=goals.gather(1,index).select(1,0)
        kind=kinds.gather(1,index).select(1,0)
        changed=(source!=t[:,0])|(kind!=t[:,2])
        t[:,0]=source.float();t[:,1]=destinations.gather(1,index).select(1,0).float();t[:,2]=kind.float()
        t[:,3]=torch.where(kind==1,torch.ones_like(kind),torch.where((kind==6)|(kind==7),torch.full_like(kind,2),
                              torch.where(kind>=8,torch.full_like(kind,3),torch.zeros_like(kind)))).float()
        t[:,4]=torch.where(changed,torch.zeros_like(t[:,4]),t[:,4])
        t[:,5]=t[:,15];t[:,12]=t[:,12]+changed.float()
        t[:,21]=(choice-17).clamp(0,4).float()
        return t

    def control(self,sem:torch.Tensor,pos:torch.Tensor,enemies:torch.Tensor,classes:torch.Tensor,
                holding:torch.Tensor,t:torch.Tensor,nav:torch.Tensor,event:bool):
        t=t.clone()
        kind=t[:,2].long()
        goal=t[:,0].long().clamp(0,575)
        goal=torch.where(kind==1,t[:,1].long().clamp(0,575),goal)
        goal=torch.where(kind==3,cell_of(enemies)[t[:,21].long().clamp(0,4)],goal)
        walk=self.tasks.walks(sem,classes)|F.one_hot(cell_of(pos),576).to(dtype=torch.bool)
        walk=walk|((kind==6)[:,None]&self.tasks.hunter[None])|((kind==7)[:,None]&self.tasks.carrier[None])
        refresh=event or bool((goal!=t[:,16].long()).any()) or bool((walk.float()!=nav[:,1]).any())
        if refresh:
            field,_=self.tasks.geometry(walk,F.one_hot(goal,576).to(dtype=torch.bool))
        else:
            field=nav[:,0]
        t[:,16]=goal.float()
        t[:,19]=field.gather(1,cell_of(pos)[:,None]).select(1,0)
        t[:,20]=t[:,19]/t[:,17].clamp(.15,15.)
        xy=(pos*24).long().clamp(0,23)
        x=xy[:,0,None]+self.crop_x[None];y=xy[:,1,None]+self.crop_y[None]
        inside=(x>=0)&(x<24)&(y>=0)&(y<24)
        idx=y.clamp(0,23)*24+x.clamp(0,23)
        planes=sem[:,idx].permute(1,0,2)*inside[:,None].float()
        planes[:,1]=torch.where(inside,planes[:,1],torch.ones_like(planes[:,1]))
        target=(idx==goal[:,None]).float()*inside.float()
        distance=(field.gather(1,idx)/48.).clamp(0,1)
        # Match compact rollout storage exactly, including deployed inference.
        local=torch.cat((planes,target[:,None],distance[:,None]),1).reshape(5,13,9,9).to(torch.float16).float()
        wv=torch.zeros((5,32),device=pos.device)
        wv[:,:3]=F.one_hot(classes,3).float();wv[:,3:9]=F.one_hot(holding,6).float()
        wv[:,9:19]=F.one_hot(kind.clamp(0,9),10).float()
        wv[:,19:23]=F.one_hot(t[:,3].long().clamp(0,3),4).float()
        wv[:,23:25]=center_of(goal)-pos;wv[:,25:27]=t[:,13:15]
        wv[:,27]=t[:,17]/10.;wv[:,28]=(t[:,19]/48.).clamp(0,1)
        wv[:,29]=(t[:,4]/21000.).clamp(0,1);wv[:,30]=t[:,19]>=9999.
        wv[:,31]=1.
        return local,wv,t,torch.stack((field,walk.float()),1)

    def finish(self,action:torch.Tensor,token:torch.Tensor,h:torch.Tensor,t:torch.Tensor,
               pos:torch.Tensor,classes:torch.Tensor,holding:torch.Tensor,side:torch.Tensor,
               present:torch.Tensor):
        t=t.clone()
        t[:,6]=token[:,118];t[:,7]=holding.float();t[:,8]=classes.float();t[:,9:11]=pos
        t[:,11]=present.float();t[:,13:15]=action;t[:,4]+=1.;t[:,15]+=1.;t[:,22]=side.float()
        history=torch.cat((token[:,None],h[:,:63]),1)
        if bool(side):
            action=action.flip(-1)
        return action*present[:,None].float(),history,t

    def planner_action(self,pos:torch.Tensor,enemies:torch.Tensor,t:torch.Tensor,nav:torch.Tensor):
        current=cell_of(pos);geometry=self.tasks.geometry
        neighbours=geometry.neighbours[current]
        walk=nav[:,1]>0
        legal=(walk.gather(1,neighbours)&geometry.bounds[current]&walk.gather(1,geometry.corner_a[current])
               &walk.gather(1,geometry.corner_b[current]))
        costs=nav[:,0].gather(1,neighbours)+geometry.cost[None]
        choice=torch.where(legal,costs,torch.full_like(costs,10000.)).argmin(-1)
        waypoint=neighbours.gather(1,choice[:,None]).select(1,0)
        waypoint=torch.where(current==t[:,16].long(),current,waypoint)
        safe_positions=center_of(neighbours)
        safety=((safe_positions[:,:,None]-enemies[None,None])**2).sum(-1).sqrt().min(-1).values
        escape=torch.where(legal,safety,torch.full_like(safety,-1.)).argmax(-1)
        waypoint=torch.where(t[:,2]==9,neighbours.gather(1,escape[:,None]).select(1,0),waypoint)
        delta=center_of(waypoint)-pos;length=delta.square().sum(-1).sqrt()
        move=delta/length.clamp_min(1e-8)[:,None]
        valid=(length>=.006)&(t[:,2]!=8)&((t[:,19]<9999.)|(t[:,2]==9))
        return move*valid[:,None].float()

    def forward(self,vector:torch.Tensor,graphic:torch.Tensor,slot:torch.Tensor,
                state_in_history:torch.Tensor,state_in_worker:torch.Tensor,
                state_in_task:torch.Tensor,state_in_nav:torch.Tensor):
        v,sem,pos,enemies,cls,holding,present,side,h,hidden,t,token,clear,event=self.prepare(
            vector,graphic,state_in_history,state_in_worker,state_in_task,state_in_nav)
        if event:
            f,legal,caps,goals,dests,kinds,walk=self.candidates(sem,pos,enemies,cls,holding,t,token,present)
            scores=self.manager(f,token,h)/self.temperature
            choice=self.assignment(scores,legal,caps)
            t=self.assign(t,choice,goals,dests,kinds)
        local,wv,t,nav=self.control(sem,pos,enemies,cls,holding,t,state_in_nav,event)
        logits,hidden=self.worker(local,wv,token,h,hidden)
        action=self.worker.directions[logits.argmax(-1)]
        if self.planner_worker:action=self.planner_action(pos,enemies,t,nav)
        action,history,t=self.finish(action,token,h,t,pos,cls,holding,side,present)
        history=torch.where(present[:,None,None],history,state_in_history)
        hidden=torch.where(present[:,None],hidden,state_in_worker)
        t=torch.where(present[:,None],t,state_in_task)
        nav=torch.where(present[:,None,None],nav,state_in_nav)
        return action,history,hidden,t,nav
