"""Sampling/training adapter. Geometry and state transitions are shared with export."""
from collections import deque
import torch
from .contracts import initial_state,validate_observation
from .assignment import sample_assignment,joint_log_prob
from .history import History


class Actor:
    def __init__(self,policy,critic,config):
        self.policy=policy.eval();self.critic=critic.eval();self.config=config
        self.device=next(policy.parameters()).device
        self.lags=History(tuple(config.get('lags',(0,4,8,16,32,64)))).to(self.device)
        self.reset()

    def reset(self):
        from .diagnostics import Diagnostics
        self.state=initial_state(self.device)
        self.prefix=deque(maxlen=128)
        self.diagnostics=Diagnostics()

    @torch.no_grad()
    def step(self,vector,graphic,phase='worker',deterministic=False,teacher_action=None,teacher_task=None):
        v=torch.as_tensor(vector,device=self.device);g=torch.as_tensor(graphic,device=self.device)
        validate_observation(v,g)
        p=self.policy
        v,sem,pos,enemies,cls,holding,present,side,h,hidden,t,token,clear,event=p.prepare(v,g,*self.state)
        self.value_phase=phase
        sampler=p.manager.temporal.history if phase=='manager' else p.worker.temporal.history
        lagged=sampler(token,h)
        value=self.critic(lagged)
        manager_event=None
        if event:
            features,legal,caps,goals,dests,kinds,_=p.candidates(sem,pos,enemies,cls,holding,t,token,present)
            scores=p.manager(features,token,h)
            sample=phase=='manager' and not deterministic
            choice=(sample_assignment(scores,caps,legal,p.temperature) if sample else
                    p.assignment(scores,legal,caps))
            if teacher_task is not None:
                # Prefer semantic task matches; unavailable teacher support is explicitly masked from BC.
                teacher_task=torch.as_tensor(teacher_task,device=self.device)
                semantic=(kinds==teacher_task[:,1,None].long())
                region_task=(kinds==6)|(kinds==7)|(kinds==8)|(kinds==9)
                match=((goals==teacher_task[:,0,None].long())|region_task)&semantic&legal
                label=match.float().argmax(-1)
                usage=torch.nn.functional.one_hot(label,32).sum(0)
                valid_label=bool(match.any(-1).all() and (usage<=caps).all())
                if valid_label:choice=label
            else:valid_label=False
            logp=joint_log_prob(scores,choice,caps,legal,p.temperature)
            manager_event=dict(features=features.cpu(),legal=legal.cpu(),caps=caps.cpu(),
                assignment=choice.cpu(),old_log_prob=logp.cpu(),bc_valid=valid_label)
            t=p.assign(t,choice,goals,dests,kinds)
        local,wv,t,nav=p.control(sem,pos,enemies,cls,holding,t,self.state[3],event)
        logits,next_hidden=p.worker(local,wv,token,h,hidden)
        distribution=torch.distributions.Categorical(logits=logits)
        action_index=logits.argmax(-1) if deterministic or phase=='manager' else distribution.sample()
        teacher_index=torch.zeros(5,dtype=torch.long,device=self.device)
        if teacher_action is not None:
            teacher=torch.as_tensor(teacher_action,device=self.device)
            if bool(side):teacher=teacher.flip(-1)
            teacher_index=(teacher@p.worker.directions.T).argmax(-1)
            teacher_index=torch.where(teacher.norm(dim=-1)<1e-5,torch.full_like(teacher_index,len(p.worker.directions)-1),teacher_index)
            action_index=teacher_index
        action_index=torch.where(present,action_index,torch.full_like(action_index,len(p.worker.directions)-1))
        logp=(distribution.log_prob(action_index)*present).sum()
        action=p.worker.directions[action_index] if teacher_action is None else teacher
        if p.planner_worker and teacher_action is None:
            action=p.planner_action(pos,enemies,t,nav)
        action,history,t=p.finish(action,token,h,t,pos,cls,holding,side,present)
        old=self.state
        self.state=(torch.where(present[:,None,None],history,old[0]),
                    torch.where(present[:,None],next_hidden,old[1]),
                    torch.where(present[:,None],t,old[2]),
                    torch.where(present[:,None,None],nav,old[3]))
        self.diagnostics.observe(old[2],self.state[2],sem,clear,event,
                                 valid_label if teacher_task is not None and event else None)
        row=dict(token=token.cpu(),local=local.to('cpu',dtype=torch.float16),vector=wv.cpu(),
            reset=clear.cpu(),present=present.cpu(),action=action_index.cpu(),old_log_prob=logp.cpu(),value=value.cpu(),
            hidden=hidden.cpu(),manager=manager_event,teacher_action=teacher_index.cpu(),
            bc_mask=(present.cpu() if teacher_task is None else
                     (((t[:,0].long()==torch.as_tensor(teacher_task,device=self.device)[:,0])|(t[:,2]>=6)) &
                      (t[:,2].long()==torch.as_tensor(teacher_task,device=self.device)[:,1])&present).cpu()),
            reward=0.,done=False)
        self.prefix.append(row)
        return action.cpu().numpy(),row

    @torch.no_grad()
    def bootstrap(self,vector,graphic):
        result=self.policy.prepare(torch.as_tensor(vector,device=self.device),torch.as_tensor(graphic,device=self.device),*self.state)
        sampler=self.policy.manager.temporal.history if getattr(self,'value_phase','worker')=='manager' else self.policy.worker.temporal.history
        return float(self.critic(sampler(result[11],result[8])))

    def prefix_snapshot(self):
        rows=list(self.prefix)
        prefix=rows[-64:]
        older=rows[:-64] if len(rows)>64 else []
        history=torch.zeros(5,64,128)
        for row in older:
            history=history.masked_fill(row['reset'][:,None,None],0)
            history=torch.where(row['present'][:,None,None],torch.cat((row['token'][:,None],history[:,:63]),1),history)
        hidden=prefix[0]['hidden'].clone() if prefix else torch.zeros(5,64)
        # If no prefix is available (new episode), state is the exact initial state.
        return dict(prefix=prefix,history=history,hidden=hidden)

    @torch.no_grad()
    def refresh_hidden(self):
        """Reconstruct with current weights after a version boundary, retaining raw task state."""
        snap=self.prefix_snapshot();h=snap['history'].to(self.device);hidden=snap['hidden'].to(self.device)
        for row in snap['prefix']:
            reset=row['reset'].to(self.device)
            h=h.masked_fill(reset[:,None,None],0);hidden=hidden.masked_fill(reset[:,None],0)
            token=row['token'].to(self.device)
            _,updated=self.policy.worker(row['local'].to(self.device,dtype=torch.float32),row['vector'].to(self.device),token,h,hidden)
            present=row['present'].to(self.device)
            hidden=torch.where(present[:,None],updated,hidden)
            h=torch.where(present[:,None,None],torch.cat((token[:,None],h[:,:63]),1),h)
        self.state=(self.state[0],hidden,self.state[2],self.state[3])
