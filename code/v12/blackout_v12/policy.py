"""v12 event-driven team policy with exact capacity-constrained assignment.

The CPU public-observation/FSM boundary actually skips neural scoring, task
generation and assignment on maintenance frames. Each instance owns one team;
call reset() at episode boundaries. Python state is not implicit ONNX state.
"""
import copy
import numpy as np
import torch
from torch import nn
from .contracts import observe, SCHEMA, STATE_SCHEMA
from .navigation import Navigation, center
from .state import TeamState, UnitState
from .tasks import generate_candidates
from .model import TaskScorer
from .features import LEARNED_FEATURE_SCHEMA
from .assignment import (map_assignment, sample_assignment, joint_log_prob,
                         sequential_assignment, sequential_log_prob)

ARMS = ('Q0','Q1','Q2','S','J','F')


class MyPolicy(nn.Module):
    def __init__(self, config=None, arm='F', decoder='map', device='cpu',
                 vector_size=96, n_channels=11):
        super().__init__()
        if vector_size!=96 or n_channels!=11:
            raise ValueError('provided runtime vector_size=96 and n_channels=11 required')
        if arm not in ARMS:
            raise ValueError('unknown v12 arm: '+arm)
        if decoder not in ('map','sample'):
            raise ValueError('decoder must be map or sample')
        self.config = dict(config or {})
        self.policy_mode = self.config.get('policy_mode','residual')
        if self.policy_mode not in ('residual','learned'):
            raise ValueError('policy_mode must be residual or learned')
        if self.policy_mode == 'learned' and arm != 'F':
            raise ValueError('learned task control requires the F arm')
        for key,legacy in (('max_tasks','task_limit'),('event_period','goal_period')):
            if key in self.config and legacy in self.config and self.config[key]!=self.config[legacy]:
                raise ValueError('conflicting config values: '+key+' and '+legacy)
            if legacy in self.config:
                self.config[key]=self.config[legacy]
        for key,default in (('max_tasks',24),('event_period',25),('cache_fields',32)):
            value=self.config.get(key,default)
            if int(value)!=value or int(value)<1:
                raise ValueError(key+' must be a positive integer')
        temperature=float(self.config.get('temperature',1.))
        if not np.isfinite(temperature) or temperature<=0:
            raise ValueError('temperature must be finite and positive')
        self.arm, self.decoder = arm,decoder
        self.scorer = TaskScorer(self.config).to(device)
        self.navigation = Navigation(self.config.get('cache_fields',32))
        # An optional external scorer+MAP call boundary (e.g. ORT) is not part
        # of the state_dict or the portable two-file submission.
        self.event_backend = None
        self.reset()

    @property
    def device(self):
        return next(self.scorer.parameters()).device

    def reset(self):
        self.runtime = TeamState()
        self.navigation.reset()
        self.last_decision = None
        self.last_event = False
        self.last_diagnostics = {}

    def runtime_state(self):
        """Explicit transferable Python state; path cache can be rebuilt."""
        return copy.deepcopy(self.runtime.snapshot())

    def load_runtime_state(self, state):
        if state.get('schema') != STATE_SCHEMA:
            raise ValueError('runtime state schema mismatch')
        restored = TeamState()
        for key,value in state.items():
            if key=='schema':
                continue
            setattr(restored,key,[UnitState(**u) for u in value] if key=='units' else copy.deepcopy(value))
        self.runtime = restored
        self.navigation.reset()

    def _decision(self, obs, deterministic):
        candidates = generate_candidates(obs,self.runtime,self.navigation,self.config,self.arm)
        tensor = lambda value: torch.as_tensor(value,device=self.device)
        features,prior = tensor(candidates.features),tensor(candidates.prior)
        legal,capacities = tensor(candidates.legal),tensor(candidates.capacities)
        temperature = candidates.temperature
        sequential = self.arm in ('Q0','Q1','S')
        if self.event_backend is not None:
            if hasattr(self.event_backend,'validate_policy'):
                self.event_backend.validate_policy(self)
            # Native MAP uses physical IDs to resolve exact ties. Keep the
            # same ordering at the backend boundary, then restore task rows.
            order=np.array(sorted(range(len(candidates.tasks)),key=lambda j:candidates.tasks[j].id))
            assignment,backend_scores=self.event_backend(candidates.features[:,order],candidates.prior[:,order],
                candidates.legal[:,order],candidates.capacities[order])
            assignment=np.asarray(assignment,dtype=np.int64)
            backend_scores=np.asarray(backend_scores,dtype=np.float32)
            if assignment.shape!=(5,) or backend_scores.shape!=candidates.prior.shape:
                raise ValueError('event backend output shape mismatch')
            if (assignment<0).any() or (assignment>=len(order)).any():
                raise ValueError('event backend assignment out of range')
            chosen=tensor(order[assignment])
            scores=tensor(backend_scores[:,np.argsort(order)])
            log_prob=joint_log_prob(scores,chosen,capacities,legal,temperature)
            if not torch.isfinite(log_prob):
                raise ValueError('event backend returned an illegal assignment')
        else:
            scores = prior if self.arm.startswith('Q') else self.scorer(features,prior)
            if sequential:
                chosen=sequential_assignment(scores,capacities,legal,temperature,sample=not deterministic)
                log_prob=sequential_log_prob(scores,chosen,capacities,legal,temperature)
            else:
                chosen=(map_assignment(scores,capacities,legal,temperature,
                                       unit_ids=obs.unit_ids.tolist(),task_ids=[t.id for t in candidates.tasks]) if deterministic else
                        sample_assignment(scores,capacities,legal,temperature))
                log_prob=joint_log_prob(scores,chosen,capacities,legal,temperature)
        selected=chosen.detach().cpu().numpy()
        rows=torch.arange(5,device=self.device)
        selected_scores=scores[rows,chosen]
        selected_prior=prior[rows,chosen]
        changed=0
        for i,j in enumerate(selected):
            task=candidates.tasks[int(j)]
            u=self.runtime.units[i]
            if u.task_id != task.id:
                changed+=1
                u.age=0
                u.no_progress,u.no_progress_seconds,u.progress_goal,u.progress_distance=0,0.,-1,-1.
                u.commitment=(int(self.config.get('commitment_steps',50))
                              if self.policy_mode != 'learned' and task.kind!='hold' else 0)
            u.task_id,u.kind=task.id,task.kind
            u.source=int(candidates.targets[i,j])
            u.destination=int(candidates.destinations[i,j])
            u.phase='return' if task.kind=='deliver' else ('hold' if task.kind=='hold' else 'approach')
        decision={
            'schema':LEARNED_FEATURE_SCHEMA['id'] if self.policy_mode == 'learned' else SCHEMA,
            'features':features.detach().cpu().clone(),
            'prior':prior.detach().cpu().clone(),'legal':legal.detach().cpu().clone(),
            'capacities':capacities.detach().cpu().clone(),'assignment':chosen.detach().cpu().clone(),
            'old_log_prob':log_prob.detach().cpu().clone(),'log_prob':log_prob.detach().cpu().clone(),
            'temperature':temperature,'task_ids':[t.id for t in candidates.tasks],
            'destinations':torch.from_numpy(candidates.destinations.copy()),
            'targets':torch.from_numpy(candidates.targets.copy()),
            'locked':torch.from_numpy(candidates.locked.copy()),
            'selected_scores':selected_scores.detach().cpu().clone(),
            'selected_prior':selected_prior.detach().cpu().clone(),
            'selected_learned_scores':(selected_scores-selected_prior).detach().cpu().clone(),
            'joint_log_partition':None if sequential else
                (selected_scores.sum()/temperature-log_prob).detach().cpu().clone(),
            'critic_features':features.detach().cpu().mean((0,1)),
            'decoder':'sequential' if sequential else 'joint',
            'event_step':self.runtime.step_count,'event_reasons':list(self.runtime.event_reasons)}
        self.runtime.last_event_step=self.runtime.step_count
        self.runtime.last_event_time=obs.time_left
        self.last_diagnostics={**candidates.diagnostics,'changed_assignments':changed,
                               'locked_units':int(candidates.locked.sum()),
                               'selected_prior_sum':float(selected_prior.sum()),
                               'selected_learned_sum':float((selected_scores-selected_prior).sum())}
        if self.config.get('assignment_diagnostics',False):
            joint=map_assignment(scores,capacities,legal,temperature,
                unit_ids=obs.unit_ids.tolist(),task_ids=[t.id for t in candidates.tasks])
            ordered=sequential_assignment(scores,capacities,legal,temperature)
            zero=map_assignment(prior,capacities,legal,temperature,
                unit_ids=obs.unit_ids.tolist(),task_ids=[t.id for t in candidates.tasks])
            self.last_diagnostics.update(joint_minus_sequential_utility=float(
                scores[rows,joint].sum()-scores[rows,ordered].sum()),
                learned_assignment_changed=bool((joint!=zero).any()))
        return decision

    @torch.no_grad()
    def step(self, vector, graphic, deterministic=False):
        if self.event_backend is not None and self.policy_mode == 'learned':
            raise ValueError('learned full task support requires the Python event decoder')
        if self.event_backend is not None and (not deterministic or self.arm in ('Q0','Q1','S')):
            raise ValueError('event backend requires deterministic joint MAP decoding')
        obs=observe(vector,graphic)
        event,reset=self.runtime.update(obs,self.config)
        if reset:
            self.navigation.reset()
        self.last_event=event
        # No synthetic log probability is assigned to a maintenance frame.
        self.last_decision=self._decision(obs,deterministic) if event else None
        action=np.zeros((5,2),np.float32)
        for i,u in enumerate(self.runtime.units):
            walk=self.navigation.walkability(self.runtime,i,obs.classes[i],obs.holding[i],u.kind,
                                             allow_loaded_transform=self.policy_mode=='learned')
            profile=(int(obs.classes[i]),int(obs.holding[i]),1)
            near_enemy=float(np.linalg.norm(obs.enemy_positions-obs.positions[i],axis=-1).min())<.055
            if self.policy_mode != 'learned' and near_enemy and obs.classes[i]!=1:
                # Fixed primitive safety control does not rewrite the sampled
                # task or its probability. The threat interrupt releases locks.
                safe_walk=self.navigation.walkability(self.runtime,i,obs.classes[i],obs.holding[i])
                move,waypoint=self.navigation.evade(obs.positions[i],obs.enemy_positions,safe_walk)
            elif u.kind=='evade':
                # The sampled assignment chooses whether to call this existing
                # geometric primitive. Enemy distance never overrides a task.
                move,waypoint=self.navigation.evade(obs.positions[i],obs.enemy_positions,walk)
            elif u.kind=='hold':
                move,waypoint=np.zeros(2,np.float32),-1
            else:
                goal=u.destination if u.phase=='return' else u.source
                move,waypoint=self.navigation.action(obs.positions[i],goal,walk,profile)
            action[i]=move
            u.previous_action=move.copy()
            u.waypoint=waypoint
        if obs.side:
            action=action[:,::-1].copy()
        self.last_diagnostics.update(event=event,reasons=self.runtime.event_reasons,
                                     path_solves=self.navigation.solves,cache_hits=self.navigation.hits,
                                     cache_bytes=self.navigation.state_bytes)
        out_device=vector.device if isinstance(vector,torch.Tensor) else self.device
        return {'action':torch.as_tensor(action,device=out_device),'event':event,
                'decision':self.last_decision,'diagnostics':dict(self.last_diagnostics)}

    def forward(self, vector, graphic):
        return self.step(vector,graphic,deterministic=self.decoder!='sample')['action']


Policy = MyPolicy
