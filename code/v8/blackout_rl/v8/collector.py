from dataclasses import asdict
from pathlib import Path
import time
import numpy as np
import torch
from blackout_rl.mappo_v6_training import central_features
from blackout_rl.batching import team_agents
from .contracts import ContractError
from .reward import RewardLedger
from .returns import gae
from .telemetry import append,Telemetry,RequestTimer
from .exposure import ExposureLedger
from .observation import compact
from .opponents import make_opponent

class Collector:
    def __init__(self,env,policy,config,seed,directory):
        self.env,self.policy,self.config=env,policy,config
        self.directory=Path(directory);self.map_rng=np.random.default_rng(seed+10000);self.opponent_rng=np.random.default_rng(seed+20000)
        self.global_step=self.episode_count=self.episode_steps=self.version=0
        self.obs=None;self.exposure=ExposureLedger(self.directory/'exposure.jsonl')
        self.telemetry=Telemetry(self.directory,**config['telemetry'],seed=seed+50000)
        self.request_timer=RequestTimer()

    def reset(self):
        self.team=(0,0,1,1,1)[self.episode_count%5]
        self.map_seed=int(self.map_rng.choice(self.config['training']['maps']))
        self.obs,info=self.env.reset(seed=self.map_seed)
        self.episode_count+=1;self.episode_steps=0
        ctx=self.env.context(self.team,self.version);self.policy.reset(ctx)
        self.reward=RewardLedger(self.team,**self.config['training']['reward'])
        self.reward.reset(ctx.run_id,ctx.episode_id,info['score_points'])
        stage=next(s for s in self.config['training']['opponent_schedule'] if self.global_step<s['until'])
        self.opponent_kind=str(self.opponent_rng.choice(['scripted','weak','target'],p=stage['probabilities']))
        self.opponent,self.opponent_hash=make_opponent(self.opponent_kind,1-self.team,self.config['opponent'])

    @torch.no_grad()
    def collect(self,count):
        if count<1:raise ContractError('positive rollout length required')
        if self.obs is None:self.reset()
        rows=[];started=time.monotonic()
        for _ in range(count):
            context=self.env.context(self.team,self.version)
            central=central_features(self.obs,self.team,'cpu')
            value=self.policy.model.critic(central).squeeze(-1)
            decision=self.request_timer.measure(self.policy.decide,self.obs,context,behavior=True)
            before=self.obs['unit_0']['vector'][:90].reshape(10,9)[:,:2].copy()
            actions=dict(decision['actions']);actions.update(self.opponent.act(self.obs,team_agents(1-self.team)))
            next_obs,terminated,truncated,info=self.env.step(actions)
            # An invalid transition raises before exposure/training, while attempt ledger records failure.
            if terminated or truncated:
                components=self.reward.finish(info['outcome']);reward=components['reward']
            else:
                reward=self.reward.observe(info['score_points'],context.decision_id)
                components=dict(score_delta=reward,terminal_bonus=0.,reward=reward)
            next_value=torch.zeros_like(value) if terminated else self.policy.model.critic(central_features(next_obs,self.team,'cpu')).squeeze(-1)
            self.global_step+=1;self.episode_steps+=1
            self.exposure.record(context,self.opponent_hash)
            row=dict(features=decision['features'],valid=decision['valid'],action=torch.tensor(decision['action']),
                     old_log_prob=decision['old_log_prob'],central=central,old_value=value,reward=torch.tensor(reward),
                     next_value=next_value,terminated=torch.tensor(terminated),truncated=torch.tensor(truncated),
                     behavior_version=torch.tensor(self.version))
            rows.append(row)
            after=next_obs['unit_0']['vector'][:90].reshape(10,9)[:,:2]
            displacement=np.linalg.norm(after-before,axis=1)
            trace=dict(context=asdict(context),global_step=self.global_step,opponent_hash=self.opponent_hash,
                       observation=compact({a:self.obs[a] for a in team_agents(self.team)}),
                       **{k:v for k,v in decision.items() if k not in ('actions','log_probs')},
                       log_probs=[float(v) if torch.isfinite(v) else None for v in decision['log_probs']],
                       requested_action=decision['actions'],executed_action=decision['actions'],
                       position_before=before,position_after=after,displacement=displacement,
                       reward_components=components,value=value,next_value=next_value,outcome=info['outcome'],
                       terminated=terminated,truncated=truncated)
            triggers=[]
            if terminated or truncated:triggers.append('outcome')
            if not all(decision['teacher_valid'].values()):triggers.append('path_failure')
            if decision['q']>.5 and decision['greedy_action']==0:triggers.append('high_q_keep')
            if decision['greedy_action']!=decision['action']:triggers.append('decoder_difference')
            own_displacement=displacement[self.team*5:self.team*5+5]
            requested=np.array(list(decision['actions'].values()))
            if np.any((np.linalg.norm(requested,axis=-1)>0)&(own_displacement<1e-6)):triggers.append('nonzero_stationary')
            self.telemetry.observe(trace,triggers)
            append(self.directory/'decisions.jsonl',dict(context=asdict(context),global_step=self.global_step,q=decision['q'],
                p_keep=decision['p_keep'],max_correction=decision['max_correction'],margin=decision['margin'],
                sampled=decision['action'],greedy=decision['greedy_action'],executed=decision['action'],
                valid_count=int(decision['valid'].sum()),displacement=own_displacement,reward=components,opponent_hash=self.opponent_hash))
            self.obs=next_obs
            if terminated or truncated:
                append(self.directory/'episodes.jsonl',dict(context=asdict(context),steps=self.episode_steps,opponent_hash=self.opponent_hash,outcome=info['outcome']))
                self.reset()
            elif self.episode_steps>=self.config['environment']['max_episode_steps']:
                raise ContractError('external watchdog: invalid attempt, not game timeout')
        batch={k:torch.stack([r[k] for r in rows]) for k in rows[0]}
        batch['advantage'],batch['return_']=gae(batch['reward'],batch['old_value'],batch['next_value'],batch['terminated'],batch['truncated'],
            self.config['training']['gamma'],self.config['training']['gae_lambda'])
        append(self.directory/'return_samples.jsonl',dict(end_step=self.global_step,behavior_version=self.version,
            reward=batch['reward'],value=batch['old_value'],next_value=batch['next_value'],advantage=batch['advantage'],return_=batch['return_'],
            action=batch['action'],old_log_prob=batch['old_log_prob'],terminated=batch['terminated'],truncated=batch['truncated']))
        return batch,dict(seconds=time.monotonic()-started,steps=count,global_step=self.global_step,policy_request=self.request_timer.report())

    def state_dict(self):
        return dict(global_step=self.global_step,episode_count=self.episode_count,behavior_version=self.version,
                    map_rng=self.map_rng.bit_generator.state,opponent_rng=self.opponent_rng.bit_generator.state,
                    exposure=self.exposure.state_dict(),policy_rng=self.policy.generator.get_state(),
                    partial=None if self.obs is None else dict(episode_id=self.env.episode,map_seed=self.map_seed,side=self.team,
                    opponent_hash=self.opponent_hash,steps=self.episode_steps),resume_kind='episode_restart_resume')

    def restore(self,state):
        self.global_step=state['global_step'];self.episode_count=state['episode_count'];self.version=state['behavior_version']
        self.map_rng.bit_generator.state=state['map_rng'];self.opponent_rng.bit_generator.state=state['opponent_rng']
        self.policy.generator.set_state(state['policy_rng']);self.exposure.load_state_dict(state['exposure'])
        self.env.counter=self.episode_count  # Never reuse prior episode IDs after process restart.
        append(self.directory/'resume.jsonl',dict(resume_kind='episode_restart_resume',discarded_partial=state['partial'],physical_state_restored=False,rollout_policy='checkpoint_at_completed_update_only'))
        self.obs=None
