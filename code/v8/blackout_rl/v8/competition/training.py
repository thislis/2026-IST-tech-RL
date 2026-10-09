"""PPO using only unmodified BlackOutEnv reset/step results and legal observations."""
import json
from pathlib import Path
import random
import time
import torch
from torch import nn
import numpy as np
from blackout_rl.mappo_v6_training import central_features
from blackout_rl.v8.checkpoints import CheckpointStore,atomic_json,fingerprints
from blackout_rl.v8.contracts import digest,file_hash
from blackout_rl.v8.returns import gae
from blackout_rl.v8.opponents import make_opponent
from blackout_rl.v8.telemetry import append
from .policy import MyPolicy


def create_policy(mode,seed,encoder_path):
    with torch.random.fork_rng():
        torch.manual_seed(seed);policy=MyPolicy();policy.configure(mode)
        payload=torch.load(encoder_path,map_location='cpu',weights_only=True)
        state={k.removeprefix('actor_critic.'):v for k,v in payload['policy_state'].items()}
        policy.encoder.load_state_dict(state,strict=True)
    return policy


def pack(obs,names):
    # Input tensors are copies; never mutate the provided observation arrays.
    vector=torch.from_numpy(np.stack([obs[n]['vector'] for n in names])).float()
    graphic=torch.from_numpy(np.stack([obs[n]['graphic'] for n in names])).float().permute(0,3,1,2)
    return vector,graphic


def joint_log_prob(log_probs,actions):
    return log_probs.gather(-1,actions[...,None]).squeeze(-1).sum(-1)


class ActorCritic(nn.Module):
    def __init__(self,policy):
        super().__init__();self.policy=policy
        self.critic=nn.Sequential(nn.Linear(299,256),nn.Tanh(),nn.Linear(256,128),nn.Tanh(),nn.Linear(128,1))


class Collector:
    def __init__(self,model,cfg,seed,env_factory,directory,opponent_factory=make_opponent,progress=None):
        self.model,self.cfg,self.factory,self.directory=model,cfg,env_factory,Path(directory)
        self.opponent_factory=opponent_factory;self.progress=progress
        self.rng=random.Random(seed+10000);self.generator=torch.Generator().manual_seed(seed+40000)
        self.global_step=0;self.episodes=0;self.env=None;self.obs=None;self.episode_step=0;self.team=0
        self.export_inputs=[];self.inputs_saved=(self.directory/'policy_inputs.pt').exists()

    def close(self):
        if self.env is not None:self.env.close();self.env=None

    def reset(self):
        # Fresh unchanged instance per episode avoids carrying known timer/cache
        # failures across episodes without patching reset or adding hidden steps.
        self.close();self.env=self.factory();self.team=self.cfg['side_cycle'][self.episodes%len(self.cfg['side_cycle'])]
        self.requested_seed=self.rng.choice(self.cfg['train_seed_requests'])
        self.obs,_=self.env.reset(seed=self.requested_seed)
        if set(self.obs)!={f'unit_{i}' for i in range(10)}:raise ValueError('complete ten-agent observation required')
        self.episodes+=1;self.episode_step=0
        self.names=[f'unit_{5*self.team+i}' for i in range(5)]
        self.other=[f'unit_{5*(1-self.team)+i}' for i in range(5)]
        stage=next(x for x in self.cfg['schedule'] if self.global_step<x['until'])
        kind=self.rng.choices(['scripted','weak','target'],weights=stage['probabilities'])[0]
        self.opponent,self.opponent_hash=self.opponent_factory(kind,1-self.team,self.cfg['opponent'])
        append(self.directory/'episodes.jsonl',dict(event='start',episode=self.episodes,side=self.team,
            requested_seed=self.requested_seed,applied_map_seed_verified=False,opponent=kind,opponent_sha256=self.opponent_hash))

    @torch.no_grad()
    def collect(self,count):
        rows=[];start=time.monotonic()
        if self.obs is None:self.reset()
        for _ in range(count):
            vector,graphic=pack(self.obs,self.names)
            if not self.inputs_saved:
                self.export_inputs.append(dict(vector=vector.clone(),graphic=graphic.clone()))
                if len(self.export_inputs)==8:
                    self.directory.mkdir(parents=True,exist_ok=True)
                    temporary=self.directory/'policy_inputs.tmp'
                    torch.save(self.export_inputs,temporary);temporary.replace(self.directory/'policy_inputs.pt')
                    self.export_inputs=[];self.inputs_saved=True
            features,baseline,valid=self.model.policy.features(vector,graphic)
            log_probs=self.model.policy.log_distribution(features,valid)
            action=torch.multinomial(log_probs.exp(),1,generator=self.generator).squeeze(-1)
            selected=self.model.policy.execute(action,baseline).numpy()
            actions={name:selected[i] for i,name in enumerate(self.names)}
            actions.update(self.opponent.act(self.obs,self.other))
            central=central_features(self.obs,self.team,'cpu');value=self.model.critic(central).squeeze(-1)
            obs,rewards,terms,truncs,infos=self.env.step(actions)
            if not all(n in rewards and n in terms and n in truncs for n in self.names):raise ValueError('incomplete transition')
            if len(set(bool(terms[n]) for n in self.names))!=1 or len(set(bool(truncs[n]) for n in self.names))!=1:
                raise ValueError('partial team termination unsupported; attempt invalid')
            terminated=all(terms[n] for n in self.names);truncated=all(truncs[n] for n in self.names)
            if terminated and truncated:raise ValueError('mixed termination/truncation')
            # Training-only team aggregation; the supplied reward itself is untouched.
            reward=sum(float(rewards[n]) for n in self.names)/5
            next_value=value.new_zeros(()) if terminated else self.model.critic(central_features(obs,self.team,'cpu')).squeeze(-1)
            rows.append(dict(features=features,valid=valid,action=action,old_log_prob=joint_log_prob(log_probs,action),
                central=central,old_value=value,reward=torch.tensor(reward),next_value=next_value,
                terminated=torch.tensor(terminated),truncated=torch.tensor(truncated)))
            self.global_step+=1;self.episode_step+=1;self.obs=obs
            append(self.directory/'exposure.jsonl',dict(step=self.global_step,episode=self.episodes,side=self.team,
                requested_seed=self.requested_seed,opponent_sha256=self.opponent_hash,reward=reward,
                action=action.tolist(),greedy=self.model.policy.decode(log_probs).tolist(),q=(1-log_probs[:,0].exp()).tolist(),
                valid_count=valid.sum(-1).tolist(),terminated=terminated,truncated=truncated))
            if self.progress is not None:self.progress(self.global_step)
            if terminated or truncated:
                append(self.directory/'episodes.jsonl',dict(event='end',episode=self.episodes,steps=self.episode_step,
                    provided_info=infos,metric='training episode; not competition win classification'))
                self.close();self.obs=None
                if len(rows)<count:self.reset()
            elif self.episode_step>=self.cfg['max_episode_steps']:
                raise RuntimeError('original environment episode watchdog; invalid attempt, not a draw/timeout win')
        batch={key:torch.stack([r[key] for r in rows]) for key in rows[0]}
        batch['advantage'],batch['return_']=gae(batch['reward'],batch['old_value'],batch['next_value'],batch['terminated'],batch['truncated'],self.cfg['gamma'],self.cfg['gae_lambda'])
        return batch,dict(steps=count,seconds=time.monotonic()-start)

    def state(self):
        return dict(global_step=self.global_step,episodes=self.episodes,rng=self.rng.getstate(),generator=self.generator.get_state(),
                    discarded_partial=None if self.obs is None else dict(episode=self.episodes,side=self.team,steps=self.episode_step),resume='episode_restart')

    def restore(self,state):
        self.global_step=state['global_step'];self.episodes=state['episodes'];self.rng.setstate(state['rng']);self.generator.set_state(state['generator'])
        self.obs=None
        append(self.directory/'resume.jsonl',dict(kind='episode_restart',partial=state['discarded_partial'],physics_restored=False))


def update(model,optimizer,batch,cfg,rng):
    advantage=batch['advantage'];advantage=(advantage-advantage.mean())/advantage.std(unbiased=False).clamp_min(1e-8)
    metrics=[]
    for _ in range(cfg['epochs']):
        for ids in torch.randperm(len(advantage),generator=rng).split(cfg['minibatch_size']):
            logs=model.policy.log_distribution(batch['features'][ids],batch['valid'][ids])
            logratio=joint_log_prob(logs,batch['action'][ids])-batch['old_log_prob'][ids]
            ratio=logratio.exp();kl=((ratio-1)-logratio).mean()
            if not torch.isfinite(kl):raise ValueError('nonfinite ratio')
            if float(kl.detach())>cfg['target_kl']:return dict(minibatches=len(metrics),early_stopped=True,metrics=metrics)
            actor=torch.maximum(-advantage[ids]*ratio,-advantage[ids]*ratio.clamp(1-cfg['clip'],1+cfg['clip'])).mean()
            value=model.critic(batch['central'][ids]).squeeze(-1)
            clipped=batch['old_value'][ids]+(value-batch['old_value'][ids]).clamp(-cfg['value_clip'],cfg['value_clip'])
            loss_v=.5*torch.maximum((value-batch['return_'][ids]).square(),(clipped-batch['return_'][ids]).square()).mean()
            entropy=-(logs.exp()*logs.masked_fill(~torch.isfinite(logs),0)).sum((-1,-2)).mean()
            loss=actor+cfg['value_loss_coef']*loss_v-cfg['entropy_coef']*entropy
            if not torch.isfinite(loss):raise ValueError('nonfinite loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            norm=nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],cfg['max_grad_norm'],error_if_nonfinite=True)
            optimizer.step();metrics.append(dict(kl=float(kl.detach()),entropy=float(entropy.detach()),value_loss=float(loss_v.detach()),gradient_norm=float(norm)))
    return dict(minibatches=len(metrics),early_stopped=False,metrics=metrics)


def train(model,collector,cfg,directory,registration_sha,stop=lambda:False,on_checkpoint=None):
    import fcntl
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    with (directory/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=cfg['ppo']['learning_rate'])
        rng=torch.Generator().manual_seed(cfg['seed']+60000);store=CheckpointStore(directory/'checkpoints');parent=None
        source=fingerprints();config_hash=digest(cfg)
        if (store.directory/'latest.json').exists():
            saved,parent=store.load()
            if saved['sources']!=source or saved['registration_sha256']!=registration_sha or saved['config_sha256']!=config_hash:raise ValueError('resume provenance mismatch')
            for path in directory.glob('*.jsonl'):
                if path.name in ('resume.jsonl','attempts.jsonl'):continue
                offset=saved['log_offsets'].get(path.name,0)
                if path.stat().st_size<offset:raise ValueError('committed log shortened')
                with path.open('rb') as stream:stream.seek(offset);suffix=stream.read()
                if suffix:
                    recovery=directory/'recovery';recovery.mkdir(exist_ok=True)
                    import hashlib
                    (recovery/(hashlib.sha256(suffix).hexdigest()+'-'+path.name)).write_bytes(suffix)
                    with path.open('r+b') as stream:stream.truncate(offset)
            model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);rng.set_state(saved['ppo_rng'])
            torch.set_rng_state(saved['torch_rng']);collector.restore(saved['collector'])
            # Recover an interrupted pointer/index write after an immutable save.
            atomic_json(directory/'checkpoint_index'/f'{collector.global_step}.json',dict(sha256=parent,step=collector.global_step))
        def save():
            nonlocal parent
            parent=store.save(dict(global_step=collector.global_step,model=model.state_dict(),policy_state=model.policy.state_dict(),
                optimizer=optimizer.state_dict(),collector=collector.state(),ppo_rng=rng.get_state(),torch_rng=torch.get_rng_state(),
                registration_sha256=registration_sha,sources=source,config=cfg,config_sha256=config_hash,
                log_offsets={p.name:p.stat().st_size for p in directory.glob('*.jsonl')}),parent)
            atomic_json(directory/'checkpoint_index'/f'{collector.global_step}.json',dict(sha256=parent,step=collector.global_step))
            if on_checkpoint is not None:on_checkpoint(collector.global_step,parent)
        try:
            if parent is None:save()
            while collector.global_step<cfg['steps']:
                previous_step=collector.global_step
                batch,timing=collector.collect(min(cfg['rollout'],cfg['steps']-collector.global_step))
                metrics=update(model,optimizer,batch,cfg['ppo'],rng)
                append(directory/'training.jsonl',dict(step=collector.global_step,collection=timing,ppo=metrics))
                stopping=stop()
                stage_boundary=any(previous_step<x['until']<=collector.global_step for x in cfg['schedule'])
                if collector.global_step%cfg['save_every']==0 or collector.global_step==cfg['steps'] or stopping or stage_boundary:save()
                atomic_json(directory/'progress.json',dict(step=collector.global_step,budget=cfg['steps'],checkpoint=parent))
                if stopping:break
            complete=collector.global_step==cfg['steps']
            atomic_json(directory/'status.json',dict(state='complete' if complete else 'paused',step=collector.global_step,checkpoint=parent))
            return parent
        except BaseException as exc:
            append(directory/'attempts.jsonl',dict(error=str(exc),last_valid_checkpoint=parent,step=collector.global_step));raise
        finally:collector.close()
