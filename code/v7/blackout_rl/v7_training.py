"""Fixed-schedule v7 collection and checkpointing shared by residual and direct experiments."""
from dataclasses import asdict
from datetime import datetime, timezone
import copy
import fcntl
import json
import os
from pathlib import Path
import platform
import random
import signal
import time
import numpy as np
import torch
from blackout_rl.batching import team_agents
from blackout_rl.connectome.graph_artifact import atomic_json,digest,sha256
from blackout_rl.env import ContractBlackOutEnv
from blackout_rl.frozen_opponent import FrozenScriptedOpponent
from blackout_rl.mappo_v6 import PlannerFeatures,ROLES,joint_distribution,apply_joint_action
from blackout_rl.mappo_v6_training import central_features,update_v6
from blackout_rl.policy import DeterministicCheckpointPolicy
from blackout_rl.ppo import PPOConfig
from blackout_rl.rollout import generalized_advantage_estimate
from blackout_rl.training_reward import TeamTrainingReward,TrainingRewardConfig,RewardMode
from blackout_rl.v7_registry import ROOT,path,make_model,source_fingerprint
from blackout_rl.v7_2.policy import DirectPolicy
from blackout_rl.v7_2.decoder import DIRECTIONS
from blackout_rl.v7_2.teammates import make_teammates
from blackout_rl.v7_2.training import update_readout


def append(path_, record):
    with Path(path_).open('a') as f:
        f.write(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),**record),allow_nan=False)+'\n')


def opponent_for(kind, team, cfg):
    if kind == 'target':
        with torch.random.fork_rng():
            return DeterministicCheckpointPolicy(path(cfg['training']['opponent_checkpoint']),team=team)
    if kind == 'weak':
        return FrozenScriptedOpponent(team,roles=ROLES,chase_radius_cells=12,policy_id='weak-win70-v1')
    if kind == 'scripted':
        return FrozenScriptedOpponent(team)
    raise ValueError('unknown opponent '+kind)


class Collector:
    def __init__(self,env,model,graph,cfg,splits,seed,run):
        self.env,self.model,self.cfg,self.run=env,model,cfg,run
        self.seeds=splits['train']; self.seed=seed
        self.map_rng=np.random.default_rng(seed+10000)
        self.opponent_rng=np.random.default_rng(seed+20000)
        self.global_step=self.episode_count=self.episode_steps=0
        self.obs=None
        self.direct=cfg['mode']=='whole_connectome_direct'
        self.fixed=self.direct and cfg['controller']['training_mode']=='F0_fixed'
        self.policy=DirectPolicy(graph,None if self.fixed else model,controlled_slots=cfg['controller']['controlled_slots'],
                                 noise_seed=seed+30000,sensor_interval=cfg['sensory']['sensor_interval_game_steps'],
                                 intervention=cfg['controller'].get('intervention','normal'),decoder_config=cfg['controller']['fixed_decoder']) if self.direct else None
        self.neural_ticks=self.frames=0
        self.lineage=[]

    def reset(self):
        # Registered exact A/B episode allocation, unaffected by win/loss.
        self.team=(0,0,1,1,1)[self.episode_count % 5]
        self.map_seed=int(self.map_rng.choice(self.seeds))
        self.episode_id=f'{self.seed}:{self.episode_count}'
        self.episode_count+=1; self.episode_steps=0
        self.obs,_=self.env.reset(seed=self.map_seed)
        self.agents=team_agents(self.team); self.other=team_agents(1-self.team)
        self.context=PlannerFeatures(self.team) if not self.direct else None
        self.reward=TeamTrainingReward(self.team,TrainingRewardConfig(mode=RewardMode.SCORE_DELTA))
        self.reward.reset()
        stage=next(s for s in self.cfg['training']['opponent_schedule'] if self.global_step < s['until'])
        self.opponent_kind=str(self.opponent_rng.choice(['scripted','weak','target'],p=stage['probabilities']))
        self.opponent=opponent_for(self.opponent_kind,1-self.team,self.cfg); self.opponent.reset()
        if self.direct:
            self.policy.reset_episode_state(self.episode_id,self.team,noise_seed=self.seed+30000+self.episode_count)
            # Only the uncontrolled slots use the explicitly registered fixed teammate policy.
            self.teammate=make_teammates(self.team,self.cfg)
            self.teammate.reset()
        self.previous_positions=None
        self.displacement=0.; self.stationary_steps=0

    def collect(self,steps):
        if self.obs is None:
            self.reset()
        rows=[]; overrides=0; start=time.monotonic()
        for _ in range(steps):
            with torch.no_grad():
                central=central_features(self.obs,self.team,'cpu')
                value=self.model.critic(central).squeeze(-1)
                if self.direct:
                    before_frames=self.policy.frames
                    rates=self.policy.features(self.obs,episode_id=self.episode_id,env_step_id=self.episode_steps,team_id=self.team)
                    self.frames+=self.policy.frames-before_frames; self.neural_ticks+=len(self.policy.slots)
                    features=torch.tensor(rates)
                    if self.fixed:
                        controlled=self.policy.act(self.obs,episode_id=self.episode_id,env_step_id=self.episode_steps,team_id=self.team)
                        action=torch.zeros(len(self.policy.slots),dtype=torch.long); logp=torch.tensor(0.)
                    else:
                        distribution=self.model.distribution(features)
                        action=distribution.sample(); logp=distribution.log_prob(action)
                        self.policy.commit_actions(action.numpy())
                        controlled={f'unit_{self.team*5+s}':DIRECTIONS[int(action[i])].copy() for i,s in enumerate(self.policy.slots)}
                    actions={} if len(self.policy.slots)==5 else self.teammate.act(self.obs,self.agents)
                    actions.update(controlled)
                    for event in getattr(self.teammate, 'last_events', []):
                        if event['kind'] == 'assignment_unreachable':
                            append(self.run/'diagnostics.jsonl',dict(event='scripted_teammate_unreachable',
                                environment_steps=self.global_step,episode_id=self.episode_id,detail=event))
                    valid=None
                else:
                    features,valid,planner,alternatives=self.context.prepare(self.obs,self.model)
                    distribution=joint_distribution(self.model.logits(features),valid,self.cfg['training']['exploration'])
                    action=distribution.sample(); logp=distribution.log_prob(action)
                    overrides+=int(action!=0)
                    array=apply_joint_action(int(action),planner,alternatives)
                    actions={a:array[i] for i,a in enumerate(self.agents)}
                actions.update(self.opponent.act(self.obs,self.other))
            next_obs,rewards,terms,truncs,infos=self.env.step(actions)
            terminated,truncated=all(terms.values()),all(truncs.values())
            if any(terms.values())!=terminated or any(truncs.values())!=truncated or (terminated and truncated):
                raise RuntimeError('invalid episode boundary')
            transformed=self.reward(rewards,terms,truncs,infos,self.agents)
            reward=float(np.mean([transformed[a] for a in self.agents]))
            with torch.no_grad():
                next_value=torch.zeros_like(value) if terminated else self.model.critic(central_features(next_obs,self.team,'cpu')).squeeze(-1)
            row=dict(features=features.detach().cpu(),central=central.cpu(),action=action.cpu(),old_log_prob=logp.cpu(),old_value=value.cpu(),
                     reward=torch.tensor(reward),next_value=next_value.cpu(),terminated=torch.tensor(terminated),truncated=torch.tensor(truncated))
            if valid is not None:
                row['valid']=valid.cpu()
            rows.append(row)
            if self.direct:
                positions=np.array([next_obs[a]['vector'][:90].reshape(10,9)[self.team*5+s,:2] for s,a in [(s,f'unit_{self.team*5+s}') for s in self.policy.slots]])
                previous=np.array([self.obs[a]['vector'][:90].reshape(10,9)[self.team*5+s,:2] for s,a in [(s,f'unit_{self.team*5+s}') for s in self.policy.slots]])
                delta=np.linalg.norm(positions-previous,axis=-1)
                self.displacement+=float(delta.sum()); self.stationary_steps+=int((delta<1e-6).sum())
                if self.global_step % 100 == 0:
                    append(self.run/'diagnostics.jsonl',dict(environment_steps=self.global_step,episode_id=self.episode_id,
                        env_step_id=self.episode_steps,team=self.team,neural_ticks=self.neural_ticks,frames=self.frames,
                        rates=rates.tolist(),requested={a:actions[a].tolist() for a in controlled},actual_displacement=delta.tolist()))
            self.global_step+=1; self.episode_steps+=1; self.obs=next_obs
            if terminated or truncated:
                winner=int(infos[self.agents[0]]['winner']) if terminated else None
                if terminated and winner not in (-1,0,1):
                    raise ValueError('terminal winner missing/invalid')
                append(self.run/'episodes.jsonl',dict(seed=self.map_seed,team=self.team,opponent=self.opponent_kind,
                    episode_id=self.episode_id,environment_steps=self.global_step,episode_steps=self.episode_steps,
                    terminal_winner=winner,result=None if winner is None else 'draw' if winner==-1 else 'win' if winner==self.team else 'loss',
                    score=list(self.reward.tracker.score),controlled_displacement=self.displacement,controlled_stationary_steps=self.stationary_steps))
                self.reset()
            elif self.episode_steps>=self.cfg['environment']['max_episode_steps']:
                raise RuntimeError('no terminal within registered episode watchdog')
        batch={key:torch.stack([r[key] for r in rows]) for key in rows[0]}
        advantage,returns=generalized_advantage_estimate(batch['reward'][:,None],batch['old_value'][:,None],batch['next_value'][:,None],
            batch['terminated'][:,None],batch['truncated'][:,None],gamma=self.cfg['training']['gamma'],gae_lambda=self.cfg['training']['gae_lambda'])
        batch['advantage'],batch['return_']=advantage[:,0],returns[:,0]
        batch['exploration']=self.cfg['training']['exploration']
        return batch,dict(environment_steps=self.global_step,unit_actions=self.global_step*5,neural_ticks=self.neural_ticks,
                          sensor_frames=self.frames,team_override_rate=overrides/steps,steps_per_second=steps/(time.monotonic()-start))


def tensorize(value):
    if isinstance(value,np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value,dict):
        return {k:tensorize(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [tensorize(v) for v in value]
    return value


def save_checkpoint(filename,model,optimizer,collector,cfg,seed,sources):
    payload=dict(schema='blackout.v7.checkpoint.v1',config=cfg,config_sha256=digest(cfg),seed=seed,sources=sources,
        lineage=collector.lineage,model=model.state_dict(),optimizer=optimizer.state_dict() if optimizer is not None else None,torch_rng=torch.get_rng_state(),python_rng=random.getstate(),
        global_step=collector.global_step,episode_count=collector.episode_count,map_rng=collector.map_rng.bit_generator.state,
        opponent_rng=collector.opponent_rng.bit_generator.state,neural_ticks=collector.neural_ticks,sensor_frames=collector.frames,
        partial_episode=dict(episode_id=collector.episode_id,steps=collector.episode_steps,resume='discard_and_reset_unity'),
        neural_state=tensorize(collector.policy.state_dict()) if collector.direct else None,
        log_offsets={name:(collector.run/name).stat().st_size if (collector.run/name).exists() else 0 for name in ('training.jsonl','episodes.jsonl','diagnostics.jsonl')})
    tmp=filename.with_suffix('.tmp'); torch.save(payload,tmp); tmp.replace(filename)


def train(cfg,graph,splits,*,seed,run,resume=False):
    run=path(run); run.mkdir(parents=True,exist_ok=True)
    with (run/'run.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('experiment already running in this directory')
        checkpoint=run/'checkpoints/latest.pt'
        if not resume and any((run/n).exists() for n in ('resolved_config.json','training.jsonl','checkpoints/latest.pt')):
            raise FileExistsError('run outputs already exist; use --resume or a new --run-dir')
        if resume and not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        (run/'checkpoints').mkdir(exist_ok=True)
        torch.set_num_threads(cfg['runtime']['torch_threads']); torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        model=make_model(cfg,graph)
        parameters=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.Adam(parameters,lr=cfg['training']['learning_rate']) if parameters else None
        torch.manual_seed(seed+40000)  # Policy sampling RNG is separate from architecture initialization.
        sources=source_fingerprint()
        saved=None
        if resume:
            saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
            if saved['global_step'] >= cfg['training']['max_environment_steps']:
                raise ValueError('registered budget already complete; do not restart this run')
            if saved['config_sha256']!=digest(cfg) or saved['sources']!=sources or saved['seed']!=seed:
                raise ValueError('resume config/source/seed mismatch')
            model.load_state_dict(saved['model'])
            if optimizer is not None: optimizer.load_state_dict(saved['optimizer'])
            torch.set_rng_state(saved['torch_rng']); random.setstate(saved['python_rng'])
            for name,offset in saved['log_offsets'].items():
                p=run/name
                if not p.exists() and offset==0: continue
                if not p.exists() or p.stat().st_size<offset: raise ValueError('resume log shorter than checkpoint')
                if p.stat().st_size>offset:
                    with p.open('rb') as f: f.seek(offset); tail=f.read()
                    (run/f'{name}.recovery_{time.time_ns()}').write_bytes(tail)
                    with p.open('r+b') as f: f.truncate(offset)
        else:
            atomic_json(run/'resolved_config.json',cfg)
            atomic_json(run/'experiment_manifest.json',dict(seed=seed,config_sha256=digest(cfg),sources=sources,
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                frozen_parameters=sum(p.numel() for p in model.parameters() if not p.requires_grad),
                selection='latest_at_registered_budget; no final-test selection',mode=cfg['mode']))
            atomic_json(run/'graph_audit.json',graph.metadata['counts'])
            atomic_json(run/'source_manifest.json',json.loads(path(cfg['data']['source_manifest']).read_text()))
        atomic_json(run/'runtime_info.json',dict(python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__,pid=os.getpid()))
        (run/'training.pid').write_text(str(os.getpid()))
        stopped=False
        def stop(signum,frame):
            nonlocal stopped
            stopped=True
        old={s:signal.signal(s,stop) for s in (signal.SIGTERM,signal.SIGINT)}
        env=None; collector=None
        try:
            env=ContractBlackOutEnv(env_path=str(path(cfg['environment']['build'])),background=True,no_graphics=False,time_scale=cfg['environment']['time_scale'])
            collector=Collector(env,model,graph,cfg,splits,seed,run)
            if saved:
                collector.lineage=saved.get('lineage',[])
                for key in ('global_step','episode_count','neural_ticks'):
                    setattr(collector,key,saved[key])
                collector.frames=saved['sensor_frames']
                collector.map_rng.bit_generator.state=saved['map_rng']; collector.opponent_rng.bit_generator.state=saved['opponent_rng']
                append(run/'training.jsonl',dict(event='resume_new_episode',discarded=saved['partial_episode']))
            atomic_json(run/'status.json',dict(status='running',pid=os.getpid(),environment_steps=collector.global_step))
            last_save=collector.global_step
            while collector.global_step<cfg['training']['max_environment_steps'] and not stopped:
                steps=min(cfg['training']['rollout_steps'],cfg['training']['max_environment_steps']-collector.global_step)
                batch,diagnostics=collector.collect(steps)
                metrics={} if collector.fixed else (update_readout if collector.direct else update_v6)(model,optimizer,batch,PPOConfig(**cfg['training']['ppo']))
                append(run/'training.jsonl',dict(**diagnostics,ppo=metrics))
                atomic_json(run/'status.json',dict(status='running',pid=os.getpid(),**diagnostics))
                print(json.dumps(dict(**diagnostics,ppo=metrics)),flush=True)
                if collector.global_step-last_save>=cfg['training']['save_every']:
                    save_checkpoint(checkpoint,model,optimizer,collector,cfg,seed,sources); last_save=collector.global_step
            save_checkpoint(checkpoint,model,optimizer,collector,cfg,seed,sources)
            atomic_json(run/'status.json',dict(status='stopped' if stopped else 'budget_complete',environment_steps=collector.global_step,pid=os.getpid()))
        except BaseException as exc:
            atomic_json(run/'status.json',dict(status='failed',error=str(exc),pid=os.getpid(),environment_steps=collector.global_step if collector else 0))
            raise
        finally:
            if env is not None: env.close()
            for sig,handler in old.items(): signal.signal(sig,handler)
            (run/'training.pid').unlink(missing_ok=True)
