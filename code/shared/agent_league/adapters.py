"""Per-match policy ownership and independent RNG streams. No learner updates."""
import hashlib
import importlib.util
import random
import subprocess
import types
import sys
import numpy as np
import torch
from .common import ROOT, resolve, verify_assets, sha


def historic_planner(data):
    """Use the checkpoint's exact planner when a historical source is recorded."""
    source=data.get('source', {}); files=source.get('files', {})
    name='blackout_rl/scripted_fsm.py'; expected=files.get(name)
    if not expected:
        from blackout_rl.scripted_fsm import ScriptedTeamController
        return ScriptedTeamController
    from project_paths import historical_source_path
    try: raw=historical_source_path(name,expected).read_bytes()
    except ValueError:
        raw=subprocess.check_output(['git','-C',str(ROOT),'show',source['git_sha']+':'+name])
    if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('historical planner checksum mismatch')
    module=types.ModuleType('_league_planner_'+expected[:12]);module.__package__='blackout_rl'
    sys.modules[module.__name__]=module
    exec(compile(raw,'<league historical planner>','exec'),module.__dict__)
    return module.ScriptedTeamController


class Adapter:
    def __init__(self,spec,team,seed,episode='league',device='cpu'):
        verify_assets(spec)
        self.spec,self.team,self.seed,self.episode,self.device=spec,team,seed,episode,device
        self.names=tuple(f'unit_{i}' for i in range(team*5,team*5+5));self.metadata={}
        # Constructor randomness must not consume either player's action stream.
        outer_rng=(np.random.get_state(),random.getstate())
        try:
            np.random.seed(seed);random.seed(seed)
            with torch.random.fork_rng():
                torch.manual_seed(seed);self._load()
        finally:
            np.random.set_state(outer_rng[0]);random.setstate(outer_rng[1])
        self.torch_rng=torch.Generator().manual_seed(seed).get_state()
        self.numpy_rng=np.random.RandomState(seed).get_state();self.python_rng=random.Random(seed).getstate()

    def _load(self):
        s=self.spec;kind=s['kind'];self.model=None;self.policy=None
        if kind=='bundle':
            path=resolve(s['path']);module_spec=importlib.util.spec_from_file_location('league_'+s['id'].replace('-','_'),path.parent/'policy.py')
            mod=importlib.util.module_from_spec(module_spec)
            previous=sys.dont_write_bytecode
            try:
                sys.dont_write_bytecode=True;module_spec.loader.exec_module(mod)
            finally:sys.dont_write_bytecode=previous
            self.model=mod.MyPolicy(vector_size=96,n_channels=11)
            self.model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['policy_state'],strict=True)
        elif kind=='v8_comp':
            from blackout_rl.v8.competition.policy import MyPolicy
            self.model=MyPolicy();self.model.load_state_dict(torch.load(resolve(s['path']),map_location='cpu',weights_only=True)['policy_state'],strict=True)
        elif kind=='legacy':
            from blackout_rl.policy import DeterministicCheckpointPolicy
            from blackout_rl.team_state import Role
            self.policy=DeterministicCheckpointPolicy(resolve(s['path']),team=self.team,seed=self.seed)
            d=self.policy.checkpoint_payload;guard=d.get('inference_guardrail')
            if guard:
                planner=historic_planner(d)(self.team,seed=self.seed,roles=tuple(Role(x) for x in guard['roles']),chase_radius_cells=guard['chase_radius_cells'])
                if self.policy._v6_policy is not None:self.policy._v6_policy.context.planner=planner
                else:self.policy._safety_controller=planner
            self.policy.reset()
        elif kind=='v7':
            from blackout_rl.v7_registry import check_config,make_model
            from blackout_rl.connectome.graph_artifact import digest
            cfg=s['config'];d=torch.load(resolve(s['path']),map_location='cpu',weights_only=True)
            if digest(cfg)!=d['config_sha256']:raise ValueError('v7 config mismatch')
            graph,_=check_config(cfg,verify_sources=False);model=make_model(cfg,graph);model.load_state_dict(d['model'],strict=True);model.eval()
            if cfg['mode']=='residual_graph':
                from blackout_rl.mappo_v6 import V6Policy
                self.policy=V6Policy(model,team=self.team,seed=self.seed);self.policy.reset()
            else:
                from blackout_rl.v7_2.policy import DirectPolicy
                from blackout_rl.v7_2.teammates import make_teammates
                c=cfg['controller'];self.policy=DirectPolicy(graph,None if c['training_mode']=='F0_fixed' else model,
                    controlled_slots=c['controlled_slots'],noise_seed=self.seed,sensor_interval=cfg['sensory']['sensor_interval_game_steps'],
                    intervention=c.get('intervention','normal'),decoder_config=c['fixed_decoder'])
                self.policy.reset_episode_state(self.episode,self.team,noise_seed=self.seed)
                self.teammates=make_teammates(self.team,cfg);self.teammates.reset()
        elif kind=='v8_legacy':
            from blackout_rl.v8.policy_factory import make_policy
            from blackout_rl.model_contract import load_checkpoint
            d=torch.load(resolve(s['path']),map_location='cpu',weights_only=True);cfg=s['config']
            if sha(resolve(cfg['encoder']['path']))!=cfg['encoder']['sha256']:raise ValueError('v8 frozen encoder hash mismatch')
            actor,_=load_checkpoint(resolve(cfg['encoder']['path']))
            self.policy=make_policy(cfg,checkpoint=d,capabilities=dict(explicit_episode_decision_id=True,canonical_agent_ids=True,stateful=True),seed=self.seed,actor=actor.actor_critic)
        elif kind=='strategy':
            from blackout_v10.opponents import StrategyOpponent
            self.policy=StrategyOpponent(s['config']['kind'],self.seed)
        elif kind=='scripted':
            from blackout_rl.frozen_opponent import FrozenScriptedOpponent
            self.policy=FrozenScriptedOpponent(self.team,seed=self.seed,enable_special_items=s['config']['special'],chase_radius_cells=s['config']['radius'])
            self.policy.reset()
        else:raise ValueError('unsupported adapter '+kind)
        if self.model is not None:self.model.to(self.device).eval()

    def act(self,observations,step):
        # Inputs contain exactly this team's public observations, never opponent-only rows.
        obs={n:observations[n] for n in self.names}
        outer=(torch.get_rng_state(),np.random.get_state(),random.getstate())
        torch.set_rng_state(self.torch_rng);np.random.set_state(self.numpy_rng);random.setstate(self.python_rng)
        try:
            with torch.inference_mode():
                if self.model is not None:
                    v=torch.from_numpy(np.stack([obs[n]['vector'] for n in self.names])).to(self.device)
                    g=torch.from_numpy(np.stack([obs[n]['graphic'] for n in self.names])).permute(0,3,1,2).to(self.device)
                    # Stochastic submissions sample on CPU so results are scheduling-independent.
                    result=dict(zip(self.names,self.model(v,g).cpu().numpy()))
                elif self.spec['kind']=='v7' and self.spec['config']['mode']=='whole_connectome_direct':
                    result=self.teammates.act(obs,self.names)
                    result.update(self.policy.act(obs,episode_id=self.episode,env_step_id=step,team_id=self.team,deterministic=True))
                elif self.spec['kind']=='v8_legacy':
                    from blackout_rl.v8.contracts import StepContext
                    ctx=StepContext('league','worker',self.episode,step,step,float(obs[self.names[0]]['vector'][95]),self.seed,self.team)
                    if step==0:self.policy.reset(ctx)
                    decision=self.policy.decide(obs,ctx)
                    self.metadata={k:float(decision[k]) for k in ('action','q','p_keep','entropy','greedy_action')}
                    result=decision['actions']
                else:result=self.policy.act(obs,self.names)
                return result
        finally:
            self.torch_rng=torch.get_rng_state();self.numpy_rng=np.random.get_state();self.python_rng=random.getstate()
            torch.set_rng_state(outer[0]);np.random.set_state(outer[1]);random.setstate(outer[2])
