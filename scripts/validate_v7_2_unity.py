#!/usr/bin/env python3
"""Short paired intervention runs measuring requested actions AND Unity displacement."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from blackout_rl.v7_registry import load_config,check_config,path
from blackout_rl.connectome.graph_artifact import atomic_json,sha256
from blackout_rl.v7_2.policy import DirectPolicy
from blackout_rl.v7_2.teammates import make_teammates
from blackout_rl.env import ContractBlackOutEnv
from blackout_rl.frozen_opponent import FrozenScriptedOpponent
from blackout_rl.mappo_v6 import ROLES
from blackout_rl.batching import team_agents


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--steps',type=int,default=256)
    p.add_argument('--output',default='reports/v7/unity_causality.json');a=p.parse_args()
    cfg=load_config(a.config);g,_=check_config(cfg);torch.set_num_threads(2)
    policy=DirectPolicy(g,None,noise_seed=30011,decoder_config=cfg['controller']['fixed_decoder'])
    env=ContractBlackOutEnv(env_path=str(path(cfg['environment']['build'])),background=True,no_graphics=False,time_scale=50.)
    results={};frames=[]
    try:
        for mode in ('normal','sensory_block','frozen_frame','output_block'):
            observations,_=env.reset(seed=3001)
            policy.intervention=mode;policy.reset_episode_state(mode,0,noise_seed=30011)
            team=make_teammates(0,cfg);team.reset()
            opponent=FrozenScriptedOpponent(1);opponent.reset()
            rows=[]
            for step in range(a.steps):
                before=observations['unit_0']['vector'][:2].copy()
                if mode=='normal' and step<10:
                    frames.append(observations['unit_0']['graphic'].copy())
                actions=team.act(observations,team_agents(0))
                actions.update(policy.act(observations,episode_id=mode,env_step_id=step,team_id=0))
                actions.update(opponent.act(observations,team_agents(1)))
                observations,_,terms,truncs,_=env.step(actions)
                after=observations['unit_0']['vector'][:2]
                rows.append(dict(step=step,action=actions['unit_0'].tolist(),displacement=float(np.linalg.norm(after-before)),
                                 position=after.tolist(),rates=policy.last_features.tolist()))
                if any(terms.values()) or any(truncs.values()):break
            results[mode]=rows
            print(mode,len(rows),'steps',sum(r['displacement'] for r in rows),'displacement',flush=True)
    finally:env.close()
    normal=results['normal']; comparisons={}
    for mode,rows in results.items():
        comparisons[mode]=dict(nonzero_requested_steps=sum(np.linalg.norm(r['action'])>0 for r in rows),
            moved_steps=sum(r['displacement']>1e-6 for r in rows),cumulative_displacement=sum(r['displacement'] for r in rows),
            different_action_steps=sum(x['action']!=y['action'] for x,y in zip(normal,rows)),
            rate_l1=float(sum(np.abs(np.array(x['rates'])-y['rates']).sum() for x,y in zip(normal,rows))))
    # Require physical movement and both sensory and output interventions to affect the control path.
    passed=comparisons['normal']['moved_steps']>0 and all(comparisons[m]['different_action_steps']>0 for m in ('sensory_block','output_block'))
    report=dict(schema='blackout.v7.unity_causality.v1',graph_sha256=cfg['data']['graph_sha256'],config=cfg,seed=3001,noise_seed=30011,
                controlled_slot=0,comparisons=comparisons,causal_gate_passed=bool(passed),source_sha256=sha256(__file__),traces=results,
                interpretation='short technical closed-loop/causality test, not a terminal win-rate measurement')
    # Convert numpy scalar counters before strict JSON serialization.
    for values in comparisons.values():
        for key,value in values.items():
            if isinstance(value,np.generic):values[key]=value.item()
    atomic_json(path(a.output),report)
    np.savez_compressed(path(a.output).with_suffix('.frames.npz'),frames=np.stack(frames))
    print(json.dumps({k:v for k,v in report.items() if k not in ('traces','config')},indent=2))
    if not passed:raise SystemExit(2)

if __name__=='__main__':main()
