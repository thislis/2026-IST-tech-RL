"""Research bundle replay. Only legal recorded observations reach the actor."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from blackout_rl.model_contract import load_checkpoint
from .contracts import ContractError,StepContext,file_hash,digest
from .config import CAPABILITIES
from .policy_factory import make_policy
from .observation import expand
from .window_codec import trace_blocks

def replay(bundle,fixture):
    bundle=Path(bundle).resolve();manifest=json.loads((bundle/'manifest.json').read_text())
    for name,h in manifest['files'].items():
        if file_hash(bundle/name)!=h:raise ContractError('export file changed: '+name)
    config=json.loads((bundle/'resolved_config.json').read_text())
    payload=torch.load(bundle/'policy.pt',map_location='cpu',weights_only=True)
    actor,_=load_checkpoint(bundle/'encoder.pt')
    torch.set_num_threads(config['runtime']['torch_threads'])
    policy=make_policy(config,payload,CAPABILITIES,seed=config['training']['seed'],actor=actor.actor_critic)
    rows=[];times=[]
    for episode in fixture:
        for i,row in enumerate(episode):
            ctx=StepContext(**row['context'])
            if i==0:policy.reset(ctx)
            obs=expand(row['observation'])
            start=time.perf_counter();decision=policy.decide(obs,ctx);times.append(time.perf_counter()-start)
            again=policy.decide(dict(reversed(list(obs.items()))),ctx)
            for name in decision['actions']:
                if not np.array_equal(decision['actions'][name],again['actions'][name]):raise ContractError('export duplicate call differs')
            rows.append(dict(context=row['context'],actions={name:a.tolist() for name,a in decision['actions'].items()},
                             candidate=decision['action'],planner_state=decision['planner_state_digest']))
    return dict(decisions=rows,decision_sha256=digest(rows),requests=len(rows),episodes=len(fixture),
                timing={f'p{q}_seconds':float(np.percentile(times,q)) for q in (50,95,99)},module_path=__file__)

def fixture_from_windows(path):
    episodes={}
    for block in trace_blocks(path):
        for row in block:
            ctx=row['context'];key=(ctx['run_id'],ctx['episode_id'],ctx['team'])
            episodes.setdefault(key,{})[ctx['decision_id']]=dict(context=ctx,observation=row['observation'])
        if len(episodes)>=2 and all(all(i in r for i in range(8)) for r in list(episodes.values())[:2]):break
    selected=[]
    for rows in episodes.values():
        if 0 not in rows:continue
        seq=[]
        for i in range(min(8,len(rows))):
            if i not in rows:break
            seq.append(rows[i])
        selected.append(seq)
        if len(selected)==2:break
    if len(selected)!=2:raise ContractError('two real episodes starting at decision 0 required for export parity')
    return selected
