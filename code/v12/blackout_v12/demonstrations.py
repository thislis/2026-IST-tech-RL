"""Optional task ranking warm start; no primitive-action cloning."""
from pathlib import Path
import torch
from .io import atomic_json


def ranking_warm_start(scorer, records, epochs=1, learning_rate=1e-4):
    """Teacher choices are preference labels, not causal success labels."""
    optimizer=torch.optim.Adam(scorer.parameters(),lr=learning_rate)
    device=next(scorer.parameters()).device
    updates=0
    for _ in range(epochs):
        for row in records:
            features=torch.as_tensor(row['features'],device=device)
            prior=torch.as_tensor(row['prior'],device=device)
            legal=torch.as_tensor(row['legal'],device=device,dtype=torch.bool)
            chosen=torch.as_tensor(row['assignment'],device=device,dtype=torch.long)
            if not legal.gather(-1,chosen[...,None]).all():
                raise ValueError('task-ranking label selects an illegal edge')
            scores=scorer(features,prior)
            selected=scores.gather(-1,chosen[...,None])
            competitors=legal.clone();competitors.scatter_(-1,chosen[...,None],False)
            if not competitors.any(): continue
            loss=torch.nn.functional.softplus(scores-selected)[competitors].mean()
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();updates+=1
    return dict(updates=updates,label='observed task choice; not counterfactual superiority',
                teacher_forced_events_used_by_ppo=0)


def collect(output_dir,cfg,steps,seed=11,workers=1):
    from .training import train
    data_cfg=dict(cfg,data_split='data',data_collection_workers=int(cfg.get('data_collection_workers',workers)))
    result=train(output_dir,data_cfg,steps,seed,'Q2',workers=workers,learner_device='cpu')
    result['provenance']='public-observation Q2 episodes; fixed scorer, no optimizer update'
    result['warm_start']='zero residual is shared initialization; task-ranking is opt-in'
    result['eligible_for_ppo']=False
    atomic_json(Path(output_dir)/'dataset.json',result)
    return result
