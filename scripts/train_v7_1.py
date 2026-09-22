#!/usr/bin/env python3
"""Run a registered v7 experiment; --check is read-only and never starts Unity."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blackout_rl.v7_registry import load_config,check_config,path,make_model


def main(expected_mode='residual_graph'):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--seed',type=int,default=11)
    p.add_argument('--run-dir');p.add_argument('--resume',action='store_true');p.add_argument('--check',action='store_true')
    a=p.parse_args();cfg=load_config(a.config)
    if cfg['mode']!=expected_mode: raise ValueError('wrong entry point for experiment mode')
    graph,splits=check_config(cfg)
    run=path(a.run_dir or f'logs/v7/{cfg["experiment_id"]}/{a.seed}')
    if a.check:
        import torch
        with torch.random.fork_rng(): model=make_model(cfg,graph)
        print(json.dumps(dict(status='ready',config=a.config,run=str(run),nodes=graph.n,edges=len(graph.edge_src),
            trainable_parameters=sum(x.numel() for x in model.parameters() if x.requires_grad)),indent=2));return
    from blackout_rl.v7_training import train
    train(cfg,graph,splits,seed=a.seed,run=run,resume=a.resume)

if __name__=='__main__':main()
