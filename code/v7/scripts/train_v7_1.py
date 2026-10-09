#!/usr/bin/env python3
"""Run a registered v7 experiment; --check is read-only and never starts Unity."""

# Locate shared modules from the physical versioned layout.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents
                    if (p / "code/shared/project_paths.py").is_file())
_layout_sys.path.insert(0, str(_layout_root / "code/shared"))
from project_paths import activate as _activate_layout
_activate_layout()
from project_paths import project_root, project_path

import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(project_root()))
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
