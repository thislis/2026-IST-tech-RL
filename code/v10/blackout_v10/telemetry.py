"""Bounded phase accounting and provisional ETA; no simulation-rate claims."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from .io import read_json


def estimate(directory,cfg):
    directory=Path(directory)
    budgets={**{f'pilot/{a}':cfg['pilot_steps'] for a in cfg['arms']},
             **{f'runs/{a}_s{s}':cfg['training_steps'] for a in cfg['arms'] for s in cfg['seeds']},
             **{f'confirmation/{a}_s{s}':cfg['training_steps'] for a in cfg['arms'] for s in cfg['confirmation_seeds']}}
    completed=0;durations=[];rates=[]
    for name,budget in budgets.items():
        path=directory/name
        progress=path/'progress.json'
        completed+=min(budget,read_json(progress)['step']) if progress.exists() else 0
        log=path/'training.jsonl'
        if log.exists():
            with log.open('rb') as f:
                size=f.seek(0,2);f.seek(max(0,size-256*1024));data=f.read().splitlines()
            if size>256*1024:data=data[1:]
            for line in data[-8:]:
                r=json.loads(line);rates.append(r['end_to_end_steps_per_second'])
    # P4/P5 initial + final, P3 pilot and P6 three locked policies.
    expected_eval=len(cfg['evaluation_families'])*2*(
        (len(cfg['seeds'])+len(cfg['confirmation_seeds']))*cfg['baseline_replicates']+
        len(cfg['arms'])*(len(cfg['seeds'])+len(cfg['confirmation_seeds']))*cfg['dev_replicates']+
        len(cfg['arms'])*cfg['pilot_replicates']+3*cfg['test_replicates'])
    family_seconds=defaultdict(list);games=0
    for path in directory.glob('**/cells/*.json'):
        if 'lifecycle' in path.parts:continue
        row=read_json(path)
        if 'opponent' in row and 'wall_seconds' in row:
            family_seconds[row['opponent']['kind']].append(row['wall_seconds']);games+=1
    remaining_train=max(0,sum(budgets.values())-completed)
    remaining_eval=max(0,expected_eval-games)
    means={k:float(np.mean(v[-60:])) for k,v in family_seconds.items()}
    known=all(k in means for k in cfg['evaluation_families']) and bool(rates)
    estimate_seconds=(remaining_train/float(np.median(rates[-16:]))+
        remaining_eval*np.mean([means[k] for k in cfg['evaluation_families']])/cfg['workers']) if known else None
    pending_gates=[name for name in ('lifecycle','value_diagnostic') if not (directory/name/'result.json').exists()]
    return dict(completed_target_steps=completed,target_steps=sum(budgets.values()),remaining_target_steps=remaining_train,
        evaluated_games=games,target_evaluation_games=expected_eval,
        provisional_eta_seconds=None if estimate_seconds is None else float(estimate_seconds),
        eta_includes_remaining_gates=not pending_gates,pending_gates=pending_gates,
        family_game_seconds=means,training_step_rate=None if not rates else float(np.median(rates[-16:])),
        caveat='recent completed-wave throughput; policy-dependent episode lengths and final-wave overrun remain uncertain')


def hardware_identity():
    import os, platform, subprocess, torch
    record=dict(platform=platform.platform(),machine=platform.machine(),cpu_count=os.cpu_count(),
                torch=str(torch.__version__),mps_available=torch.backends.mps.is_available())
    try:
        result=subprocess.run(['system_profiler','SPHardwareDataType','-json'],capture_output=True,text=True,timeout=15,check=True)
        data=json.loads(result.stdout)['SPHardwareDataType'][0]
        record['hardware']={k:data.get(k) for k in ('chip_type','machine_model','number_processors','physical_memory')}
    except (OSError,subprocess.SubprocessError,KeyError,ValueError):
        record['hardware_probe_unavailable']=True
    return record


def memory_pressure():
    import subprocess
    try:
        result=subprocess.run(['sysctl','-n','kern.memorystatus_vm_pressure_level'],capture_output=True,text=True,timeout=2,check=True)
        level=int(result.stdout.strip())
        return dict(verified=True,level=level,critical=level>=4)
    except (OSError,subprocess.SubprocessError,ValueError):
        return dict(verified=False,level=None,critical=False)
