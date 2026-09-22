#!/usr/bin/env python3
"""Same-noise intervention comparison on a real full graph; no gameplay success claim."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from blackout_rl.v7_registry import load_config,path
from blackout_rl.connectome.graph_artifact import GraphArtifact,atomic_json,sha256
from blackout_rl.v7_2.dynamics import WholeBrain
from blackout_rl.v7_2.decoder import fixed_decode


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--ticks',type=int,default=256)
    p.add_argument('--output',default='reports/v7/whole_brain_causality.json');a=p.parse_args()
    cfg=load_config(a.config);g=GraphArtifact.load(path(cfg['data']['graph_artifact']),real=True)
    brain=WholeBrain(g,slots=1,seed=30011);saved=brain.state_dict();names=[p['name'] for p in g.metadata['output_ports']]
    results={};times=[];n=len(brain.input_indices)
    for mode in ('normal','sensory_block','frozen_frame','time_shuffled','output_block'):
        brain.load_state_dict(saved);headings=np.zeros(1);rates=[];actions=[];start=time.monotonic()
        order=np.random.default_rng(918).permutation(a.ticks) if mode=='time_shuffled' else np.arange(a.ticks)
        for tick in range(a.ticks):
            # Artificial stimulus is deliberately labeled offline, not a recorded game success.
            frame=int(order[tick])
            current=np.zeros((1,n),np.float32)
            if mode!='sensory_block':
                side='left' if (frame//32)%2==0 or mode=='frozen_frame' else 'right'
                current[0]=np.array([m['side']==side for m in g.metadata['retinal_mapping']],np.float32)
            z=brain.step(current)
            if mode=='output_block':z[:]=0
            action,headings=fixed_decode(z,names,headings,**cfg['controller']['fixed_decoder'])
            rates.append(z.tolist());actions.append(action.tolist())
        times.append((time.monotonic()-start)/a.ticks);results[mode]=dict(rates=rates,actions=actions)
    normal=np.asarray(results['normal']['rates']);normal_actions=np.asarray(results['normal']['actions'])
    comparisons={k:dict(rate_l1=float(np.abs(normal-np.asarray(v['rates'])).sum()),
                        different_action_ticks=int((normal_actions!=np.asarray(v['actions'])).any(axis=1).sum())) for k,v in results.items() if k!='normal'}
    passed=all(v['rate_l1']>0 for v in comparisons.values()) and comparisons['sensory_block']['different_action_ticks']>0 and comparisons['output_block']['different_action_ticks']>0
    report=dict(graph_sha256=sha256(path(cfg['data']['graph_artifact'])),ticks=a.ticks,seed=30011,comparisons=comparisons,
        median_tick_seconds=float(np.median(times)),causal_gate_passed=passed,scope='offline artificial retinal current; real game displacement still required',
        source_sha256=sha256(__file__),traces=results)
    atomic_json(path(a.output),report);print(json.dumps({k:v for k,v in report.items() if k!='traces'},indent=2))
    if not passed:raise SystemExit(2)

if __name__=='__main__':main()
