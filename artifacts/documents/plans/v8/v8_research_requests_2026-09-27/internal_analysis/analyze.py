"""Read-only audit of registered train/dev artifacts; no Unity or training runs.

Run from project root: .venv/bin/python v8_research_requests_2026-09-27/internal_analysis/analyze.py
Writes only beside this script. No final-test result or held-out seed list is opened.
"""
import collections
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from blackout_rl.connectome.graph_artifact import GraphArtifact, digest
from blackout_rl.v7_registry import make_model

INPUTS = {}
def sha(p):
    p = Path(p)
    key = str(p.resolve().relative_to(ROOT)) if p.resolve().is_relative_to(ROOT) else str(p.resolve())
    if key not in INPUTS:
        h = hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
        INPUTS[key] = dict(sha256=h.hexdigest(), bytes=p.stat().st_size)
    return INPUTS[key]['sha256']

def read(p):
    sha(p)
    return json.loads(Path(p).read_text())

def lines(p):
    sha(p)
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]

def write(name, data):
    (OUT/name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False,
                                  default=lambda x: x.item() if isinstance(x,np.generic) else str(x))+'\n')

def stats(values):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    return {} if not len(v) else dict(n=len(v),mean=float(v.mean()),min=float(v.min()),max=float(v.max()),
                                     p50=float(np.median(v)),p95=float(np.quantile(v,.95)))

def label(e):
    return 'F1' if 'v7_2' in e else 'A1' if 'matched_mlp' in e else 'A0' if 'mlp_control' in e else 'A2' if 'rewired' in e else 'A3'

def interval_bound(m, cfg, slack=1e-4):
    """Conservative interval propagation with unrestricted input drives.

    slack widens every graph-round/node bound and final output to tolerate float32
    rounding. This is a numerical bound, not a certified floating-point proof.
    """
    src=m['residual.src'].numpy(); dst=m['residual.dst'].numpy()
    ew=m['residual.edge_weight'].numpy(); n=len(m['residual.bias'])
    normalizer=torch.zeros(n).index_add_(0,torch.from_numpy(dst),torch.from_numpy(ew).abs()).clamp_min(1).numpy()
    w=(ew/normalizer[dst]).astype(float)
    W=np.zeros((n,n)); np.add.at(W,(dst,src),w)
    P,N=np.maximum(W,0),np.minimum(W,0)
    lo=np.zeros_like(m['residual.bias'].numpy(),dtype=float); hi=lo.copy()
    bias=m['residual.bias'].double().numpy(); driven=m['residual.pin'].abs().sum(1).numpy()>0
    leak=cfg['model']['leak']
    for _ in range(cfg['model']['graph_rounds']):
        a=P@lo+N@hi+bias-slack; b=P@hi+N@lo+bias+slack
        a[driven]=-np.inf; b[driven]=np.inf
        lo=(1-leak)*lo+leak*np.tanh(a)-slack
        hi=(1-leak)*hi+leak*np.tanh(b)+slack
    q=m['residual.qout'].double().numpy()
    assert (q>=0).all()
    l=(q@lo).ravel(); h=(q@hi).ravel()
    ow=m['residual.output_head.weight'].double().numpy(); ob=m['residual.output_head.bias'].double().numpy()
    lower=np.maximum(ow,0)@l+np.minimum(ow,0)@h+ob-slack
    upper=np.maximum(ow,0)@h+np.minimum(ow,0)@l+ob+slack
    return dict(lower=lower.tolist(),upper=upper.tolist(),max_upper=float(upper.max()),
                keep_logit=0.,all_finite_inputs_greedy_keep_with_numerical_margin=bool(upper.max()<0),
                slack_per_round=slack,driven_nodes=int(driven.sum()))

def main():
    started=time.monotonic(); torch.set_num_threads(2)
    inventory=[]; evaluations=[]; training=[]; exposure=[]; bounds=[]; direct=[]; checkpoints=[]
    graphs={}; reference_episodes=None; graph_equal=True
    manager=read(ROOT/'logs/v7/main_study/accelerated/summary.json')
    for run in manager['runs']:
        ep=Path(run['result']); p=ep.parent; cfg=read(p/'resolved_config.json'); lab=label(run['experiment_id'])
        ev=read(ep); manifest=read(p/'experiment_manifest.json'); runtime=read(p/'acceleration_runtime.json')
        ckfile=p/'checkpoints/latest.pt'; ckhash=sha(ckfile)
        ck=torch.load(ckfile,map_location='cpu',weights_only=True)
        for aux in ['source_manifest.json','graph_audit.json','runtime_info.json','status.json']:
            read(p/aux)
        gp=ROOT/cfg['data']['graph_artifact']; gh=sha(gp); mh=sha(gp.with_suffix('.json'))
        op=ROOT/cfg['training']['opponent_checkpoint']; oh=sha(op)
        bh=sha(ROOT/cfg['environment']['executable']); split_hash=sha(ROOT/cfg['evaluation']['split_manifest'])
        source_mismatch=[f for f,h in ck['sources'].items() if not (ROOT/f).exists() or sha(ROOT/f)!=h]
        result_ck=sha(Path(ep.parent/'evaluation_checkpoints')/(ev['checkpoint_sha256']+'.pt'))
        tl=lines(p/'training.jsonl'); updates=[x for x in tl if 'environment_steps' in x and 'ppo' in x]
        es=lines(p/'episodes.jsonl')
        artifacts=[dict(path=str(x.relative_to(ROOT)),sha256=sha(x)) for x in p.glob('*.json*') if x.name != ep.name and not x.name.endswith('.progress.json')]
        ckpaths=list((p/'checkpoints').glob('*.pt'))+list((p/'evaluation_checkpoints').glob('*.pt'))
        backup=ROOT/'logs/v7/main_study/pre_acceleration_backup'/cfg['experiment_id']/str(run['seed'])/'checkpoints/latest.pt'
        if backup.exists(): ckpaths.append(backup)
        for x in ckpaths:
            z=torch.load(x,map_location='cpu',weights_only=True)
            checkpoints.append(dict(run=cfg['experiment_id'],seed=run['seed'],path=str(x.relative_to(ROOT)),sha256=sha(x),step=z['global_step']))
        inventory.append(dict(model=lab,experiment_id=cfg['experiment_id'],seed=run['seed'],run=str(p.relative_to(ROOT)),
            result=str(ep.relative_to(ROOT)),result_sha256=sha(ep),checkpoint_sha256=ckhash,evaluation_checkpoint_sha256=result_ck,
            graph_path=str(gp.relative_to(ROOT)),graph_sha256=gh,graph_metadata_sha256=mh,opponent_sha256=oh,build_sha256=bh,
            config_sha256=digest(cfg),source_fingerprint_digest=digest(ck['sources']),current_source_mismatches=source_mismatch,
            config_matches=ck['config_sha256']==digest(cfg)==manifest['config_sha256'],graph_matches=gh==cfg['data']['graph_sha256'],
            metadata_matches=mh==cfg['data']['graph_metadata_sha256'],opponent_matches=oh==cfg['training']['opponent_checkpoint_sha256'],
            build_matches=bh==cfg['environment']['executable_sha256'],split_manifest=cfg['evaluation']['split_manifest'],
            split_hash_matches=split_hash==cfg['evaluation']['split_sha256'],runtime=runtime,lineage=ck['lineage'],
            first_update_at=updates[0]['at'],last_update_at=updates[-1]['at'],status=read(p/'status.json'),
            artifacts=artifacts,partial_episode=ck['partial_episode'],resume_records=[x for x in tl if x.get('event')]))
        rows=ev['episodes']
        for i,row in enumerate(rows): evaluations.append(dict(model=lab,train_seed=run['seed'],result=str(ep.relative_to(ROOT)),row=i,**row))
        if lab in ('A2','A3'):
            if reference_episodes is None: reference_episodes=rows
            graph_equal &= rows==reference_episodes
        prev=0; override_count=0; collection_seconds=0
        for x in updates:
            count=x['environment_steps']-prev;prev=x['environment_steps']
            override_count+=count*x['team_override_rate'];collection_seconds+=count/x['steps_per_second']
        metrics={key:stats([x['ppo'].get(key) for x in updates]) for key in ['policy_loss','value_loss','entropy','approximate_kl','clip_fraction','gradient_norm','explained_variance','epochs_completed','minibatches']}
        training.append(dict(model=lab,seed=run['seed'],updates=len(updates),steps=prev,sampled_override_rate=override_count/prev,
            last_override_rate=updates[-1]['team_override_rate'],metrics=metrics,
            early_stops=sum(bool(x['ppo'].get('early_stopped')) for x in updates),
            gradient_norm_above_clip=sum(x['ppo'].get('gradient_norm',0)>.5 for x in updates),
            collection_seconds_estimate=collection_seconds,first_to_last_update_seconds=(datetime.fromisoformat(updates[-1]['at'])-datetime.fromisoformat(updates[0]['at'])).total_seconds(),
            window_100k=[dict(end=end,override=stats([x['team_override_rate'] for x in updates if end-100000<x['environment_steps']<=end]),
                entropy=stats([x['ppo'].get('entropy') for x in updates if end-100000<x['environment_steps']<=end]),
                value_loss=stats([x['ppo'].get('value_loss') for x in updates if end-100000<x['environment_steps']<=end])) for end in range(100000,2000001,100000)]))
        groups=collections.defaultdict(list)
        for x in es: groups[(x['seed'],x['team'],x['opponent'])].append(x)
        for (seed,team,opp),g in groups.items():
            exposure.append(dict(model=lab,train_seed=run['seed'],map_seed=seed,side=team,opponent=opp,episodes=len(g),
                steps=sum(x['episode_steps'] for x in g),wins=sum(x['result']=='win' for x in g),draws=sum(x['result']=='draw' for x in g)))
        if lab!='F1':
            if str(gp) not in graphs: graphs[str(gp)]=GraphArtifact.load(gp,real=True)
            graph=graphs[str(gp)]
            torch.manual_seed(run['seed']); initial=make_model(cfg,graph)
            initstate={k:v.clone() for k,v in initial.state_dict().items()}
            deltas={}
            for prefix in ['actor_model','residual','critic']:
                parts=[(ck['model'][k]-v).double().flatten() for k,v in initstate.items() if k.startswith(prefix) and v.is_floating_point()]
                deltas[prefix]=float(torch.linalg.vector_norm(torch.cat(parts)))
            initial.load_state_dict(ck['model']); initial.eval()
            item=dict(model=lab,seed=run['seed'],checkpoint_sha256=ckhash,parameter_l2_change_from_seed_reconstruction=deltas)
            if lab in ('A2','A3'):
                item['bound']=interval_bound(ck['model'],cfg)
                item['loaded_graph_matches']=bool(np.array_equal(ck['model']['residual.src'].numpy(),graph.edge_src) and np.array_equal(ck['model']['residual.dst'].numpy(),graph.edge_dst))
                item['edge_weight_l2_change']=float(torch.linalg.vector_norm(ck['model']['residual.edge_weight']-initstate['residual.edge_weight']))
            # No fabricated observations: these are explicitly synthetic feature stress inputs.
            gen=torch.Generator().manual_seed(927)
            x=torch.randn(96,5,177,generator=gen)*torch.logspace(-3,3,96)[:,None,None]
            with torch.no_grad():
                logits=initial.logits(x);probs=logits.softmax(-1);greedy=logits.argmax(-1)
            item['synthetic_unmasked_features']=dict(n=96,greedy_override_count=int((greedy!=0).sum()),
                p_keep=stats(probs[:,0].tolist()),max_correction_logit=float(logits[:,1:].max()),
                p_override_gt_half_and_greedy_keep=int(((probs[:,0]<.5)&(greedy==0)).sum()))
            bounds.append(item)
        else:
            ds=lines(p/'diagnostics.jsonl'); samples=[x for x in ds if 'rates' in x]
            rates=np.asarray([x['rates'][0] for x in samples]); displacement=np.asarray([x['actual_displacement'][0] for x in samples])
            w=ck['model']['readout.weight'].numpy(); b=ck['model']['readout.bias'].numpy()
            logits=rates@w.T+b; final_actions=logits.argmax(1)
            bias_action=int(b.argmax()); singular=np.linalg.svd(rates-rates.mean(0),compute_uv=False)
            direct.append(dict(seed=run['seed'],sample_interval_global_steps=100,n=len(samples),
                rates_mean=rates.mean(0).tolist(),rates_std=rates.std(0).tolist(),rates_min=rates.min(0).tolist(),rates_max=rates.max(0).tolist(),
                rate_zero_fraction=(rates==0).mean(0).tolist(),rate_correlation=np.corrcoef(rates.T).tolist(),
                centered_rank=int(np.linalg.matrix_rank(rates-rates.mean(0))),singular_values=singular.tolist(),
                effective_rank=float(np.exp(-np.sum((singular/singular.sum())*np.log((singular/singular.sum()).clip(1e-30))))),
                final_readout_action_counts=dict(collections.Counter(map(int,final_actions))),bias_only_action=bias_action,
                final_readout_bias_only_agreement=float((final_actions==bias_action).mean()),
                stationary_fraction_sampled=float((displacement<1e-6).mean()),
                suspicious_displacement_gt_one_cell_count=int((displacement>1/24).sum()),
                requested_nonzero_stationary_count=sum(np.linalg.norm(next(iter(x['requested'].values())))>1e-6 and x['actual_displacement'][0]<1e-6 for x in samples),
                diagnostics_events=dict(collections.Counter(x.get('event','sample') for x in ds)),
                readout_weight=w.tolist(),readout_bias=b.tolist(),
                examples=[dict(source=str((p/'diagnostics.jsonl').relative_to(ROOT)),**x) for x in samples[:2]]))
    cube=collections.defaultdict(list)
    for x in evaluations:cube[(x['model'],x['train_seed'],x['team'])].append(x)
    evaluation_summary=[]
    for (lab,seed,side),g in sorted(cube.items()):
        evaluation_summary.append(dict(model=lab,train_seed=seed,side=side,n=len(g),wins=sum(x['win'] for x in g),draws=sum(x['draw'] for x in g),
            zero_score=sum(x['score_0']==x['score_1']==0 for x in g),zero_score_winner=sum(x['score_0']==x['score_1']==0 and x['winner']!=-1 for x in g),
            steps=stats([x['steps'] for x in g]),length_21003=sum(x['steps']==21003 for x in g),
            negative_scores=sum(x['score_0']<0 or x['score_1']<0 for x in g),over_one_scores=sum(x['score_0']>1 or x['score_1']>1 for x in g)))
    write('artifact_inventory.json',dict(aliases={'ROOT':str(ROOT),'GAME':str(ROOT.parent/'blackout'),'API':str(ROOT.parent/'blackout-env')},runs=inventory,checkpoints=checkpoints))
    write('evaluation_cube.json',evaluations); write('evaluation_summary.json',dict(by_model_seed_side=evaluation_summary,all_A2_A3_episode_arrays_equal=graph_equal))
    write('training_summary.json',training); write('training_exposure_cube.json',exposure)
    write('behavior_audit.json',bounds); write('direct_diagnostics.json',direct)
    v6=lines(ROOT/'logs/mappo_planner_residual_v6/training.jsonl')
    vu=[x for x in v6 if 'update' in x and 'ppo' in x]
    event=[x for x in v6 if x.get('record_type') not in ('start',None)]
    windows=[]
    for step in (567296,925696):
        windows.append(dict(rollback_step=step,updates=[x for x in vu if step-6144<=x['global_step']<=step+6144]))
    dev=[]
    for p in sorted((ROOT/'logs/mappo_planner_residual_v6').glob('target_eval_step_*.json')):
        d=read(p);dev.append(dict(file=str(p.relative_to(ROOT)),step=int(p.stem.split('_')[-1]),summary=d.get('summary'),keys=list(d)))
    v6ck=[]
    for p in list((ROOT/'checkpoints/mappo_v6_snapshots').glob('*.pt'))+list((ROOT/'checkpoints').glob('mappo_planner_residual_v6_*.pt')):
        sha(p);v6ck.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size))
    write('event_timeline.json',dict(v6_events=event,v6_rollback_windows=windows,v6_dev=sorted(dev,key=lambda x:x['step']),v6_checkpoints=v6ck,
        v6_updates=vu,v6_completed_episodes=lines(ROOT/'logs/mappo_planner_residual_v6/training_episodes.jsonl')))
    # Degree-preserving control; only partial graphs loaded, not full held-out data.
    a=graphs[str(ROOT/'data/connectomes/fafb783/cx_primary.npz')];b=graphs[str(ROOT/'data/connectomes/fafb783/cx_rewired_1729.npz')]
    ae=set(zip(a.edge_src.tolist(),a.edge_dst.tolist()));be=set(zip(b.edge_src.tolist(),b.edge_dst.tolist()))
    write('graph_comparison.json',dict(nodes=a.n,edges=len(a.edge_src),shared_edges=len(ae&be),different_edges=len(ae-be),
        node_ids_equal=bool(np.array_equal(a.node_ids,b.node_ids)),ports_equal=all(a.metadata[k]==b.metadata[k] for k in ('input_ports','output_ports')),
        in_degree_equal=bool(np.array_equal(np.bincount(a.edge_dst,minlength=a.n),np.bincount(b.edge_dst,minlength=b.n))),
        out_degree_equal=bool(np.array_equal(np.bincount(a.edge_src,minlength=a.n),np.bincount(b.edge_src,minlength=b.n))),
        actual_reachable=a.reachable(4),rewired_reachable=b.reachable(4)))
    for p in [ROOT/'requirements.lock',ROOT/'requirements-v7.lock',ROOT/'versions.md',ROOT/'.gitignore',Path(__file__)]:sha(p)
    write('input_hashes.json',INPUTS)
    write('audit_execution.json',dict(command=f'.venv/bin/python {Path(__file__).relative_to(ROOT)}',python=platform.python_version(),numpy=np.__version__,torch=str(torch.__version__),
        current_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        current_branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        elapsed_seconds=time.monotonic()-started,scope='local original train/dev artifacts; no Unity, final test, new training, or resume'))
    print('Wrote audit files; runs:',len(inventory),'episodes:',len(evaluations),'elapsed:',time.monotonic()-started)

if __name__=='__main__':main()
