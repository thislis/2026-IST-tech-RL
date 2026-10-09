"""Additional train/dev-only statistics and artifact lifecycle checks."""

# Locate shared modules from the physical versioned layout.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents
                    if (p / "code/shared/project_paths.py").is_file())
_layout_sys.path.insert(0, str(_layout_root / "code/shared"))
from project_paths import activate as _activate_layout
_activate_layout()
from project_paths import project_root, project_path

import collections
import io
import json
from pathlib import Path
import sys
import zlib
import numpy as np
import torch
ROOT=project_root();sys.path.insert(0,str(ROOT))
from analyze import OUT,read,lines,sha,write,INPUTS,interval_bound,stats
from blackout_rl.model_contract import load_checkpoint

def main():
    torch.set_num_threads(2)
    inventory=read(OUT/'artifact_inventory.json');cube=read(OUT/'evaluation_cube.json')
    exposure=[];failures=[];examples=[];direct_interventions=[]
    for run in inventory['runs']:
        p=project_path(run['run'], root=ROOT);es=lines(p/'episodes.jsonl');side=collections.Counter();opp=collections.Counter()
        episode_side=collections.Counter()
        for e in es:side[e['team']]+=e['episode_steps'];episode_side[e['team']]+=1;opp[e['opponent']]+=e['episode_steps']
        complete=sum(side.values());discarded=0
        for e in run['resume_records']:
            if 'discarded' in e:
                part=e['discarded'];n=part['steps'];discarded+=n;side[(0,0,1,1,1)[int(part['episode_id'].split(':')[-1])%5]]+=n
        part=run['partial_episode'];side[(0,0,1,1,1)[int(part['episode_id'].split(':')[-1])%5]]+=part['steps']
        exposure.append(dict(model=run['model'],train_seed=run['seed'],completed_episodes=dict(episode_side),completed_episode_steps=complete,
            exact_side_steps_with_partial=dict(side),step_coverage=sum(side.values()),discarded_steps=discarded,final_partial_steps=part['steps'],
            opponent_steps_completed=dict(opp),opponent_unknown_steps=2000000-complete))
        if run['model']=='F1':
            ck=torch.load(p/'checkpoints/latest.pt',weights_only=True,map_location='cpu');sha(p/'checkpoints/latest.pt')
            ds=[x for x in lines(p/'diagnostics.jsonl') if 'rates' in x]
            r=np.asarray([x['rates'][0] for x in ds]);w=ck['model']['readout.weight'].numpy();b=ck['model']['readout.bias'].numpy()
            normal=(r@w.T+b).argmax(1);zero=np.broadcast_to(b,r.shape[:1]+b.shape).argmax(1)
            shuffled=(r[:,[5,4,3,2,1,0]]@w.T+b).argmax(1)
            direct_interventions.append(dict(seed=run['seed'],n=len(r),scope='offline final F1 readout on historical training rates, not full closed loop',
                normal_counts=dict(collections.Counter(map(int,normal))),zero_rate_counts=dict(collections.Counter(map(int,zero))),
                reversed_pool_counts=dict(collections.Counter(map(int,shuffled))),zero_rate_differences=int((normal!=zero).sum()),
                reversed_pool_differences=int((normal!=shuffled).sum())))
            # Preserve real source line numbers; a sparse diagnostic cannot supply adjacent frames.
            pdiag=p/'diagnostics.jsonl'
            for predicate,kind in [(lambda x:x['actual_displacement'][0]<1e-6 and np.linalg.norm(next(iter(x['requested'].values())))>0,'requested_but_stationary'),
                                   (lambda x:0<x['actual_displacement'][0]<.01,'normal_small_movement'),
                                   (lambda x:x['actual_displacement'][0]>.04,'possible_respawn_or_reset')]:
                found=next(((i,x) for i,x in enumerate(ds) if predicate(x)),None)
                if found:
                    i,x=found;examples.append(dict(kind=kind,model='F1',train_seed=run['seed'],path=str(pdiag.relative_to(ROOT)),line=i+1,
                        trace=ds[max(0,i-1):i+2],spacing_note='100 global steps; not adjacent decision frames'))
    for cond,kind in [(lambda e:e['score_0']==e['score_1']==0 and e['winner']!=-1,'zero_with_winner'),
                      (lambda e:e['score_0']!=0 or e['score_1']!=0,'nonzero_terminal_untrusted'),
                      (lambda e:e['winner']==-1,'wrapper_draw'),(lambda e:e['steps']==21003,'timeout_length_only')]:
        examples.extend(dict(kind=kind,**e) for e in [x for x in cube if cond(x)][:2])
    write('exposure_summary.json',exposure);write('case_traces.json',examples);write('F1_readout_interventions.json',direct_interventions)
    lifecycle=[]
    for rec in read(project_path('logs/v7/reports/acceleration_resume_points.json', root=ROOT)):
        current=Path(rec['run'])/'checkpoints/latest.pt';backup=Path(rec['backup'])/'checkpoints/latest.pt'
        ck=torch.load(current,weights_only=True,map_location='cpu');bc=torch.load(backup,weights_only=True,map_location='cpu')
        lifecycle.append(dict(run=rec['run'],registered=rec['checkpoint_sha256'],current=sha(current),backup=sha(backup),
            current_step=ck['global_step'],backup_step=bc['global_step'],backup_matches=sha(backup)==rec['checkpoint_sha256'],lineage=ck.get('lineage',[])))
    reg=read(project_path('logs/v7/reports/acceleration_registration.json', root=ROOT));mismatches=[]
    for f,h in reg['files'].items():
        if not (project_path(f, root=ROOT)).exists() or sha(project_path(f, root=ROOT))!=h:mismatches.append(f)
    # Compare no-terminal boolean field hashes against zero arrays for the tested window.
    parity=[]
    import hashlib
    for fn in ['acceleration_equivalence_baseline','acceleration_direct_baseline']:
        d=read(project_path(f'logs/v7/reports/{fn}.json', root=ROOT));zero_hash=hashlib.sha256(np.zeros(d['steps'],dtype=bool).tobytes()).hexdigest()
        parity.append(dict(file=fn,steps=d['steps'],no_terminals_in_window=d['batch_sha256']['terminated']==zero_hash,
                          no_truncations_in_window=d['batch_sha256']['truncated']==zero_hash))
    write('runtime_audit.json',dict(lifecycle=lifecycle,overlay_files=len(reg['files']),overlay_mismatches=mismatches,
        parity_window_scope=parity,hardware=reg['hardware'],registration_sha256=sha(project_path('logs/v7/reports/acceleration_registration.json', root=ROOT))))
    # Independently inspect final training replay; no fit, no held-out test use.
    v3path=project_path('artifacts/checkpoints/mappo_teacher_curriculum_v3_latest.pt', root=ROOT);sha(v3path)
    ck=torch.load(v3path,weights_only=True,map_location='cpu');replay=ck['teacher_replay_state'];size=replay['size']
    slots=replay['slot_id'][:size].numpy();target=replay['action_index'][:size].numpy()
    summary=[];selected=[];rng=np.random.default_rng(927)
    for slot in range(5):
        ix=np.flatnonzero(slots==slot);selected.extend(rng.choice(ix,min(512,len(ix)),replace=False).tolist())
        summary.append(dict(slot=slot,n=len(ix),label_counts=np.bincount(target[ix],minlength=9).tolist(),noop_fraction=float((target[ix]==0).mean())))
    selected=np.array(sorted(selected));graphics=torch.load(io.BytesIO(zlib.decompress(replay['graphic_ids_zlib'])),weights_only=True,map_location='cpu').numpy()
    model,_=load_checkpoint(v3path);model.eval();pred=[]
    with torch.no_grad():
        for chunk in np.array_split(selected,40):
            g=torch.nn.functional.one_hot(torch.from_numpy(graphics[chunk].astype(np.int64)),11).permute(0,3,1,2).float()
            output=model.actor_critic(replay['vector'][chunk],g,replay['slot_id'][chunk])
            logits=output[0] if isinstance(output,tuple) else output
            if isinstance(logits,dict):logits=logits['logits']
            pred.extend(logits.argmax(-1).tolist())
    pred=np.array(pred)
    for rec in summary:
        mask=slots[selected]==rec['slot'];truth=target[selected][mask];pr=pred[mask];conf=np.zeros((9,9),int);np.add.at(conf,(truth,pr),1)
        moving=truth!=0
        rec.update(sample_n=int(mask.sum()),sample_accuracy=float((truth==pr).mean()),
            moving_label_accuracy=float((truth[moving]==pr[moving]).mean()) if moving.any() else None,
            confusion_true_rows_pred_columns=conf.tolist())
    logs=lines(project_path('logs/v3/mappo_teacher_curriculum_v3/training.jsonl', root=ROOT))
    warmup=[x for x in logs if x.get('stage')=='dagger_warmup' and 'imitation' in x]
    write('v3_replay_audit.json',dict(step=ck['training']['global_step'],replay_size=size,scope='training replay only; no independent generalization or closed-loop claim',
        overall_noop=float((target==0).mean()),sample_n=len(selected),overall_sample_accuracy=float((pred==target[selected]).mean()),
        slots=summary,replay_missing_group_fields=['map_seed','episode_id','env_step_id'],
        last_warmup_log=warmup[-1] if warmup else None,last_log=logs[-1]))
    write('supplement_input_hashes.json',INPUTS)
    print('Supplement complete.')

if __name__=='__main__':main()
