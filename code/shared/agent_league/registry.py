"""Discover representatives without using new league outcomes for selection."""
import re
from collections import defaultdict
import torch
from .common import ROOT, read, sha, digest, relative, resolve


def payload(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def stem_group(stem):
    # Preserve training-method/version distinctions; remove checkpoint-role suffixes only.
    stem = re.sub(r'_gen_\d+$', '', stem)
    stem = re.sub(r'_fit(_v\d+)?$', lambda m: m.group(1) or '', stem)
    return re.sub(r'_(eval_latest|stage_best|target_best|eval_candidate|latest|best|initial|candidate)$', '', stem)


def discover():
    candidates, excluded = [], []
    def add(path, group, kind, *, config=None, seed=0, step=0, selected=False, complete=True, main=True, assets=None):
        row = dict(id=group, kind=kind, path=relative(path) if path else None, config=config,
                   seed=int(seed or 0), step=int(step or 0), selected=selected, complete=complete, main=main,
                   contract='provided_two_tensor' if kind in ('bundle','v8_comp') else 'legacy_identity_state',
                   assets=assets or {})
        if path:
            row['assets'][relative(path)] = sha(path)
        candidates.append(row)
    # Initial and v1-v6 policies carry their original guardrails in the checkpoint.
    for folder in sorted((ROOT/'artifacts/checkpoints').iterdir()):
        if not folder.is_dir() or folder.name not in ('pre_v1','v1','v2','v3','v4','v5','v6'): continue
        for p in sorted(folder.rglob('*.pt')):
            if p.parent != folder:
                excluded.append(dict(path=relative(p), reason='intermediate snapshot')); continue
            try:
                d = payload(p)
            except Exception as e:
                raise ValueError(f'Cannot inspect local candidate {p}: {e}') from e
            if 'policy_state' not in d or 'model_config' not in d:
                excluded.append(dict(path=relative(p),reason='not a project policy checkpoint')); continue
            name = stem_group(p.stem) if folder.name=='pre_v1' else folder.name
            # Keep changed inference contracts distinct even inside one named experiment.
            cfg = dict(model=d['model_config'], guard=d.get('inference_guardrail'))
            group = f'{folder.name}-{name}' if folder.name=='pre_v1' else name
            signature = digest(cfg)[:8]
            add(p,group+'-'+signature,'legacy',config=cfg,step=d.get('training',{}).get('global_step',0),
                seed=d.get('training',{}).get('seed',0),selected=folder.name=='v6' and p.name.endswith('_latest.pt'))
    # Version 7: final training checkpoint per configuration; graph is never reduced.
    for p in sorted((ROOT/'logs/v7').glob('*/*/resolved_config.json')):
        run=p.parent; cfg=read(p); cp=run/'checkpoints/latest.pt'
        if 'smoke' in str(run) or not cp.exists():
            excluded.append(dict(path=relative(run),reason='smoke or missing final checkpoint')); continue
        status=read(run/'status.json') if (run/'status.json').exists() else {}
        d=payload(cp)
        identity={k:cfg.get(k) for k in ('mode','model','controller','sensory','brain')}
        identity['graph_sha256']=cfg['data']['graph_sha256']
        # Main/pilot budget does not define a new inference architecture.
        key=f"v7-{cfg['mode']}-{digest(identity)[:10]}"
        assets={relative(p):sha(p)}
        for section in ('data','model'):
            for k,v in cfg.get(section,{}).items():
                if isinstance(v,str) and k in ('graph_artifact','frozen_encoder_checkpoint','source_manifest'):
                    f=resolve(v)
                    if not f.exists():
                        excluded.append(dict(path=relative(run),reason='missing asset '+v));break
                    assets[relative(f)]=sha(f)
                    if k=='graph_artifact' and f.with_suffix('.json').exists():assets[relative(f.with_suffix('.json'))]=sha(f.with_suffix('.json'))
            else:continue
            break
        else:
            add(cp,key,'v7',config=cfg,seed=d['seed'],step=d['global_step'],assets=assets,
                complete=status.get('status')=='budget_complete',main='main_' in run.parent.name)
    # Version 8 has two different execution contracts; both are real variants.
    for study,kind in [('provided_competition_v1','v8_comp'),('accelerated_pilot_v1','v8_legacy')]:
        for run in sorted((ROOT/'logs/v8'/study/'runs').glob('*')):
            if not run.is_dir():continue
            if kind=='v8_comp':
                if not (run/'progress.json').exists():continue
                progress=read(run/'progress.json');cp=run/'checkpoints'/(progress['checkpoint']+'.pt')
            else:
                if not (run/'checkpoints/latest.json').exists():continue
                info=read(run/'checkpoints/latest.json');cp=run/'checkpoints'/info['path']
            if not cp.exists():excluded.append(dict(path=relative(run),reason='missing checkpoint'));continue
            d=payload(cp);cfg=d['config'];group=f'v8-{kind}-{run.name.rsplit("_s",1)[0]}'
            add(cp,group,kind,config=cfg,seed=int(run.name.rsplit('_s',1)[1]),step=d['global_step'])
    # v9/v10 completed final exports; one representative per arm, honoring prior selection.
    for v,study in [('v9','attention_original_v1'),('v10','mapcritic_v1')]:
        base=ROOT/'logs'/v/study
        lock=read(base/'selection_lock.json') if (base/'selection_lock.json').exists() else {}
        if not lock and (base/'summary.json').exists():
            summary=read(base/'summary.json'); selected=summary.get('selected',{})
            if isinstance(selected,dict): lock=selected
        if v=='v9' and not lock.get('run'): lock={'run':'B_s22'}
        for p in sorted(base.glob('runs/*/exports/final/checkpoint.pt')):
            run=p.parents[2]; arm,seed=run.name.rsplit('_s',1)
            progress=read(run/'progress.json')
            add(p,f'{v}-{arm}','bundle',seed=int(seed),step=progress['step'],
                selected=lock.get('run')==run.name,assets={relative(p.parent/'policy.py'):sha(p.parent/'policy.py')})
    for kind in ('noop','random','collector','rush','raider','defender'):
        add(None,'scripted-'+kind,'strategy',config={'kind':kind})
    for name,special,radius in [('default',False,5),('special',True,5),('weak',False,12)]:
        add(None,'scripted-'+name,'scripted',config=dict(special=special,radius=radius))
    groups=defaultdict(list)
    for row in candidates:groups[row['id']].append(row)
    models=[]
    for key,rows in sorted(groups.items()):
        rows.sort(key=lambda r:(not r['selected'],not(r['main'] and r['complete']),
                               r['seed'] if r['main'] and r['complete'] else -r['step'], -r['step'], r['path'] or ''))
        chosen=rows[0];chosen['selection_reason']='prior selection' if chosen['selected'] else 'completed main / lowest seed / final step; otherwise largest completed budget'
        chosen['label']=key
        if chosen['kind']=='v7':
            c=chosen['config'];chosen['label']=c['experiment_id']
        # Pin external inference dependencies referenced by the selected config.
        def dependencies(value):
            if isinstance(value,dict):
                for child in value.values(): dependencies(child)
            elif isinstance(value,list):
                for child in value: dependencies(child)
            elif isinstance(value,str) and ('/' in value):
                asset=resolve(value)
                if asset.is_file() and (value.endswith(('.pt','.npz','.json','.yaml'))):
                    chosen['assets'][relative(asset)]=sha(asset)
        dependencies(chosen.get('config'))
        models.append(chosen)
        for row in rows[1:]:excluded.append(dict(path=row['path'],reason='same configuration representative: '+key))
    # Aliases with exactly identical inference weights/config collapse to one participant.
    seen={};unique=[]
    for row in models:
        if row['kind']=='legacy':
            d=payload(resolve(row['path']))
            h=__import__('hashlib').sha256(digest(row['config']).encode())
            for field in ('policy_state','planner_residual_v2_state','v6_model_state'):
                for k,t in sorted(d.get(field,{}).items()):
                    if torch.is_tensor(t):h.update(k.encode());h.update(t.cpu().contiguous().numpy().tobytes())
            identity=h.hexdigest()
            if identity in seen:
                excluded.append(dict(path=row['path'],reason='identical inference policy alias: '+seen[identity]));continue
            seen[identity]=row['id']
        unique.append(row)
    return dict(schema='blackout.league.roster.v1', models=unique, excluded=excluded,
                selection='existing selection, otherwise completed main lowest seed final; pilot fallback largest step')
