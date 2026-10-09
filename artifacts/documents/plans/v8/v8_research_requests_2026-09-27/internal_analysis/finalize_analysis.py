"""Reproducible provenance, paired descriptive statistics, and report tables."""
import json
import subprocess
from pathlib import Path
import numpy as np
from analyze import ROOT, OUT, read, sha, write, INPUTS

def main():
    provenance=[]
    for dataset in ('fafb783','malecns_v1'):
        manifest=read(ROOT/'data/connectomes'/dataset/'source_manifest.json')
        for f in manifest['files']:
            p=ROOT/'data/connectomes'/dataset/'raw'/f['name']
            actual=sha(p)
            provenance.append(dict(path=str(p.relative_to(ROOT)),expected=f['sha256'],actual=actual,matches=actual==f['sha256']))
    repos={}
    for alias,p in [('ROOT',ROOT),('GAME',ROOT.parent/'blackout'),('API',ROOT.parent/'blackout-env')]:
        repos[alias]=dict(path=str(p),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip(),
                         status=subprocess.check_output(['git','status','--short'],cwd=p,text=True))
    for p in list((ROOT/'blackout_rl').rglob('*.py'))+list((ROOT/'scripts').glob('*.py'))+list((ROOT/'tests').rglob('*.py')):
        sha(p)
    for root,extension in [(ROOT.parent/'blackout'/'Assets/Project/Runtime/Scripts','*.cs'),(ROOT.parent/'blackout-env'/'blackout_env','*.py')]:
        for p in root.rglob(extension):sha(p)
    for p in (ROOT/'v8_research_requests_2026-09-27').glob('*.md'):
        if p.name!='internal_result.md':sha(p)
    for name in ('history.md','README.md','issues/1st_issues_v0-v7.md',
                 'reports/v7/acceleration_baseline_profile.json','reports/v7/acceleration_baseline.json',
                 'reports/v7/acceleration_mps_comparison.json','reports/v7/acceleration_equivalence.json',
                 'reports/v7/acceleration_direct_final.json','reports/v7/acceleration_and_resume.md',
                 'reports/v7/unity_causality_teammate_v2.json','data/connectomes/malecns_v1/whole_brain_audit.json'):
        sha(ROOT/name)
    inv=read(OUT/'artifact_inventory.json');cube=read(OUT/'evaluation_cube.json');exp=read(OUT/'exposure_summary.json');train=read(OUT/'training_summary.json')
    comparisons=[];rng=np.random.default_rng(927)
    for model in ('A0','A1','A3','F1'):
        rows=[r for r in cube if r['model']==model];seeds=sorted({r['train_seed'] for r in rows});maps=sorted({r['seed'] for r in rows})
        lookup={(r['model'],r['train_seed'],r['seed'],r['team']):r['win'] for r in cube}
        diff=np.array([[[lookup[(model,s,m,t)]-lookup[('A2',s,m,t)] for t in (0,1)] for m in maps] for s in seeds])
        estimates=[]
        for _ in range(10000):
            si=rng.integers(len(seeds),size=len(seeds));mi=rng.integers(len(maps),size=len(maps))
            estimates.append(diff[si][:,mi].mean())
        comparisons.append(dict(model=model,baseline='A2',training_seeds=seeds,n_paired_map_side=len(maps)*2,
            scope='descriptive wrapper-winner only; crossed resampling training-seed and paired-map blocks; not verified engine victory',
            mean_win_difference=float(diff.mean()),illustrative_crossed_bootstrap_95=np.quantile(estimates,[.025,.975]).tolist(),
            new_wins=int((diff==1).sum()),lost_wins=int((diff==-1).sum()),unchanged=int((diff==0).sum()),
            paired_changes=[dict(train_seed=s,map_seed=m,side=t,win_difference=int(diff[i,j,t])) for i,s in enumerate(seeds) for j,m in enumerate(maps) for t in (0,1) if diff[i,j,t]!=0]))
    write('paired_statistics.json',comparisons)
    aggregate=[]
    for model in ('A0','A1','A2','A3','F1'):
        es=[r for r in exp if r['model']==model];ts=[r for r in train if r['model']==model]
        sides={t:sum(r['exact_side_steps_with_partial'][str(t)] for r in es) for t in (0,1)}
        eps={t:sum(r['completed_episodes'][str(t)] for r in es) for t in (0,1)}
        aggregate.append(dict(model=model,episode_A_fraction=eps[0]/sum(eps.values()),step_A_fraction=sides[0]/sum(sides.values()),
            sampled_override_mean=float(np.mean([r['sampled_override_rate'] for r in ts])),
            collector_hours_range=[min(r['collection_seconds_estimate']/3600 for r in ts),max(r['collection_seconds_estimate']/3600 for r in ts)],
            first_to_last_update_hours_range=[min(r['first_to_last_update_seconds']/3600 for r in ts),max(r['first_to_last_update_seconds']/3600 for r in ts)]))
    write('aggregate_statistics.json',aggregate)
    table=['# 23-run 산출물 색인','', '전체 SHA256·설정·source·build·opponent·runtime·lineage는 artifact_inventory.json에 있다. 아래 hash는 표시용 앞 12자리다. 시간은 UTC이며 시작/완료 시각 대신 첫/마지막 update 기록이다. 모든 final checkpoint는 2,000,000 step이다.','', '| 모델 | seed | final ckpt | graph | config | 첫 update / 마지막 update (UTC) | runtime 기록 | 결과 파일 |','|---|---:|---|---|---|---|---|---|']
    for r in sorted(inv['runs'],key=lambda x:(x['model'],x['seed'])):
        table.append(f"| {r['model']} | {r['seed']} | `{r['checkpoint_sha256'][:12]}` | `{r['graph_sha256'][:12]}` | `{r['config_sha256'][:12]}` | {r['first_update_at']} / {r['last_update_at']} | {r['runtime']['brain_backend']} | [result](../../{r['result']}) |")
    (OUT/'run_inventory.md').write_text('\n'.join(table)+'\n')
    write('provenance_verification.json',dict(repositories=repos,raw_connectome_files=provenance))
    sha(Path(__file__));write('finalize_input_hashes.json',INPUTS)
    print('raw manifests:',len(provenance),'mismatches:',sum(not r['matches'] for r in provenance),'paired comparisons:',len(comparisons))

if __name__=='__main__':main()
