"""Paired score-based evaluation with reusable, immutable per-match records."""
from collections import defaultdict
from pathlib import Path
import numpy as np
from .io import atomic_json,read_json,digest


class MapRegistry:
    def __init__(self,path):
        import sqlite3
        self.db=sqlite3.connect(path)
        self.db.execute('CREATE TABLE IF NOT EXISTS maps (hash TEXT PRIMARY KEY, split TEXT NOT NULL)')

    def add(self,map_hash,split):
        row=self.db.execute('SELECT split FROM maps WHERE hash=?',(map_hash,)).fetchone()
        if row is not None and row[0]!=split:return False
        self.db.execute('INSERT OR IGNORE INTO maps VALUES (?,?)',(map_hash,split));self.db.commit();return True

    def close(self):self.db.close()


def schedule(opponents,pairs,base_seed,baseline=False,prefix='dev'):
    result=[]
    for k,opponent in enumerate(opponents):
        for repeat in range(pairs):
            seed=base_seed+k*1000+repeat
            for side in (0,1):
                key=f'{prefix}_{"baseline" if baseline else "candidate"}_{opponent["id"]}_{seed}_{side}'
                result.append(dict(id=key,seed=seed,side=side,opponent=opponent,baseline=baseline))
    return result


def summarize(rows):
    groups=defaultdict(list)
    for row in rows:groups[row['opponent']].append(row['score_rate'])
    rates={k:float(np.mean(v)) for k,v in sorted(groups.items())}
    return dict(completed=len(rows),wins=sum(r['outcome']>0 for r in rows),
        draws=sum(r['outcome']==0 for r in rows),losses=sum(r['outcome']<0 for r in rows),
        macro_score_rate=float(np.mean(list(rates.values()))) if rates else 0.,
        worst_opponent_rate=min(rates.values(),default=0.),opponents=rates)


def cached_matches(tasks,directories):
    """Read exact paired tasks from caller-verified caches of the same policy version."""
    found=[]
    for task in tasks:
        for directory in directories:
            path=Path(directory)/(task['id']+'.json')
            if not path.exists():continue
            row=read_json(path)
            expected=dict(id=task['id'],seed=task['seed'],side=task['side'],
                          opponent=task['opponent']['id'],baseline=bool(task.get('baseline')))
            if any(row.get(k)!=v for k,v in expected.items()) or not row.get('natural_terminal'):
                raise ValueError('cached match does not match its scheduled task: '+str(path))
            found.append(row);break
    return found


def seed_cache(tasks,directories,destination):
    """Reuse completed games when expanding a screening evaluation to its full set."""
    destination=Path(destination)
    for row in cached_matches(tasks,directories):
        path=destination/(row['id']+'.json')
        if not path.exists():atomic_json(path,row)


def evaluate(pool,tasks,weights,directory,registry,split,stop=lambda:False):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    records=[];pending=[];hashes={};identity=digest([dict(t) for t in tasks])
    # Caller uses a version-specific directory; match identity prevents accidental cache reuse.
    manifest=directory/'schedule.json'
    if manifest.exists() and read_json(manifest)['sha256']!=identity:raise ValueError('evaluation schedule changed')
    atomic_json(manifest,dict(sha256=identity,tasks=tasks))
    for task in tasks:
        path=directory/(task['id']+'.json')
        if path.exists():records.append(read_json(path))
        else:pending.append(task)
    def save(row):
        if not registry.add(row['map_sha256'],split):
            raise ValueError('evaluation map overlaps another split; new registered seed schedule required')
        key=(row['opponent'],row['seed'])
        if key in hashes and hashes[key]!=row['map_sha256']:raise ValueError('paired initial maps differ')
        hashes[key]=row['map_sha256'];records.append(row)
        atomic_json(directory/(row['id']+'.json'),row)
        atomic_json(directory/'progress.json',dict(completed=len(records),expected=len(tasks)))
    for row in records:
        key=(row['opponent'],row['seed'])
        if key in hashes and hashes[key]!=row['map_sha256']:raise ValueError('cached paired maps differ')
        hashes[key]=row['map_sha256']
    pool.evaluate(pending,weights,save,stop)
    report=summarize(records);report['expected']=len(tasks);atomic_json(directory/'summary.json',report)
    return report,records


def paired_confirmation(candidate,baseline,direct,seed=1301,repeats=5000):
    base={(r['opponent'],r['seed'],r['side']):r for r in baseline}
    groups=defaultdict(lambda:defaultdict(list))
    for r in candidate:
        key=(r['opponent'],r['seed'],r['side'])
        if key in base:
            if r['map_sha256']!=base[key]['map_sha256']:raise ValueError('baseline/candidate maps differ')
            groups[r['opponent']][r['seed']].append(r['score_rate']-base[key]['score_rate'])
    blocks=[np.array([np.mean(pair) for pair in seeds.values() if len(pair)==2]) for seeds in groups.values()]
    if not blocks or any(not len(b) for b in blocks):raise ValueError('paired confirmation is incomplete')
    rng=np.random.default_rng(seed)
    samples=np.mean([rng.choice(b,(repeats,len(b)),replace=True).mean(1) for b in blocks],axis=0)
    direct_groups=defaultdict(list)
    for r in direct:direct_groups[r['seed']].append(r['score_rate'])
    db=np.array([np.mean(v) for v in direct_groups.values() if len(v)==2])
    if not len(db):raise ValueError('direct confirmation is incomplete')
    ds=rng.choice(db,(repeats,len(db)),replace=True).mean(1)
    delta_ci=np.quantile(samples,[.025,.975]).tolist();direct_ci=np.quantile(ds,[.025,.975]).tolist()
    return dict(macro_delta=float(np.mean([b.mean() for b in blocks])),macro_delta_ci95=delta_ci,
                direct_rate=float(db.mean()),direct_ci95=direct_ci,
                statistical_gate=delta_ci[0]>0 and direct_ci[0]>.5,
                method='opponent-stratified paired-seed block bootstrap; exploratory at boundary outcomes')
