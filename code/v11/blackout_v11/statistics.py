"""Opponent/family/side summaries and paired actual-map uncertainty."""
from collections import defaultdict
import numpy as np


def points(row,key='winner'):
    value=row.get(key)
    return .5 if value is None else float(value==0)


def interval(values,alpha=.05):
    # Wilson guard prevents a degenerate all-win bootstrap claiming certainty.
    a=np.asarray(values,dtype=float)
    if not len(a): return [0.,1.]
    from statistics import NormalDist
    n=len(a);p=float(a.mean());z=NormalDist().inv_cdf(1-alpha/2)
    center=(p+z*z/(2*n))/(1+z*z/n)
    radius=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    rng=np.random.default_rng(202611)
    samples=a[rng.integers(n,size=(2000,n))].mean(1)
    lo,hi=np.quantile(samples,[alpha/2,1-alpha/2])
    return [max(0.,min(float(lo),float(center-radius))),min(1.,max(float(hi),float(center+radius)))]


def grouped_bound(rows,key='winner',alpha=.05):
    maps=defaultdict(list)
    for r in rows: maps[r['initial_map_hash']].append(points(r,key))
    return interval([np.mean(x) for x in maps.values()],alpha)


def summarize(rows,invalid=0):
    if not rows: raise ValueError('empty evaluation')
    def group(items):
        values=[points(r) for r in items]
        scores=[points(r,'visible_score_winner') for r in items]
        return dict(n=len(items),wins=values.count(1.),draws=values.count(.5),losses=values.count(0.),
            win_rate=values.count(1.)/len(items),score_rate=float(np.mean(values)),
            visible_score_rate=float(np.mean(scores)),map_interval=grouped_bound(items),
            unique_maps=len({r['initial_map_hash'] for r in items}),
            delivery_proxy=sum(r.get('delivery_proxy',0) for r in items),
            inference_p95_ms=float(np.quantile([r.get('inference_p95_ms',0) for r in items],.95)))
    by_opp=defaultdict(list);by_family=defaultdict(list);by_side=defaultdict(list)
    for r in rows:
        by_opp[r['opponent']['id']].append(r)
        by_family[r['opponent'].get('family','unknown')].append(r)
        by_side[str(r['side'])].append(r)
    opponents={k:group(v) for k,v in sorted(by_opp.items())}
    families={k:group(v) for k,v in sorted(by_family.items())}
    sides={k:group(v) for k,v in sorted(by_side.items())}
    return dict(**group(rows),opponents=opponents,families=families,sides=sides,
        opponent_macro=float(np.mean([g['score_rate'] for g in opponents.values()])),
        family_macro=float(np.mean([g['score_rate'] for g in families.values()])),
        worst_family=min(g['score_rate'] for g in families.values()),
        worst_side=min(g['score_rate'] for g in sides.values()),
        invalid=invalid,invalid_rate=invalid/(invalid+len(rows)),
        objective='provided_runner_v1',official_server_certified=False,
        interval_scope='actual-map cluster bootstrap with Wilson guard; not requested seed count')


def selection_key(report,parameters=0):
    strong=[v['score_rate'] for k,v in report['opponents'].items() if k.startswith(('v4-','v6-'))]
    return (float(np.mean(strong)) if strong else 0.,report['family_macro'],report['worst_side'],
            -report['inference_p95_ms'],-parameters)


def economic_comparison(candidate,baseline):
    index={(r['opponent']['id'],r['replicate'],r['side']):r for r in baseline}
    clusters=defaultdict(list); unmatched=0
    for r in candidate:
        b=index.get((r['opponent']['id'],r['replicate'],r['side']))
        if b is None: continue  # The fixed reference cannot play itself.
        if r['initial_map_hash']!=b['initial_map_hash']:
            unmatched+=1; continue
        clusters[r['initial_map_hash']].append(points(r,'visible_score_winner')-points(b,'visible_score_winner'))
    values=np.asarray([np.mean(v) for v in clusters.values()])
    if not len(values): return dict(passed=False,pairs=0,unmatched_maps=unmatched,lower_bound=-1.,mean=None)
    rng=np.random.default_rng(202612)
    samples=values[rng.integers(len(values),size=(2000,len(values)))].mean(1)
    lower=float(np.quantile(samples,.05)) if len(values)>1 else -1.
    return dict(passed=bool(lower>=-.05 and len(values)>=10),pairs=sum(map(len,clusters.values())),map_clusters=len(values),
                unmatched_maps=unmatched,lower_bound=float(lower),mean=float(values.mean()))


def final_gates(report,rows,baseline_rows):
    strong={k:v for k,v in report['opponents'].items() if k.startswith(('v4-','v6-'))}
    bounds={k:grouped_bound([r for r in rows if r['opponent']['id']==k],alpha=.025)[0] for k in strong}
    scripted=[v for k,v in report['opponents'].items() if k.startswith('scripted-')]
    planner_sides=[]
    for side in (0,1):
        values=[points(r) for r in rows if r['side']==side and r['opponent']['family']=='planner']
        planner_sides.append(float(np.mean(values)) if values else 0.)
    return dict(G4=dict(passed=len(strong)==2 and all(v['score_rate']>=.6 and bounds[k]>.5 for k,v in strong.items()),
                        strong_lower_bounds=bounds,multiplicity='Bonferroni 95% simultaneous (conservative Holm sufficient condition)'),
                G5=dict(passed=report['opponent_macro']>=.9 and all(v['score_rate']>=.8 for v in scripted)
                        and min(planner_sides)>=.5,planner_side_rates=planner_sides),
                G6=economic_comparison(rows,baseline_rows))
