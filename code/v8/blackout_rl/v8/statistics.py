"""Independent algorithm-run rows crossed with shared paired-map blocks."""
import numpy as np
from .contracts import ContractError

def compare(candidate,reference,*,reference_fixed=False,replicates=20000,seed=0):
    # Shapes run,map,side (opponent/replicate are fixed strata or predeclared averages).
    a,b=np.asarray(candidate,float),np.asarray(reference,float)
    if a.ndim!=3 or b.ndim!=3 or a.shape[1:]!=b.shape[1:] or a.shape[2]!=2:
        raise ContractError('complete run x map x two sides arrays required')
    if min(a.shape)<1 or min(b.shape)<1 or not np.isfinite(a).all() or not np.isfinite(b).all():raise ContractError('missing/invalid cells cannot be dropped')
    if np.any((a<0)|(a>1)) or np.any((b<0)|(b>1)):raise ContractError('expected W/N observations')
    if reference_fixed and b.shape[0]!=1:raise ContractError('fixed reference must have exactly one row')
    if replicates<1:raise ContractError('positive bootstrap count')
    rng=np.random.default_rng(seed)
    samples={k:[] for k in ('crossed','fixed_map_run','fixed_policy_map')}
    sides=[]
    for _ in range(replicates):
        maps=rng.integers(a.shape[1],size=a.shape[1])
        ai=rng.integers(len(a),size=len(a));bi=np.array([0]) if reference_fixed else rng.integers(len(b),size=len(b))
        difference=a[ai][:,maps].mean(0)-b[bi][:,maps].mean(0)
        samples['crossed'].append(difference.mean());sides.append(difference.mean(0))
        samples['fixed_map_run'].append(a[ai].mean()-b[bi].mean())
        samples['fixed_policy_map'].append(a[:,maps].mean()-b[:,maps].mean())
    return dict(delta=float(a.mean()-b.mean()),candidate_run_means=a.mean((1,2)).tolist(),reference_run_means=b.mean((1,2)).tolist(),
                candidate_side=a.mean((0,1)).tolist(),reference_side=b.mean((0,1)).tolist(),
                intervals={k:np.quantile(v,[.025,.975]).tolist() for k,v in samples.items()},
                side_difference_intervals=np.quantile(sides,[.025,.975],axis=0).T.tolist(),
                independent_run_rows=True,shared_paired_map_indices=True,reference_fixed=reference_fixed,
                replicates=replicates,seed=seed,scope='conditional on registered build/opponent/maps; few-run coverage limited')
