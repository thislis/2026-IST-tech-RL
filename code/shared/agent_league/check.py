"""Offline adapter checks; no Unity, no optimizer, no training."""
import argparse
import gc
import time
import numpy as np
import torch
from .common import read,atomic,verify_assets
from .adapters import Adapter


def fixture():
    obs={};g=np.zeros((96,96,11),np.float32);g[:,:,0]=1
    for y,x,c in [(5,5,2),(85,85,3),(40,40,6)]:g[y:y+4,x:x+4,0]=0;g[y:y+4,x:x+4,c]=1
    e=np.zeros((10,9),np.float32);e[:,:2]=np.array([[.15+(i%5)*.12,.2 if i<5 else .8] for i in range(10)]);e[:,3]=1
    for i in range(10):
        v=np.zeros(96,np.float32);e[:,2]=np.where(np.arange(10)//5==i//5,1,-1);v[:90]=e.reshape(-1);v[90+i%3]=1;v[95]=1
        obs[f'unit_{i}']=dict(vector=v,graphic=g.copy())
    return obs


def check_one(spec):
    torch.set_num_threads(1);obs=fixture();start=time.monotonic();measurements=[]
    for side in (0,1):
        p=Adapter(spec,side,1234);q=Adapter(spec,side,1234)
        for step in range(2):
            t=time.perf_counter();a=p.act(obs,step);elapsed=time.perf_counter()-t;b=q.act(obs,step)
            if set(a)!=set(p.names):raise ValueError('wrong team keys')
            for n in p.names:
                x=a[n]
                if x.shape!=(2,) or x.dtype!=np.float32 or not np.isfinite(x).all() or (np.abs(x)>1).any():raise ValueError('action contract')
                np.testing.assert_array_equal(x,b[n],err_msg='RNG/state not isolated')
            measurements.append(elapsed)
            # Exercise an advancing public clock, keeping fixture predictable.
            for row in obs.values():row['vector'][95]-=.0001
        del p,q;gc.collect()
    verify_assets(spec)
    return dict(id=spec['id'],passed=True,teams=[0,1],calls=4,wall_seconds=time.monotonic()-start,
                max_inference_seconds=max(measurements),live_game=False,source_behavior='original loader and decoder')


def main():
    p=argparse.ArgumentParser();p.add_argument('spec');p.add_argument('output');args=p.parse_args()
    try:result=check_one(read(args.spec))
    except Exception as e:
        import traceback
        result=dict(passed=False,error=str(e),traceback=traceback.format_exc())
    atomic(args.output,result)
    if not result['passed']:raise SystemExit(1)

if __name__=='__main__':main()
