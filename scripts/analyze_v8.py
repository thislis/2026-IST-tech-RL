#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blackout_rl.v8.statistics import compare
from blackout_rl.v8.checkpoints import atomic_json

def main():
    p=argparse.ArgumentParser(description='Compare complete registered run,map,side W/N arrays.')
    p.add_argument('--candidate',required=True);p.add_argument('--reference',required=True);p.add_argument('--output',required=True)
    p.add_argument('--fixed-reference',action='store_true');p.add_argument('--replicates',type=int,default=20000);p.add_argument('--seed',type=int,default=0);a=p.parse_args()
    result=compare(json.loads(Path(a.candidate).read_text()),json.loads(Path(a.reference).read_text()),reference_fixed=a.fixed_reference,replicates=a.replicates,seed=a.seed)
    result['input_files']={p:__import__('hashlib').sha256(Path(p).read_bytes()).hexdigest() for p in (a.candidate,a.reference)}
    atomic_json(a.output,result);print(a.output)
if __name__=='__main__':main()
