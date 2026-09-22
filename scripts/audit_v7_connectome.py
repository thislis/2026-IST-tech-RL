#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blackout_rl.v7_registry import load_config,check_config
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    graph,_=check_config(load_config(a.config));print(json.dumps(graph.audit(),indent=2))
