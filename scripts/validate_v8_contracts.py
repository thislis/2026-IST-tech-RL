#!/usr/bin/env python3
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser();p.add_argument('--live',action='store_true');p.add_argument('--config',default='configs/v8/experiments/planner.yaml')
    p.add_argument('--output',default='reports/v8/contract_validation.json');a=p.parse_args()
    if not a.live:
        from blackout_rl.v8.checkpoints import atomic_json
        run=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests/v8','-v'],cwd=ROOT,capture_output=True,text=True)
        report=dict(scope='synthetic_offline',returncode=run.returncode,stdout=run.stdout,stderr=run.stderr,live_passed=False)
        atomic_json(a.output,report);print(a.output);raise SystemExit(run.returncode)
    from _v8_cli import setup
    cfg=setup(a.config)
    from blackout_rl.v8.checkpoints import atomic_json
    from blackout_rl.v8.live_validation import validate
    report=validate(cfg)
    atomic_json(a.output,report);print(a.output)
if __name__=='__main__':main()
