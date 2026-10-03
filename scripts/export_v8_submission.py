#!/usr/bin/env python3
"""Export a newly trained competition checkpoint as policy.py + checkpoint.pt."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    from blackout_rl.v8.competition import study
    from blackout_rl.v8.competition.exporting import export_policy,verify_export
    from blackout_rl.v8.checkpoints import CheckpointStore,atomic_json
    from blackout_rl.v8.contracts import file_hash
    p=argparse.ArgumentParser();p.add_argument('--checkpoint-store',required=True);p.add_argument('--sha');p.add_argument('--output',required=True);p.add_argument('--report',required=True);args=p.parse_args()
    study.validate();payload,h=CheckpointStore(args.checkpoint_store).load(args.sha)
    if payload.get('registration_sha256')!=file_hash(study.REGISTRATION):raise ValueError('not a checkpoint from the new original-environment experiment')
    directory=export_policy(payload,args.output)
    report=verify_export(directory,payload['policy_state'],Path(args.checkpoint_store).parent/'policy_inputs.pt')
    atomic_json(args.report,dict(report,checkpoint_sha256=h,global_step=payload['global_step']))
    print(directory)


if __name__=='__main__':main()
