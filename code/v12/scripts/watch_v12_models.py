"""Visible, normal-speed v12 versus v12 exhibition using the exported policies."""
from pathlib import Path
import sys

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'code/shared/project_paths.py').is_file())
sys.path.insert(0,str(ROOT/'code/shared'))
from project_paths import activate
activate()

import argparse
import time
import torch
from blackout_rl.env import ContractBlackOutEnv
from blackout_v11.trajectory import canonical_names
from blackout_v12.export import load_export
from blackout_v12.env import executable
from blackout_v12.io import atomic_json, file_hash
from blackout_v12.runtime import FileLease, process_tree_guard


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',default='joint_trip_v12')
    parser.add_argument('--speed',type=float,default=1.)
    parser.add_argument('--seed',type=int,default=20261027)
    args=parser.parse_args()
    if not 0<args.speed<=50 or Path(args.study).name!=args.study:
        parser.error('simple study name and 0 < speed <= 50 required')
    torch.set_num_threads(1)
    bundle=ROOT/'artifacts/submission/v12'/args.study
    policies=[load_export(bundle),load_export(bundle)]
    session=ROOT/'logs/v12/watch'/str(time.time_ns())
    session.mkdir(parents=True)
    atomic_json(session/'manifest.json',dict(blue=str(bundle),red=str(bundle),speed=args.speed,
        seed=args.seed,checkpoint_sha256=file_hash(bundle/'checkpoint.pt'),exhibition_only=True))
    env=None
    with FileLease(ROOT/'logs/runtime/compute.lock',blocking=False),process_tree_guard():
        try:
            env=ContractBlackOutEnv(env_path=str(executable({})),no_graphics=False,
                background=False,time_scale=args.speed,worker_id=0,base_port=None)
            obs,_=env.reset(seed=args.seed)
            atomic_json(session/'status.json',dict(state='running',steps=0))
            print('V12 vs V12: Blue and Red use separate F/seed11/MAP policy instances. Speed='+str(args.speed),flush=True)
            for step in range(22000):
                actions={}
                with torch.inference_mode():
                    for side,policy in enumerate(policies):
                        names=canonical_names(side)
                        actions.update(policy.act({name:obs[name] for name in names}))
                obs,rewards,terms,truncs,_=env.step(actions)
                if any(truncs.values()):
                    raise RuntimeError('Exhibition truncated')
                if terms and all(terms.values()):
                    atomic_json(session/'status.json',dict(state='complete',steps=step+1))
                    print('Exhibition finished.',flush=True)
                    break
                if (step+1)%250==0:
                    atomic_json(session/'status.json',dict(state='running',steps=step+1))
                    print('steps='+str(step+1),flush=True)
            else:
                raise RuntimeError('Exhibition step limit reached')
        except BaseException as exc:
            atomic_json(session/'status.json',dict(state='closed',reason=str(exc)))
            raise
        finally:
            if env is not None:
                env.close()


if __name__=='__main__':
    main()
