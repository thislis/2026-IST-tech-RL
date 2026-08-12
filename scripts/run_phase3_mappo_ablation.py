#!/usr/bin/env python3
"""Equal-budget live IPPO/MAPPO ablation against the frozen scripted opponent."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from blackout_rl import (ContractBlackOutEnv,FrozenScriptedOpponent,ParallelRolloutCollector,
                         PPOConfig,SubmissionPolicy,TeamTrainingReward,checkpoint_payload,
                         load_checkpoint,ppo_update,save_checkpoint)
from blackout_rl.evaluation_protocol import SeedSplits,paired_seed_bootstrap_ci
from blackout_rl.mappo import (AblationArm,MAPPOParallelRolloutCollector,compare_ippo_mappo,
                               initialize_mappo_from_ippo,mappo_update)
from scripts.train_ippo_vs_scripted import evaluate_checkpoint


def save_actor(path: Path, model, *, step: int, seed: int) -> None:
    actor=model.actor_model if hasattr(model,"actor_model") else model
    config=dict(actor.model_config); config.pop("n_actions",None)
    policy=SubmissionPolicy(**config); policy.actor_critic.load_state_dict(actor.state_dict())
    save_checkpoint(path,checkpoint_payload(policy,global_step=step,training_seed=seed,
                    source={"phase":"phase3-live-ablation"}))


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--steps",type=int,default=128)
    parser.add_argument("--workers",type=int,default=5); args=parser.parse_args()
    build=ROOT/"builds/BlackOut.app"; initial=ROOT/"checkpoints/base_r13_bc_warm_start.pt"
    splits=SeedSplits.from_dict(json.loads((ROOT/"configs/seed_splits_v1.json").read_text()))
    dev=list(splits.seeds_for("dev",purpose="model_selection")); train_seed=splits.train[0]
    outputs={}; ppo_config=PPOConfig(update_epochs=2,minibatch_size=128,target_kl=.03)
    for algorithm in ("ippo","mappo"):
        torch.manual_seed(33001); np.random.seed(33001)
        policy,_=load_checkpoint(initial); ippo=policy.actor_critic
        env=ContractBlackOutEnv(env_path=str(build),no_graphics=False,time_scale=50.)
        try:
            if algorithm=="ippo":
                model=ippo; collector=ParallelRolloutCollector(env,model,FrozenScriptedOpponent(1),
                    learning_team=0,reward_transform=TeamTrainingReward(0))
                batch=collector.collect(args.steps,seed=train_seed).as_batch(gamma=.999,gae_lambda=.95)
                diagnostics=ppo_update(model,torch.optim.Adam(model.parameters(),3e-4),batch,config=ppo_config)
            else:
                model=initialize_mappo_from_ippo(ippo)
                collector=MAPPOParallelRolloutCollector(env,model,FrozenScriptedOpponent(1),
                    learning_team=0,reward_transform=TeamTrainingReward(0))
                batch=collector.collect(args.steps,seed=train_seed).as_batch(gamma=.999,gae_lambda=.95)
                diagnostics=mappo_update(model,torch.optim.Adam(model.parameters(),3e-4),batch,config=ppo_config)
        finally: env.close()
        checkpoint=ROOT/f"checkpoints/phase3_{algorithm}_{args.steps}.pt"; save_actor(checkpoint,model,step=args.steps,seed=33001)
        series=evaluate_checkpoint(build=build,checkpoint=checkpoint,
            opponent_config=ROOT/"configs/policies/scripted_battery_v1.json",seeds=dev,
            workers=args.workers,time_scale=50.,max_episode_steps=22000,global_step=args.steps,
            output=ROOT/f"logs/phase3_{algorithm}_{args.steps}_dev.json")
        summary=series["summary"]
        outputs[algorithm]={"diagnostics":diagnostics.to_dict(),"summary":summary,
            "win_rate_ci":paired_seed_bootstrap_ci(series["episodes"],metric="win_rate",resamples=10000,seed=33001).to_dict(),
            "score_diff_ci":paired_seed_bootstrap_ci(series["episodes"],metric="score_diff",resamples=10000,seed=33002).to_dict()}
    comparison=compare_ippo_mappo(
        AblationArm("ippo",tuple(dev),"scripted-battery-v1",args.steps,outputs["ippo"]["summary"]["win_rate"],outputs["ippo"]["summary"]["mean_model_score_diff"]),
        AblationArm("mappo",tuple(dev),"scripted-battery-v1",args.steps,outputs["mappo"]["summary"]["win_rate"],outputs["mappo"]["summary"]["mean_model_score_diff"]))
    payload={"schema_version":"blackout.phase3_mappo_ablation.v1","train_seed":train_seed,
             "dev_seeds":dev,"environment_steps_per_arm":args.steps,"opponent":"scripted-battery-v1",
             "arms":outputs,"comparison":comparison}
    path=ROOT/f"logs/phase3_mappo_ablation_{args.steps}.json"; path.write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps({"output":str(path),"comparison":comparison},indent=2))


if __name__=="__main__": main()
