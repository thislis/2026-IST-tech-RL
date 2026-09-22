#!/usr/bin/env python3
"""Paired-side evaluation; final test requires a pre-existing model lock."""
import argparse,json,sys,time,shutil,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from blackout_rl.connectome.graph_artifact import atomic_json,sha256,digest
from blackout_rl.v7_registry import path,check_config,make_model,source_fingerprint
from blackout_rl.v7_training import opponent_for
from blackout_rl.env import ContractBlackOutEnv
from blackout_rl.mappo_v6 import V6Policy,ROLES
from blackout_rl.v7_2.policy import DirectPolicy
from blackout_rl.v7_2.teammates import make_teammates
from blackout_rl.batching import team_agents
from blackout_rl.frozen_opponent import FrozenScriptedOpponent
from blackout_rl.evaluation_protocol import paired_seed_bootstrap_ci


def freeze_checkpoint(run, checkpoint):
    directory=run/'evaluation_checkpoints';directory.mkdir(exist_ok=True)
    temporary=directory/(uuid.uuid4().hex+'.tmp')
    shutil.copyfile(checkpoint,temporary)
    target=directory/(sha256(temporary)+'.pt')
    if target.exists():
        if sha256(target)!=sha256(temporary):raise ValueError('immutable checkpoint corrupted')
        temporary.unlink()
    else:temporary.replace(target)
    return target


def model_lock(run,checkpoint):
    cfg=json.loads((run/'resolved_config.json').read_text())
    return dict(schema='blackout.v7.model_lock.v1',checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),
        config_sha256=digest(cfg),split_sha256=sha256(path(cfg['evaluation']['split_manifest'])),
        evaluator_sha256=sha256(__file__),sources=source_fingerprint())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--split',choices=['dev','confirmation','test'],default='dev')
    p.add_argument('--checkpoint');p.add_argument('--lock');p.add_argument('--create-lock',action='store_true')
    p.add_argument('--opponent',choices=['scripted','weak','target'],default='target')
    p.add_argument('--max-maps',type=int,help='development smoke only; forbidden for final test')
    a=p.parse_args();run=path(a.run);checkpoint=path(a.checkpoint) if a.checkpoint else run/'checkpoints/latest.pt'
    cfg=json.loads((run/'resolved_config.json').read_text());graph,splits=check_config(cfg)
    if a.split=='test' and a.lock and not a.checkpoint:
        checkpoint=path(json.loads(path(a.lock).read_text())['checkpoint'])
    checkpoint=freeze_checkpoint(run,checkpoint)
    lock=model_lock(run,checkpoint)
    if a.create_lock:
        destination=run/'final_model_lock.json'
        if destination.exists():raise FileExistsError(destination)
        atomic_json(destination,lock);print(destination);return
    if a.split=='test':
        if not a.lock or a.max_maps is not None:raise ValueError('final test requires --lock and all registered maps')
        if json.loads(path(a.lock).read_text())!=lock:raise ValueError('model lock no longer matches checkpoint/config/code')
        output=run/'final_test.json'
        if output.exists() or (run/'final_test.started.json').exists():raise FileExistsError('final test already started; do not silently rerun selection')
    else:
        output=run/f'eval_{a.split}_{a.opponent}_{time.time_ns()}.json'
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if payload['config_sha256']!=digest(cfg) or payload['sources']!=source_fingerprint():raise ValueError('checkpoint provenance mismatch')
    torch.set_num_threads(cfg['runtime']['torch_threads'])
    with torch.random.fork_rng():model=make_model(cfg,graph)
    model.load_state_dict(payload['model']);model.eval()
    if a.split=='test':atomic_json(run/'final_test.started.json',lock)
    direct=cfg['mode']=='whole_connectome_direct';rows=[]
    maps=splits[a.split][:a.max_maps] if a.max_maps is not None else splits[a.split]
    env=ContractBlackOutEnv(env_path=str(path(cfg['environment']['build'])),background=True,no_graphics=False,time_scale=cfg['environment']['time_scale'])
    brain_policy=DirectPolicy(graph,None if cfg['controller'].get('training_mode')=='F0_fixed' else model,
        controlled_slots=cfg['controller']['controlled_slots'],noise_seed=payload['seed']+30000,decoder_config=cfg['controller']['fixed_decoder']) if direct else None
    try:
        for seed in maps:
            for team in (0,1):
                torch.manual_seed(payload['seed'])
                observations,_=env.reset(seed=seed);agents=team_agents(team);other=team_agents(1-team)
                opponent=opponent_for(a.opponent,1-team,cfg);opponent.reset()
                policy=brain_policy if direct else V6Policy(model,team=team)
                if direct:
                    policy.reset_episode_state(f'eval:{seed}:{team}',team,noise_seed=seed+30000)
                    teammate=make_teammates(team,cfg);teammate.reset()
                else:policy.reset()
                for step in range(cfg['environment']['max_episode_steps']):
                    if direct:
                        actions={} if len(policy.slots)==5 else teammate.act(observations,agents)
                        actions.update(policy.act(observations,episode_id=f'eval:{seed}:{team}',env_step_id=step,team_id=team))
                    else:actions=policy.act(observations,agents)
                    actions.update(opponent.act(observations,other))
                    observations,_,terms,truncs,infos=env.step(actions)
                    if all(terms.values()):
                        info=infos[agents[0]];winner=int(info['winner'])
                        if winner not in (-1,0,1):raise ValueError('invalid terminal winner')
                        rows.append(dict(seed=seed,team=team,winner=winner,win=int(winner==team),draw=int(winner==-1),
                            score_0=float(info['score_0']),score_1=float(info['score_1']),steps=step+1))
                        break
                    if any(truncs.values()):raise RuntimeError('truncated match cannot be scored as terminal outcome')
                else:raise RuntimeError('evaluation episode watchdog')
                atomic_json(output.with_suffix('.progress.json'),dict(episodes=rows,opponent=a.opponent,split=a.split))
    finally:env.close()
    paired=np.array([np.mean([r['win'] for r in rows if r['seed']==s]) for s in maps])
    rng=np.random.default_rng(0)
    means=paired[rng.integers(len(maps),size=(10000,len(maps)))].mean(-1)
    atomic_json(output,dict(schema='blackout.v7.evaluation.v1',checkpoint_sha256=sha256(checkpoint),split=a.split,opponent=a.opponent,
        win_rate=float(np.mean([r['win'] for r in rows])),draw_rate=float(np.mean([r['draw'] for r in rows])),
        paired_map_bootstrap_ci=np.quantile(means,[.025,.975]).tolist(),training_seed=payload['seed'],
        statistical_scope='within_one_training_seed; aggregate independent runs before architecture claims',episodes=rows))
    print(output)

if __name__=='__main__':main()
