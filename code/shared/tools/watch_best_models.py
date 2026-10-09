"""Visible, real-time exhibition of pinned v6 and v7-1 A3 models. No training."""

# Locate shared modules from the physical versioned layout.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents
                    if (p / "code/shared/project_paths.py").is_file())
_layout_sys.path.insert(0, str(_layout_root / "code/shared"))
from project_paths import activate as _activate_layout
_activate_layout()
from project_paths import project_root, project_path, verify_relocated_source

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import types

ROOT=project_root()
sys.path.insert(0,str(ROOT))
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'

import numpy as np
import torch
from blackout_rl.batching import team_agents
from blackout_rl.mappo_v6 import V6Policy,ROLES,model_from_payload
from blackout_rl.v7_registry import make_model,check_config
from blackout_rl.connectome.graph_artifact import sha256,digest
from blackout_rl.v8.provided_environment import verify_original

V6_PATH='artifacts/checkpoints/v6/mappo_planner_residual_v6_latest.pt'
V6_SHA='6429337e23e71c84e4fb618b3d0ff5edd102212be487d71b07b1e1721062be41'
V7_PATH='logs/v7/v7_1_fafb783_flywire_main_2m_v1/11/evaluation_checkpoints/3598c52f6a8c3edd85af1eb86fe9f84f9ab397e6182b1aa84875f76dfa154ce0.pt'
V7_SHA='3598c52f6a8c3edd85af1eb86fe9f84f9ab397e6182b1aa84875f76dfa154ce0'
NAMES=('v6 final (1,159,168 step)','v7-1 A3 FlyWire / seed 11 (2,000,000 step)')


def pinned_payload(path,expected):
    path=project_path(path, root=ROOT)
    if not path.is_file():raise FileNotFoundError(f'Local model artifact required: {path}')
    if sha256(path)!=expected:raise ValueError(f'Pinned model changed: {path}')
    return torch.load(path,map_location='cpu',weights_only=True)


def load_models():
    verify_original()
    v6=pinned_payload(V6_PATH,V6_SHA);v7=pinned_payload(V7_PATH,V7_SHA)
    assert v6['training']['global_step']==1159168 and v7['global_step']==2000000
    # Preserve the exact v6 planner source, without changing the current v7
    # planner module or any provided game/environment implementation.
    original=v6['source']['files'];old_path='blackout_rl/scripted_fsm.py'
    for sources in (original,v7['sources']):
        for name,h in sources.items():
            if name.startswith('blackout_rl/') and not (sources is original and name==old_path):
                verify_relocated_source(name,h)
    source=subprocess.check_output(['git','-C',str(ROOT),'show',v6['source']['git_sha']+':'+old_path])
    if hashlib.sha256(source).hexdigest()!=original[old_path]:raise ValueError('v6 planner snapshot mismatch')
    module=types.ModuleType('blackout_rl._viewer_v6_planner');module.__package__='blackout_rl'
    sys.modules[module.__name__]=module
    exec(compile(source,'<pinned v6 scripted_fsm>', 'exec'),module.__dict__)
    cfg=v7['config']
    if digest(cfg)!=v7['config_sha256']:raise ValueError('v7 checkpoint config changed')
    graph,_=check_config(cfg,verify_sources=False)
    with torch.random.fork_rng():second=make_model(cfg,graph)
    second.load_state_dict(v7['model'],strict=True)
    return (model_from_payload(v6),second.eval()),module.ScriptedTeamController


def policies_for(models,legacy_planner,swap):
    policies=[]
    for i,model in enumerate(models):
        team=i^int(swap);policy=V6Policy(model,team=team)
        if i==0:
            policy.context.planner=legacy_planner(team,roles=ROLES,chase_radius_cells=48)
        policy.reset();policies.append(policy)
    return policies


def visible_env(speed):
    # Direct supplied API and original app: no wrapper, seed workaround or
    # window-hiding launcher. This viewer deliberately displays the game.
    from blackout_env import BlackOutEnv
    return BlackOutEnv(env_path=str(project_path('artifacts/builds/BlackOut.app', root=ROOT)),no_graphics=False,time_scale=speed)


def play(models,legacy_planner,*,games=2,speed=1.,max_steps=22000,env_factory=visible_env):
    for game in range(games):
        swap=bool(game%2);policies=policies_for(models,legacy_planner,swap)
        print(f'\n경기 {game+1}/{games} | A: {NAMES[int(swap)]} | B: {NAMES[1-int(swap)]}',flush=True)
        env=env_factory(speed)
        try:
            obs,_=env.reset();totals=[0.,0.]
            for step in range(max_steps):
                actions={}
                with torch.inference_mode():
                    for policy in policies:actions.update(policy.act(obs,team_agents(policy.team)))
                if set(actions)!=set(env.agents):raise ValueError('Incomplete ten-agent action batch')
                for action in actions.values():
                    if action.shape!=(2,) or not np.isfinite(action).all() or (np.abs(action)>1).any():
                        raise ValueError('Invalid model action')
                obs,rewards,terms,truncs,infos=env.step(actions)
                for team in (0,1):totals[team]+=sum(float(rewards.get(n,0)) for n in team_agents(team))
                if any(truncs.values()):raise RuntimeError('Exhibition truncated; no result awarded')
                if not env.agents:
                    print(f'경기 종료: {step+1} step | 제공 누적 reward A={totals[0]:.2f}, B={totals[1]:.2f}',flush=True)
                    print('관전용 경기입니다. 공식 승률이나 과거 평가 재현으로 집계하지 않습니다.',flush=True)
                    break
                if (step+1)%500==0:print(f'  진행: {step+1} step',flush=True)
            else:raise RuntimeError(f'{max_steps}-step watchdog: stopped without awarding a result')
        finally:env.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true',help='Load and validate both models; do not launch Unity')
    parser.add_argument('--games',type=int,default=2,help='Number of visible games, swapping sides (default: 2)')
    parser.add_argument('--speed',type=float,default=1.,help='Unity time scale (default: 1, normal speed)')
    args=parser.parse_args()
    if args.games<1 or not np.isfinite(args.speed) or not 0<args.speed<=50:parser.error('games >= 1; 0 < speed <= 50 required')
    os.chdir(ROOT);torch.set_num_threads(1)
    models,legacy=load_models()
    print('모델/원본 게임 검사 완료. 학습·가중치 변경 없음.',flush=True)
    if args.check:
        print(json.dumps(dict(models=list(NAMES),checkpoint_sha256=[V6_SHA,V7_SHA],unity_started=False)));return
    print('게임 창에서 관전합니다. 종료: 이 터미널에서 Ctrl+C.',flush=True)
    def interrupt(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupt)
    try:play(models,legacy,games=args.games,speed=args.speed)
    except KeyboardInterrupt:print('\n관전을 종료했습니다. Unity를 정리했습니다.',flush=True)


if __name__=='__main__':main()
