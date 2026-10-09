"""Small source-linked synthetic reproductions; no game process or checkpoint edits."""
import ast
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
OUT=Path(__file__).resolve().parent
from blackout_rl.mappo_v6 import PlannerFeatures,V6Model,joint_distribution
from blackout_rl.navigation import PathNotFound
from blackout_rl.reward import ScoreDeltaRewardTracker
from blackout_rl.v7_1.model import GraphResidualHead
from blackout_rl.v7_2.policy import DirectPolicy
from blackout_rl.v7_2.decoder import ReadoutModel
from blackout_rl.v7_2.dynamics import WholeBrain,DynamicsConfig
from tests.test_mappo_v6 import all_observations
from tests.test_ppo_training import small_model
from tests.v7.test_contracts import graph
from blackout_env import BlackOutEnv
from blackout_env.env.obs_preprocessor import ObsPreprocessor,load_semantic_config
import blackout_env

def run():
    torch.set_num_threads(1);torch.manual_seed(927)
    results={}
    # Execute the evaluator's exact terminal branch, including its rows.append,
    # inside a one-iteration loop. Engine truth is fixture-only, never policy input.
    src=(ROOT/'scripts/evaluate_v7.py').read_text();tree=ast.parse(src)
    branch=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=='all(terms.values())')
    loop=ast.For(target=ast.Name(id='_',ctx=ast.Store()),iter=ast.List(elts=[ast.Constant(0)],ctx=ast.Load()),body=[branch],orelse=[])
    mod=ast.fix_missing_locations(ast.Module(body=[loop],type_ignores=[]))
    scope=dict(terms={'unit_0':True},infos={'unit_0':dict(winner=0,score_0=0.,score_1=0.)},agents=['unit_0'],seed=3001,team=0,step=5,rows=[])
    exec(compile(mod,'evaluate_v7.py:terminal-branch','exec'),scope)
    results['terminal_score']=dict(evidence='synthetic execution of actual evaluator AST',source_line=branch.lineno,
        expected_engine_final_normalized=[1.,.42],observed_saved=[scope['rows'][0]['score_0'],scope['rows'][0]['score_1']],
        contract_assertion_passed=[scope['rows'][0]['score_0'],scope['rows'][0]['score_1']]==[1.,.42])
    # Real upstream winner extraction, fake terminal step reward +0.1 in a draw.
    env=BlackOutEnv.__new__(BlackOutEnv);env.agents=['unit_0'];env._latest_winner=None;env._agent_name_cache={}
    env._collect_map_obs=lambda:None;env._resolve_behavior_names=lambda:['b']
    class Steps:
        agent_id=[0]
        def __getitem__(self,k):return SimpleNamespace(reward=.1)
    env._unity_env=SimpleNamespace(get_steps=lambda _: (SimpleNamespace(agent_id=[]),Steps()))
    env._extract_step=lambda *_:('unit_0',np.zeros(45,np.float32))
    env._preprocess=lambda *_:{};env._extract_scalars=lambda *_:{}
    env._collect_obs()
    results['draw_plus_shaping_winner']=dict(expected_engine_winner=-1,synthetic_terminal_reward=.1,actual_wrapper_winner=env._latest_winner,
        evidence='actual _collect_obs with mocked terminal Steps',contract_assertion_passed=env._latest_winner==-1)
    tracker=ScoreDeltaRewardTracker();tracker.update_points((97,42))
    terminal=tracker.update_normalized(0.,0.,terminated=True,winner=0)
    after=tracker.score;tracker.reset();fresh=tracker.update_points((0,0))
    results['score_delta']=dict(preterminal=[97,42],engine_final_fixture=[100,42],terminal_delta=list(terminal.score_delta),
        preserved_score=list(after),terminal_bonus=list(terminal.terminal_bonus),fresh_episode_delta=list(fresh.score_delta))
    timers=['episode_old','absorb_old'];trace=[]
    for i in range(len(timers)-1,-1,-1):
        trace.append(dict(index=i,timers=list(timers)))
        if timers[i]=='episode_old':
            timers.clear();timers.extend(['episode_new','absorb_new']);timers.pop(i)
    results['timer']=dict(evidence='Python mirror of C# callback/list ordering, not Unity execution',trace=trace,remaining=timers)
    pre=ObsPreprocessor(load_semantic_config(Path(blackout_env.__file__).parent/'semantic_map_config.json'),n_items=5,n_classes=3)
    raw=np.array([[[1/255],[1.5/255],[2/255]]],np.float32);graphic=pre.preprocess_graphic(raw)
    results['semantic_interpolation']=dict(raw_scaled=(raw[:,:,0]*255).tolist(),ids=graphic.argmax(-1).tolist(),one_hot_sum=graphic.sum(-1).tolist())
    model=V6Model(small_model());ctx=PlannerFeatures(0);obs=all_observations()
    normal=ctx.prepare(obs,model)[2]
    with patch.object(ctx.planner,'act',side_effect=PathNotFound('fixture one-unit route failure')):
        _,mask,failed,_=ctx.prepare(obs,model)
    results['team_stop_amplification']=dict(normal_nonzero_slots=int(np.any(normal!=0,axis=1).sum()),failure_nonzero_slots=int(np.any(failed!=0,axis=1).sum()),
        failures=ctx.failures,keep_legal=bool(mask[0]),evidence='injected planner exception through actual PlannerFeatures.prepare')
    head=GraphResidualHead(12,graph());x=torch.randn(2,5,12);opt=torch.optim.SGD(head.parameters(),lr=.1)
    grads=[]
    for index in range(2):
        opt.zero_grad();head(x).square().mean().backward()
        grads.append({k:float(getattr(head,k).grad.norm()) for k in ('edge_weight','bias')})
        if index == 0:opt.step()
    results['zero_init_gradient']=dict(two_synthetic_backward_steps=grads,updates_only_in_memory=True)
    # Config-driven intervention is absent from the real evaluation constructor.
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='DirectPolicy']
    results['evaluation_intervention']=dict(constructor_keywords=[[k.arg for k in n.keywords] for n in calls],
        missing_intervention=all('intervention' not in [k.arg for k in n.keywords] for n in calls))
    # A learned output bias still yields motion when rates are zeroed. Output block
    # cuts neural features, not necessarily the final actuator in F1.
    m=ReadoutModel(3)
    with torch.no_grad():m.readout.bias[1]=1
    policy=DirectPolicy(graph(),m,intervention='output_block');policy.reset_episode_state('fixture',0)
    act=policy.act(obs,episode_id='fixture',env_step_id=0,team_id=0)
    results['F1_output_block']=dict(rates=policy.last_features.tolist(),action=act['unit_0'].tolist(),
        evidence='synthetic learned bias; output_block zeroes rates before readout, not post-readout actions')
    # Demonstrate a 13-tick average is not a 13-tick pure delay.
    from dataclasses import replace
    brain=WholeBrain(graph(),config=replace(DynamicsConfig(),noise_hz=0.,tonic=0.,sensory_ema=0.,sensory_gain=1.))
    rates=[brain.step(np.ones((1,1),np.float32)).tolist() for _ in range(20)]
    nonzero=[i for i,r in enumerate(rates) if np.any(r)]
    results['pooling_step_response']=dict(first_nonzero_output_tick=nonzero[0] if nonzero else None,window_ticks=13,rates=rates,
        evidence='four-node synthetic graph, not MaleCNS timing measurement')
    (OUT/'minimal_repros.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in results.items() if k!='pooling_step_response'},ensure_ascii=False,indent=2))

if __name__=='__main__':run()
