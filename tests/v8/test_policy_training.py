import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from blackout_rl.batching import team_agents
from blackout_rl.scripted_fsm import ScriptedTeamController
from blackout_rl.mappo_v6 import ROLES
from blackout_rl.navigation import PathNotFound
from blackout_rl.v8.contracts import StepContext,ContractError
from blackout_rl.v8.config import CAPABILITIES
from blackout_rl.v8.policy_factory import make_policy
from blackout_rl.v8.planner_adapter import IsolatedPlanner
from blackout_rl.v8.candidates import execute
from blackout_rl.v8.trainer import update,DEFAULT_PPO
from tests.test_ppo_training import small_model
from tests.test_mappo_v6 import all_observations

class PolicyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.policy=make_policy({'policy':{}},capabilities=CAPABILITIES,actor=small_model(),seed=11)
        self.context=StepContext('r','e','ep',0,0,0.,3001,0)
        self.obs=all_observations();self.policy.reset(self.context)
    def test_retry_canonical_order_and_distinct_equal_frame(self):
        first=self.policy.decide(self.obs,self.context,behavior=True);rng=self.policy.generator.get_state().clone()
        again=self.policy.decide(dict(reversed(list(self.obs.items()))),self.context,behavior=True)
        self.assertEqual(first['action'],again['action']);self.assertTrue(torch.equal(rng,self.policy.generator.get_state()))
        self.policy.decide(self.obs,replace(self.context,decision_id=1),behavior=True)
        self.assertEqual(self.policy.context.planner._step,2)
        with self.assertRaises(ContractError):self.policy.decide(self.obs,replace(self.context,decision_id=3))
        with self.assertRaises(ContractError):self.policy.decide(self.obs,replace(self.context,env_id='other',decision_id=2))
    def test_same_id_different_observation_is_error(self):
        self.policy.decide(self.obs,self.context)
        obs=copy.deepcopy(self.obs);obs['unit_0']['vector'][-1]-=.001
        with self.assertRaises(ContractError):self.policy.decide(obs,self.context)
    def test_noop_is_not_teacher_label_for_unreachable(self):
        original=ScriptedTeamController._navigate
        def injected(controller,state):
            if state.agent=='unit_0':raise PathNotFound('one-slot fixture')
            return original(controller,state)
        reference=IsolatedPlanner(0);expected=reference.act(self.obs,team_agents(0))
        with patch.object(ScriptedTeamController,'_navigate',injected):actual=self.policy.decide(self.obs,self.context)
        self.assertFalse(actual['teacher_valid']['unit_0']);np.testing.assert_array_equal(actual['actions']['unit_0'],[0,0])
        for name in team_agents(0)[1:]:np.testing.assert_array_equal(actual['actions'][name],expected[name])
    def test_normal_planner_matches_legacy_both_sides(self):
        for side in (0,1):
            old=ScriptedTeamController(side,roles=ROLES,chase_radius_cells=48);new=IsolatedPlanner(side)
            a=old.act(self.obs,team_agents(side));b=new.act(self.obs,team_agents(side))
            for name in a:np.testing.assert_array_equal(a[name],b[name])
    def test_bad_factory_setting_cannot_be_ignored(self):
        for setting in [dict(intervention='sensory_block'),dict(sensor_interval=9),dict(controlled_slots=[0])]:
            with self.assertRaises(ContractError):make_policy({'policy':setting},capabilities=CAPABILITIES,actor=small_model())
        with self.assertRaises(ContractError):make_policy({'policy':{}},capabilities={},actor=small_model())
    def test_zero_map_and_silent_executor_rewrite_are_errors(self):
        bad=copy.deepcopy(self.obs);bad['unit_0']['graphic'][:]=0
        with self.assertRaises(ContractError):self.policy.decide(bad,self.context)
        with self.assertRaises(ContractError):execute(1,np.zeros((5,2)),np.zeros((5,8,2)),torch.zeros(40,dtype=torch.bool))
    def test_small_remainder_ppo_and_frozen_encoder(self):
        model=self.policy.model;n=3;features=torch.randn(n,5,81);valid=torch.ones(n,40,dtype=torch.bool)
        with torch.no_grad():
            dist=model.distribution(features,valid);actions=dist.sample();logp=dist.log_prob(actions)
        batch=dict(features=features,valid=valid,action=actions,old_log_prob=logp,central=torch.randn(n,299),old_value=torch.zeros(n),
                   advantage=torch.tensor([-1.,0.,1.]),return_=torch.tensor([-.2,.1,.7]),behavior_version=torch.zeros(n,dtype=torch.long))
        before={k:v.clone() for k,v in model.actor_model.state_dict().items()}
        optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-4)
        metrics=update(model,optimizer,batch,dict(DEFAULT_PPO,epochs=2,target_kl=None))
        self.assertEqual(metrics['minibatches'],2);self.assertGreater(metrics['module_parameter_delta']['correction'],0)
        for k,v in before.items():torch.testing.assert_close(v,model.actor_model.state_dict()[k],rtol=0,atol=0)
        batch['behavior_version'][1]=1
        with self.assertRaises(ContractError):update(model,optimizer,batch,DEFAULT_PPO)
