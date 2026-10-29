"""Short synthetic replay/export checks; no Unity, real training or matches."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from blackout_v12.assignment import sample_assignment
from blackout_v12.collector import collect_episode
from blackout_v12.critic import Critic, critic_features
from blackout_v12.evaluation import summarize
from blackout_v12.export import export_policy, load_export
from blackout_v12.map_splits import MapSeedMismatch, MapSplitCollision, register_map
from blackout_v12.model import TaskScorer
from blackout_v12.policy import MyPolicy
from blackout_v12.ppo import event_log_prob, update
from blackout_v12.rewards import gae_episode, shape_episode
from blackout_v12.training import make_batch
from test_policy import fixture


class LearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_frozen_replay_ratio_and_ppo_gradient(self):
        torch.manual_seed(17)
        scorer, critic = TaskScorer(), Critic()
        batch=dict(features=torch.randn(4,5,3,64),prior=torch.randn(4,5,3),
                   legal=torch.ones(4,5,3,dtype=torch.bool),
                   capacities=torch.tensor([[2,2,5]]).expand(4,-1),temperature=torch.ones(4))
        with torch.no_grad():
            scores=scorer(batch['features'],batch['prior'])
            batch['assignment']=sample_assignment(scores,batch['capacities'],batch['legal'])
            batch['old_log_prob']=event_log_prob(scorer,batch,'J').detach()
        batch.update(critic_features=critic_features(batch['features'],batch['legal']),
                     on_policy=torch.ones(4,dtype=torch.bool),
                     advantages=torch.tensor([-2.,-.5,.5,2.]),returns=torch.tensor([1.,-.4,.5,-1.]))
        before=scorer.layers[-1].weight.detach().clone()
        report=update(scorer,critic,torch.optim.Adam(scorer.parameters(),lr=1e-3),
                      torch.optim.Adam(critic.parameters(),lr=1e-3),batch,'J',{'ppo_epochs':1})
        self.assertEqual(report['old_new_logprob_max_error'],0.)
        self.assertEqual(report['updates'],1)
        self.assertGreater(float((scorer.layers[-1].weight.detach()-before).abs().max()),0.)
        batch['on_policy'][0]=False
        with self.assertRaisesRegex(ValueError,'foreign-policy'):
            update(scorer,critic,None,None,batch,'J',{})

    def test_natural_duration_and_missing_terminal_score(self):
        vector,graphic=fixture()
        def observations(t):
            v=vector.copy();v[:,95]-=.01*t
            if t:
                e=v[:,:90].reshape(5,10,9);e[:,0,3]=0;e[:,0,4]=1
            return {f'unit_{i}':dict(vector=v[i%5].copy(),graphic=graphic[i%5].transpose(1,2,0).copy())
                    for i in range(10)}
        class Env:
            t=0
            closed=False
            def reset(self,seed=None): return observations(0),{}
            def step(self,actions):
                self.t+=1;done=self.t==3
                rewards={f'unit_{i}':float(done and i==0) for i in range(10)}
                terms={n:done for n in rewards};truncs={n:False for n in rewards}
                return ({} if done else observations(self.t)),rewards,terms,truncs,{}
            def close(self): self.closed=True
        class Policy:
            t=0
            def reset(self): self.t=0
            def step(self,v,g,deterministic=False):
                event=self.t in (0,2);self.t+=1
                decision=dict(features=torch.zeros(5,2,64),prior=torch.zeros(5,2),
                    legal=torch.ones(5,2,dtype=torch.bool),capacities=torch.tensor([5,5]),
                    assignment=torch.zeros(5,dtype=torch.long),log_prob=torch.tensor(-1.),
                    temperature=1.,task_ids=['collect:0','hold'],destinations=torch.zeros(5,2,dtype=torch.long))
                return dict(event=event,decision=decision,action=torch.zeros(5,2))
        class Opponent:
            def act(self,obs,names):return {n:np.zeros(2,np.float32) for n in names}
        with tempfile.TemporaryDirectory() as temporary:
            task=dict(side=0,config={'map_split_registry':str(Path(temporary)/'maps.json')},
                      requested_seed=1,action_seed=2,id='synthetic',opponent={'id':'noop'})
            env=Env();episode=collect_episode(Policy(),Opponent(),task,lambda:env)
        self.assertTrue(env.closed)
        self.assertEqual([e['duration'] for e in episode['events']],[2,1])
        self.assertEqual(episode['raw_points'],1.)
        self.assertFalse(episode['score_valid'])
        self.assertIsNone(episode['score_after_terminal'])
        self.assertGreater(episode['trip_ledger']['ambiguous'],0)
        batch=make_batch([episode],Critic())
        self.assertEqual(len(batch['returns']),2)
        episode['natural_terminal']=False
        with self.assertRaisesRegex(ValueError,'complete on-policy'):
            make_batch([episode],Critic())
        rewards=shape_episode([.1,-.1],[2,3],1,gamma=.9)
        advantages,_=gae_episode(rewards,[0.,0.],[2,3],gamma=.9,lam=1.)
        self.assertAlmostEqual(float(advantages[0]),.9**4-.1,places=6)

    def test_map_split_collision_and_failed_score_denominator(self):
        with tempfile.TemporaryDirectory() as temporary:
            config={'map_split_registry':str(Path(temporary)/'maps.json')}
            register_map(config,'train','battery-storage-wall','observation-a',1,0)
            register_map(config,'train','battery-storage-wall','observation-b',2,1)
            with self.assertRaises(MapSeedMismatch):
                register_map(config,'train','different-battery-storage','observation-d',1,1)
            with self.assertRaises(MapSplitCollision):
                register_map(config,'test','battery-storage-wall','observation-c',3,0)
        spec={'id':'scripted-noop','family':'scripted'}
        rows=[dict(opponent=spec,valid=True,raw_points=1.,score_points=1.,score_valid=True,side=0),
              dict(opponent=spec,valid=False,raw_points=0.,score_points=None,score_valid=False,side=1)]
        summary=summarize(rows)
        self.assertEqual(summary['opponents']['scripted-noop']['score'],.5)
        self.assertFalse(summary['completeness'])

    def test_two_file_provided_loader_sequence(self):
        vector,graphic=fixture()
        policy=MyPolicy(arm='F',decoder='map').eval()
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)/'submission'
            export_policy(policy,directory)
            loaded=load_export(directory)
            for t in range(3):
                v=vector.copy();v[:,95]-=t*.02/420
                obs={f'unit_{i}':dict(vector=v[i],graphic=graphic[i].transpose(1,2,0)) for i in range(5)}
                received=loaded.act(obs)
                np.testing.assert_allclose(policy(v,graphic).numpy(),np.stack(list(received.values())),atol=1e-6)
            self.assertEqual({p.name for p in directory.iterdir()},{'policy.py','checkpoint.pt'})
            self.assertEqual(policy.runtime_state()['step_count'],loaded._net.runtime_state()['step_count'])
            checkpoint_bytes=(directory/'checkpoint.pt').stat().st_size
            self.assertLess(checkpoint_bytes,256*1024)
            print(f'synthetic_export_checkpoint_bytes={checkpoint_bytes}')


if __name__=='__main__':unittest.main()
