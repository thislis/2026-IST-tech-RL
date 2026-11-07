import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from blackout_v13.collector import Session
from blackout_v13.export import synthetic_observation
from blackout_v13.policy import Policy
from blackout_v13.critic import Critic
from blackout_v13.io import ROOT,read_json,cpu_state
from blackout_v13.evaluation import MapRegistry,schedule,summarize,paired_confirmation
from blackout_v13.study import new_learner,save_checkpoint,load_checkpoint


class FakeEnv:
    def __init__(self):self.resets=0;self.closed=False
    def observation(self):
        v,g=synthetic_observation();v[:,95]=1-self.step_count*.02/420
        rows={}
        for i in range(10):
            vec=v[i%5].numpy().copy()
            if i>=5:vec[:90].reshape(10,9)[:,2]*=-1
            rows[f'unit_{i}']=dict(vector=vec,graphic=g[0].permute(1,2,0).numpy())
        return rows
    def reset(self,seed=None):self.resets+=1;self.step_count=0;return self.observation(),{}
    def step(self,actions):
        self.step_count+=1;done=self.step_count==2
        names=[f'unit_{i}' for i in range(10)]
        info={n:dict(score_0=1.,score_1=0.,winner=1) for n in names}
        return self.observation(),{n:-10. for n in names},{n:done for n in names},{n:False for n in names},info
    def close(self):self.closed=True


class FakeOpponent:
    def __init__(self,spec,side,seed,episode):self.side=side
    def act(self,obs,step):return {f'unit_{i}':np.zeros(2,np.float32) for i in range(self.side*5,self.side*5+5)}


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg=read_json(ROOT/'code/v13/configs/default.json')
        self.cfg.update(script_geometry=False,training_opponents=[dict(id='fake')])

    def test_immediate_episode_refill(self):
        env=FakeEnv();s=Session(self.cfg,0,lambda:env,FakeOpponent)
        p=Policy();c=Critic()
        s.load(dict(version=0,policy=cpu_state(p),critics={'worker':cpu_state(c)}),'worker')
        result=s.collect(5)
        self.assertEqual(env.resets,3)
        self.assertEqual([r['done'] for r in result['rows']],[False,True,False,True,False])
        self.assertEqual(len(result['results']),2)
        # Global score and learner side, deliberately opposite the misleading raw rewards/winner field.
        self.assertEqual(result['results'][0]['outcome'],-1.)
        self.assertEqual(result['results'][1]['outcome'],1.)
        s.close();self.assertTrue(env.closed)

    def test_checkpoint_roundtrip_and_identity(self):
        learner=new_learner(self.cfg,'cpu')
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'latest.pt';progress=dict(stage=2,env_steps=50)
            save_checkpoint(path,learner,progress,self.cfg,'fixed')
            restored=new_learner(self.cfg,'cpu')
            self.assertEqual(load_checkpoint(path,restored,'fixed'),progress)
            for a,b in zip(learner.policy.parameters(),restored.policy.parameters()):torch.testing.assert_close(a,b)
            with self.assertRaises(ValueError):load_checkpoint(path,restored,'changed')

    def test_map_split_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            r=MapRegistry(Path(d)/'maps.sqlite3')
            self.assertTrue(r.add('one','train'));self.assertTrue(r.add('one','train'))
            self.assertFalse(r.add('one','dev'));r.close()

    def test_paired_schedule(self):
        tasks=schedule([dict(id='a'),dict(id='b')],5,30000000)
        self.assertEqual(len(tasks),20)
        for i in range(0,len(tasks),2):self.assertEqual(tasks[i]['seed'],tasks[i+1]['seed'])

    def test_teacher_bundle_and_scripted_session(self):
        from blackout_v13.opponents import Teacher
        from blackout_v13.actor import Actor
        v,g=synthetic_observation();teacher=Teacher()
        action,task=teacher.act(v.numpy(),g.numpy())
        self.assertEqual(action.shape,(5,2));self.assertEqual(task.shape,(5,2))
        self.assertTrue(np.isfinite(action).all());teacher.reset()
        config={**self.cfg,'script_geometry':True}
        env=FakeEnv();session=Session(config,0,lambda:env,FakeOpponent)
        learner=new_learner(config,'cpu')
        session.load(dict(version=0,policy=cpu_state(learner.policy),
                           critics={k:cpu_state(c) for k,c in learner.critics.items()}),'worker')
        chunk=session.collect(3,imitation=True)
        self.assertEqual(chunk['env_steps'],3)
        self.assertTrue(any(bool(r['bc_mask'].any()) for r in chunk['rows']))
        session.close()


if __name__=='__main__':unittest.main()
