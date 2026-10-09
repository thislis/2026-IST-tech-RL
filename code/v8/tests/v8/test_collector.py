import json
from pathlib import Path
import tempfile
import unittest
import torch
from blackout_rl.v8.config import load_config,CAPABILITIES
from blackout_rl.v8.policy_factory import make_policy
from blackout_rl.v8.collector import Collector
from blackout_rl.v8.trainer import train
from blackout_rl.v8.checkpoints import CheckpointStore
from tests.test_ppo_training import small_model
from tests.v8.fixture_env import FixtureEnv

class CollectorTests(unittest.TestCase):
    def config(self):
        cfg=load_config('configs/v8/experiments/c1.yaml',require_build=False)
        cfg['training'].update(steps=24,rollout=8,save_every=8)
        cfg['training']['ppo'].update(epochs=1,minibatch_size=5,target_kl=None)
        cfg['telemetry'].update(window=2,max_pending=1)
        return cfg
    def policy(self,cfg):return make_policy(cfg,capabilities=CAPABILITIES,actor=small_model(),seed=11)
    def test_rollout_keeps_episode_and_recomputes_behavior_ratio(self):
        torch.set_num_threads(1);cfg=self.config();env=FixtureEnv(length=7);policy=self.policy(cfg)
        with tempfile.TemporaryDirectory() as directory:
            c=Collector(env,policy,cfg,11,directory)
            b,_=c.collect(3);self.assertEqual(env.reset_count,1);self.assertEqual(env.decision,3)
            with torch.no_grad():actual=policy.model.distribution(b['features'],b['valid']).log_prob(b['action'])
            torch.testing.assert_close(actual,b['old_log_prob'],rtol=0,atol=0)
            b,_=c.collect(5);self.assertEqual(env.reset_count,2);self.assertEqual(c.exposure.steps,8)
            self.assertEqual(int(b['terminated'].sum()),1)
            self.assertEqual(float(b['next_value'][b['terminated']][0]),0.)
            c.telemetry.close()
    def test_interrupted_partial_is_preserved_without_duplicate_exposure(self):
        torch.set_num_threads(1);cfg=self.config()
        with tempfile.TemporaryDirectory() as directory:
            c=Collector(FixtureEnv(fail_after=19),self.policy(cfg),cfg,11,directory)
            with self.assertRaisesRegex(RuntimeError,'transport crash'):train(c,directory)
            saved,old=CheckpointStore(Path(directory)/'checkpoints').load();self.assertEqual(saved['global_step'],16)
            resumed=Collector(FixtureEnv(),self.policy(cfg),cfg,11,directory)
            final=train(resumed,directory,resume=True)
            self.assertNotEqual(old,final);self.assertEqual(resumed.exposure.steps,24)
            rows=[json.loads(x) for x in (Path(directory)/'exposure.jsonl').read_text().splitlines()]
            self.assertEqual([x['global_step'] for x in rows],list(range(1,25)))
            self.assertTrue(list((Path(directory)/'recovery').glob('*exposure.jsonl')))
            self.assertEqual(CheckpointStore(Path(directory)/'checkpoints').load(old)[0]['global_step'],16)
