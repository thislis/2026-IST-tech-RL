"""Offline fixtures only: no Unity, no real training run, no registered dev games."""

from project_paths import project_root, project_path

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from blackout_rl.v8.acceleration import compact_shared,validate_shared,vectorized_mask,installed
from blackout_rl.v8 import observation,candidates,telemetry
from blackout_rl.v8.journal import Journal
from blackout_rl.v8.window_codec import decode_window
from blackout_rl.v8.collector import Collector
from blackout_rl.v8.trainer import train
from blackout_rl.v8.checkpoints import CheckpointStore
from tests.v8 import test_collector as fixtures
from tests.v8.fixture_env import FixtureEnv
from tests.test_mappo_v6 import all_observations

SCRIPTS=project_root()/'scripts';sys.path.insert(0,str(SCRIPTS))
import v8_experiments as runner
import v8_study as study


class AccelerationTests(unittest.TestCase):
    def test_mask_exact_reasons_near_edges_and_both_teams(self):
        rng=np.random.default_rng(927)
        for i in range(80):
            obs=all_observations()
            pos=rng.uniform(-.02,1.02,(10,2)).astype(np.float32)
            if i%4==0:pos[:,0]=np.nextafter(np.float32(i%25/24),np.float32(0))
            for o in obs.values():
                o['vector'][:90].reshape(10,9)[:,:2]=pos
                o['graphic'][...,1]=rng.random((96,96))>.8
            alternatives=rng.uniform(-1,1,(5,8,2)).astype(np.float32)
            planner=alternatives[:,0,:].copy()
            for team in (0,1):
                a,ra=candidates.build_mask(obs,team,planner,alternatives)
                b,rb=vectorized_mask(obs,team,planner,alternatives)
                self.assertTrue(torch.equal(a,b));self.assertEqual(ra,rb)

    def test_shared_maps_are_cached_only_within_call(self):
        obs=all_observations()
        for i in range(5):obs[f'unit_{i}']['graphic']=obs['unit_0']['graphic']
        self.assertEqual(observation.compact(obs),compact_shared(obs))
        self.assertEqual(observation.validate_and_hash(obs,0),validate_shared(obs,0))
        before=validate_shared(obs,0)
        obs['unit_0']['graphic'][0,0,:]=0;obs['unit_0']['graphic'][0,0,1]=1
        self.assertNotEqual(before,validate_shared(obs,0))
        self.assertEqual(observation.validate_and_hash(obs,0),validate_shared(obs,0))
        obs['unit_1']['graphic']=obs['unit_1']['graphic'].copy();obs['unit_1']['graphic'][:]=0
        with self.assertRaises(ValueError):validate_shared(obs,0)

    def test_encoded_windows_match_reference_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'a';b=Path(tmp)/'b'
            def feed(t):
                for i in range(12):t.observe(dict(i=i,tensor=torch.tensor([i,2]),array=np.ones(2)),['collision'] if i%2 else [])
                t.close()
            feed(telemetry.Telemetry(a,window=2,max_pending=2,control_probability=.2,seed=77))
            with installed(b):
                from blackout_rl.v8 import collector
                feed(collector.Telemetry(b,window=2,max_pending=2,control_probability=.2,seed=77))
            lookup={}
            for line in (b/'window_rows.jsonl').read_text().splitlines():
                block=decode_window(json.loads(line))
                for i,row in zip(block['row_ids'],block['rows']):
                    self.assertNotIn(i,lookup);lookup[i]=row
            for name in ('windows.jsonl','counters.jsonl'):
                self.assertEqual([json.loads(x) for x in (a/name).read_text().splitlines()],
                                 [decode_window(json.loads(x),lookup) for x in (b/name).read_text().splitlines()])

    def test_writer_order_barrier_and_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'journal.jsonl';writer=Journal(capacity=2)
            for i in range(30):writer.append(p,dict(i=i))
            writer.flush();self.assertEqual([json.loads(x)['i'] for x in p.read_text().splitlines()],list(range(30)))
            writer.close();writer.close()
            with self.assertRaises(RuntimeError):writer.append(p,{})
            bad=Journal();bad.raw(Path(tmp),b'x')
            with self.assertRaises(RuntimeError):bad.flush()
            with self.assertRaises(RuntimeError):bad.close()

    def test_optimized_rollout_and_update_match_reference(self):
        fixture=fixtures.CollectorTests();cfg=fixture.config();torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            models=[]
            for accelerated in (False,True):
                d=Path(tmp)/str(accelerated);torch.manual_seed(927);p=fixture.policy(cfg);c=Collector(FixtureEnv(),p,cfg,11,d)
                if accelerated:
                    with installed(d):
                        c=Collector(FixtureEnv(),p,cfg,11,d);h=train(c,d)
                else:h=train(c,d)
                models.append(CheckpointStore(d/'checkpoints').load(h)[0])
            for k in models[0]['model']:
                torch.testing.assert_close(models[0]['model'][k],models[1]['model'][k],rtol=0,atol=0)
            self.assertEqual(models[0]['collector']['exposure'],models[1]['collector']['exposure'])

    def test_journal_crash_resume_and_graceful_stop_boundary(self):
        fixture=fixtures.CollectorTests();cfg=fixture.config();torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            with installed(tmp):
                c=Collector(FixtureEnv(fail_after=19),fixture.policy(cfg),cfg,11,tmp)
                with self.assertRaisesRegex(RuntimeError,'transport crash'):train(c,tmp)
            self.assertEqual(CheckpointStore(Path(tmp)/'checkpoints').load()[0]['global_step'],16)
            with installed(tmp):
                c=Collector(FixtureEnv(),fixture.policy(cfg),cfg,11,tmp);train(c,tmp,resume=True)
            rows=[json.loads(x) for x in (Path(tmp)/'exposure.jsonl').read_text().splitlines()]
            self.assertEqual([r['global_step'] for r in rows],list(range(1,25)))
        with tempfile.TemporaryDirectory() as tmp:
            indexes=[]
            with installed(tmp):
                c=Collector(FixtureEnv(),fixture.policy(cfg),cfg,11,tmp)
                train(c,tmp,stop_requested=lambda:True,on_checkpoint=lambda s,h:indexes.append(s))
            self.assertEqual(indexes,[0,8]);self.assertEqual(json.loads((Path(tmp)/'status.json').read_text())['state'],'paused')
            with installed(tmp):
                c=Collector(FixtureEnv(),fixture.policy(cfg),cfg,11,tmp);train(c,tmp,resume=True)
            self.assertEqual(json.loads((Path(tmp)/'status.json').read_text())['global_step'],24)
            with installed(tmp):
                c=Collector(FixtureEnv(),fixture.policy(cfg),cfg,11,tmp)
                train(c,tmp,resume=True,on_checkpoint=lambda s,h:indexes.append(s))
            self.assertEqual(indexes[-1],24)
            self.assertEqual(c.global_step,24)

if __name__=='__main__':unittest.main()
