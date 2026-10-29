"""Data-only batching and legacy partial-batch migration; no Unity."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from blackout_v12 import collector, demonstrations, training
from blackout_v12.execution import compatible_manifest, schedule_allocation, semantic_config
from blackout_v12.io import atomic_torch, digest, file_hash, read_json


class DataParallelTests(unittest.TestCase):
    def test_data_uses_free_slots_while_ppo_stays_capped(self):
        cfg=dict(workers=10,workers_per_run=2,rollout_episodes_per_wave=2,parallel_train_runs=4)
        self.assertEqual(schedule_allocation([dict(kind='data')],cfg)[0]['workers'],10)
        mixed=schedule_allocation([dict(kind='train'),dict(kind='data')],cfg)
        self.assertEqual([task['workers'] for task in mixed],[2,8])
        ppo=schedule_allocation([dict(kind='train') for _ in range(4)],cfg)
        self.assertEqual([task['workers'] for task in ppo],[2,2,2,2])
        newer=dict(cfg,data_collection_workers=10)
        self.assertEqual(semantic_config(cfg),semantic_config(newer))
        self.assertTrue(compatible_manifest({'config':cfg},{'config':newer}))

    def test_legacy_partial_batch_finishes_before_expansion_without_seed_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);output=root/'run'
            cfg=dict(workers_per_run=2,rollout_episodes_per_wave=2,actor_threads=1,
                splits={'data':[1,999999]},data_split='data',
                opponents=[{'id':'scripted-noop'}],train_family_weights={'scripted':1.})
            generated=[];dispatches=[];interrupt=True

            def episode(policy,opponent,task,env_factory):
                generated.append(dict(task))
                return dict(natural_terminal=True,policy_sha256=task['policy_sha256'],
                    episode_id=task['id'],events=[],trip_ledger={'rows':[]},
                    env_steps=1,event_count=0,family=task['family'],side=task['side'])

            def wave(tasks,workers):
                dispatches.append((workers,[dict(task) for task in tasks]))
                rows=[]
                for index,task in enumerate(tasks):
                    if interrupt and task['id']=='w00001_01':
                        raise InterruptedError('mock cancellation after one cached episode')
                    rows.append(collector.run_episode(task))
                return rows

            with patch.object(training,'ROOT',root),patch.object(training,'run_wave',side_effect=wave), \
                    patch.object(collector,'collect_episode',side_effect=episode), \
                    patch('blackout_v12.opponents.make_opponent',return_value=object()), \
                    patch('blackout_v12.env.make_env') as env,patch.object(training,'update') as optimizer:
                with self.assertRaises(InterruptedError):
                    training.train(output,cfg,11,11,'Q2',workers=2)
                latest=output/'latest.pt'
                payload=torch.load(latest,weights_only=True)
                self.assertEqual(payload['counters']['episodes'],2)
                self.assertEqual(payload['counters']['wave'],1)
                # Exercise the original signature and missing scheduling audits.
                payload['signature']=digest(dict(config=cfg,seed=11,arm='Q2',budget=11,initial=None))
                atomic_torch(latest,payload)
                data_root=next((root/'artifacts/data/v12').iterdir())
                for audit in data_root.glob('wave_*/data_schedule.json'):
                    audit.unlink()
                protected=[path for path in data_root.rglob('*') if path.is_file() and
                           path.name in ('plan.json','behavior.pt','episode.pt','task.json')]
                original={path:file_hash(path) for path in protected}
                partial_plan=read_json(data_root/'wave_00001/plan.json')
                interrupt=False
                report=demonstrations.collect(output,dict(cfg,data_collection_workers=6),11,11,workers=6)
                env.assert_not_called();optimizer.assert_not_called()

            self.assertEqual([len(tasks) for _,tasks in dispatches],[2,2,2,6,6])
            self.assertEqual([workers for workers,_ in dispatches],[2,2,6,6,6])
            self.assertEqual(dispatches[2][1],partial_plan['tasks'])
            self.assertEqual({path:file_hash(path) for path in protected},original)
            self.assertEqual(report['episodes'],16)
            self.assertEqual(len(generated),16)  # Three cached episodes were not replayed.
            self.assertEqual(len({task['id'] for task in generated}),16)
            plans=[read_json(path) for path in sorted(data_root.glob('wave_*/plan.json'))]
            tasks=[task for plan in plans for task in plan['tasks']]
            seeds=[]
            for serial,task in enumerate(tasks):
                h=int(digest(['v12',11,serial])[:12],16)
                self.assertEqual(task['action_seed'],h%(2**31-1))
                self.assertEqual(task['requested_seed'],1+h%999999)
                seeds.append(task['action_seed'])
            self.assertEqual(len(set(seeds)),16)
            audits=[read_json(path) for path in sorted(data_root.glob('wave_*/data_schedule.json'))]
            self.assertEqual([(row['episode_serial_start'],row['episode_serial_end_exclusive'])
                              for row in audits],[(2,4),(4,10),(10,16)])
            recorded=[json.loads(line) for line in (output/'data_dispatches.jsonl').read_text().splitlines()]
            self.assertTrue(recorded[2]['reused_plan'])
            self.assertEqual(recorded[2]['episode_count'],2)
            self.assertEqual(recorded[3]['episode_count'],6)
            self.assertEqual(report['data_scheduling']['target_episode_batch_size'],6)
            self.assertFalse(report['eligible_for_ppo'])


if __name__=='__main__': unittest.main()
