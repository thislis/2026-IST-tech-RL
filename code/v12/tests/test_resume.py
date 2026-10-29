"""Runtime migration and allocation fixtures; no Unity or real experiment."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from blackout_v12.execution import compatible_manifest, schedule_allocation, semantic_config
from blackout_v12.io import atomic_json, file_hash, read_json
from blackout_v12 import resume, runner


class ResumeTests(unittest.TestCase):
    def test_resources_do_not_change_science_or_fixed_episode_wave(self):
        original=dict(workers=4,workers_per_run=2,parallel_runs=2,ppo={'minibatch_events':128},seed=11)
        changed=dict(original,workers=8,workers_per_run=4,rollout_episodes_per_wave=2,
                     parallel_train_runs=4,worker_allocation_file='runtime.json')
        self.assertEqual(semantic_config(original),semantic_config(changed))
        self.assertTrue(compatible_manifest({'config':original,'workers':2},{'config':changed,'workers':4}))
        changed['rollout_episodes_per_wave']=4
        self.assertFalse(compatible_manifest({'config':original},{'config':changed}))

    def test_allocation_preserves_remainder_and_limits_training_waves(self):
        cfg=dict(workers=5,parallel_runs=2,parallel_train_runs=4,rollout_episodes_per_wave=2)
        evaluations=[dict(id=str(i),kind='evaluate') for i in range(3)]
        self.assertEqual([t['workers'] for t in schedule_allocation(evaluations,cfg)],[3,2])
        self.assertEqual([t['workers'] for t in schedule_allocation(evaluations[:1],cfg)],[5])
        training=[dict(id=str(i),kind='train') for i in range(9)]
        allocated=schedule_allocation(training,dict(cfg,workers=8))
        self.assertEqual([t['workers'] for t in allocated],[2,2,2,2])

    def test_prepare_preserves_valid_matches_then_tunes_without_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); directory=root/'logs/v12/example'
            cfg=dict(workers='auto',worker_cap=8,parallel_runs=2,ppo={'minibatch_events':128})
            original=dict(config=cfg,source_sha256={'code/v12/blackout_v12/runtime.py':'old'},
                packages={},python='test',executable='python',feature_schema='feature',state_schema='state',objective='provided')
            atomic_json(directory/'registration.json',original)
            atomic_json(directory/'resolved_config.json',dict(cfg,workers=4,workers_per_run=2))
            atomic_json(directory/'runtime_profile.json',dict(selected={'workers':4,'minibatch_events':128}))
            atomic_json(directory/'status.json',dict(state='stopped'))
            folder=directory/'evaluation/baseline/Q0_map'
            valid=folder/'cells/valid.json'; cancelled=folder/'cells/cancelled.json'; failed=folder/'cells/failed.json'
            atomic_json(valid,dict(valid=True,env_steps=100))
            atomic_json(cancelled,dict(valid=False,error='evaluation interrupted'))
            atomic_json(failed,dict(valid=False,error='watchdog'))
            atomic_json(folder/'job_result.json',dict(completeness=False))
            original_hash=file_hash(directory/'registration.json')
            valid_hash,failed_hash=file_hash(valid),file_hash(failed)
            evidence=root/'tuning_plan.json'
            atomic_json(evidence,dict(selected_workers=4,measurement_pending=True))
            def registered(config):
                result=copy.deepcopy(original); result['config']=copy.deepcopy(config)
                result['source_sha256']['code/v12/blackout_v12/runtime.py']='new'
                return result
            with patch.object(resume,'ROOT',root),patch.object(runner,'registration',side_effect=registered):
                record=resume.prepare(directory,4,evidence)
                self.assertEqual(record['preserved_valid_matches'],1)
                self.assertEqual(record['retry_interrupted_matches'],1)
                self.assertEqual(file_hash(valid),valid_hash)
                self.assertEqual(file_hash(failed),failed_hash)
                self.assertFalse(cancelled.exists())
                self.assertFalse((folder/'job_result.json').exists())
                self.assertTrue(resume.check(directory)['measurement_pending'])
                atomic_json(evidence,dict(selected_workers=6,throughput_only=True,
                    evaluation_evidence=False,selection_rule='fixture only; no Unity started'))
                selected=resume.activate_tuning(directory,evidence)
                self.assertEqual(selected['workers'],6)
                self.assertEqual(selected['rollout_episodes_per_wave'],2)
                self.assertFalse(resume.check(directory)['measurement_pending'])
            self.assertEqual(file_hash(directory/'registration.json'),original_hash)
            self.assertEqual(read_json(directory/'status.json')['state'],'stopped')


if __name__=='__main__': unittest.main()
