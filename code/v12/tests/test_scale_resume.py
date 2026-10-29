"""Data-stage runtime migration preserves completed evidence; no Unity."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from blackout_v12 import resume, runner
from blackout_v12.io import atomic_json, digest, file_hash, read_json


class ScaleResumeTests(unittest.TestCase):
    def test_scale_retains_data_checkpoint_and_matches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); directory=root/'logs/v12/example'
            revision=directory/'runtime_revisions/0002'
            config=dict(workers=8,worker_cap=8,rollout_episodes_per_wave=2,
                        ppo={'minibatch_events':128},objective='provided')
            original=dict(config=config,source_sha256={},packages={},python='test',
                executable='python',feature_schema='feature',state_schema='state',objective='provided')
            atomic_json(directory/'registration.json',original)
            atomic_json(revision/'registration.json',original)
            atomic_json(revision/'runtime_profile.json',dict(selected={'workers':8}))
            atomic_json(directory/'resume_preparation.json',dict(state='ready',revision='runtime_revisions/0002',
                registration_sha256=digest(original),original_registration_sha256=file_hash(directory/'registration.json'),
                rollout_episodes_per_wave=2,retry_interrupted_matches=0))
            atomic_json(directory/'status.json',dict(state='stopped'))
            atomic_json(directory/'resolved_config.json',config)
            cell=directory/'evaluation/baseline/Q0/cells/one.json'
            atomic_json(cell,dict(valid=True))
            checkpoint=directory/'runs/data/latest.pt'
            checkpoint.parent.mkdir(parents=True); checkpoint.write_bytes(b'preserved checkpoint fixture')
            episode=root/'artifacts/data/v12/example/runs/data/wave_00000/episode_00/episode.pt'
            episode.parent.mkdir(parents=True); episode.write_bytes(b'preserved trajectory fixture')
            before={p:file_hash(p) for p in (cell,checkpoint,episode,directory/'registration.json')}
            evidence=root/'evidence.json'
            atomic_json(evidence,dict(selected_workers=14,throughput_only=True,evaluation_evidence=False,
                candidates=[{'workers':14}],selection_rule='fixture best',
                source_file_sha256=file_hash(Path(resume.__file__).with_name('throughput_probe.py'))))
            def register(cfg):
                result=copy.deepcopy(original); result['config']=copy.deepcopy(cfg); return result
            with patch.object(resume,'ROOT',root),patch.object(runner,'registration',side_effect=register):
                record=resume.prepare_scale(directory,evidence)
                self.assertEqual(record['data_collection_workers'],14)
                self.assertEqual(record['parallel_train_runs'],7)
                self.assertEqual(record['rollout_episodes_per_wave'],2)
                self.assertTrue(resume.check(directory)['ready'])
            self.assertEqual(before,{p:file_hash(p) for p in before})
            self.assertEqual(read_json(directory/'status.json')['state'],'stopped')
            self.assertEqual((directory/record['revision']/'previous_data_checkpoint.pt').read_bytes(),checkpoint.read_bytes())


if __name__=='__main__': unittest.main()
