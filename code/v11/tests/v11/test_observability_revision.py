import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from blackout_v11.io import atomic_json, read_json, file_hash, digest
from blackout_v11.revision import create_evaluation_revision, create_observability_revision, validated_revision
from blackout_v11.runner import prepare_registration


class ObservabilityRevisionTests(unittest.TestCase):
    def fixture(self, root):
        original = dict(config={'arms':['R','G','F','D'],'seeds':[11,22,33],
                                'confirmation_seeds':[44,55],'training_steps':2097152},
                        packages={'torch':'same'},sources={'code/v11/blackout_v11/policy.py':'actor',
                            'code/v11/blackout_v11/runner.py':'initial'})
        before = copy.deepcopy(original)
        before['sources']['code/v11/blackout_v11/runner.py']='approved'
        for name,value in {'registration.json':original,'runtime_profile.json':{'same':True},
                           'task_plan.json':{'same':True},'status.json':{'state':'stopped','operation':'study'}}.items():
            atomic_json(root/name,value)
        with patch('blackout_v11.revision.registration',return_value=before):
            create_evaluation_revision(root,original['config'])
        atomic_json(root/'observability_revisions/stop-events-v1/before_source_registration.json',before)
        after=copy.deepcopy(before)
        after['sources']['code/v11/blackout_v11/runner.py']='logged'
        after['sources']['code/v11/blackout_v11/stop_events.py']='logger'
        return original,before,after

    def test_preserves_prior_registration_and_evaluation_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);original,before,after=self.fixture(root)
            old=file_hash(root/'evaluation_revision.json')
            with patch('blackout_v11.revision.registration',return_value=after):
                create_observability_revision(root,original['config'])
                validated_revision(root,original['config'])
            with patch('blackout_v11.runner.registration',return_value=after):
                self.assertEqual(prepare_registration(root,original['config']),digest(original))
            self.assertEqual(file_hash(root/'evaluation_revision.json'),old)
            self.assertEqual(read_json(root/'registration.json'),original)
            changed=copy.deepcopy(after);changed['sources']['code/v11/blackout_v11/runner.py']='later'
            with self.assertRaises(ValueError):validated_revision(root,original['config'],changed)

    def test_rejects_algorithm_or_config_changes(self):
        for kind in ['actor','config']:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);original,before,after=self.fixture(root)
                if kind=='actor':after['sources']['code/v11/blackout_v11/policy.py']='changed'
                else:after['config']['training_steps']=1
                with patch('blackout_v11.revision.registration',return_value=after):
                    with self.assertRaises(ValueError):create_observability_revision(root,original['config'])
                self.assertFalse((root/'observability_revision.json').exists())
