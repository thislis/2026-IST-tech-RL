"""Exercise study selection and repeat-save without launching any experiment."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from blackout_v12 import study, opponents
from blackout_v12.io import ROOT, read_json


class SelectionSaved(Exception):
    pass


class SelectionSerializationTests(unittest.TestCase):
    def test_learned_winner_and_baseline_tie_save_and_resume(self):
        for learned_raw, expected in ((.9,'J'),(.5,'Q2')):
            with self.subTest(winner=expected), tempfile.TemporaryDirectory() as temporary:
                directory=Path(temporary)
                checkpoint=directory/'fixture.pt'
                checkpoint.write_bytes(b'no model is loaded')
                cfg=read_json(ROOT/'code/v12/configs/default.json')
                cfg.update(arms=['J'],seeds=[11],opponents=[],workers_per_run=2)

                def jobs(tasks,*args):
                    result={}
                    for task in tasks:
                        if task['kind']=='evaluate':
                            raw=learned_raw if task['id'].startswith('main_dev') else .5
                            result[task['id']]=dict(completeness=True,opponent_macro_raw=raw,
                                opponent_macro_score=.8,opponents={})
                        else:
                            result[task['id']]=dict(checkpoint=str(checkpoint),arm=task.get('arm','Q2'))
                    return result

                def progress(**row):
                    if row.get('phase') in ('P4_additional_seeds','P4_confirmation'):
                        raise SelectionSaved()

                with patch.object(study,'_checkpoint',return_value=str(checkpoint)), \
                     patch.object(study,'run_jobs',side_effect=jobs), \
                     patch.object(opponents,'dev_catalog',return_value=[]), \
                     patch.object(opponents,'confirmation_catalog',return_value=[]):
                    with self.assertRaises(SelectionSaved):
                        study.run_study(directory,cfg,progress=progress)
                    lock=directory/'selection_lock.json'
                    original=lock.read_bytes()
                    selected=read_json(lock)
                    self.assertEqual(selected['arm'],expected)
                    self.assertEqual(selected['decoder'],'map')
                    self.assertFalse(selected['test_used'])
                    self.assertTrue(all('options' not in option and 'seed' not in option
                                        for option in selected['options']))
                    with self.assertRaises(SelectionSaved):
                        study.run_study(directory,cfg,progress=progress)
                    self.assertEqual(lock.read_bytes(),original)


if __name__=='__main__': unittest.main()
