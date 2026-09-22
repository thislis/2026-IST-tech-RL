"""Orchestration tests: reuse only complete paired results; preserve failure status."""
import fcntl
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts import pilot_evaluation as evaluation


class PilotEvaluationTests(unittest.TestCase):
    def test_reuse_requires_exact_checkpoint_and_full_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = dict(run=str(root), seed=11, maps=[41000, 41001], checkpoint_sha256='expected')
            result = dict(schema='blackout.v7.evaluation.v1', checkpoint_sha256='expected',
                          training_seed=11, split='dev', opponent='target',
                          episodes=[dict(seed=seed, team=team, winner=0)
                                    for seed in job['maps'] for team in (0, 1)])
            partial = root / 'eval_dev_target_1.progress.json'
            partial.write_text(json.dumps(result))
            self.assertIsNone(evaluation.completed_evaluation(job))
            file = root / 'eval_dev_target_2.json'
            file.write_text(json.dumps(result))
            self.assertEqual(evaluation.completed_evaluation(job)[0], file)
            result['episodes'][-1] = result['episodes'][0]
            file.write_text(json.dumps(result))
            self.assertIsNone(evaluation.completed_evaluation(job))
            result['episodes'][-1] = dict(seed=41001, team=1, winner=0)
            result['checkpoint_sha256'] = 'other'
            file.write_text(json.dumps(result))
            self.assertIsNone(evaluation.completed_evaluation(job))

    def test_worker_skips_complete_results_and_builds_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = dict(run=str(root), seed=11, experiment_id='test')
            result = dict(episodes=[dict(team=0, winner=0), dict(team=1, winner=-1)],
                          win_rate=.5, draw_rate=.5, paired_map_bootstrap_ci=[0., 1.])
            with (root/'evaluation.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch.object(evaluation, 'DIRECTORY', root), patch.object(evaluation, 'preflight', return_value=[job]), \
                     patch.object(evaluation, 'completed_evaluation', return_value=(root/'result.json', result)), \
                     patch.object(evaluation.subprocess, 'Popen') as popen, patch.object(evaluation.signal, 'signal'):
                    evaluation.worker(os.dup(lock.fileno()))
                    popen.assert_not_called()
            self.assertEqual(json.loads((root/'status.json').read_text())['status'], 'complete')
            self.assertTrue(json.loads((root/'summary.json').read_text())['complete'])
            self.assertFalse((root/'evaluation.pid').exists())

    def test_worker_stops_on_failure_without_success_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = dict(run=str(root), seed=11, experiment_id='test')
            with (root/'evaluation.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch.object(evaluation, 'DIRECTORY', root), patch.object(evaluation, 'preflight', return_value=[job, job]), \
                     patch.object(evaluation, 'completed_evaluation', return_value=None), \
                     patch.object(evaluation.subprocess, 'Popen') as popen, patch.object(evaluation.signal, 'signal'):
                    popen.return_value.wait.return_value = 1
                    with self.assertRaisesRegex(RuntimeError, 'evaluation failed'):
                        evaluation.worker(os.dup(lock.fileno()))
                    self.assertEqual(popen.call_count, 1)
            self.assertEqual(json.loads((root/'status.json').read_text())['status'], 'failed')
            self.assertFalse((root/'summary.json').exists())
            self.assertFalse((root/'evaluation.pid').exists())


if __name__ == '__main__':
    unittest.main()
