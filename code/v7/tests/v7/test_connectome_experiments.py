"""Regression checks for skip/resume and fail-fast combined orchestration."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import connectome_experiments as combined
from scripts import pilot_evaluation as evaluation


class CombinedExperimentTests(unittest.TestCase):
    def test_pipeline_resumes_only_incomplete_run_then_evaluates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            complete = dict(run=str(root / 'done'), seed=11, experiment_id='done', complete=True)
            partial = dict(run=str(root / 'partial'), seed=22, experiment_id='partial', complete=False,
                           checkpoint_sha256='saved', config='/registered.yaml', script='run_v7_2.py')
            result = dict(episodes=[dict(team=0, winner=0), dict(team=1, winner=-1)],
                          win_rate=.5, draw_rate=.5, paired_map_bootstrap_ci=[0., 1.])
            with (root / 'lock').open('a') as lock, \
                    patch.object(evaluation, 'DIRECTORY', root), patch.object(evaluation, 'PIPELINE', True), \
                    patch.object(evaluation, 'preflight', side_effect=[[complete, partial], [complete, partial]]), \
                    patch.object(evaluation, 'completed_evaluation', return_value=(root / 'result.json', result)), \
                    patch.object(evaluation.signal, 'signal'), patch.object(evaluation.subprocess, 'Popen') as popen:
                popen.return_value.wait.return_value = 0
                evaluation.worker(os.dup(lock.fileno()))
                self.assertEqual(popen.call_count, 1)
                command = popen.call_args.args[0]
                self.assertIn('--resume', command)
                self.assertEqual(command[command.index('--run-dir') + 1], partial['run'])
            self.assertEqual(json.loads((root / 'status.json').read_text())['status'], 'complete')
            self.assertEqual(len(json.loads((root / 'summary.json').read_text())['runs']), 2)

    def test_training_failure_never_starts_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = dict(run=str(root), seed=11, complete=False, config='/config.yaml',
                       script='run_v7_2.py', checkpoint_sha256=None)
            with (root / 'lock').open('a') as lock, \
                    patch.object(evaluation, 'DIRECTORY', root), patch.object(evaluation, 'PIPELINE', True), \
                    patch.object(evaluation, 'preflight', return_value=[job]), \
                    patch.object(evaluation, 'completed_evaluation') as completed, \
                    patch.object(evaluation.signal, 'signal'), patch.object(evaluation.subprocess, 'Popen') as popen:
                popen.return_value.wait.return_value = 1
                with self.assertRaisesRegex(RuntimeError, 'training failed'):
                    evaluation.worker(os.dup(lock.fileno()))
                completed.assert_not_called()
                self.assertNotIn('--resume', popen.call_args.args[0])
            self.assertEqual(json.loads((root / 'status.json').read_text())['status'], 'failed')

    def test_combined_failure_prevents_second_suite_and_removes_pid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (root / 'lock').open('a') as lock, patch.object(combined, 'DIRECTORY', root), \
                    patch.object(combined.signal, 'signal'), patch.object(combined.subprocess, 'Popen') as popen:
                popen.return_value.wait.return_value = 1
                with self.assertRaisesRegex(RuntimeError, 'v7-1 failed'):
                    combined.worker(os.dup(lock.fileno()))
                self.assertEqual(popen.call_count, 1)
            self.assertEqual(json.loads((root / 'status.json').read_text())['status'], 'failed')
            self.assertFalse(json.loads((root / 'summary.json').read_text())['complete'])
            self.assertFalse((root / 'experiment.pid').exists())

    def test_combined_collects_both_summaries_sequentially(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for suite in ('v7_1', 'v7_2'):
                output = root / f'logs/v7/pilot_{suite}_dev_evaluation'
                output.mkdir(parents=True)
                (output / 'summary.json').write_text(json.dumps(dict(complete=True, runs=[suite])))
            with (root / 'lock').open('a') as lock, patch.object(combined, 'ROOT', root), \
                    patch.object(combined, 'SUITE_DIRECTORIES', {suite: root / f'logs/v7/pilot_{suite.replace("-", "_")}_dev_evaluation' for suite in combined.SUITES}), \
                    patch.object(combined, 'DIRECTORY', root), patch.object(combined.signal, 'signal'), \
                    patch.object(combined.subprocess, 'Popen') as popen:
                popen.return_value.wait.return_value = 0
                combined.worker(os.dup(lock.fileno()))
                calls = [call.args[0] for call in popen.call_args_list]
                self.assertEqual([cmd[cmd.index('--suite') + 1] for cmd in calls], ['v7-1', 'v7-2'])
                self.assertTrue(all('--foreground' in cmd for cmd in calls))
            self.assertEqual(json.loads((root / 'status.json').read_text())['status'], 'complete')
            summary = json.loads((root / 'summary.json').read_text())
            self.assertTrue(summary['complete'])
            self.assertEqual(set(summary['suites']), {'v7-1', 'v7-2'})


if __name__ == '__main__':
    unittest.main()
